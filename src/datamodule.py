""" Lightning Data Module for all data-related code """

import math
import shutil
from pathlib import Path
from typing import Literal, List, Callable
import torch
import polars as pl
import lightning as L
import pyarrow.dataset as ds
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from src.pipeline import DataPipeline
from src.dataset import LMDBDataset, FinetuneLMDBDataset
from src.collate_fn import Collate, MaskCollate, CensorCollate
from src.utils import (
    get_background_length,
    calculate_abspos,
    create_weights,
    flex_attn_padding,
)
from src.paths import FPATH, check_and_copy_file_or_dir
from torch.nn.attention.flex_attention import create_block_mask


# pylint: disable=arguments-differ
class BaseLightningDataModule(L.LightningDataModule):
    """Base Lightning Data Module for shared pretrain and finetune code"""

    def __init__(
        self,
        dir_path: Path,
        sources: List[ds.Dataset],
        background: pl.DataFrame,
        cls_token: bool,
        sep_token: bool,
        train_person_ids,
        val_person_ids,
        segment=False,
        fill_nulls=False,
        subset_background=False,
        collate_method="flatten_and_expand",
        num_workers=0,
        batch_size=128,
        max_seq_len=512,
        cutoff=0,
    ):
        super().__init__()
        # Init data related stuff
        self.fill_nulls = fill_nulls
        self.sources = sources
        self.background = background
        self.segment = segment
        self.train_person_ids = train_person_ids
        self.val_person_ids = val_person_ids
        # Init Path related stuff
        self.dir_path = dir_path
        check_and_copy_file_or_dir(self.dir_path)

        if (pipeline_path := dir_path / "pipeline.pt").exists():
            print("Loading old pipeline")
            self.pipeline = torch.load(pipeline_path, weights_only=False)
        else:

            print("Creating new pipeline")
            self.pipeline = DataPipeline(
                cls_token=cls_token,
                sep_token=sep_token,
                fill_nulls=fill_nulls,
                subset_background=subset_background,
                cutoff=cutoff,
            )

        # Check and copy files between dirs
        self.dir_path.mkdir(parents=True, exist_ok=True)

        # Init other arg-related stuff
        self.collate_method = collate_method
        self.num_workers = num_workers
        self.batch_size = batch_size

        # Init length-related stuff
        self.background_length = (
            get_background_length(background) + int(cls_token) + int(sep_token)
        )
        self.truncate_length = max_seq_len - self.background_length
        self.cls_token = cls_token

        # Avoid lint complaints
        self.dataset = None
        self.train_dataset, self.val_dataset, self.predict_dataset = None, None, None

    def prepare_data(self):
        """Not on all workers"""
        features_df = self.pipeline(self.sources, self.background, self.dir_path)
        torch.save(self.pipeline, self.dir_path / "pipeline.pt")

        # This reuses lmdb if exists or creates
        self.dataset = self._create_dataset(features_df, self.dir_path / "dataset.lmdb")

    def _create_dataset(self, df: pl.LazyFrame, path: Path):
        return LMDBDataset(df, path)

    def setup(self, stage: Literal["fit"] = None):
        """Defaults random splitting"""
        train, val = self.dataset.split_by_person_ids(
            self.train_person_ids, self.val_person_ids
        )
        self.train_dataset = train
        self.val_dataset = val

    def collate_fn(self):
        """Returns the Collate function for the DataModule"""
        return Collate(
            self.truncate_length,
            self.background_length,
            segment=self.segment,
        )

    def get_dataloader(self, dataset: LMDBDataset, sampler=None):
        """Returns a generic DataLoader with given attributes from self and kwargs"""
        sampler = BatchSampler(
            range(len(dataset)) if sampler is None else sampler,
            batch_size=self.batch_size,
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=False,
            collate_fn=self.collate_fn(),
            sampler=sampler,
            pin_memory=True,
        )

    def train_dataloader(self):
        """Returns the train dataloader for self.train_dataset"""
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self):
        """Returns the val dataloader for self.val_dataset"""
        return self.get_dataloader(self.val_dataset)

    def predict_dataloader(self):
        """Returns the prediction dataloader for self.predict_dataset"""
        return self.get_dataloader(self.predict_dataset)

    def get_steps_per_train_epoch(self):
        """Returns length of dataloader (calls setup and teardown)"""
        if self.train_dataset is None:
            self.setup()
        return math.ceil(len(self.train_dataset) / self.batch_size)

    def teardown(self, stage: str = None):
        """Copies all contents from dir_path to opposite drive if they do not exist."""
        swapped_path = FPATH.swap_drives(self.dir_path)
        swapped_path.mkdir(parents=True, exist_ok=True)

        for item in self.dir_path.iterdir():
            dest = swapped_path / item.name
            if item.is_dir():
                if not dest.exists():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                if not dest.exists():
                    shutil.copy2(item, dest)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, v in batch.items():
            if torch.is_tensor(v):
                batch[key] = v.to(device, non_blocking=True)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        bs, seq_len = batch["event"].shape
        # Create block mask
        batch["attn_mask"] = create_block_mask(
            flex_attn_padding(batch["last_data_idx"]),
            bs,
            None,
            seq_len,
            seq_len,
            _compile=True,
        )
        return batch


class PretrainLifeLightningDataModule(BaseLightningDataModule):
    """Lightning Data Module for MLM pretraining"""

    def collate_fn(self):
        """Returns the Collate function for the DataModule"""
        return MaskCollate(
            vocab=self.pipeline.vocab,
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )


class FinetuneLifeLightningDataModule(BaseLightningDataModule):
    """Lightning Data Module for binary finetuning"""

    def __init__(
        self,
        *args,
        outcomes: pl.DataFrame,
        inference_type: str = "test",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = outcomes  # .with_columns(
        # pl.col("censor").alias("filtering_date"), calculate_abspos(pl.col("censor"))
        # )  # .to_dict(as_series=False)
        self.inference_type = inference_type

    def _create_dataset(self, df: pl.LazyFrame, path: Path):
        outcome_dict = self.outcomes.with_columns(
            calculate_abspos(pl.col("censor"))
        ).to_dict(as_series=False)
        return FinetuneLMDBDataset(df, path, outcome_dict)

    def collate_fn(self):
        """Returns the Collate function for the DataModule"""
        return CensorCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )

    def setup(self, stage: Literal["fit"] = None):
        """On all workers"""
        if stage == "fit" or stage is None:
            dev = self.outcomes.filter(pl.col("censor").dt.year() < 2016)
            dev_dataset = self.dataset.subset(
                dev.with_columns(calculate_abspos(pl.col("censor"))).to_dict(
                    as_series=False
                )
            )
            train, val = dev_dataset.split_by_person_ids(
                self.train_person_ids, self.val_person_ids
            )
            self.train_dataset = train
            self.val_dataset = val

        if stage == "predict":
            if self.inference_type == "test":
                # Define what dataset you want to use for prediction (e.g., test or new data)
                predict = self.outcomes.filter(
                    pl.col("censor").dt.year() == 2016
                )  # Example: using 2016 data
                self.predict_dataset = self.dataset.subset(
                    predict.with_columns(calculate_abspos(pl.col("censor"))).to_dict(
                        as_series=False
                    )
                )
            elif self.inference_type == "val":
                dev = self.outcomes.filter(pl.col("censor").dt.year() < 2016)
                dev_dataset = self.dataset.subset(
                    dev.with_columns(calculate_abspos(pl.col("censor"))).to_dict(
                        as_series=False
                    )
                )
                train, val = dev_dataset.split_by_person_ids(
                    self.train_person_ids, self.val_person_ids
                )
                self.predict_dataset = val
            else:
                raise Exception(
                    "Unknown inference_type. Only supports 'test' or 'val' at current moment"
                )

    def train_dataloader(self):
        train_outcomes = self.train_dataset.observations["target"]
        weights = create_weights(train_outcomes, op=math.sqrt)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_outcomes), replacement=True
        )
        return self.get_dataloader(self.train_dataset, sampler)
