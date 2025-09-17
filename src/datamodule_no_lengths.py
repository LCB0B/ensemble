""" Lightning Data Module for all data-related code """

import math
import shutil
from pathlib import Path
from typing import List, Literal

import lightning as L
import polars as pl
import pyarrow.dataset as ds
import torch
from flash_attn.bert_padding import unpad_input
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from src.collate_fn import CensorCollate, Collate, MaskCollate
from src.dataset import FinetuneLMDBDataset, LMDBDataset
from src.paths import FPATH, check_and_copy_file_or_dir
from src.pipeline import DataPipeline
from src.sampler import UnpadSampler
from src.utils import (
    calculate_abspos,
    create_weights,
    get_background_length,
    set_posix_windows,
)


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
        n_tokens=8e5,
        max_seq_len=512,
        cutoff=0,
        source_dir=None,
        lengths="lengths",
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
        self.source_dir = source_dir
        self.lengths = lengths

        if (pipeline_path := dir_path / "pipeline.pt").exists():
            print("Loading old pipeline")
            print(pipeline_path)
            try:
                self.pipeline = torch.load(pipeline_path, weights_only=False)
            except NotImplementedError as e:
                if "cannot instantiate 'PosixPath'" in str(e):
                    with set_posix_windows():
                        self.pipeline = torch.load(pipeline_path, weights_only=False)
                else:
                    raise e
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
        self.n_tokens = n_tokens

        # Init length-related stuff
        self.background_length = (
            get_background_length(background) + int(cls_token) + int(sep_token)
        )
        self.truncate_length = max_seq_len - self.background_length
        self.cls_token = cls_token
        self.max_seq_len = max_seq_len

        # Avoid lint complaints
        self.dataset = None
        self.train_dataset, self.val_dataset, self.predict_dataset = None, None, None

    def prepare_data(self):
        """Not on all workers"""
        features_df = self.pipeline(
            self.sources, self.background, self.dir_path, self.source_dir
        )
        torch.save(self.pipeline, self.dir_path / "pipeline.pt")

        # This reuses lmdb if exists or creates
        self.dataset = self._create_dataset(features_df, self.dir_path / "dataset.lmdb")
        tokenized_path = self.dir_path / "tokenized.parquet"
        if tokenized_path.is_file():
            shutil.copy2(
                tokenized_path,
                FPATH.NETWORK_DATA / self.source_dir / f"{self.dir_path.name}.parquet",
            )
            tokenized_path.unlink()
        print("Lengths:", self.lengths)
        # Define lengths for UnpadSampler
        if isinstance(self.lengths, str) and (
            (self.dir_path / (self.lengths + ".parquet")).exists()
        ):
            print(f'Info: Using unpadding lengths "{self.lengths}"')
            self._lengths = pl.read_parquet(self.dir_path / (self.lengths + ".parquet"))
        else:
            print(
                f"Warning: No lengths.parquet found - Using {self.max_seq_len} unpadding lengths"
            )
            self._lengths = [self.max_seq_len]

    def _create_dataset(self, df: pl.LazyFrame, path: Path):
        return LMDBDataset(df, path)

    def setup(self, stage: Literal["fit"] = None):
        subsets = self.dataset.split(
            {"train": self.train_person_ids, "val": self.val_person_ids}
        )
        self.train_dataset = subsets["train"]
        self.val_dataset = subsets["val"]

    def collate_fn(self):
        """Returns the Collate function for the DataModule"""
        return Collate(
            self.truncate_length,
            self.background_length,
            segment=self.segment,
        )

    def get_dataloader(self, dataset: LMDBDataset, sampler=None):
        """Returns a generic DataLoader with given attributes from self and kwargs"""
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn(),
            sampler=self.get_sampler(dataset, sampler),
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
        )

    def get_sampler(self, dataset, sampler=None):
        """Returns a UnpadSampler"""
        # If lengths.parquet, subset and re-order
        if isinstance(self._lengths, pl.DataFrame):
            pnrs = pl.from_dict({"person_id": dataset.observations["person_id"]})
            subset = pnrs.join(self._lengths, on="person_id", how="left")
            pnr_to_length = dict(zip(subset["person_id"], subset["length"]))
            lengths = [pnr_to_length[pnr] for pnr in dataset.observations["person_id"]]
        else:
            lengths = self._lengths * len(dataset)
        return UnpadSampler(
            lengths,
            n_tokens=self.n_tokens,
            max_seq_len=self.max_seq_len,
            sampler=range(len(dataset)) if sampler is None else sampler,
        )

    @staticmethod
    def worker_init_fn(worker_id):
        """Initializes the dataset"""
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset._init_db()

    def train_dataloader(self):
        """Returns the train dataloader for self.train_dataset"""
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self):
        """Returns the val dataloader for self.val_dataset"""
        return self.get_dataloader(self.val_dataset)

    def predict_dataloader(self):
        """Returns the prediction dataloader for self.predict_dataset"""
        return self.get_dataloader(self.predict_dataset)

    def teardown(self, stage: str = None):
        """Copies all contents from dir_path to opposite drive if they do not exist."""
        swapped_path = FPATH.swap_drives(self.dir_path)
        swapped_path.mkdir(parents=True, exist_ok=True)

        for item in self.dir_path.iterdir():
            dest = swapped_path / item.name
            if item.is_dir():
                if not dest.exists():
                    shutil.copytree(item, dest)
            else:
                if not dest.exists():
                    shutil.copy2(item, dest)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key, v in batch.items():
            if torch.is_tensor(v):
                batch[key] = v.to(device, non_blocking=True)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # TODO: Should be part of collate_fn!
        batch["attn_mask"] = batch["event"] != 0

        # Unpad inputs
        _, indices, cu_seqlens, max_seqlen_in_batch, total = unpad_input(
            batch["event"].unsqueeze(-1), batch["attn_mask"]
        )
        batch["indices"] = indices
        batch["max_seqlen_in_batch"] = max_seqlen_in_batch
        batch["cu_seqlens"] = cu_seqlens
        batch["total"] = total.sum()

        # Flatten inputs
        batch["event"] = batch["event"].flatten()[batch["indices"]]
        batch["abspos"] = batch["abspos"].flatten()[batch["indices"]]
        batch["age"] = batch["age"].flatten()[batch["indices"]]
        batch["segment"] = batch["segment"].flatten()[batch["indices"]]

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
        cutoff_year: int = 2018,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = outcomes
        self.inference_type = inference_type
        self.cutoff_year = cutoff_year

    def _create_dataset(self, df: pl.LazyFrame, path: Path):
        outcome_dict = (
            self.outcomes.with_columns(calculate_abspos(pl.col("censor")))
            .to_pandas()
            .to_dict(orient="list")
        )
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
            dev = self.outcomes.filter(pl.col("censor").dt.year() < self.cutoff_year)
            dev_dataset = self.dataset.subset(
                dev.with_columns(calculate_abspos(pl.col("censor")))
                .to_pandas()
                .to_dict(orient="list")
            )
            subsets = dev_dataset.split(
                {"train": self.train_person_ids, "val": self.val_person_ids}
            )
            self.train_dataset = subsets["train"]
            self.val_dataset = subsets["val"]

        if stage == "predict":
            if self.inference_type == "test":
                # Define what dataset you want to use for prediction (e.g., test or new data)
                predict = self.outcomes.filter(
                    pl.col("censor").dt.year() == self.cutoff_year
                )
                self.predict_dataset = self.dataset.subset(
                    predict.with_columns(calculate_abspos(pl.col("censor")))
                    .to_pandas()
                    .to_dict(orient="list")
                )
            elif self.inference_type == "val":
                dev = self.outcomes.filter(
                    pl.col("censor").dt.year() < self.cutoff_year
                )
                dev_dataset = self.dataset.subset(
                    dev.with_columns(calculate_abspos(pl.col("censor")))
                    .to_pandas()
                    .to_dict(orient="list")
                )
                subsets = dev_dataset.split({"predict": self.val_person_ids})
                self.predict_dataset = subsets["predict"]
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
