""" Lightning Data Module for all data-related code """

import math
import shutil
from pathlib import Path
from typing import List, Literal

import lightning as L
import polars as pl
import pyarrow.dataset as ds
import torch
from torch.nn.attention.flex_attention import and_masks, create_block_mask
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler

from src.earlylife.src.collate_fn import (
    CensorCollate,
    Collate,
    GridCensorCollate,
    MaskCollate,
    MaskGridCollate,
    ParentGridCensorCollate,
    ParentMaskGridCollate,
)
from src.earlylife.src.dataset import (
    FinetuneLMDBDataset,
    LMDBDataset,
    ParentsFinetuneLMDBDataset,
)
from src.earlylife.src.paths import FPATH, check_and_copy_file_or_dir
from src.earlylife.src.pipeline import DataPipeline
from src.earlylife.src.utils import (
    calculate_abspos,
    create_weights,
    flex_attn_causal_event,
    flex_attn_padding,
    get_background_length,
)

ONE_YEAR_ABSPOS = 365.25 * 24


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
        segment=False,
        fill_nulls=False,
        subset_background=False,
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
        # Init Path related stuff
        self.dir_path = dir_path
        check_and_copy_file_or_dir(self.dir_path)

        if (pipeline_path := dir_path / "pipeline.pt").exists():
            print("Loading pipeline")
            self.pipeline = torch.load(pipeline_path, weights_only=False)
        else:
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
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.segment = segment

        # Init length-related stuff
        self.background_length = (
            get_background_length(background) + int(cls_token) + int(sep_token)
        )
        self.truncate_length = (
            max_seq_len - self.background_length
        )  # TODO: Reverse, so we give max_seq_len anad calculate truncate_length in collate
        self.cls_token = cls_token
        self.max_seq_len = max_seq_len

        # Avoid lint complaints
        self.dataset = None
        self.train_dataset, self.val_dataset, self.predict_dataset = None, None, None

    def prepare_data(self):
        """Not on all workers"""
        features_df = self.pipeline(self.sources, self.background, self.dir_path)
        torch.save(self.pipeline, self.dir_path / "pipeline.pt")

        # This reuses lmdb if exists or creates
        self.dataset = self._create_dataset(features_df, self.dir_path / "dataset.lmdb")

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        return LMDBDataset(dataset, path)

    def setup(self, stage: Literal["fit"] = None):
        """Defaults random splitting"""
        if stage == "fit" or stage is None:
            train, val = self.dataset.split(0.8)
            self.train_dataset = train
            self.val_dataset = val
        elif stage == "predict":
            self.predict_dataset = self.val_dataset

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

        persist_workers = False
        if self.num_workers > 0:
            persist_workers = True

        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=persist_workers,
            collate_fn=self.collate_fn(),
            sampler=sampler,
            pin_memory=True,
            worker_init_fn=self.worker_init_fn,
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
        bs, seq_len = batch["event"].shape

        # Create block mask
        batch["attn_mask"] = create_block_mask(
            flex_attn_padding(batch["sequence_lens"]),
            bs,
            None,
            seq_len,
            seq_len,
            _compile=True,
        )
        return batch


class PretrainDataModule(BaseLightningDataModule):
    """Lightning Data Module for MLM pretraining"""

    def collate_fn(self):
        return MaskCollate(
            vocab=self.pipeline.vocab,
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )


class PretrainCausalGridDataModule(PretrainDataModule):
    """Lightning Data Module for grid MLM pretraining"""

    def __init__(
        self,
        *args,
        outcomes: pl.DataFrame,
        predictions_per_year: float = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.predictions_per_year = predictions_per_year
        self.outcomes = outcomes

    def collate_fn(self):
        return MaskGridCollate(
            vocab=self.pipeline.vocab,
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        outcomes_dict = (
            self.outcomes.with_columns(
                calculate_abspos(pl.col("censor")),
                calculate_abspos(pl.col("event_final_date")),
            )
            .to_pandas()
            .to_dict(orient="list")
        )
        return FinetuneLMDBDataset(dataset, path, outcomes_dict)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        bs, seq_len = batch["event"].shape

        # Create block mask
        batch["attn_mask"] = create_block_mask(
            and_masks(
                flex_attn_causal_event(batch["abspos"]),
                flex_attn_padding(batch["sequence_lens"]),
            ),
            bs,
            None,
            seq_len,
            seq_len,
            _compile=True,
        )
        return batch


class FinetuneLifeLightningDataModule(BaseLightningDataModule):
    """Lightning Data Module for binary finetuning"""

    def __init__(
        self,
        *args,
        outcomes: pl.DataFrame,
        negative_censor: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = outcomes
        self.negative_censor = negative_censor

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        outcomes_dict = (
            self.outcomes.with_columns(
                calculate_abspos(pl.col("censor")),
                calculate_abspos(pl.col("event_final_date")),
            )
            .to_pandas()
            .to_dict(orient="list")
        )
        return FinetuneLMDBDataset(dataset, path, outcomes_dict)

    def collate_fn(self):
        return CensorCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
        )

    def train_dataloader(self):
        train_outcomes = self.train_dataset.observations["outcome"]
        weights = create_weights(train_outcomes, op=math.sqrt)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_outcomes), replacement=True
        )
        return self.get_dataloader(self.train_dataset, sampler=sampler)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        return batch


class RiskFinetuneLifeLightningDataModule(FinetuneLifeLightningDataModule):
    """Lightning Data Module for risk trajectories finetuning"""

    def __init__(
        self,
        *args,
        prediction_windows: List[float],
        predictions_per_year: float = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prediction_windows = prediction_windows
        self.predictions_per_year = predictions_per_year
        self.collate = None

    def collate_fn(self):
        self.collate = GridCensorCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
            segment=self.segment,
            predictions_per_year=self.predictions_per_year,
        )
        return self.collate

    def on_after_batch_transfer(self, batch, dataloader_idx):
        bs, seq_len = batch["event"].shape

        batch["event_mask"] = self.create_event_mask(batch)

        # Create block mask
        batch["attn_mask"] = create_block_mask(
            and_masks(
                flex_attn_causal_event(batch["abspos"]),
                flex_attn_padding(batch["sequence_lens"]),
            ),
            bs,
            None,
            seq_len,
            seq_len,
            _compile=True,
        )
        return batch

    def create_event_mask(self, batch):
        """Creates the event mask based on the prediction type (grid or per event)"""
        n_predictions = batch["target"].size(1)
        first_grid_token = (
            (batch["event"] == self.collate.grid_token_id).long().argmax(1)
        )
        first_abspos = batch["abspos"][
            torch.arange(len(first_grid_token)), first_grid_token
        ]
        grid_interval = self.collate.grid_interval
        grid = first_abspos.unsqueeze(1) + torch.arange(
            0,
            grid_interval * n_predictions,
            grid_interval,
            device=batch["abspos"].device,
        )
        return (batch["abspos"].unsqueeze(1) <= grid.unsqueeze(-1)).half()


class ParentsRiskFinetuneLifeLightningDataModule(RiskFinetuneLifeLightningDataModule):
    """Lightning Data Module for parent risk trajectories finetuning"""

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        outcomes_dict = (
            self.outcomes.with_columns(
                calculate_abspos(pl.col("censor")),
                calculate_abspos(pl.col("event_final_date")),
            )
            .to_pandas()
            .to_dict(orient="list")
        )
        return ParentsFinetuneLMDBDataset(dataset, path, outcomes_dict)

    def collate_fn(self):
        self.collate = ParentGridCensorCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
            segment=self.segment,
        )
        return self.collate


class ParentPretrainCausalGridDataModule(PretrainCausalGridDataModule):
    """Lightning Data Module for grid MLM pretraining"""

    def collate_fn(self):
        return ParentMaskGridCollate(
            vocab=self.pipeline.vocab,
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        outcomes_dict = (
            self.outcomes.with_columns(
                calculate_abspos(pl.col("censor")),
                calculate_abspos(pl.col("event_final_date")),
            )
            .to_pandas()
            .to_dict(orient="list")
        )
        return ParentsFinetuneLMDBDataset(dataset, path, outcomes_dict)
