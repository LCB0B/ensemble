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
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.earlylife.src.collate_fn import (
    AutoregressiveCollate,
    CensorCollate,
    Collate,
    MaskCollate,
    PredictCensorCollate,
)
from src.earlylife.src.dataset import (
    FinetuneLMDBDataset,
    LMDBDataset,
    ParentsFinetuneLMDBDataset,
)
from src.earlylife.src.paths import FPATH, check_and_copy_file_or_dir
from src.earlylife.src.pipeline import DataPipeline
from src.earlylife.src.sampler import UnpadSampler
from src.earlylife.src.utils import (
    calculate_abspos,
    create_weights,
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
        # Init Path related stuff
        self.dir_path = dir_path
        check_and_copy_file_or_dir(self.dir_path)
        self.source_dir = source_dir
        self.lengths = lengths

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
        self.n_tokens = n_tokens
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
        self._lengths = None
        self.train_dataset, self.val_dataset, self.predict_dataset = None, None, None

    def prepare_data(self):
        """Not on all workers"""
        if not (self.dir_path / "dataset.lmdb").exists():
            features_df = self.pipeline(self.sources, self.background, self.dir_path)
            torch.save(self.pipeline, self.dir_path / "pipeline.pt")
        else:
            features_df = None

        # This reuses lmdb if exists or creates
        self.dataset = self._create_dataset(features_df, self.dir_path / "dataset.lmdb")
        tokenized_path = self.dir_path / "tokenized.parquet"
        if tokenized_path.is_file():
            shutil.move(
                tokenized_path,
                FPATH.NETWORK_DATA / self.source_dir / f"{self.dir_path.name}.parquet",
            )

        # Define lengths for UnpadSampler
        if self._lengths is None:
            if isinstance(self.lengths, str) and (
                (self.dir_path / (self.lengths + ".parquet")).exists()
            ):
                print(f'Info: Using unpadding lengths "{self.lengths}.parquet"')
                self._lengths = pl.read_parquet(
                    self.dir_path / (self.lengths + ".parquet")
                ).cast({"person_id": pl.String})
            else:
                print(
                    f"Warning: No {self.lengths}.parquet found - Using {self.max_seq_len} unpadding lengths"
                )
                self._lengths = [self.max_seq_len]

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        return LMDBDataset(dataset, path)

    def setup(self, stage: Literal["fit"] = None):
        """Defaults random splitting"""
        subsets = self.dataset.split({"train": 0.8, "val": 0.2})
        self.train_dataset = subsets["train"]
        self.val_dataset = subsets["val"]
        self.predict_dataset = self.val_dataset
        # if stage == "fit" or stage is None:
        #     subsets = self.dataset.split({"train": 0.8, "val": 0.2})
        #     self.train_dataset = subsets["train"]
        #     self.val_dataset = subsets["val"]
        # elif stage == "predict":
        #     if self.val_dataset is None:
        #         print("No val_dataset, init new")
        #         subsets = self.dataset.split({"train": 0.8, "val": 0.2})
        #         self.val_dataset = subsets["val"]
        #     self.predict_dataset = self.val_dataset

    def collate_fn(self):
        """Returns the Collate function for the DataModule"""
        return Collate(
            self.truncate_length,
            self.background_length,
            segment=self.segment,
        )

    def get_dataloader(self, dataset: LMDBDataset, sampler=None):
        """Returns a generic DataLoader with given attributes from self and kwargs"""
        if self.num_workers == 0:
            dataset._init_db()
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
            pnrs = pl.from_dict({"person_id": dataset.observations["person_id"]}).cast(
                {"person_id": pl.String}
            )
            subset = pnrs.join(self._lengths, on="person_id", how="left")
            pnr_to_length = dict(zip(subset["person_id"], subset["length"]))
            lengths = [
                pnr_to_length[str(pnr)] for pnr in dataset.observations["person_id"]
            ]
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

        batch = self.unpad(batch)

        return batch

    def unpad(self, batch):
        # Unpad inputs
        _, indices, cu_seqlens, max_seqlen_in_batch, total = unpad_input(
            batch["event"].unsqueeze(-1), batch["attn_mask"]
        )
        batch["indices"] = indices
        batch["max_seqlen_in_batch"] = max_seqlen_in_batch
        batch["cu_seqlens"] = cu_seqlens
        batch["total"] = total.sum().item()

        # Flatten inputs
        batch["event"] = batch["event"].flatten()[batch["indices"]]
        batch["abspos"] = batch["abspos"].flatten()[batch["indices"]]
        batch["age"] = batch["age"].flatten()[batch["indices"]]
        batch["segment"] = batch["segment"].flatten()[batch["indices"]]
        return batch


class PretrainDataModule(BaseLightningDataModule):
    """Lightning Data Module for MLM pretraining"""

    def __init__(self, *args, pretrain_style: str, masking_ratio=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain_style = pretrain_style.upper()
        self.masking_ratio = masking_ratio

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        batch["target"] = batch["target"].flatten()[batch["indices"]]

        return batch

    def collate_fn(self):
        if self.pretrain_style == "MLM":
            return MaskCollate(
                vocab=self.pipeline.vocab,
                truncate_length=self.truncate_length,
                background_length=self.background_length,
                segment=self.segment,
                mask_prob=self.masking_ratio,
            )
        elif self.pretrain_style == "AR":
            return AutoregressiveCollate(
                truncate_length=self.truncate_length,
                background_length=self.background_length,
                segment=self.segment,
            )
        raise ValueError(self.pretrain_style)


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
        self.outcomes = outcomes.cast({"person_id": pl.String})

    def collate_fn(self):
        return MaskGridCollate(
            vocab=self.pipeline.vocab,
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
            predictions_per_year=self.predictions_per_year,
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


class PredictFinetuneLifeLDM(BaseLightningDataModule):
    def __init__(
        self,
        *args,
        outcomes: pl.DataFrame,
        prediction_windows: List[float],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = outcomes
        self.prediction_windows = prediction_windows
        self.collate = None

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        outcomes_dict = (
            self.outcomes.with_columns(
                calculate_abspos(pl.col("censor")),
                calculate_abspos(pl.col("outcome")),
                pl.col("predict").list.eval(calculate_abspos(pl.element())),
                # calculate_abspos(pl.col("predict")),
            )
            .to_pandas()
            .to_dict(orient="list")
        )
        return FinetuneLMDBDataset(dataset, path, outcomes_dict)

    def train_dataloader(self):
        train_outcomes = self.train_dataset.observations["outcome"]
        train_outcomes = [1 - math.isnan(out) for out in train_outcomes]
        weights = create_weights(train_outcomes, op=math.sqrt)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_outcomes), replacement=True
        )
        return self.get_dataloader(self.train_dataset, sampler=sampler)

    def collate_fn(self):
        self.collate = PredictCensorCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
        )
        return self.collate

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch["predict_tokens"] = batch["event"] == self.collate.predict_token_id
        batch = super().on_after_batch_transfer(batch, dataloader_idx)

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
        batch["grid_tokens"] = batch["event"] == self.collate.grid_token_id
        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        return batch


class ScreenRiskFinetuneLifeLightningDataModule(RiskFinetuneLifeLightningDataModule):
    """Lightning Data Module for risk trajectories finetuning"""

    def collate_fn(self):
        self.collate = ScreenGridCensorCollate(
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
        batch["og_event"] = batch["event"]
        batch["og_abspos"] = batch["abspos"]
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        return batch


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
