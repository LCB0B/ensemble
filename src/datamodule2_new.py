""" Lightning Data Module for all data-related code """

import math
import shutil
import random
from pathlib import Path
from typing import List, Literal, Dict

import lightning as L
import polars as pl
import pyarrow.dataset as ds
import torch
from flash_attn.bert_padding import unpad_input
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.collate_fn2_new import (
    Collate,
    AutoregressiveCollate,
    AutoregressivePredictCollate,
    FamilyCensorAutoregressivePredictCollate,
    AutoregressiveCensorPredictCollate,
    FamilyPredictCensorCollate,
    FamilyPredictCensorRegressionCollate,
    PredictCensorCollate,
)
from src.dataset_new import LMDBDataset, FinetuneLMDBDataset, FamilyFinetuneLMDBDataset
from src.paths import FPATH, check_and_copy_file_or_dir
from src.pipeline import DataPipeline
from src.sampler import UnpadSampler
from src.utils import calculate_abspos, create_weights, get_background_length

ONE_YEAR_ABSPOS = 365.25 * 24


# pylint: disable=arguments-differ
class BaseLightningDataModule(L.LightningDataModule):
    """Base Lightning Data Module for shared pretrain and finetune code"""

    def __init__(
        self,
        dir_path: Path,
        sources: List[ds.Dataset],
        background: pl.DataFrame,
        cohorts: dict = None,
        fill_nulls=False,
        subset_background=False,
        num_workers=0,
        n_tokens=8e5,
        max_seq_len=512,
        cutoff=0,
        source_dir=None,
    ):
        super().__init__()
        # Init data related stuff
        self.fill_nulls = fill_nulls
        self.sources = sources
        self.background = background
        self.cohorts = cohorts
        # Init Path related stuff
        self.dir_path = dir_path
        check_and_copy_file_or_dir(self.dir_path, verbosity=2)
        self.source_dir = source_dir

        if (pipeline_path := dir_path / "pipeline.pt").exists():
            print("Loading pipeline")
            self.pipeline = torch.load(pipeline_path, weights_only=False)
        else:
            self.pipeline = DataPipeline(
                cls_token=False,
                sep_token=False,
                fill_nulls=fill_nulls,
                subset_background=subset_background,
                cutoff=cutoff,
            )

        # Check and copy files between dirs
        self.dir_path.mkdir(parents=True, exist_ok=True)

        # Init other arg-related stuff
        self.num_workers = num_workers
        self.n_tokens = n_tokens

        # Init length-related stuff
        self.background_length = get_background_length(background)
        self.max_seq_len = max_seq_len

        self.prepared = 0
        # Avoid lint complaints
        self.dataset = None
        self.lengths = None
        self.train_dataset, self.val_dataset = None, None
        self.predict_dataset, self.test_dataset = None, None

    def prepare_data(self):
        """Not on all workers"""
        self.prepared += 1
        if self.prepared > 1:
            return
        if not (self.dir_path / "dataset.lmdb").exists():
            features_df = self.pipeline(
                self.sources, self.background, self.dir_path, self.source_dir
            )  # ADDED self.source_dir
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
        self.lengths = self.dataset.get_lengths()

        if self.cohorts is not None:
            for key, cohort in self.cohorts.items():
                self.cohorts[key] = self.prep_cohort(cohort)

    def prep_cohort(self, cohort):
        """Prepares the cohort"""
        return cohort.to_pandas().to_dict(orient="list")

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        return LMDBDataset(dataset, path)

    def setup(self, stage: Literal["fit"] = None):
        """Defaults random splitting"""
        if self.train_dataset is not None:
            return
        if self.cohorts is None:
            print("Info: Using random splits (80/20)")
            subsets = self.dataset.split({"train": 0.8, "val": 0.2})
            self.train_dataset = subsets["train"]
            self.val_dataset = subsets["val"]
            self.predict_dataset = self.val_dataset
        else:
            print("Info: Using predefined splits", self.cohorts.keys())
            if "train" in self.cohorts:
                self.train_dataset = self.dataset.subset(self.cohorts["train"])
            if "val" in self.cohorts:
                self.val_dataset = self.dataset.subset(self.cohorts["val"])
            if "test" in self.cohorts:
                self.predict_dataset = self.dataset.subset(self.cohorts["test"])
                self.test_dataset = self.dataset.subset(self.cohorts["test"])

    def collate_fn(self):
        """Returns the Collate function for the DataModule"""
        return Collate(
            self.max_seq_len,
            self.background_length,
        )

    def get_dataloader(self, dataset: LMDBDataset, sampler=None):
        """Returns a generic DataLoader with given attributes from self and kwargs"""
        if self.num_workers == 0:
            dataset.init_db()
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
        lengths = [self.lengths[str(pnr)] for pnr in dataset.observations["person_id"]]

        if sampler is None:
            sampler = list(range(len(dataset)))
            random.shuffle(sampler)
        return UnpadSampler(
            lengths,
            n_tokens=self.n_tokens,
            max_seq_len=self.max_seq_len,
            sampler=sampler,
        )

    @staticmethod
    def worker_init_fn(worker_id):
        """Initializes the dataset"""
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.init_db()

    def train_dataloader(self):
        """Returns the train dataloader for self.train_dataset"""
        assert self.train_dataset is not None
        return self.get_dataloader(self.train_dataset)

    def val_dataloader(self):
        """Returns the val dataloader for self.val_dataset"""
        assert self.val_dataset is not None
        return self.get_dataloader(self.val_dataset)

    def predict_dataloader(self):
        """Returns the prediction dataloader for self.predict_dataset"""
        assert self.predict_dataset is not None
        return self.get_dataloader(self.predict_dataset)

    def test_dataloader(self):
        """Returns the test dataloader for self.test_dataset"""
        assert self.test_dataset is not None
        return self.get_dataloader(self.test_dataset)

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


class OutcomesBaseLDM(BaseLightningDataModule):
    """Base LDM with outcomes"""

    def __init__(self, *args, outcomes, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = self.prep_outcomes(outcomes)

    def prep_outcomes(self, outcomes):
        """Prepares the outcomes with needed conversions"""
        return outcomes

    def prep_cohort(self, cohort) -> Dict[str, List]:
        """Subsets self.outcomes it on cohort"""
        df = cohort.to_pandas().merge(self.outcomes.to_pandas(), on="person_id")
        df = df.to_dict(orient="list")
        return df

    @staticmethod
    def _create_dataset(dataset: ds.Dataset, path: Path):
        return FinetuneLMDBDataset(dataset, path, None)


class FamilyLifeLDM(BaseLightningDataModule):
    def __init__(self, *args, feature_set: List[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_set = feature_set

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        return FamilyFinetuneLMDBDataset(dataset, path, None)

    def unpad(self, batch):
        batch = super().unpad(batch)
        batch["family_type"] = batch["family_type"].flatten()[batch["indices"]]
        return batch

    def prep_cohort(self, cohort):
        cohort = cohort.join(
            self._lengths.cast({"person_id": pl.Int64}), on="person_id"
        )  # join length_dict onto
        return super().prep_cohort(cohort)


class PretrainLDM(BaseLightningDataModule):
    """Lightning Data Module for AR pretraining"""

    def unpad(self, batch):
        batch = super().unpad(batch)
        batch["target"] = batch["target"].flatten()[batch["indices"]]
        return batch

    def collate_fn(self):
        return AutoregressiveCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
        )


class PretrainPredictLDM(PretrainLDM, OutcomesBaseLDM):
    """Lightning Data Module for PREDICT token pretraining"""

    def prep_outcomes(self, outcomes):
        return outcomes.with_columns(
            pl.col("predict").list.eval(calculate_abspos(pl.element())),
        )

    def collate_fn(self):
        return AutoregressivePredictCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            # predict_token_id=1, # DEFAULTED
        )


class PretrainCensorPredictLDM(PretrainPredictLDM):
    """PretrainPredict with censoring"""

    def prep_outcomes(self, outcomes):
        return outcomes.with_columns(
            calculate_abspos(pl.col("censor")),
            pl.col("predict").list.eval(calculate_abspos(pl.element())),
        )

    def collate_fn(self):
        return AutoregressiveCensorPredictCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            # predict_token_id=1, # DEFAULTED
        )


class PredictFinetuneLifeLDM(OutcomesBaseLDM):
    def __init__(
        self,
        *args,
        prediction_windows: List[float],
        padding_side: Literal["left", "right"] = "left",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prediction_windows = prediction_windows
        self.padding_side = padding_side
        self.collate = None
        self.pprint = True

    def prep_outcomes(self, outcomes):
        return outcomes.with_columns(
            calculate_abspos(pl.col("censor")),
            calculate_abspos(pl.col("outcome")),
            pl.col("predict").list.eval(calculate_abspos(pl.element())),
        )

    def train_dataloader(self):
        train_outcomes = self.train_dataset.observations["outcome"]
        train_outcomes = [1 - math.isnan(out) for out in train_outcomes]
        weights = create_weights(train_outcomes, op=math.sqrt, pprint=self.pprint)
        self.pprint = False
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_outcomes), replacement=True
        )
        return self.get_dataloader(self.train_dataset, sampler=sampler)

    def collate_fn(self):
        self.collate = PredictCensorCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            padding_side=self.padding_side,
        )
        return self.collate

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch["predict_tokens"] = batch["event"] == self.collate.predict_token_id
        batch["og_abspos"] = batch["abspos"]  # TODO: Not ideal
        batch["og_age"] = batch["age"].round(decimals=2)  # TODO: Not ideal
        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        return batch


class FamilyPredictFinetuneLifeLDM(FamilyLifeLDM, PredictFinetuneLifeLDM):
    def collate_fn(self):
        self.collate = FamilyPredictCensorCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            feature_set=self.feature_set,
            padding_side=self.padding_side,
        )
        return self.collate


class FamilyRegressionFinetuneLifeLDM(FamilyPredictFinetuneLifeLDM):
    def collate_fn(self):
        self.collate = FamilyPredictCensorRegressionCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            feature_set=self.feature_set,
        )
        return self.collate


class FamilyPretrainCensorPredictLDM(FamilyLifeLDM, PretrainCensorPredictLDM):
    def collate_fn(self):
        return FamilyCensorAutoregressivePredictCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            feature_set=self.feature_set,
        )
