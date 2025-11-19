""" Lightning Data Module for all data-related code """

import math
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Literal

import lightning as L
import polars as pl
import pyarrow.dataset as ds
import torch
from flash_attn.bert_padding import unpad_input
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from src.collate_fn import (
    AutoregressiveCollate,
    AutoregressivePredictCollate,
    Collate,
    FamilyAutoregressiveCollate,
    FamilyPredictCensorCollate,
    FamilyPredictCensorRegressionCollate,
    MaskCollate,
    PredictCensorCollate,
)
from src.dataset import (
    FamilyFinetuneLMDBDataset,
    FinetuneLMDBDataset,
    LMDBDataset,
)
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
        check_and_copy_file_or_dir(self.dir_path, verbosity=2)
        self.source_dir = source_dir
        self.lengths = lengths

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

        # Avoid lint complaints
        self.dataset = None
        self._lengths = None
        self.train_dataset, self.val_dataset, self.predict_dataset = None, None, None

    def prepare_data(self):
        """Not on all workers"""
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

    def collate_fn(self):
        """Returns the Collate function for the DataModule"""
        return Collate(
            self.max_seq_len,
            self.background_length,
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
                max_seq_len=self.max_seq_len,
                background_length=self.background_length,
                mask_prob=self.masking_ratio,
            )
        elif self.pretrain_style == "AR":
            return AutoregressiveCollate(
                max_seq_len=self.max_seq_len,
                background_length=self.background_length,
            )
        raise ValueError(self.pretrain_style)


class PretrainPredictDataModule(PretrainDataModule):
    """Lightning Data Module for PREDICT token pretraining"""

    def __init__(
        self,
        *args,
        outcomes: pl.DataFrame,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = outcomes

    def collate_fn(self):
        return AutoregressivePredictCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            # predict_token_id=1, # DEFAULTED
        )

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


class PredictFinetuneLifeLDM(BaseLightningDataModule):
    def __init__(
        self,
        *args,
        outcomes: Dict[str, pl.DataFrame],
        prediction_windows: List[float],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = outcomes
        self.prediction_windows = prediction_windows
        self.collate = None

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        return FinetuneLMDBDataset(
            dataset,
            path,
            self.outcomes.with_columns(
                calculate_abspos(pl.col("censor")),
                calculate_abspos(pl.col("outcome")),
                pl.col("predict").list.eval(calculate_abspos(pl.element())),
            )
            .to_pandas()
            .to_dict(orient="list"),
        )

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
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
        )
        return self.collate

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch["predict_tokens"] = batch["event"] == self.collate.predict_token_id
        batch["og_abspos"] = batch["abspos"]
        batch["event_mask"] = self.create_event_mask(batch)
        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        return batch

    def create_event_mask(self, batch):
        """Creates the event mask based on the prediction type (grid or per event)"""
        n_predictions = batch["target"].size(1)
        first_grid_token = (
            (batch["event"] == self.collate.predict_token_id).long().argmax(1)
        )
        first_abspos = batch["abspos"][
            torch.arange(len(first_grid_token)), first_grid_token
        ]
        grid_interval = ONE_YEAR_ABSPOS
        grid = first_abspos.unsqueeze(1) + torch.arange(
            0,
            grid_interval * n_predictions,
            grid_interval,
            device=batch["abspos"].device,
        )
        return (batch["abspos"].unsqueeze(1) <= grid.unsqueeze(-1)).half()


class FamilyPredictFinetuneLifeLDM(PredictFinetuneLifeLDM):
    def __init__(self, *args, feature_set: List[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_set = feature_set

    def _create_dataset(
        self,
        dataset: ds.Dataset,
        path: Path,
    ):
        outcomes_dict = (
            self.outcomes.with_columns(
                calculate_abspos(pl.col("censor")),
                calculate_abspos(pl.col("outcome")),
                pl.col("predict").list.eval(calculate_abspos(pl.element())),
            )
            .to_pandas()
            .to_dict(orient="list")
        )
        return FamilyFinetuneLMDBDataset(dataset, path, outcomes_dict)

    def collate_fn(self):
        self.collate = FamilyPredictCensorCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            feature_set=self.feature_set,
        )
        return self.collate

    def unpad(self, batch):
        batch = super().unpad(batch)
        batch["family_type"] = batch["family_type"].flatten()[batch["indices"]]
        return batch


class FamilyRegressionFinetuneLifeLDM(FamilyPredictFinetuneLifeLDM):
    def __init__(
        self,
        *args,
        train_person_ids=None,
        val_person_ids=None,
        test_person_ids=None,
        inference_type=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_person_ids = train_person_ids
        self.val_person_ids = val_person_ids
        self.test_person_ids = test_person_ids
        self.inference_type = inference_type

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

    def train_dataloader(self):
        """Returns the train dataloader for self.train_dataset"""
        return self.get_dataloader(
            self.train_dataset, sampler=RandomSampler(range(len(self.train_dataset)))
        )

    def setup(self, stage: Literal["fit"] = None):
        """On all workers"""
        if stage == "fit" or stage is None:
            if (self.train_person_ids is None) or (self.val_person_ids is None):
                raise Exception(
                    "Expected both train_person_ids and val_person_ids to be passed during training"
                )

            subsets = self.dataset.split(
                {"train": self.train_person_ids, "val": self.val_person_ids}
            )
            self.train_dataset = subsets["train"]
            self.val_dataset = subsets["val"]

        if stage == "predict":
            if self.inference_type == "test":
                if self.test_person_ids is None:
                    raise Exception("Expected test_person_ids")

                subsets = self.dataset.split({"test": self.test_person_ids})
                self.predict_dataset = subsets["test"]

            elif self.inference_type == "val":
                if self.val_person_ids is None:
                    raise Exception("Expected val_person_ids")
                subsets = self.dataset.split({"val": self.val_person_ids})
                self.predict_dataset = subsets["val"]

            elif self.inference_type == "train":
                if self.train_person_ids is None:
                    raise Exception("Expected train_person_ids")
                subsets = self.dataset.split({"train": self.train_person_ids})
                self.predict_dataset = subsets["train"]
            else:
                raise Exception(
                    "Unknown inference_type. Only supports 'test', 'val' or 'train' at current moment"
                )

    def prepare_data(self):
        """Prepares data and adjusts unpadding lengths based on actual truncation logic, with caching."""
        super().prepare_data()

        if not isinstance(self._lengths, pl.DataFrame):
            raise ValueError(
                "Expected self._lengths to be a DataFrame after super().prepare_data()"
            )

        # Build cache filename
        feature_str = "-".join(sorted(self.feature_set))
        fname = (
            f"{self.lengths}_fs={feature_str}-max_seq_len={self.max_seq_len}.parquet"
        )
        path = FPATH.swap_drives(self.dir_path) / fname
        path.parent.mkdir(exist_ok=True)

        if path.exists():
            print(f"Loading feature-set-aware lengths from {fname}")
            self._lengths = pl.read_parquet(path)
            return

        print("Computing feature-set-aware lengths with family-aware truncation")

        # Start with child lengths
        df = self._lengths.rename({"length": "Child"})

        # Find parents
        parent_map = (
            self.outcomes.select(
                [pl.col("person_id").cast(pl.String), pl.col("parents")]
            )
            .explode("parents")
            .unnest("parents")
            .select(
                [
                    pl.col("person_id"),
                    pl.col("parent_id").cast(str),
                    pl.col("relation_details"),
                ]
            )
        )
        # Find parent lengths
        parent_lengths = (
            parent_map.join(
                self._lengths.rename(
                    {"person_id": "parent_id", "length": "parent_length"}
                ),
                on="parent_id",
                how="left",
            )
            .pivot(
                values="parent_length", index="person_id", columns="relation_details"
            )
            .fill_null(0)
        )

        # Join parent lengths
        df = df.join(parent_lengths, on="person_id", how="left").fill_null(0)

        # Ensure all family columns exist (even if 0)
        for col in ["Child", "Mother", "Father"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))

        # Mask for which are included in feature_set
        included = [
            col for col in ["Child", "Mother", "Father"] if col in self.feature_set
        ]

        # Compute n_present_family_members (non-zero + in feature_set)
        df = df.with_columns(
            [
                pl.sum_horizontal(
                    [(pl.col(col) > 0).cast(pl.Int8) for col in included]
                ).alias("n_present_family")
            ]
        )

        # Compute truncate length per person
        df = df.with_columns(
            [
                ((pl.lit(self.max_seq_len) // pl.col("n_present_family")))
                .cast(pl.Int32)
                .alias("truncate_length")
            ]
        )

        # Apply min(trunc_len, member_len) for each included member
        for col in included:
            df = df.with_columns(
                [
                    pl.min_horizontal(pl.col(col), pl.col("truncate_length")).alias(
                        f"{col}_used"
                    )
                ]
            )

        # Final effective length = sum of used lengths
        df = df.with_columns(
            [
                pl.sum_horizontal([pl.col(f"{col}_used") for col in included])
                .cast(pl.Int32)
                .alias("length")
            ]
        )

        self._lengths = df.select(["person_id", "length"])
        self._lengths.write_parquet(path)
        print(f"Done determining family lengths. Saved to {fname}")


class FamilyAutoRegressiveDataModule(PretrainDataModule):
    def __init__(
        self,
        *args,
        outcomes,
        feature_set: List[str],
        train_person_ids=None,
        val_person_ids=None,
        test_person_ids=None,
        inference_type=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = outcomes  # We need outcomes, which is just dataframe of person id and parents
        self.feature_set = feature_set
        self.train_person_ids = train_person_ids
        self.val_person_ids = val_person_ids
        self.test_person_ids = test_person_ids
        self.inference_type = inference_type

    def collate_fn(self):
        self.collate = FamilyAutoregressiveCollate(
            max_seq_len=self.max_seq_len,
            background_length=self.background_length,
            feature_set=self.feature_set,
        )
        return self.collate

    def setup(self, stage: Literal["fit"] = None):
        """On all workers"""
        if stage == "fit" or stage is None:
            if (self.train_person_ids is None) or (self.val_person_ids is None):
                raise Exception(
                    "Expected both train_person_ids and val_person_ids to be passed during training"
                )

            subsets = self.dataset.split(
                {"train": self.train_person_ids, "val": self.val_person_ids}
            )
            self.train_dataset = subsets["train"]
            self.val_dataset = subsets["val"]

    def _create_dataset(
        self,
        dataset: ds.Dataset,
        path: Path,
    ):

        # If child is not in feature_set, we need to subset to those who have at least one of the inputted features ("Mother", "Father", or both) to avoid empty sequences
        if "Child" not in self.feature_set:
            self.outcomes = self.outcomes.filter(
                pl.col("parents")
                .list.eval(
                    pl.element()
                    .struct.field("relation_details")
                    .is_in(self.feature_set)
                )
                .list.any()
            )
        outcomes_dict = self.outcomes.to_pandas().to_dict(orient="list")
        return FamilyFinetuneLMDBDataset(dataset, path, outcomes_dict)

    def unpad(self, batch):
        batch = super().unpad(batch)
        batch["family_type"] = batch["family_type"].flatten()[batch["indices"]]
        return batch

    def prepare_data(self):
        """Prepares data and adjusts unpadding lengths based on actual truncation logic, with caching."""
        super().prepare_data()

        if not isinstance(self._lengths, pl.DataFrame):
            raise ValueError(
                "Expected self._lengths to be a DataFrame after super().prepare_data()"
            )

        # Build cache filename
        feature_str = "-".join(sorted(self.feature_set))
        fname = (
            f"{self.lengths}_fs={feature_str}-max_seq_len={self.max_seq_len}.parquet"
        )
        path = FPATH.swap_drives(self.dir_path) / fname
        path.parent.mkdir(exist_ok=True)

        if path.exists():
            print(f"Loading feature-set-aware lengths from {fname}")
            self._lengths = pl.read_parquet(path)
            return

        print("Computing feature-set-aware lengths with family-aware truncation")

        # Start with child lengths
        df = self._lengths.rename({"length": "Child"})

        # Find parents
        parent_map = (
            self.outcomes.select(
                [pl.col("person_id").cast(pl.String), pl.col("parents")]
            )
            .explode("parents")
            .unnest("parents")
            .select(
                [
                    pl.col("person_id"),
                    pl.col("parent_id").cast(str),
                    pl.col("relation_details"),
                ]
            )
        )
        # Find parent lengths
        parent_lengths = (
            parent_map.join(
                self._lengths.rename(
                    {"person_id": "parent_id", "length": "parent_length"}
                ),
                on="parent_id",
                how="left",
            )
            .pivot(
                values="parent_length", index="person_id", columns="relation_details"
            )
            .fill_null(0)
        )

        # Join parent lengths
        df = df.join(parent_lengths, on="person_id", how="left").fill_null(0)

        # Ensure all family columns exist (even if 0)
        for col in ["Child", "Mother", "Father"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))

        # Mask for which are included in feature_set
        included = [
            col for col in ["Child", "Mother", "Father"] if col in self.feature_set
        ]

        # Compute n_present_family_members (non-zero + in feature_set)
        df = df.with_columns(
            [
                pl.sum_horizontal(
                    [(pl.col(col) > 0).cast(pl.Int8) for col in included]
                ).alias("n_present_family")
            ]
        )

        # Compute truncate length per person
        df = df.with_columns(
            [
                ((pl.lit(self.max_seq_len) // pl.col("n_present_family")))
                .cast(pl.Int32)
                .alias("truncate_length")
            ]
        )

        # Apply min(trunc_len, member_len) for each included member
        for col in included:
            df = df.with_columns(
                [
                    pl.min_horizontal(pl.col(col), pl.col("truncate_length")).alias(
                        f"{col}_used"
                    )
                ]
            )

        # Final effective length = sum of used lengths
        df = df.with_columns(
            [
                pl.sum_horizontal([pl.col(f"{col}_used") for col in included])
                .cast(pl.Int32)
                .alias("length")
            ]
        )

        self._lengths = df.select(["person_id", "length"])
        self._lengths.write_parquet(path)
        print(f"Done determining family lengths. Saved to {fname}")

    def train_dataloader(self):
        return self.get_dataloader(
            self.train_dataset, sampler=RandomSampler(range(len(self.train_dataset)))
        )
