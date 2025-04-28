""" Lightning Data Module for all data-related code """

import math
import shutil
from pathlib import Path
from typing import Literal, List
import torch
import polars as pl
import lightning as L
import pyarrow.dataset as ds
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torch.nn.attention.flex_attention import create_block_mask, and_masks
from src.pipeline import DataPipeline
from src.dataset import LMDBDataset, FinetuneLMDBDataset, ParentsFinetuneLMDBDataset
from src.collate_fn2 import (
    Collate,
    MaskCollate,
    CausalCollate,
    CensorCollate,
    CausalEventCollate,
    ParentCausalEventCollate,
)
from src.utils import get_background_length, calculate_abspos, create_weights
from src.paths import FPATH, check_and_copy_file_or_dir
from src.utils import flex_attn_causal_event, flex_attn_padding

import pdb
import os 

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

        # # Init length-related stuff
        # self.background_length = (
        #     get_background_length(background) + int(cls_token) + int(sep_token)
        # )
        # self.truncate_length = (
        #     max_seq_len - self.background_length
        # )  # TODO: Reverse, so we give max_seq_len anad calculate truncate_length in collate
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


        def causal_mask(b,h,q_idx,kv_idx):
            return q_idx >= kv_idx

        if __debug__:
            pdb.set_trace()
        # Create block mask
        batch["attn_mask"] = create_block_mask(
            and_masks(
                causal_mask,
                flex_attn_padding(batch["sequence_lens"]),
            ),
            bs,
            None,
            seq_len,
            seq_len,
            _compile=True)

        return batch


class PretrainLifeLightningDataModule(BaseLightningDataModule):
    """Lightning Data Module for MLM pretraining"""

    def collate_fn(self):
        return MaskCollate(
            vocab=self.pipeline.vocab,
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )

class LifeLightningDataModule(BaseLightningDataModule):
    """Lightning Data Module for Causal pretraining"""

    def collate_fn(self):
        return CausalCollate(
            vocab=self.pipeline.vocab,
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )


    
    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     bs, seq_len = batch["event"].shape
    #     range_tensor = torch.arange(seq_len, device=batch["event"].device)

    #     if __debug__:
    #         pdb.set_trace()
        
    #     batch["event_mask"] = self.create_event_mask(batch,range_tensor)

    #     range_tensor_exp = range_tensor.expand(
    #         batch["acc_event_lens"].size(0), -1
    #     ).contiguous()
    #     # Create mapping
    #     causal_mapping = torch.searchsorted(
    #         batch["acc_event_lens"], range_tensor_exp, right=True
    #     )

    #     # Create block mask
    #     batch["attn_mask"] = create_block_mask(
    #         and_masks(
    #             flex_attn_causal_event(causal_mapping),
    #             flex_attn_padding(batch["sequence_lens"]),
    #         ),
    #         bs,
    #         None,
    #         seq_len,
    #         seq_len,
    #         _compile=True,
    #     )
    #     return batch

    # def create_event_mask(self, batch,range_tensor):
    #     """Creates the event mask based on the prediction type (grid or per event)"""
    #     # num_buckets = batch["target"].size(1)
    #     # grid = batch["first_abspos"].unsqueeze(1) + torch.arange(
    #     #     0,
    #     #     ONE_YEAR_ABSPOS * (num_buckets // 2),
    #     #     ONE_YEAR_ABSPOS // 2,
    #     #     device=batch["abspos"].device,
    #     # )
    #     # return (batch["abspos"].unsqueeze(1) <= grid.unsqueeze(-1)).half()
    #     return (
    #         (range_tensor < batch["acc_event_lens"].unsqueeze(-1))
    #         & (range_tensor < batch["sequence_lens"].unsqueeze(-1)).unsqueeze(1)
    #     ).half()



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
        outcomes_dict = self.outcomes.with_columns(
            calculate_abspos(pl.col("censor"))
        ).to_dict(as_series=False)
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prediction_windows = prediction_windows

    def get_dataloader(self, dataset: LMDBDataset, sampler=None):
        """Returns a generic DataLoader with given attributes from self and kwargs"""
        self.max_abspos = dataset.get_max_abspos()
        return super().get_dataloader(dataset, sampler)

    def collate_fn(self):
        return CausalEventCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
            segment=self.segment,
            max_abspos=self.max_abspos,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        bs, seq_len = batch["event"].shape
        range_tensor = torch.arange(seq_len, device=batch["event"].device)

        batch["event_mask"] = self.create_event_mask(batch)

        range_tensor_exp = range_tensor.expand(
            batch["acc_event_lens"].size(0), -1
        ).contiguous()
        # Create mapping
        causal_mapping = torch.searchsorted(
            batch["acc_event_lens"], range_tensor_exp, right=True
        )

        # Create block mask
        batch["attn_mask"] = create_block_mask(
            and_masks(
                flex_attn_causal_event(causal_mapping),
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
        num_buckets = batch["target"].size(1)
        grid = batch["first_abspos"].unsqueeze(1) + torch.arange(
            0,
            ONE_YEAR_ABSPOS * (num_buckets // 2),
            ONE_YEAR_ABSPOS // 2,
            device=batch["abspos"].device,
        )
        return (batch["abspos"].unsqueeze(1) <= grid.unsqueeze(-1)).half()
        # return (
        #     (range_tensor < batch["acc_event_lens"].unsqueeze(-1))
        #     & (range_tensor < batch["sequence_lens"].unsqueeze(-1)).unsqueeze(1)
        # ).half()


class ParentsFinetuneLifeLightningDataModule(FinetuneLifeLightningDataModule):
    """Lightning Data Module for parent risk trajectories finetuning"""

    def __init__(
        self,
        *args,
        parents: pl.DataFrame,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = self.outcomes.join(parents, on="person_id", how="left")

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        outcomes_dict = self.outcomes.with_columns(
            calculate_abspos(pl.col("censor"))
        ).to_dict(as_series=False)
        return ParentsFinetuneLMDBDataset(dataset, path, outcomes_dict)

    def collate_fn(self):
        return ParentCausalEventCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
            segment=self.segment,
            max_abspos=self.max_abspos,
        )


class ParentsRiskFinetuneLifeLightningDataModule(RiskFinetuneLifeLightningDataModule):
    """Lightning Data Module for parent risk trajectories finetuning"""

    def __init__(
        self,
        *args,
        parents: pl.DataFrame,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.outcomes = self.outcomes.join(parents, on="person_id", how="left")

    def _create_dataset(self, dataset: ds.Dataset, path: Path):
        outcomes_dict = self.outcomes.with_columns(
            calculate_abspos(pl.col("censor"))
        ).to_dict(as_series=False)
        return ParentsFinetuneLMDBDataset(dataset, path, outcomes_dict)

    def collate_fn(self):
        return ParentCausalEventCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
            segment=self.segment,
        )
