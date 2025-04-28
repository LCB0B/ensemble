
import math
import shutil
import pickle # Added for loading vocab/pnr
from pathlib import Path
from typing import Literal, List, Optional # Added Optional
import torch
import polars as pl # For background and outcomes/parents in finetuning
import lightning as L
# import pyarrow.dataset as ds # No longer needed for sources
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torch.nn.attention.flex_attention import create_block_mask, and_masks

# Assuming these imports remain valid and contain necessary classes/functions
from src.dataset import LMDBDataset, FinetuneLMDBDataset, ParentsFinetuneLMDBDataset
from src.collate_fn2 import (
    Collate,
    MaskCollate,
    CausalCollate,
    CensorCollate,
    CausalEventCollate,
    ParentCausalEventCollate,
)
from src.utils import get_background_length # Reactivated for background calculation
from src.utils import calculate_abspos, create_weights
from src.paths import FPATH, check_and_copy_file_or_dir
from src.utils import flex_attn_causal_event, flex_attn_padding

import pdb
import os
import json

ONE_YEAR_ABSPOS = 365.25 * 24

# Function to load vocabulary (adjust based on how it's saved)
def load_json(path: Path):
    """Loads a dictionary or list from a JSON file."""
    print(f"Loading JSON object from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f: # Use utf-8 encoding
            obj = json.load(f)
        print(f"Successfully loaded JSON object from {path}")
        return obj
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {path}: {e}")
        raise
    except Exception as e:
        print(f"Error loading JSON object from {path}: {e}")
        raise

# pylint: disable=arguments-differ
class BaseLightningDataModule(L.LightningDataModule):
    """
    Base Lightning Data Module modified to use pre-compiled LMDB, vocab, pnr_to_idx,
    and to accept a background DataFrame similar to datamodule2.
    """

    def __init__(
        self,
        dir_path: Path, # Path for logs, checkpoints, etc.
        lmdb_path: Path, # Path to the pre-built LMDB dataset directory/file
        vocab_path: Path, # Path to the pre-built vocabulary file (e.g., vocab.pkl)
        pnr_to_idx_path: Optional[Path] = None, # Path to pnr_to_idx mapping (optional)
        background: Optional[pl.DataFrame] = None, # Added background parameter
        cls_token: bool = True, # Still needed for config/collate
        sep_token: bool = False, # Still needed for config/collate
        segment: bool = False,
        fill_nulls: bool = False, # Added for background handling
        subset_background: bool = False, # Added for background handling
        background_length: Optional[int] = None, # Now optional, calculated if background provided
        num_workers: int = 0,
        batch_size: int = 128,
        max_seq_len: int = 512,
        cutoff: int = 0, # Added for compatibility
    ):
        super().__init__()
        # Init Path related stuff
        self.dir_path = dir_path
        self.lmdb_path = lmdb_path
        self.vocab_path = vocab_path
        self.pnr_to_idx_path = pnr_to_idx_path
        check_and_copy_file_or_dir(self.dir_path) # Keep for output dir mgmt if needed

        # Load pre-compiled vocabulary
        self.vocab = load_json(self.vocab_path)

        # Load pre-compiled pnr_to_idx map (if path provided)
        self.pnr_to_idx = None
        if self.pnr_to_idx_path:
            if self.pnr_to_idx_path.exists():
                self.pnr_to_idx = load_json(self.pnr_to_idx_path)
            else:
                print(f"Warning: pnr_to_idx path provided but file not found: {self.pnr_to_idx_path}")
        else:
            print("Info: No pnr_to_idx path provided.")

        # Background handling
        self.background = background
        self.fill_nulls = fill_nulls
        self.subset_background = subset_background
        self.cutoff = cutoff

        # Init other arg-related stuff
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.segment = segment
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.max_seq_len = max_seq_len

        # Calculate background_length if background provided, otherwise use provided value
        if background is not None:
            self.background_length = get_background_length(background) + int(cls_token) + int(sep_token)
            print(f"Calculated background_length: {self.background_length} based on provided background DataFrame")
        elif background_length is not None:
            self.background_length = background_length
            print(f"Using provided background_length: {self.background_length}")
        else:
            self.background_length = 0
            print("Warning: Neither background nor background_length provided. Setting background_length to 0.")
        
        # Calculate truncate_length
        self.truncate_length = self.max_seq_len - self.background_length
        if self.truncate_length <= 0:
            raise ValueError(f"max_seq_len ({max_seq_len}) must be greater than background_length ({self.background_length})")

        # Dir path creation
        self.dir_path.mkdir(parents=True, exist_ok=True)

        # Avoid lint complaints / Setup dataset holders
        self.dataset: Optional[LMDBDataset] = None # Main dataset holder
        self.train_dataset: Optional[LMDBDataset] = None
        self.val_dataset: Optional[LMDBDataset] = None
        self.test_dataset: Optional[LMDBDataset] = None
        self.predict_dataset: Optional[LMDBDataset] = None

    def prepare_data(self):
        """Data is assumed to be pre-compiled. This method does nothing."""
        print(f"Data preparation step skipped: Using pre-compiled data from {self.lmdb_path}")
        # Verify LMDB path exists
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB dataset not found at specified path: {self.lmdb_path}")

    def _load_dataset(self, path: Path) -> LMDBDataset:
        """Loads the base LMDB dataset from the specified path."""
        print(f"Attempting to load LMDBDataset from: {path}")
        # Use the MODIFIED LMDBDataset constructor (only needs path)
        return LMDBDataset(lmdb_path=path) # Pass lmdb_path keyword argument
        
    def setup(self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None):
        """Loads the dataset and splits it if necessary."""
        print(f"Setting up data for stage: {stage}")
        # Load the dataset only once
        if self.dataset is None:
            self.dataset = self._load_dataset(self.lmdb_path)
            print(f"Dataset loaded successfully. Size: {len(self.dataset)}")

        # Split based on stage
        if stage == "fit" or stage is None:
            if self.train_dataset is None or self.val_dataset is None:
                print("Splitting dataset for 'fit' stage (80/20 split)")
                self.train_dataset, self.val_dataset = self.dataset.split(0.8)
                print(f"Train dataset size: {len(self.train_dataset)}")
                print(f"Validation dataset size: {len(self.val_dataset)}")
            else:
                print("Using existing train/validation split.")

        elif stage == "validate":
            if self.val_dataset is None:
                # If only validate is called, we might need to create the split
                _, self.val_dataset = self.dataset.split(0.8)
                print(f"Validation dataset created (size: {len(self.val_dataset)})")

        elif stage == "test":
            # Assuming validation set is used for testing if no dedicated test set
            if self.val_dataset is None:
                _, self.val_dataset = self.dataset.split(0.8) # Or load a dedicated test set
            self.test_dataset = self.val_dataset # Assign to test_dataset attribute
            print(f"Test dataset assigned (using validation set, size: {len(self.test_dataset)})")

        elif stage == "predict":
            # Often prediction uses the validation or a separate prediction set
            if self.predict_dataset is None:
                self.predict_dataset = self.val_dataset # Default to val set for prediction
                if self.predict_dataset is None: # If val_dataset wasn't created
                    _, self.predict_dataset = self.dataset.split(0.8)
                print(f"Predict dataset assigned (using validation set, size: {len(self.predict_dataset)})")

        else:
            print(f"Setup stage '{stage}' not explicitly handled for splitting, using full dataset if needed later.")

    def collate_fn(self):
        """Returns the base Collate function."""
        return Collate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )

    def get_dataloader(self, dataset: LMDBDataset, sampler=None, shuffle=False):
        """Returns a generic DataLoader with given attributes from self and kwargs"""
        if dataset is None:
            raise ValueError("Dataset is not loaded. Ensure setup() has been called.")
        if len(dataset) == 0:
            print("Warning: Dataset is empty.")
            return None

        # Create batch_sampler based on sampler or shuffle parameters
        if sampler is None and shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)
        elif sampler is not None:
            batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)
        else: # No sampler, no shuffle -> sequential
            sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=False)

        # Use keyword arguments to avoid passing any conflicting parameters
        loader_kwargs = {
            'dataset': dataset,
            'batch_sampler': batch_sampler,
            'num_workers': self.num_workers,
            'persistent_workers': self.num_workers > 0,
            'collate_fn': self.collate_fn(),
            'pin_memory': True
    }
    
        # Create dataloader with only compatible parameters
        return DataLoader(**loader_kwargs)

    def train_dataloader(self):
        """Returns the train dataloader for self.train_dataset"""
        if self.train_dataset is None:
            self.setup('fit') # Ensure dataset is split
        return self.get_dataloader(self.train_dataset, shuffle=True) # Typically shuffle train data

    def val_dataloader(self):
        """Returns the val dataloader for self.val_dataset"""
        if self.val_dataset is None:
            self.setup('fit') # Ensure dataset is split
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """Returns the test dataloader."""
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            self.setup('test') # Ensure test dataset exists
        return self.get_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        """Returns the prediction dataloader for self.predict_dataset"""
        if self.predict_dataset is None:
            self.setup('predict') # Ensure predict dataset exists
        return self.get_dataloader(self.predict_dataset, shuffle=False)

    def get_steps_per_train_epoch(self):
        """Returns length of dataloader (calls setup and teardown)"""
        if self.train_dataset is None:
            self.setup('fit')
        if self.train_dataset is None or len(self.train_dataset) == 0:
            return 0
        return math.ceil(len(self.train_dataset) / self.batch_size)

    def teardown(self, stage: str = None):
        """Copies all contents from dir_path to opposite drive if they do not exist."""
        print(f"Teardown called for stage: {stage}")
        swapped_path = FPATH.swap_drives(self.dir_path)
        if swapped_path == self.dir_path:
            print("Source and destination paths for teardown are the same. Skipping copy.")
            return

        swapped_path.mkdir(parents=True, exist_ok=True)
        print(f"Checking/copying contents of {self.dir_path} to {swapped_path}")

        copied_count = 0
        for item in self.dir_path.iterdir():
            dest = swapped_path / item.name
            try:
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(item, dest)
                        print(f"  Copied directory: {item.name}")
                    else:
                        shutil.copy2(item, dest)
                        print(f"  Copied file: {item.name}")
                    copied_count += 1
            except Exception as e:
                print(f"  Error copying {item.name}: {e}")
        if copied_count == 0:
            print("No new items needed copying during teardown.")
        else:
            print(f"Copied {copied_count} items during teardown.")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # This method should be fine as it operates on the batch *after* loading/collation
        for key, v in batch.items():
            if torch.is_tensor(v):
                batch[key] = v.to(device, non_blocking=True)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # This method should also be fine as it operates on the transferred batch
        if "event" not in batch or not torch.is_tensor(batch["event"]):
            print("Warning: 'event' key missing or not a tensor in batch during on_after_batch_transfer. Skipping mask creation.")
            return batch
        if "sequence_lens" not in batch or not torch.is_tensor(batch["sequence_lens"]):
            print("Warning: 'sequence_lens' key missing or not a tensor in batch. Padding mask might be incorrect.")
            return batch

        bs, seq_len = batch["event"].shape

        # Define the causal mask function locally or import if used elsewhere consistently
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Use try-except for potentially missing keys if padding mask is optional
        try:
            padding_mask = flex_attn_padding(batch["sequence_lens"])
        except KeyError:
            print("Warning: 'sequence_lens' not found in batch. Cannot apply padding mask.")
            # Create a mask assuming no padding if sequence_lens is missing
            padding_mask = lambda b, h, q_idx, kv_idx: True

        # Combine masks
        combined_mask_fn = and_masks(causal_mask, padding_mask)

        # Create block mask
        if bs > 0 and seq_len > 0:
            batch["attn_mask"] = create_block_mask(
                combined_mask_fn,
                bs,
                None, # num_heads is inferred or not needed by create_block_mask here
                seq_len,
                seq_len,
                _compile=True
            )
        elif "attn_mask" not in batch: # Handle empty batch case
            batch["attn_mask"] = None

        return batch


# --- Subclasses Modifications ---
# All other subclasses remain the same as in the previous version

class PretrainLifeLightningDataModule(BaseLightningDataModule):
    """Lightning Data Module for MLM pretraining (using pre-compiled data)"""

    def collate_fn(self):
        return MaskCollate(
            vocab=self.vocab,
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )

class LifeLightningDataModule(BaseLightningDataModule):
    """Lightning Data Module for Causal pretraining (using pre-compiled data)"""

    def collate_fn(self):
        return CausalCollate(
            vocab=self.vocab, # Use loaded vocab
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
        )


class FinetuneLifeLightningDataModule(BaseLightningDataModule):
    """Lightning Data Module for binary finetuning (using pre-compiled data)"""

    def __init__(
        self,
        *args, # Pass Base class args through
        outcomes: pl.DataFrame, # Keep outcomes DataFrame argument
        negative_censor: float,
        **kwargs, # Pass Base class kwargs through
    ):
        # Call Base __init__ first to load vocab, set up paths etc.
        super().__init__(*args, **kwargs)
        # Finetuning specific setup
        self.outcomes = outcomes
        # Check if outcomes df is empty
        if self.outcomes.height == 0:
             print("Warning: Provided outcomes DataFrame is empty.")
        self.negative_censor = negative_censor

    # Override _load_dataset to use the correct LMDBDataset class
    def _load_dataset(self, path: Path) -> FinetuneLMDBDataset:
        """Loads the FinetuneLMDBDataset and injects outcomes."""
        print(f"Attempting to load FinetuneLMDBDataset from: {path}")
        if self.outcomes is None or self.outcomes.height == 0:
             raise ValueError("Outcomes DataFrame is missing or empty, required for FinetuneLMDBDataset.")

        # Prepare outcomes_dict (same logic as before)
        # Ensure 'censor' column exists
        if "censor" not in self.outcomes.columns:
             raise ValueError("Outcomes DataFrame must contain a 'censor' column.")
        outcomes_dict = self.outcomes.with_columns(
            calculate_abspos(pl.col("censor")) # Assumes calculate_abspos handles potential nulls if any
        ).to_dict(as_series=False)

        # IMPORTANT: Ensure FinetuneLMDBDataset can be initialized with path and outcomes_dict
        return FinetuneLMDBDataset(path=path, observations=outcomes_dict) # Adjust constructor call as needed

    def collate_fn(self):
        # Collate function specific to finetuning
        return CensorCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            segment=self.segment,
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
        )

    def train_dataloader(self):
        # Uses WeightedRandomSampler based on outcomes
        if self.train_dataset is None:
             self.setup('fit')
        if not isinstance(self.train_dataset, FinetuneLMDBDataset):
             raise TypeError("Train dataset is not of type FinetuneLMDBDataset, cannot get outcomes for sampler.")
        if self.train_dataset is None or len(self.train_dataset) == 0:
            print("Warning: Train dataset is empty or None in Finetune train_dataloader.")
            return None # Or handle appropriately

        # Access outcomes - Ensure 'observations' attribute exists and has 'outcome'
        try:
            train_outcomes = self.train_dataset.observations["outcome"]
            if not isinstance(train_outcomes, (list, tuple)) or len(train_outcomes) == 0:
                 raise ValueError("Outcomes are empty or not in expected format.")
        except (AttributeError, KeyError, TypeError) as e:
             raise ValueError(f"Could not retrieve 'outcome' from train_dataset.observations: {e}")


        weights = create_weights(train_outcomes, op=math.sqrt)
        num_samples = len(train_outcomes)
        if num_samples == 0:
             print("Warning: No samples in train outcomes, cannot create sampler.")
             return self.get_dataloader(self.train_dataset, shuffle=False) # Fallback? Or error?

        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True
        )
        # Pass the sampler to get_dataloader, ensure shuffle=False if sampler is used
        return self.get_dataloader(self.train_dataset, sampler=sampler, shuffle=False)

    # on_after_batch_transfer is inherited from Base, should be fine


class RiskFinetuneLifeLightningDataModule(FinetuneLifeLightningDataModule):
    """Lightning Data Module for risk trajectories finetuning (using pre-compiled data)"""

    def __init__(
        self,
        *args,
        prediction_windows: List[float],
        # max_abspos is now loaded from dataset in get_dataloader, remove from init?
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prediction_windows = prediction_windows
        self.max_abspos: Optional[float] = None # Will be set when data loaded

    def _load_dataset(self, path: Path) -> FinetuneLMDBDataset:
        # Override needed if Risk uses a specific LMDB class, otherwise inherit FineTune's
        # Assuming it uses the same FinetuneLMDBDataset for now
        dataset = super()._load_dataset(path)
        # Get max_abspos after loading
        try:
             # Ensure the loaded dataset object has this method
             self.max_abspos = dataset.get_max_abspos()
             print(f"Max abspos from dataset: {self.max_abspos}")
        except AttributeError:
             print("Warning: Loaded dataset does not have get_max_abspos method. Max abspos set to None.")
             self.max_abspos = None # Or raise error if critical
        return dataset


    # get_dataloader modification removed from here, handled in _load_dataset now

    def collate_fn(self):
        # Uses the specific CausalEventCollate
        if self.max_abspos is None:
             print("Warning: max_abspos not set before creating collate_fn. Trying to load dataset to get it.")
             # Attempt to load dataset if not already loaded to get max_abspos
             # This might be inefficient if called multiple times. Best to ensure setup() runs first.
             if self.dataset is None:
                  self.setup() # Call setup to load the dataset and set max_abspos
             if self.max_abspos is None:
                   raise ValueError("max_abspos could not be determined from the dataset, but is required for CausalEventCollate.")

        return CausalEventCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
            segment=self.segment,
            max_abspos=self.max_abspos, # Use the loaded max_abspos
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # This method creates specific masks for risk prediction. Should be okay if
        # CausalEventCollate produces the required keys ('event', 'acc_event_lens',
        # 'sequence_lens', 'target', 'first_abspos', 'abspos').
        required_keys = ['event', 'acc_event_lens', 'sequence_lens', 'target', 'first_abspos', 'abspos']
        for key in required_keys:
             if key not in batch:
                 print(f"Warning: Key '{key}' missing in batch for RiskFinetune on_after_batch_transfer. Mask creation might fail.")
                 # Depending on severity, return batch or raise error
                 # return batch # Return early if essential keys are missing
                 raise KeyError(f"Essential key '{key}' missing in batch for RiskFinetune mask creation.")


        bs, seq_len = batch["event"].shape
        if bs == 0 or seq_len == 0:
             print("Warning: Empty batch received in RiskFinetune on_after_batch_transfer.")
             return batch # Handle empty batch

        # Event mask creation (grid-based) - seems okay
        try:
            batch["event_mask"] = self.create_event_mask(batch)
        except Exception as e:
             print(f"Error creating event mask: {e}")
             raise

        # Attention mask creation (causal event-based) - seems okay
        try:
            range_tensor = torch.arange(seq_len, device=batch["event"].device)
            range_tensor_exp = range_tensor.expand(
                batch["acc_event_lens"].size(0), -1
            ).contiguous()
            causal_mapping = torch.searchsorted(
                batch["acc_event_lens"], range_tensor_exp, right=True
            )
            batch["attn_mask"] = create_block_mask(
                and_masks(
                    flex_attn_causal_event(causal_mapping),
                    flex_attn_padding(batch["sequence_lens"]),
                ),
                bs, None, seq_len, seq_len, _compile=True
            )
        except Exception as e:
            print(f"Error creating attention mask: {e}")
            raise # Re-raise after logging

        return batch

    def create_event_mask(self, batch):
        """Creates the event mask based on the prediction type (grid or per event)"""
        # This logic depends only on batch contents provided by the collate function.
        # Ensure 'target', 'first_abspos', 'abspos' are present and correct.
        if not all(k in batch for k in ['target', 'first_abspos', 'abspos']):
             raise KeyError("Missing required keys for create_event_mask")

        num_buckets = batch["target"].size(1)
        device = batch["abspos"].device
        grid_step = ONE_YEAR_ABSPOS / 2 # Example: half-year steps
        # Calculate end point based on num_buckets (e.g., if num_buckets is 10, it covers 5 years)
        grid_end = ONE_YEAR_ABSPOS * (num_buckets // 2)
        # Create grid tensor relative to first_abspos
        grid_rel = torch.arange(0, grid_end, grid_step, device=device)
        grid = batch["first_abspos"].unsqueeze(1) + grid_rel

        # Compare event's absolute position to the grid points
        # Mask is True if event time <= grid time point
        event_mask = (batch["abspos"].unsqueeze(1) <= grid.unsqueeze(-1))

        return event_mask.half() # Use half precision if appropriate


# --- Parent-related Modules ---
# Need careful check of _load_dataset overrides and collate functions

class ParentsFinetuneLifeLightningDataModule(FinetuneLifeLightningDataModule):
    """Lightning Data Module for parent finetuning (using pre-compiled data)"""

    def __init__(
        self,
        *args,
        parents: pl.DataFrame, # Keep parents DataFrame argument
        # prediction_windows parameter was missing here but present in collate fn usage? Add it.
        prediction_windows: List[float], # Added based on collate function needs
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Check parents df
        if parents is None or parents.height == 0:
             print("Warning: Provided parents DataFrame is empty.")
             # Decide how to handle: error or proceed? Depends on logic.
             raise ValueError("Parents DataFrame is required for ParentsFinetune module but is missing or empty.")

        # Join outcomes with parents (ensure 'person_id' exists)
        if "person_id" not in self.outcomes.columns or "person_id" not in parents.columns:
             raise ValueError("Both outcomes and parents DataFrames must contain 'person_id' for joining.")
        # Perform the join, handle potential missing matches if needed (how='left')
        self.outcomes = self.outcomes.join(parents, on="person_id", how="left")
        # Store prediction windows if needed by collate fn (inherited Risk uses it)
        self.prediction_windows = prediction_windows


    # Override _load_dataset to use the correct LMDBDataset class
    def _load_dataset(self, path: Path) -> ParentsFinetuneLMDBDataset:
        """Loads the ParentsFinetuneLMDBDataset and injects combined outcomes."""
        print(f"Attempting to load ParentsFinetuneLMDBDataset from: {path}")
        if self.outcomes is None or self.outcomes.height == 0:
             raise ValueError("Outcomes DataFrame (potentially joined with parents) is missing or empty.")

        # Prepare outcomes_dict (now includes parent info)
        if "censor" not in self.outcomes.columns:
            raise ValueError("Joined outcomes DataFrame must still contain a 'censor' column.")
        outcomes_dict = self.outcomes.with_columns(
            calculate_abspos(pl.col("censor"))
        ).to_dict(as_series=False)

        # IMPORTANT: Ensure ParentsFinetuneLMDBDataset constructor is correct
        return ParentsFinetuneLMDBDataset(path=path, observations=outcomes_dict)

    # This module had collate_fn defined as ParentCausalEventCollate
    # but inherited RiskFinetune which had CausalEventCollate. Resolve this.
    # Assuming ParentCausalEventCollate is the correct one for this class.
    def collate_fn(self):
        # Determine max_abspos similarly to RiskFinetune
        if not hasattr(self, 'max_abspos') or self.max_abspos is None:
             if self.dataset: # Check if dataset loaded
                 try:
                     self.max_abspos = self.dataset.get_max_abspos()
                 except AttributeError:
                      print("Warning: Parent dataset has no get_max_abspos.")
                      self.max_abspos = None # Handle missing max_abspos
             else:
                 # Try loading if not loaded (might happen if collate_fn called before setup completes fully)
                 print("Warning: max_abspos not set. Attempting setup() to load dataset.")
                 self.setup()
                 if not hasattr(self, 'max_abspos') or self.max_abspos is None:
                       raise ValueError("max_abspos could not be determined from the dataset for ParentCausalEventCollate.")


        # Use ParentCausalEventCollate
        return ParentCausalEventCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            # Pass prediction windows, negative censor, segment, max_abspos
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
            segment=self.segment,
            max_abspos=self.max_abspos, # Pass determined max_abspos
        )


# ParentsRiskFinetuneLifeLightningDataModule seems like a duplicate or alternative naming
# of ParentsFinetuneLifeLightningDataModule, as both inherit from Finetune/RiskFinetune
# and perform the parent join. I'll keep it but ensure it's consistent.
# It inherits from RiskFinetune, so it gets its `on_after_batch_transfer` and mask logic.
class ParentsRiskFinetuneLifeLightningDataModule(RiskFinetuneLifeLightningDataModule):
    """Lightning Data Module for parent risk trajectories finetuning (using pre-compiled data)"""

    def __init__(
        self,
        *args,
        parents: pl.DataFrame,
        **kwargs,
    ):
        # Call RiskFinetune's init first
        super().__init__(*args, **kwargs)
        # Join outcomes (loaded in RiskFinetune's super() call) with parents
        if parents is None or parents.height == 0:
             raise ValueError("Parents DataFrame is required for ParentsRiskFinetune module but is missing or empty.")
        if "person_id" not in self.outcomes.columns or "person_id" not in parents.columns:
             raise ValueError("Both outcomes and parents DataFrames must contain 'person_id' for joining.")

        self.outcomes = self.outcomes.join(parents, on="person_id", how="left")

    # Override _load_dataset again to ensure the final dataset type is correct
    # and uses the joined outcomes.
    def _load_dataset(self, path: Path) -> ParentsFinetuneLMDBDataset:
        """Loads the ParentsFinetuneLMDBDataset using joined outcomes."""
        print(f"Attempting to load ParentsFinetuneLMDBDataset (via ParentsRisk) from: {path}")
        if self.outcomes is None or self.outcomes.height == 0:
             raise ValueError("Joined outcomes DataFrame is missing or empty.")

        # Prepare outcomes_dict from the *joined* self.outcomes
        if "censor" not in self.outcomes.columns:
             raise ValueError("Joined outcomes DataFrame must contain a 'censor' column.")
        outcomes_dict = self.outcomes.with_columns(
            calculate_abspos(pl.col("censor"))
        ).to_dict(as_series=False)

        dataset = ParentsFinetuneLMDBDataset(path=path, observations=outcomes_dict)

        # Get max_abspos after loading the correct dataset type
        try:
             self.max_abspos = dataset.get_max_abspos()
             print(f"Max abspos from ParentsRisk dataset: {self.max_abspos}")
        except AttributeError:
             print("Warning: Loaded Parents dataset does not have get_max_abspos method. Max abspos set to None.")
             self.max_abspos = None
        return dataset


    # Override collate_fn to use ParentCausalEventCollate
    def collate_fn(self):
        if not hasattr(self, 'max_abspos') or self.max_abspos is None:
             # Try to ensure dataset is loaded
             if self.dataset is None: self.setup()
             if not hasattr(self, 'max_abspos') or self.max_abspos is None:
                 raise ValueError("max_abspos could not be determined for ParentCausalEventCollate.")

        return ParentCausalEventCollate(
            truncate_length=self.truncate_length,
            background_length=self.background_length,
            prediction_windows=[
                ONE_YEAR_ABSPOS * window for window in self.prediction_windows
            ],
            negative_censor=ONE_YEAR_ABSPOS * self.negative_censor,
            segment=self.segment,
            max_abspos=self.max_abspos, # Use the loaded max_abspos
        )

    # Inherits on_after_batch_transfer from RiskFinetuneLifeLightningDataModule, which should be appropriate
    # if ParentCausalEventCollate produces the same required keys.