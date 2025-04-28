# --- Modified src/dataset.py ---

import json
import random
from typing import Self, Optional, Dict, List, Tuple, Union
from pathlib import Path
import lmdb
import pickle
import lz4.frame
import polars as pl
from tqdm import tqdm
from torch.utils.data import Dataset
from src.utils import split_dict, print_main
# from src.chunking import yield_chunks # No longer needed if creation is removed
# import pyarrow.dataset as ds # No longer needed if creation is removed
import os

class LMDBDataset(Dataset):
    """
    LMDB implementation for larger-than-memory datasets.
    MODIFIED: Assumes LMDB database and pnr_to_database_idx.json already exist.
    Creation logic has been removed.
    """

    def __init__(
        self,
        lmdb_path: Path, # Renamed path to lmdb_path for clarity
        observations: Optional[Dict] = None,
        # REMOVED: data: ds.Dataset argument
    ):
        self.lmdb_path = lmdb_path
        # self.data = None # Data attribute no longer needed

        # --- LMDB Existence Check ---
        if not lmdb_path.exists():
            raise FileNotFoundError(
                f"LMDB database not found at specified path: {lmdb_path}. "
                f"This class now assumes the database already exists."
            )

        # --- PNR Mapping File Check ---
        self.pnr_to_database_idx_path = (
            self.lmdb_path.parent / "pnr_to_database_idx.json"
        )
        if not self.pnr_to_database_idx_path.exists():
            raise FileNotFoundError(
                f"Required PNR mapping file not found: {self.pnr_to_database_idx_path}"
            )

        print_main(f"Loading existing LMDB from: {lmdb_path}")
        print_main(f"Loading PNR mapping from: {self.pnr_to_database_idx_path}")

        # --- Load PNR Mapping ---
        try:
            with open(
                self.pnr_to_database_idx_path, "r", encoding="utf-8"
            ) as json_file:
                self.pnr_to_database_idx = json.load(json_file)
        except Exception as e:
            print(f"Error loading {self.pnr_to_database_idx_path}: {e}")
            raise

        # --- Handle Observations ---
        if observations is None:
            # Default observations are all persons in the loaded mapping
            print_main("No specific observations provided, using all persons from PNR mapping.")
            observations = {"person_id": list(self.pnr_to_database_idx.keys())}
        elif not isinstance(observations, dict) or "person_id" not in observations:
            raise ValueError("Observations must be a dictionary containing at least the 'person_id' key.")

        self.observations = observations
        if len(self.observations.get("person_id", [])) == 0:
            print_main("Warning: Observations list is empty.")

        # --- Initialize LMDB Environment ---
        self.env = None # Initialize as None
        try:
            self.env = self._init_db()
            print_main(f"LMDB environment initialized successfully for {lmdb_path}.")
        except lmdb.Error as e:
            print(f"Error initializing LMDB environment at {lmdb_path}: {e}")
            # Decide if this is fatal or if env can remain None
            raise # Re-raise as it's likely fatal

    # REMOVED: _estimate_map_size method
    # REMOVED: _create_db method

    def _init_db(self):
        """Initializes the LMDB environment in read-only mode."""
        return lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,        # Readonly, no lock needed (usually)
            readahead=False,   # Performance tuning (False often better for random reads)
            meminit=False,     # Performance tuning
            create=False,      # DO NOT CREATE if missing
        )

    def __len__(self) -> int:
        # Length is based on the provided/defaulted observations
        return len(self.observations.get("person_id", []))

    def __getitem__(self, idx: Union[int, List[int]]): # Return type depends on input
        if self.env is None:
            raise RuntimeError("LMDB environment is not initialized.")

        try:
             with self.env.begin() as txn: # Open transaction here
                 if isinstance(idx, int):
                     if not (0 <= idx < len(self)):
                          raise IndexError(f"Index {idx} out of range for dataset length {len(self)}")
                     pnr = self.index_to_pnr(idx)
                     database_idx = self.pnr_to_db_idx(pnr)
                     if database_idx is None:
                          raise KeyError(f"PNR {pnr} (from observations index {idx}) not found in pnr_to_database_idx mapping.")
                     value = txn.get(self.encode_key(database_idx))
                     if value is None:
                          raise KeyError(f"Key {database_idx} (for PNR {pnr}) found in mapping but not in LMDB database {self.lmdb_path}.")
                     data = self.decode(value) # Single dictionary
                 elif isinstance(idx, list):
                     # Validate all indices in the list
                     if not all(0 <= i < len(self) for i in idx):
                          invalid_indices = [i for i in idx if not (0 <= i < len(self))]
                          raise IndexError(f"Indices {invalid_indices} out of range for dataset length {len(self)}")
                     # --- Call the RENAMED internal method with the transaction ---
                     data = self._getitems_internal(idx, txn) # List of dictionaries
                 else:
                     raise TypeError(f"Index must be int or list, not {type(idx)}")
        except lmdb.Error as e:
             print(f"LMDB error during __getitem__: {e}")
             raise

        return data

    # --- RENAMED internal helper method ---
    def _getitems_internal(self, indices: list, txn: lmdb.Transaction) -> List[dict]:
        """Internal helper to get multiple items efficiently within a transaction."""
        # (Keep the implementation of the original __getitems__ method here)
        data_list = []
        pnrs = [self.index_to_pnr(i) for i in indices]
        database_idxs = [self.pnr_to_db_idx(pnr) for pnr in pnrs]
        keys_to_fetch = [self.encode_key(db_idx) for db_idx in database_idxs if db_idx is not None]

        missing_pnrs = [pnrs[i] for i, db_idx in enumerate(database_idxs) if db_idx is None]
        if missing_pnrs:
            raise KeyError(f"PNRs {missing_pnrs} (from observations indices) not found in pnr_to_database_idx mapping.")

        fetched_map = {}
        if keys_to_fetch: # Only query if there are valid keys
             try:
                 with txn.cursor() as cur:
                     fetched_values = cur.getmulti(keys_to_fetch)
                     # Filter out None values right after fetching
                     fetched_map = {key: self.decode(val) for key, val in fetched_values if val is not None}
             except lmdb.Error as e:
                 print(f"LMDB error during getmulti: {e}")
                 raise # Or handle more gracefully if appropriate

        # Reconstruct the data list in the original order of indices
        result_list = []
        for i, db_idx in enumerate(database_idxs):
            key_bytes = self.encode_key(db_idx)
            if key_bytes in fetched_map:
                 result_list.append(fetched_map[key_bytes])
            else:
                 # This could happen if the key was in mapping but missing/null in LMDB or if getmulti failed silently
                 raise KeyError(f"Key {db_idx} (for PNR {pnrs[i]}) expected but data not found or null in LMDB result from {self.lmdb_path}.")

        return result_list

    def close_db(self):
        """Closes database if active"""
        if self.env is not None:
            print_main(f"Closing LMDB environment for {self.lmdb_path}")
            self.env.close()
            self.env = None # Set to None after closing

    def split(self, split_ratio: float) -> Tuple[Self, Self]: # Correct return type hint
        """Splits the Dataset class in two by splitting self.observations based on split_ratio"""
        print_main(f"Splitting observations with ratio {split_ratio}")
        obs1, obs2 = split_dict(self.observations, split_ratio)
        print_main(f"Split sizes: {len(obs1.get('person_id',[]))}, {len(obs2.get('person_id',[]))}")
        d1 = self.subset(obs1)
        d2 = self.subset(obs2)
        return d1, d2

    def split_dict_by_person_ids(
        self, data: Dict[str, List], ids_split1: List[int], ids_split2: List[int]
    ) -> Tuple[Dict[str, List], Dict[str, List]]:
        """
        Splits a dictionary into two based on two lists of person_ids.
        (Static method potential, or move to utils?)
        """
        # Convert ids lists to sets for faster lookup
        ids_set1 = set(map(str, ids_split1)) # Ensure comparison uses strings if PNRs are strings
        ids_set2 = set(map(str, ids_split2))

        # Convert observation person_ids to strings for comparison
        current_person_ids = [str(p) for p in data.get("person_id", [])]

        # Find indices corresponding to each split
        indices1 = [i for i, pid in enumerate(current_person_ids) if pid in ids_set1]
        indices2 = [i for i, pid in enumerate(current_person_ids) if pid in ids_set2]

        # Create new dictionaries by selecting based on indices
        split1 = {key: [value[i] for i in indices1] for key, value in data.items()}
        split2 = {key: [value[i] for i in indices2] for key, value in data.items()}

        return split1, split2

    def split_by_person_ids(self, ids_split1: list[int], ids_split2: list[int]) -> Tuple[Self, Self]: # Correct hint
        """
        Splits the Dataset class into two by filtering self.observations based on lists of person IDs.
        """
        print_main(f"Splitting observations by provided person ID lists.")
        obs1, obs2 = self.split_dict_by_person_ids(
            self.observations, ids_split1, ids_split2
        )
        print_main(f"Split sizes: {len(obs1.get('person_id',[]))}, {len(obs2.get('person_id',[]))}")
        d1 = self.subset(obs1)
        d2 = self.subset(obs2)
        return d1, d2

    def subset(self, observations_subset: dict) -> Self:
        """
        Creates a new Dataset instance pointing to the same LMDB
        but with a different subset of observations.
        """
        print_main(f"Creating subset with {len(observations_subset.get('person_id',[]))} observations.")
        # MODIFIED: Does not pass self.data anymore
        return self.__class__(self.lmdb_path, observations_subset)

    # --- Helper methods ---
    def index_to_pnr(self, idx: int) -> str: # Return str consistently
        """Converts dataset index to associated pnr (string)."""
        try:
            return str(self.observations["person_id"][idx])
        except IndexError:
            raise IndexError(f"Index {idx} is out of range for observations list of length {len(self.observations['person_id'])}.")
        except KeyError:
            raise KeyError("Observations dictionary missing 'person_id' key.")

    def pnr_to_db_idx(self, pnr: Union[str, int]) -> Optional[int]: # Return Optional[int]
        """Finds the database idx (int) for the pnr (str or int). Returns None if not found."""
        return self.pnr_to_database_idx.get(str(pnr)) # Returns None if key doesn't exist

    # --- Encoding/Decoding (Static methods as they don't depend on self) ---
    @staticmethod
    def encode(val) -> bytes:
        """The byte encoding using for the values"""
        return lz4.frame.compress(pickle.dumps(val))

    @staticmethod
    def encode_key(key: Union[str, int, None]) -> bytes: # Allow None for dummy key if needed elsewhere
        """Simple utf-8 encoding for keys"""
        return str(key).encode("utf-8")

    @staticmethod
    def decode(val: bytes) -> dict: # Expect bytes, return dict
        """Decode byte data using LZ4 and pickle."""
        if val is None:
            # This case should ideally be caught before calling decode
            print_main("Warning: Attempted to decode a None value.")
            return {} # Or raise error? Returning empty dict might hide issues.
            # raise ValueError("Attempted to decode a None value.")
        try:
            # Attempt LZ4 decompression then Pickle loading
            decompressed = lz4.frame.decompress(val)
            return pickle.loads(decompressed)
        except (lz4.frame.LZ4Error, pickle.UnpicklingError, EOFError, TypeError, ValueError) as e:
            # Catch potential errors during decompression or unpickling
            # Log the error and the first few bytes for debugging
            val_preview = val[:60] # Show first 60 bytes
            print(f"Error decoding value (preview: {val_preview!r}): {e}")
            # Decide how to handle: re-raise, return default, etc.
            raise ValueError(f"Failed to decode LZ4/Pickle value (preview: {val_preview!r})") from e
        # Removed JSON fallback as encode uses Pickle


class FinetuneLMDBDataset(LMDBDataset):
    """
    Finetuning version of LMDBDataset.
    MODIFIED: Assumes LMDB exists, takes outcomes dict as observations.
    """
    def __init__(
        self,
        lmdb_path: Path,
        outcomes: dict, # Use outcomes dict directly as observations
        # REMOVED: data: ds.Dataset argument
    ):
        # Pass outcomes directly as the observations dict to the base class
        super().__init__(lmdb_path=lmdb_path, observations=outcomes)

    def __getitem__(self, idx: Union[int, List[int]]):
        # Get the core data using the base class method
        # This will return either a single dict (for int idx) or a list of dicts (for list idx)
        core_data = super().__getitem__(idx)

        # Retrieve corresponding outcome info from self.observations
        if isinstance(idx, int):
            # Get the single outcome dict for the given index
            # Base class __getitem__ already validated the index
            outcome_info = {key: val[idx] for key, val in self.observations.items()}
            return core_data, outcome_info # Return tuple (dict, dict)
        elif isinstance(idx, list):
            # Get a list of outcome dicts corresponding to the indices
            # Base class __getitem__ already validated the indices
            outcome_info_list = [
                {key: val[i] for key, val in self.observations.items()}
                for i in idx # Iterate through original list of indices
            ]
            # core_data is already a list of dicts from super().__getitem__(idx)
            return core_data, outcome_info_list # Return tuple (list[dict], list[dict])
        else:
            # This case should be caught by base class, but added for robustness
            raise TypeError(f"Index must be int or list, not {type(idx)}")

    # subset method is inherited and should work correctly as it calls
    # self.__class__(self.lmdb_path, observations_subset)


class ParentsFinetuneLMDBDataset(FinetuneLMDBDataset):
    """
    Parents version of FinetuneLMDBDataset.
    MODIFIED: Assumes LMDB exists. Loads parent data based on outcome info.
    """
    # __init__ is inherited from FinetuneLMDBDataset, which correctly calls
    # the modified base LMDBDataset.__init__

    def __getitem__(self, idx: Union[int, List[int]]):
        # Get core data and outcome info from the parent class (FinetuneLMDBDataset)
        core_data, outcome_info = super().__getitem__(idx)

        if self.env is None:
            raise RuntimeError("LMDB environment is not initialized.")

        # Prepare to fetch parent data
        all_parents_data = []

        # Determine if we are processing a single item or a batch
        is_single_item = isinstance(idx, int)
        outcome_list = [outcome_info] if is_single_item else outcome_info

        # Collect all parent database keys needed across the batch
        parent_keys_to_fetch = []
        parent_key_map = {} # Map key back to its position in the batch/single item
        parent_rel_map = {} # Map key back to relation details

        for item_idx, single_outcome in enumerate(outcome_list):
            parents_in_outcome = single_outcome.get("parents") # Safely get parents list
            if isinstance(parents_in_outcome, list): # Check if it's a list
                for parent_info in parents_in_outcome:
                    # Check if parent_info is a dict and has required keys
                    if isinstance(parent_info, dict) and "parent_id" in parent_info and "relation_details" in parent_info:
                        parent_pnr = str(parent_info["parent_id"])
                        relation = parent_info["relation_details"]
                        parent_db_idx = self.pnr_to_db_idx(parent_pnr)
                        if parent_db_idx is not None:
                            parent_key = self.encode_key(parent_db_idx)
                            parent_keys_to_fetch.append(parent_key)
                            # Store where this key belongs and its relation
                            parent_key_map[parent_key] = item_idx
                            parent_rel_map[parent_key] = relation
                        else:
                            print_main(f"Warning: Parent PNR {parent_pnr} not found in mapping file. Skipping.")
                    else:
                        print_main(f"Warning: Invalid parent info format in outcome index {idx if is_single_item else idx[item_idx]}: {parent_info}. Skipping.")
            elif parents_in_outcome is not None:
                print_main(f"Warning: 'parents' field in outcome index {idx if is_single_item else idx[item_idx]} is not a list: {parents_in_outcome}. Skipping.")

        # Fetch parent data in one go if needed
        fetched_parent_values = {}
        if parent_keys_to_fetch:
            try:
                with self.env.begin() as txn:
                    with txn.cursor() as cur:
                        # Use getmulti for efficiency
                        fetched_parent_values_list = cur.getmulti(list(set(parent_keys_to_fetch))) # Fetch unique keys
                        fetched_parent_values = {key: self.decode(val) for key, val in fetched_parent_values_list if val is not None}
            except lmdb.Error as e:
                print(f"LMDB error fetching parent data: {e}")
                # Decide how to handle - skip parents? raise error?
                # For now, we'll proceed without the fetched data if there's an error
                fetched_parent_values = {} # Reset to empty

        # Structure the fetched parent data back into the batch/single item structure
        structured_parents = [{} for _ in outcome_list] # Initialize list of dicts for parents
        for key, decoded_data in fetched_parent_values.items():
            if key in parent_key_map and key in parent_rel_map:
                item_idx = parent_key_map[key]
                relation = parent_rel_map[key]
                # Add parent data only if it has events (or adjust logic as needed)
                if decoded_data and decoded_data.get("event"): # Check if data exists and has 'event' key
                    structured_parents[item_idx][relation] = decoded_data
                # else: # Optional: Log if parent had no event data
                #    print_main(f"Debug: Parent data for key {key.decode()} (relation: {relation}) was empty or lacked events.")

        if is_single_item:
            # Return tuple (dict, dict, dict)
            return core_data, outcome_info, structured_parents[0]
        else:
            # Return tuple (list[dict], list[dict], list[dict])
            return core_data, outcome_info, structured_parents