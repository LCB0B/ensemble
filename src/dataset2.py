""" Implements the LMDB dataset"""

import json
import random
from typing import Self, Optional, Dict, List, Tuple
from pathlib import Path
import lmdb
import pickle
import lz4.frame
import polars as pl
from tqdm import tqdm
from torch.utils.data import Dataset
from src.utils import split_dict, print_main
from src.chunking import yield_chunks
import pyarrow.dataset as ds


class LMDBDataset(Dataset):
    """LMDB implementation for larger-than-memory datasets"""

    def __init__(
        self, data: ds.Dataset, lmdb_path: Path, observations: Optional[Dict] = None
    ):
        self.lmdb_path = lmdb_path
        self.data = data

        self.pnr_to_database_idx_path = (
            self.lmdb_path.parent / "pnr_to_database_idx.json"
        )  # TODO: We never check to copy_from_opposite_drive
        self.pnr_to_database_idx = {}

        if not lmdb_path.exists():
            print_main("Creating", lmdb_path.stem)
            self._create_db(data, str(self.lmdb_path))
        else:
            print_main("Loading", lmdb_path.stem)
            with open(
                self.pnr_to_database_idx_path, "r", encoding="utf-8"
            ) as json_file:
                self.pnr_to_database_idx = json.load(json_file)

        if observations is None:
            observations = {"person_id": list(self.pnr_to_database_idx.keys())}
        self.observations = observations

        # Do not init here due to pickle requirement
        self.env = None
        self.txn = None

    def _estimate_map_size(
        self, data: ds.Dataset, buffer: int = 0.5, sample_size=100_000
    ) -> int:
        """
        Estimate the required map size for the LMDB database.

        Args:
            data (ds.Dataset): The Dataset containing the data.
            buffer (int): The buffer to add to the map_size estimation

        Returns:
            int: The estimated map size in bytes.
        """
        num_rows = data.count_rows()
        sample_idxs = random.choices(range(num_rows), k=sample_size)
        first_chunk = data.take(sample_idxs).to_pydict()

        # Estimate size of each person
        total_size = 0
        for i in range(len(first_chunk["person_id"])):
            person = {
                key: val[i] for key, val in first_chunk.items() if key != "person_id"
            }
            # As i is from 0 to len(first_chunk) here, this will underestimate the idx encoding size
            # Assume that most keys are in the millions
            key = str(1_000_000 + i).encode("utf-8")
            value = self.encode(person)
            total_size += len(key) + len(value)

        # Estimate map size
        estimated_map_size = int((total_size / (i + 1)) * num_rows * (1 + buffer))
        return estimated_map_size

    def _create_db(self, data: ds.Dataset, lmdb_path: Path, map_size=None):
        if map_size is None:
            map_size = self._estimate_map_size(data)
        idx = 0
        with lmdb.open(str(lmdb_path), map_size=map_size) as env:
            with env.begin(write=True) as txn:
                for chunk_df in tqdm(yield_chunks(data), "Creating database"):
                    for person in (
                        chunk_df.group_by("person_id")
                        .agg(pl.all().sort_by("abspos"))
                        .iter_rows(named=True)
                    ):
                        pnr = person.pop("person_id")
                        key = str(idx).encode("utf-8")
                        value = self.encode(person)
                        txn.put(key, value, overwrite=True)
                        self.pnr_to_database_idx[str(pnr)] = idx
                        idx += 1

                # Add dummy person (if no events) to database
                dummy_person = {key: [] for key in person}
                dummy_key = str(None).encode("utf-8")
                txn.put(dummy_key, self.encode(dummy_person))

        with open(self.pnr_to_database_idx_path, "w", encoding="utf-8") as json_file:
            json.dump(self.pnr_to_database_idx, json_file, indent=4)

    def _init_db(self):
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            create=False,
        )
        # There's a buffers argument here which may be interesting
        self.txn = self.env.begin(buffers=False)

    def __len__(self) -> int:
        return len(self.observations["person_id"])

    def __getitem__(self, idx: int) -> dict:
        if self.env is None:
            self._init_db()
        obs_pnr = self.observations["person_id"][idx]
        database_idx = self.pnr_to_database_idx.get(str(obs_pnr))
        value = self.txn.get(str(database_idx).encode("utf-8"))
        data = self.decode(value)

        return data

    def close_db(self):
        """Closes database if activate"""
        if self.env is not None:
            self.env.close()

    def split(self, split_ratio: float) -> Self:
        """Splits the Dataset class in two by splitting self.observations based on split_ratio"""
        obs1, obs2 = split_dict(self.observations, split_ratio)
        d1 = self.subset(obs1)
        d2 = self.subset(obs2)
        return d1, d2

    def split_dict_by_person_ids(
        self, data: Dict[str, List], ids_split1: List[int], ids_split2: List[int]
    ) -> Tuple[Dict[str, List], Dict[str, List]]:
        """
        Splits a dictionary into two based on two lists of person_ids, with progress bars for tracking.

        Args:
            data (dict): The dictionary containing data with a 'person_id' key.
            ids_split1 (list of int): List of person_ids for the first split.
            ids_split2 (list of int): List of person_ids for the second split.

        Returns:
            A tuple of dictionaries, one for each split.
        """
        # Convert ids lists to sets for faster lookup
        ids_set1 = set(ids_split1)
        ids_set2 = set(ids_split2)

        # Map person_id to indices for faster lookup
        person_id_to_index = {pid: idx for idx, pid in enumerate(data["person_id"])}
        person_id_outcome_sequence = data["person_id"]

        # Get indices for each split
        idxs_split1 = [
            person_id_to_index[pid]
            for pid in person_id_outcome_sequence
            if pid in ids_set1
        ]

        idxs_split2 = [
            person_id_to_index[pid]
            for pid in person_id_outcome_sequence
            if pid in ids_set2
        ]

        # Generate each split dictionary
        split1 = {key: [values[i] for i in idxs_split1] for key, values in data.items()}
        split2 = {key: [values[i] for i in idxs_split2] for key, values in data.items()}

        return split1, split2

    def split_by_person_ids(self, ids_split1: list[int], ids_split2: list[int]) -> Self:
        """
        Splits the Dataset class into two by filtering self.observations based on lists of person IDs.

        Args:
            ids_split1 (list of int): List of person_ids for the first split.
            ids_split2 (list of int): List of person_ids for the second split.

        Returns:
            A tuple of two Dataset instances, each corresponding to a split of the original data.
        """
        obs1, obs2 = self.split_dict_by_person_ids(
            self.observations, ids_split1, ids_split2
        )
        d1 = self.subset(obs1)
        d2 = self.subset(obs2)
        return d1, d2

    def subset(self, observations_subset: dict) -> Self:
        """
        Subset of the dataset based on the observations subset.

        Returns:
            Self: A new Self object with the observations subset.
        """
        return self.__class__(self.data, self.lmdb_path, observations_subset)

    @staticmethod
    def encode(val):
        """The byte encoding using for the values"""
        return lz4.frame.compress(pickle.dumps(val))
        # return json.dumps(val).encode("utf-8")

    @staticmethod
    def decode(val):
        """The byte decoding using for the values"""
        return pickle.loads(lz4.frame.decompress(val))
        # return json.loads(val.decode("utf-8"))


class FinetuneLMDBDataset(LMDBDataset):
    """Finetuning version of LMDBDataset, which uses outcomes as observations"""

    def __init__(self, data: pl.LazyFrame, lmdb_path: Path, outcomes: dict):
        super().__init__(data=data, lmdb_path=lmdb_path, observations=outcomes)

    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        outcome_info = {key: val[idx] for key, val in self.observations.items()}
        return data, outcome_info


class ParentsFinetuneLMDBDataset(FinetuneLMDBDataset):
    """Parents version of LMDBDataset, which uses outcomes as observations"""

    def __getitem__(self, idx: int):
        data, outcome_info = super().__getitem__(idx)
        parents = {}
        if outcome_info["parents"] is not None:
            for parent in outcome_info["parents"]:
                person = self.get_person(parent["parent_id"])
                if person["event"]:
                    parents[parent["relation_details"]] = person
        return data, outcome_info, parents

    def get_person(self, pnr: str):
        """Returns the person base on pnr (__getitem__ used idx)"""
        database_idx = self.pnr_to_database_idx.get(str(pnr))
        value = self.txn.get(str(database_idx).encode("utf-8"))
        return pickle.loads(lz4.frame.decompress(value))
