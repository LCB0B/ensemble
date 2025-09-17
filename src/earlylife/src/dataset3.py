""" Implements the LMDB dataset"""

import gc
import json
import lzma
import random
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple, Union

import lmdb
import msgpack
import polars as pl
import pyarrow.dataset as ds
from torch.utils.data import Dataset

from src.earlylife.src.chunking import yield_chunks
from src.earlylife.src.paths import check_and_copy_file_or_dir
from src.earlylife.src.utils import print_main  # , split_dict


# CHANGES TO COMPRESSION METHOD
class LMDBDataset(Dataset):
    """LMDB implementation for larger-than-memory datasets"""

    def __init__(
        self, data: ds.Dataset, lmdb_path: Path, observations: Optional[Dict] = None
    ):
        self.lmdb_path = lmdb_path
        self.data = data

        check_and_copy_file_or_dir(self.lmdb_path.parent)
        self.pnr_to_database_idx_path = (
            self.lmdb_path.parent / "pnr_to_database_idx.json"
        )
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

        self.data_counts_before_reset = 10_000
        self.env = None
        # The GC is running amok, so we increase the thresholds (default 700, 10, 10) by 10x to achieve a high speedup and use less num_workers in the DataLoader
        gc.set_threshold(700 * 10, 10 * 10, 10 * 10)

    def _estimate_map_size(
        self, data: ds.Dataset, buffer: int = 0.3, sample_size=100_000
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
        chunk = data.take(sample_idxs).to_pydict()

        # Estimate size of each person
        total_size = 0
        for i in range(len(chunk["person_id"])):
            person = {key: val[i] for key, val in chunk.items() if key != "person_id"}
            # Assume that most keys are in the millions
            key = str(i).encode("utf-8")
            value = self.encode(person)
            total_size += len(key) + len(value)

        # Estimate map size
        # (total size divided by amount of rows in sample) multiplied by total amount of rows multiplied by (1 + buffer)
        estimated_map_size = int((total_size / (i + 1)) * num_rows * (1 + buffer))
        return estimated_map_size

    def _create_db(self, data: ds.Dataset, lmdb_path: Path, map_size=None):
        if map_size is None:
            map_size = self._estimate_map_size(data)
        idx = 0
        with lmdb.open(str(lmdb_path), map_size=map_size) as env:
            with env.begin(write=True) as txn:
                for chunk_df in yield_chunks(data):
                    for person in (
                        chunk_df.group_by("person_id")
                        .agg(pl.all().sort_by("abspos"))
                        .iter_rows(named=True)
                    ):
                        pnr = person.pop("person_id")
                        key = self.encode_key(idx)
                        value = self.encode(person)
                        txn.put(key, value, overwrite=True)
                        self.pnr_to_database_idx[str(pnr)] = idx
                        idx += 1

                # Add dummy person (if no events) to database
                dummy_person = {key: [] for key in person}
                dummy_key = self.encode_key(None)
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
        self.data_counts = 0

    def __len__(self) -> int:
        return len(self.observations["person_id"])

    def __getitem__(self, idx: Union[int, List[int]]) -> dict:
        # Restart env to free up memory
        self.data_counts += 1 if isinstance(idx, int) else len(idx)
        if self.data_counts > self.data_counts_before_reset:
            self.env.close()
            self._init_db()

        with self.env.begin() as txn:
            if isinstance(idx, int):
                pnr = self.index_to_pnr(idx)
                database_idx = self.pnr_to_db_idx(pnr)
                value = txn.get(self.encode_key(database_idx))
                data = self.decode(value)
            elif isinstance(idx, list):
                data = self.__getitems__(idx, txn)
            else:
                raise IndexError("Only int or list indices supported")

        return {"data": data}

    def __getitems__(self, indices: list, txn: lmdb.Transaction) -> dict:
        with txn.cursor() as cur:
            pnrs = [self.index_to_pnr(i) for i in indices]
            database_idxs = [self.pnr_to_db_idx(pnr) for pnr in pnrs]
            vals = cur.getmulti([self.encode_key(key) for key in database_idxs])
            decoded = [self.decode(val) for key, val in vals]
        return decoded

    def close_db(self):
        """Closes database if activate"""
        if self.env is not None:
            self.env.close()

    def split(self, split_ratio: float) -> Self:
        """Splits the Dataset class in two by splitting self.observations based on split_ratio"""
        raise NotImplementedError
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

        # Convert to df -> filter -> dict seemed faster than dict filtering in testing
        df = pl.DataFrame(data).with_columns(pl.col("person_id").cast(str))
        split1 = df.filter(pl.col("person_id").is_in(ids_set1)).to_dict(as_series=False)
        split2 = df.filter(pl.col("person_id").is_in(ids_set2)).to_dict(as_series=False)

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

    def index_to_pnr(self, idx: int):
        """Converts idx to associated pnr"""
        return self.observations["person_id"][idx]

    def pnr_to_db_idx(self, pnr: Union[str, int]):
        """Finds the database idx for the pnr"""
        return self.pnr_to_database_idx.get(str(pnr))

    @staticmethod
    def encode(val):
        """The byte encoding using for the values"""
        return lzma.compress(msgpack.packb(val, use_bin_type=True))

    @staticmethod
    def encode_key(key: Union[str, int, List[str], List[int]]):
        """Simple utf-8 encoding for keys"""
        return str(key).encode("utf-8")

    @staticmethod
    def decode(val):
        """The byte decoding using for the values"""
        return msgpack.unpackb(lzma.decompress(val), raw=False)


class FinetuneLMDBDataset(LMDBDataset):
    """Finetuning version of LMDBDataset, which uses outcomes as observations"""

    def __init__(self, data: ds.Dataset, lmdb_path: Path, outcomes: dict):
        super().__init__(data=data, lmdb_path=lmdb_path, observations=outcomes)

    def __getitem__(self, idx: Union[int, List[int]]):
        output = super().__getitem__(idx)
        if isinstance(idx, int):
            outcome_info = {key: val[idx] for key, val in self.observations.items()}
        elif isinstance(idx, list):
            outcome_info = [
                {key: val[i] for key, val in self.observations.items()} for i in idx
            ]
        else:
            raise IndexError("Only int or List[int] allowed")
        output.update({"outcome_info": outcome_info})
        return output


class ParentsFinetuneLMDBDataset(FinetuneLMDBDataset):
    """Parents version of LMDBDataset, which uses outcomes as observations"""

    def __getitem__(self, idx: Union[int, List[int]]):
        output = super().__getitem__(idx)

        all_family = []
        with self.env.begin() as txn:
            for i, info in enumerate(output["outcome_info"]):
                family = {"Child": output["data"][i]}
                if info["parents"] is not None:
                    for parent in info["parents"]:
                        database_idx = self.pnr_to_database_idx.get(
                            str(parent["parent_id"])
                        )
                        person = self.decode(txn.get(str(database_idx).encode("utf-8")))
                        person = {
                            key: v[1:] for key, v in person.items()
                        }  # Remove background from parents
                        if person["event"]:
                            family[parent["relation_details"]] = person
                all_family.append(family)

        return {"data": all_family, "outcome_info": output["outcome_info"]}
