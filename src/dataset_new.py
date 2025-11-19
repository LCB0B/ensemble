""" Implements the LMDB dataset"""

import gc
import lzma
import random
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple, Union

import lmdb
import msgpack
import numpy as np
import polars as pl
import pyarrow.dataset as ds
from torch.utils.data import Dataset

from src.chunking import yield_chunks
from src.paths import check_and_copy_file_or_dir
from src.utils import print_main


class LMDBDataset(Dataset):
    """LMDB implementation for larger-than-memory datasets"""

    def __init__(
        self,
        data: ds.Dataset,
        lmdb_path: Path,
        observations: Optional[Dict] = None,
        verbosity=3,
    ):
        self.lmdb_path = lmdb_path
        self.data = data

        check_and_copy_file_or_dir(self.lmdb_path.parent, verbosity=verbosity)

        if not lmdb_path.exists():
            print_main("Creating", lmdb_path.stem)
            self._create_db(data, str(self.lmdb_path))

        if observations is None:
            with lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                readahead=False,
                meminit=False,
                create=False,
                max_dbs=4,
            ) as env:
                db = env.open_db(b"pop", create=False)
                with env.begin() as txn:
                    observations = {
                        "person_id": self.decode(
                            txn.get(b"pnrs", db=db), decompress=False
                        )
                    }
        self.observations = observations

        self.env = None
        self.data_db = None
        self.lengths_db = None
        self.pop_db = None
        # The GC is running amok, so we increase the thresholds (default 700, 10, 10) by 10x to achieve a high speedup and useless num_workers in the DataLoader
        gc.set_threshold(700 * 10, 10 * 10, 10 * 10)

    def _estimate_map_size(
        self,
        data: ds.Dataset,
        buffer: int = 0.3,
        sample_size=100_000,
        min_size=5_000_000_000,
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
            key = self.encode(str(i), compress=False)
            value = self.encode(person, compress=True)
            total_size += len(key) + len(value)

        # Estimate map size
        # (total size divided by amount of rows in sample) multiplied by total amount of rows multiplied by (1 + buffer)
        estimated_map_size = int((total_size / (i + 1)) * num_rows * (1 + buffer))
        # Have min sized buffer
        estimated_map_size = max(estimated_map_size, min_size)

        return estimated_map_size

    def _create_db(self, data: ds.Dataset, lmdb_path: Path, map_size=None):
        if map_size is None:
            map_size = self._estimate_map_size(data)

        pop_info = {
            "total_lengths": [],
            "pnrs": [],
        }
        with lmdb.open(str(lmdb_path), map_size=map_size, max_dbs=4) as env:
            data_db = env.open_db(b"data")
            lengths_db = env.open_db(b"event_lengths")
            pop_db = env.open_db(b"pop")
            with env.begin(write=True) as txn:
                for chunk_df in yield_chunks(data):
                    for person in (
                        chunk_df.group_by("person_id")
                        .agg(pl.all().sort_by("abspos"))
                        .iter_rows(named=True)
                    ):
                        pnr = str(person.pop("person_id"))
                        key = self.encode(pnr, compress=False)
                        txn.put(key, self.encode(person, compress=True), db=data_db)

                        event_lengths = [len(event) for event in person["event"]]
                        txn.put(
                            key,
                            self.encode(event_lengths, compress=True),
                            db=lengths_db,
                        )

                        pop_info["total_lengths"].append(sum(event_lengths))
                        pop_info["pnrs"].append(pnr)

                txn.put(
                    b"total_lengths",
                    self.encode(pop_info["total_lengths"], compress=True),
                    db=pop_db,
                )
                txn.put(
                    b"pnrs", self.encode(pop_info["pnrs"], compress=False), db=pop_db
                )

    def init_db(self):
        """Initialises the self.env and databases required in __getitem__"""
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            # lock=False,
            readahead=False,
            meminit=False,
            create=False,
            max_dbs=4,
        )
        self.data_db = self.env.open_db(b"data", create=False)
        self.lengths_db = self.env.open_db(b"event_lengths", create=False)

    def __len__(self) -> int:
        return len(self.observations["person_id"])

    def __getitem__(self, idx: Union[int, List[int]]) -> dict:
        with self.env.begin() as txn:
            if isinstance(idx, int):
                pnr = self.index_to_pnr(idx)
                pnr = self.encode(str(pnr), compress=False)

                output = {
                    "data": self.decode(txn.get(pnr, db=self.data_db), decompress=True),
                    "lengths": self.decode(
                        txn.get(pnr, db=self.lengths_db), decompress=True
                    ),
                }
            elif isinstance(idx, str):
                pnr = self.encode(str(idx), compress=False)
                output = {
                    "data": self.decode(txn.get(pnr, db=self.data_db), decompress=True),
                    "lengths": self.decode(
                        txn.get(pnr, db=self.lengths_db), decompress=True
                    ),
                }
            elif isinstance(idx, list):
                output = self.__getitems__(idx, txn)
            else:
                raise IndexError("Only int or list indices supported")

        return output

    def __getitems__(self, indices: list, txn: lmdb.Transaction) -> dict:
        pnrs = [self.index_to_pnr(i) for i in indices]
        pnrs = [self.encode(pnr, compress=False) for pnr in pnrs]
        with txn.cursor(db=self.data_db) as cur:
            vals = cur.getmulti(pnrs)
            decoded = [self.decode(val, decompress=True) for key, val in vals]
        with txn.cursor(db=self.lengths_db) as cur:
            vals = cur.getmulti(pnrs)
            # event_lengths = [self.decode(val, decompress=False) for key, val in vals]
        return {"data": decoded, "event_lengths": event_lengths}

    def get_lengths(self):
        """Returns a dict of (pnr, length)"""
        with lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            readahead=False,
            meminit=False,
            create=False,
            max_dbs=4,
        ) as env:
            db = env.open_db(b"pop", create=False)
            with env.begin() as txn:
                pnrs = self.decode(txn.get(b"pnrs", db=db), decompress=False)
                length = self.decode(txn.get(b"total_lengths", db=db), decompress=True)
        return dict(zip(pnrs, length))

    def split(self, splits: Dict[str, Union[float, List[int]]]) -> Tuple[Self, Self]:
        """If splits is float, use random slitting, if List, split based on lists"""
        first_key = next(iter(splits))
        if isinstance(splits[first_key], float):
            obs = self.split_ratio(self.observations, splits)
        elif isinstance(splits[first_key], list):
            obs = self.split_list(self.observations, splits)
        else:
            raise TypeError

        datasets = {}
        for key, subset in obs.items():
            datasets[key] = self.subset(subset)
        return datasets

    @staticmethod
    def split_ratio(data: Dict[str, List], splits: Dict[str, float]):
        """Splits a dictionary using one or more split ratios"""
        assert sum(splits.values()) == 1
        N = len(data["person_id"])
        idxs = list(range(N))

        random.shuffle(idxs)

        last_ratio = 0
        new_splits = {}
        for i, (key, split_ratio) in enumerate(splits.items()):
            start_idx, end_idx = int(N * last_ratio), int(
                N * (last_ratio + split_ratio)
            )
            if i == (len(splits) - 1):
                end_idx = None

            subset = {
                key: [values[i] for i in idxs[start_idx:end_idx]]
                for key, values in data.items()
            }
            last_ratio += split_ratio
            new_splits[key] = subset

        return new_splits

    @staticmethod
    def split_list(data: Dict[str, List], splits: Dict[str, List[int]]):
        """Splits a dictionary based on Lists of pnrs"""
        first_key = next(iter(splits))
        list_type = type(splits[first_key][0])
        df = pl.DataFrame(data).with_columns(pl.col("person_id").cast(list_type))

        new_splits = {}
        for key, split in splits.items():
            ids_set = set(split)
            subset = df.filter(pl.col("person_id").is_in(ids_set)).to_dict(
                as_series=False
            )
            new_splits[key] = subset
        return new_splits

    def subset(self, observations_subset: dict) -> Self:
        """
        Subset of the dataset based on the observations subset.

        Returns:
            Self: A new Self object with the observations subset.
        """
        return self.__class__(
            self.data, self.lmdb_path, observations_subset, verbosity=2
        )

    def index_to_pnr(self, idx: int):
        """Converts idx to associated pnr"""
        return str(self.observations["person_id"][idx])

    @staticmethod
    def encode(val, compress: bool):
        """The byte encoding using for the values"""
        val = msgpack.packb(val, use_bin_type=True)
        if compress:
            val = lzma.compress(val)
        return val

    @staticmethod
    def decode(val, decompress: bool):
        """The byte decoding using for the values"""
        if decompress:
            val = lzma.decompress(val)
        val = msgpack.unpackb(val, raw=False)
        return val


class FinetuneLMDBDataset(LMDBDataset):
    """Finetuning version of LMDBDataset, which uses outcomes as observations"""

    def __init__(self, data: ds.Dataset, lmdb_path: Path, outcomes: dict, **kwargs):
        super().__init__(
            data=data, lmdb_path=lmdb_path, observations=outcomes, **kwargs
        )

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


class FamilyFinetuneLMDBDataset(FinetuneLMDBDataset):
    """Family version of LMDBDataset, which uses outcomes as observations"""

    def __getitem__(self, idx: Union[int, List[int]]):
        output = super().__getitem__(idx)

        all_family = []
        with self.env.begin() as txn:
            for i, info in enumerate(output["outcome_info"]):
                family = {"Child": output["data"][i]}
                if info.get("parents") is not None:
                    for parent in info["parents"]:
                        key = self.encode(str(parent["parent_id"]), compress=False)
                        person = self.decode(txn.get(key), decompress=True)
                        if person["event"]:
                            family[parent["relation_details"]] = person
                all_family.append(family)

        return {"data": all_family, "outcome_info": output["outcome_info"]}

    @staticmethod
    def split_list(
        data: Dict[str, List], splits: Dict[str, List[int]]
    ) -> Dict[str, Dict[str, List]]:
        """
        Splits a dictionary of lists based on provided person_id lists.

        Args:
            data (Dict[str, List]): Dictionary where each key maps to a list of values, including "person_id".
            splits (Dict[str, List[int]]): Dictionary mapping split names to lists of person_ids.

        Returns:
            Dict[str, Dict[str, List]]: Dictionary mapping split names to subsets of the original dictionary.
        """
        person_ids = data["person_id"]

        # Ensure consistent dtype for when we check if person_ids in given list
        person_id_target_type = type(person_ids[0])

        id_to_idx = {pid: i for i, pid in enumerate(person_ids)}

        new_splits = {}
        for split_name, raw_ids in splits.items():
            casted_ids = [person_id_target_type(pid) for pid in raw_ids]
            indices = [id_to_idx[pid] for pid in casted_ids if pid in id_to_idx]
            subset = {key: [values[i] for i in indices] for key, values in data.items()}
            new_splits[split_name] = subset

        return new_splits
