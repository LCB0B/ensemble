""" Implements the LMDB dataset"""

import gc
import json
import lzma
import random
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple, Union

import lmdb
import msgpack
import pandas as pd
import polars as pl
import pyarrow.dataset as ds
from torch.utils.data import Dataset

from src.chunking import yield_chunks
from src.paths import check_and_copy_file_or_dir
from src.utils import print_main
from src.log_data import log_sequence_length_stats, calculate_sequence_length_statistics


class LMDBDataset(Dataset):
    """LMDB implementation for larger-than-memory datasets"""

    def __init__(
        self, data: ds.Dataset, lmdb_path: Path, observations: Optional[Dict] = None, log_dir: Optional[Path] = None
    ):
        self.lmdb_path = lmdb_path
        self.data = data
        self.log_dir = log_dir

        dir_path = lmdb_path.parent
        check_and_copy_file_or_dir(self.lmdb_path.parent)

        if not lmdb_path.exists():
            print_main("Creating", lmdb_path.stem)
            self._create_db(data, str(self.lmdb_path), dir_path)

        with open(
            dir_path / "pnr_to_database_idx.json", "r", encoding="utf-8"
        ) as json_file:
            self.pnr_to_database_idx = json.load(json_file)

        if observations is None:
            observations = {"person_id": list(self.pnr_to_database_idx.keys())}
        self.observations = observations

        self.data_counts_before_reset = 10_000
        self.env = None
        # The GC is running amok, so we increase the thresholds (default 700, 10, 10) by 10x to achieve a high speedup and useless num_workers in the DataLoader
        gc.set_threshold(700 * 10, 10 * 10, 10 * 10)

    def _estimate_map_size(
        self,
        data: ds.Dataset,
        buffer: int = 0.3,
        sample_size=100_000,
        min_size=1_000_000,
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

        # Have min sized buffer
        estimated_map_size = max(estimated_map_size, min_size)
        return estimated_map_size

    def _create_db(
        self, data: ds.Dataset, lmdb_path: Path, dir_path: Path, map_size=None
    ):
        if map_size is None:
            map_size = self._estimate_map_size(data)
        pnr_to_database_idx = {}
        lengths = {"person_id": [], "length": []}

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
                        lengths["person_id"].append(pnr)
                        lengths["length"].append(
                            sum([len(event) for event in person["event"]])
                        )
                        pnr_to_database_idx[str(pnr)] = idx

                        key = self.encode_key(idx)
                        value = self.encode(person)
                        txn.put(key, value, overwrite=True)
                        idx += 1

        with open(
            dir_path / "pnr_to_database_idx.json", "w", encoding="utf-8"
        ) as json_file:
            json.dump(pnr_to_database_idx, json_file, indent=4)

        lengths_df = pd.DataFrame(lengths)
        lengths_df.to_parquet(dir_path / "lengths.parquet")

        # Log sequence length statistics if logging is enabled
        if self.log_dir:
            length_list = lengths["length"]
            stats = calculate_sequence_length_statistics(length_list)
            stats["dataset_phase"] = "lmdb_creation"
            stats["total_sequences"] = len(length_list)
            log_sequence_length_stats(stats, self.log_dir, filename="lmdb_sequence_length_stats.json")

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

    def split(self, splits: Dict[str, Union[float, List[int]]]) -> Tuple[Self, Self]:
        """If splits is float, use random slitting, if List, split based on lists"""
        first_key = next(iter(splits))
        if isinstance(splits[first_key], float):
            obs = self.split_ratio(self.observations, splits)
        elif isinstance(splits[first_key], list):
            obs = self.split_list(self.observations, splits)

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

    def __init__(self, data: ds.Dataset, lmdb_path: Path, outcomes: dict, log_dir: Optional[Path] = None):
        super().__init__(data=data, lmdb_path=lmdb_path, observations=outcomes, log_dir=log_dir)

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
                        database_idx = self.pnr_to_database_idx.get(
                            str(parent["parent_id"])
                        )
                        person = self.decode(txn.get(str(database_idx).encode("utf-8")))
                        # person = {
                        #     key: v[1:] for key, v in person.items()
                        # }  # Remove background from parents
                        if person["event"]:
                            family[parent["relation_details"]] = person
                all_family.append(family)

        return {"data": all_family, "outcome_info": output["outcome_info"]}

    # @staticmethod
    # def split_list(data: Dict[str, List], splits: Dict[str, List[int]]):
    #     """Splits a dictionary based on Lists of pnrs"""
    #     first_key = next(iter(splits))
    #     list_type = type(splits[first_key][0])
    #     print("data keys", data.keys())

    #     for key in data.keys():
    #         print(key, "type", type(data[key]))

    #     # Having arrays of parents with nones causes errors when dict is converted to df.
    #     # Remove if there and add later
    #     parents_list = data.pop("parents", None)

    #     df = pl.DataFrame(data).with_columns(pl.col("person_id").cast(list_type))

    #     if parents_list:
    #         df = df.with_columns(
    #             pl.Series([x if x is not None else [] for x in parents_list])
    #             .map_elements(lambda x: None if len(x) == 0 else x)
    #             .alias("parents")
    #         )

    #     print(df.head())

    #     new_splits = {}
    #     for key, split in splits.items():
    #         ids_set = set(split)
    #         subset = df.filter(pl.col("person_id").is_in(ids_set)).to_dict(
    #             as_series=False
    #         )
    #         new_splits[key] = subset

    #         print(key)
    #         for _key in subset.keys():
    #             print(_key, "type", type(subset[_key]))
    #     return new_splits

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
