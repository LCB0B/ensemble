""" File for tokenization and event creation """

import json
from pathlib import Path
from typing import List

import polars as pl
import pyarrow.dataset as ds

from src.features import (
    create_cls_source,
    create_tokenized_events,
)
from src.paths import FPATH
from src.tokenize import create_vocab
from src.utils import get_pnrs


class DataPipeline:
    """Class for handling everything related to data processing of the Datamodule"""

    def __init__(
        self,
        cls_token: bool,
        sep_token: bool,
        fill_nulls: bool = False,
        subset_background: bool = False,
        cutoff: int = 0,
    ):
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.fill_nulls = fill_nulls
        self.subset_background = subset_background
        self.cutoff = cutoff

        # Assigned during __call__
        self.dir_path = None
        self.source_dir = None
        self.vocab = None

    def __call__(
        self,
        sources: List[ds.Dataset],
        background: pl.DataFrame,
        dir_path: Path = None,
        source_dir: Path = None,
    ):
        """Does all data processing required to create features"""
        assert {"person_id", "date_col"}.issubset(
            background.columns
        ), "Required cols: person_id, date_col"
        self.dir_path = dir_path
        self.source_dir = source_dir
        # Subset background on sources
        if self.subset_background:
            background = self.get_background_subset(sources, background)
        birthdates = background.select("person_id", birthday="date_col")

        # Prepend background to sources
        background_source = None
        if len(background.columns) > 2:
            background_source = background

        # Create or insert CLS DataFrame
        if self.cls_token:
            if background_source is None:
                background_source = create_cls_source(
                    birthdates.rename({"birthday": "date_col"})
                )
            else:
                background_source = background_source.clone().insert_column(
                    0, pl.lit("[CLS]").alias("CLS_COL")
                )

        if background_source is not None:
            sources = [ds.dataset(background_source.to_arrow())] + sources

        # Get vocab if not computed
        self.vocab = self.get_vocab(sources)

        # Get tokenized event Dataset
        tokenized_event = self.get_tokenized_event(sources, self.vocab, birthdates)

        return tokenized_event

    @staticmethod
    def _load_if_exists(path: Path, backend=None):
        if path.exists():
            # print("Loading", path.stem)
            if path.suffix == ".parquet":
                if backend == "arrow":
                    return ds.dataset(path, format="parquet")
                elif backend == "polars":
                    return pl.read_parquet(path)
                else:
                    raise ValueError(
                        "Only 'arrow' or 'polars' backend supported", backend
                    )
            elif path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                raise ValueError("Only .parquet and .json files allowed")
        else:
            print("Creating", path.stem)
            return None

    def get_background_subset(
        self, sources: List[ds.Dataset], background_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Load or subset background with pnrs of sources"""
        background_subset_path = self.dir_path / "background_subset.parquet"

        # Load or subset background
        if (
            background_subset := self._load_if_exists(
                background_subset_path, backend="polars"
            )
        ) is None:
            sources_pnrs = get_pnrs(sources)
            background_subset = background_df.join(
                pl.DataFrame({"person_id": sources_pnrs.tolist()}), on="person_id"
            )
            background_subset.write_parquet(background_subset_path)
        return background_subset

    def get_vocab(self, sources: List[ds.Dataset]) -> dict:
        """Load or create the vocabulary"""
        vocab_path = self.dir_path / "vocab.json"

        if (vocab := self._load_if_exists(vocab_path)) is None:
            vocab = create_vocab(sources, cutoff=self.cutoff)
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f)
        return vocab

    def get_tokenized_event(
        self, sources: List[ds.Dataset], vocab: dict, birthdates: pl.DataFrame
    ) -> ds.Dataset:
        """Load or create the tokenized event dataframe"""
        # Tokenize parquet is moved to network to conserve space on local IO
        tokenized_path_local = self.dir_path / "tokenized.parquet"
        tokenized_path_network = (
            FPATH.NETWORK_DATA / self.source_dir / f"{self.dir_path.name}.parquet"
        )
        # Load first from network IO
        if (
            tokenized_event := self._load_if_exists(
                tokenized_path_network, backend="arrow"
            )
        ) is None:
            # Load from local io (backwards compatibility)
            if (
                tokenized_event := self._load_if_exists(
                    tokenized_path_local, backend="arrow"
                )
            ) is None:
                tokenized_event = create_tokenized_events(
                    sources=sources,
                    vocab=vocab,
                    birthdates=birthdates,
                    sep_token=self.sep_token,
                    dir_path=self.dir_path,
                    fill_nulls=self.fill_nulls,
                )  # tokenized_event is saved within create_tokenized_events function
        return tokenized_event
