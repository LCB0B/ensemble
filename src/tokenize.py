""" File for creating the vocabulary and tokenizing the dataframes """

from typing import List, Dict
from collections import Counter
import polars as pl
import polars.selectors as cs

import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa


def create_time_tokens_vocab() -> Dict[str, int]:
    """
    Create vocabulary entries for year and age tokens

    Returns:
        Dictionary mapping time token strings to token IDs
    """
    time_vocab = {}

    # Year tokens: YEAR_1900 to YEAR_2030 (131 tokens)
    for year in range(1900, 2031):
        time_vocab[f"YEAR_{year}"] = len(time_vocab)

    # Age tokens: AGE_0 to AGE_100 (101 tokens)
    for age in range(101):
        time_vocab[f"AGE_{age}"] = len(time_vocab)

    return time_vocab


def create_vocab(sources: List[ds.Dataset], cutoff=0, time_encoding="time2vec") -> dict:
    """Creates vocabulary based on sources by iterating through string columns.
    Default vocab includes {[PAD]: 0, [CLS]: 1, [SEP]: 2, [UNK]: 3, [MASK]: 4}

    Args:
        sources (list[ds.Dataset]): List of sources to create vocab from
        cutoff (int): Minimum number of counts to add token to vocab
        time_encoding (str): Type of time encoding ("time2vec" or "time_tokens")
    """
    print(f'Creating vocab with cutoff {cutoff}, time_encoding={time_encoding}')

    # Base vocabulary with special tokens
    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[MASK]": 4,
    }

    # Add time tokens if using time_tokens encoding
    if time_encoding == "time_tokens":
        time_vocab = create_time_tokens_vocab()
        # Add time tokens after special tokens
        for token, _ in time_vocab.items():
            vocab[token] = len(vocab)
        print(f"Added {len(time_vocab)} time tokens to vocabulary")

    # Process sources for domain-specific tokens
    counts = Counter()
    for source in sources:
        string_columns = [
            field.name
            for field in source.schema
            if pa.types.is_large_string(field.type)  # large_string is default
        ]
        for column in string_columns:
            value_counts = pc.value_counts(source.to_table(columns=[column])[column])
            value_counts = {
                ele["values"]: ele["counts"] for ele in value_counts.tolist()
            }
            counts += Counter(value_counts)

    # Add domain tokens that meet cutoff threshold
    for key, value in counts.items():
        if key:
            if value >= cutoff and key not in vocab:
                vocab[key] = len(vocab)

    print(f"Final vocabulary size: {len(vocab)}")
    return vocab


def tokenize(df: pl.DataFrame, vocab: dict):
    """Tokenize all String columns of the dataframe"""
    # Copy just to make sure nothing goes wrong down the line with vocab
    hack_vocab = vocab.copy()
    hack_vocab[None] = None

    return df.with_columns(
        cs.string().replace_strict(
            hack_vocab, default=vocab["[UNK]"], return_dtype=pl.Int64
        )
    )
 # type: ignore