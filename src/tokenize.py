""" File for creating the vocabulary and tokenizing the dataframes """

from typing import List
from collections import Counter
import polars as pl
import polars.selectors as cs

import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa


def create_vocab(sources: List[ds.Dataset], cutoff=0) -> dict:
    """Creates vocabulary based on sources by iterating through string columns.
    Default vocab includes {[PAD]: 0, [CLS]: 1, [SEP]: 2, [UNK]: 3, [MASK]: 4}
    Args:
        sources (list[ds.Dataset]): List of sources to create vocab from
        cutoff (int): Minimum number of counts to add token to vocab
    """
    print(f'Creating vocab with cutoff {cutoff}')
    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[MASK]": 4,
    }
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

    for key, value in counts.items():
        if key:
            if value >= cutoff and key not in vocab:
                vocab[key] = len(vocab)

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