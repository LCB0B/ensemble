""" Misc files for utils """

import math
import pathlib
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch


# Taken from https://github.com/huggingface/transformers/blob/75f15f39a0434fe7a61385c4677f2700542a7ba6/src/transformers/data/data_collator.py#L817
def mask_inputs(
    inputs: torch.Tensor,
    vocab: dict,
    mask_prob=0.15,
    replace_prob=0.8,
    random_prob=0.1,
    special_token_border=None,
):
    """Masks inputs using the 80-10-10 strategy"""
    assert (replace_prob + random_prob) <= 1
    assert 0 <= mask_prob < 1
    # inputs must be pre-padded and a tensor
    targets = inputs.clone().long()
    probability_matrix = torch.full(targets.shape, mask_prob)
    special_tokens_mask = get_special_tokens_mask(targets, vocab, special_token_border)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    targets[~masked_indices] = -100  # Only compute loss for masked indices

    # Replace tokens with [MASK] with probability replace_prob
    indices_replaced = (
        torch.bernoulli(torch.full(targets.shape, replace_prob)).bool() & masked_indices
    )
    inputs[indices_replaced] = vocab["[MASK]"]

    # Replace tokens with random with probability random_prob
    random_prob = random_prob / (
        1 - replace_prob
    )  # Adjust probability to account for already masked tokens
    indices_random = (
        torch.bernoulli(torch.full(targets.shape, random_prob)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(vocab), targets.shape, dtype=inputs.dtype)
    inputs[indices_random] = random_words[indices_random]

    # Ignore tokens with probability 1 - replace_prob - random_prob
    return inputs, targets


def get_special_tokens_mask(
    inputs: torch.Tensor, vocab: dict, special_token_border: int = None
):
    """Gets the special token mask for inputs"""
    if special_token_border is None:
        special_token_border = get_max_special_token_value(vocab)
    return inputs <= special_token_border


def get_max_special_token_value(vocab: dict):
    """Get the highest value for a special token - the special tokens must be from 0...max_value"""
    special_tokens = [v for k, v in vocab.items() if k.startswith("[")]
    assert len(special_tokens) == (
        max(special_tokens) + 1
    )  # Asserts that it is a range from 0...max_value
    return max(special_tokens)


def get_background_length(background: pl.DataFrame):
    """Returns number of background cols (not accounting person_id and date)"""
    return len(background.columns) - 2  # -2 for person_id and date
    # return background.group_by("person_id").len().select(pl.col("len").first()).item() (In case of events)


def get_pnrs(sources: Union[ds.Dataset, List[ds.Dataset]]) -> pa.Array:
    """Gets unique pnrs from Dataset or List of Dataset"""
    if isinstance(sources, ds.Dataset):
        return pc.unique(sources.to_table(columns=["person_id"])["person_id"])
    if isinstance(sources, list):
        return pc.unique(
            pa.concat_arrays(
                pc.unique(source.to_table(columns=["person_id"])["person_id"])
                for source in sources
            )
        )
    else:
        raise TypeError(
            f"{type(sources)} is not supported, only ds.Dataset and List[ds.Dataset]"
        )


def create_weights(outcomes: list[int], op=lambda x: x) -> torch.Tensor:
    """
    Creates weights for a outcome list to be used in a WeightedRandomSampler.

    Args:
        outcomes (list[int]): A list of integer outcomes.
        op (Optional: Callable): An optional operation to apply to the denominator of the weights
    Returns:
        list: A list of weights for each outcome.
    """
    # Count the frequency of each outcome
    counter = Counter(outcomes)

    # Calculate weight for each class with optional operation
    weights = {key: 1 / op(count) for key, count in counter.items()}
    print("Sampling with weights: ", weights)

    # Assign weights to each element in the outcomes list
    sample_weights = [weights[outcome] for outcome in outcomes]

    return sample_weights


def print_main(*args, **kwargs):
    """Prints to main process when running distributed systems"""
    rank = -99
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    if __name__ == "main" or "get_ipython" in dir() or rank == 0:
        print(*args, **kwargs)


def calculate_abspos(
    date_col: pl.Expr, origin_point=pl.datetime(2020, 1, 1, time_unit="ns")
):
    """Calculate absolute position (abspos) given datetime col and origin_point (default 1/1/2020)"""
    # return date_col.dt.epoch(time_unit="s") / (60 * 60)

    # return date_col.dt.epoch(time_unit="d").truediv(365.25) # without div for days, of course
    return (date_col - origin_point).dt.total_seconds() / 60 / 60  # hours

    # return (date_col - origin_point).dt.total_seconds() / 60 / 60 / 24 # days
 

def calculate_datetime_from_abspos(
    abspos_col: pl.Expr, origin_point=pl.datetime(2020, 1, 1, time_unit="ns")
) -> pl.Expr:
    """
    Calculate the datetime from absolute position (abspos) in hours, relative to an origin point with nanosecond precision.

    Args:

        abspos_col (pl.Expr): Column with absolute position values in hours from the origin point.
        origin_point (pl.Expr): The origin datetime (default 1/1/2020).

    Returns:
        pl.Expr: Expression that converts abspos back to datetime in nanoseconds.
    """

    return origin_point + (abspos_col * 60 * 60 * 1_000_000_000).cast(pl.Duration("ns"))


def filter_sources_by_date(
    sources: List[pl.LazyFrame], pretrain_cutoff: pl.datetime
) -> List[pl.LazyFrame]:
    """
    Filters a list of LazyFrames by a common 'date_col' column based on a cutoff date.

    Args:
        sources (List[pl.LazyFrame]): List of LazyFrames to filter.
        pretrain_cutoff (pl.datetime): The cutoff date for filtering the LazyFrames.

    Returns:
        List[pl.LazyFrame]: Filtered LazyFrames.
    """
    filtered_sources = [
        lf.filter(pl.col("date_col") < pretrain_cutoff) for lf in sources
    ]
    return filtered_sources


def filter_source_paths_by_date(
    source_paths: List[Path], pretrain_cutoff: pa.Scalar
) -> List[Path]:
    """
    Filters a list of Parquet files by a common 'date_col' column based on a cutoff date,
    using a streaming approach for handling large data.
    Saves the filtered data to new Parquet files with '_cutoff' appended to the filename.
    The new files are saved in the same directory as the original files.

    Args:ds
        source_paths (List[Path]): List of Path objects to the Parquet files to filter.
        pretrain_cutoff (pa.Timestamp): The cutoff date for filtering the dataset.

    Returns:
        List[Path]: List of paths to the newly created Parquet files.
    """
    new_paths = []
    for source_path in source_paths:
        print(f"Cutoff on: {source_path}")

        # Define output file name in the same directory as the original file
        new_path = source_path.with_name(
            f"{source_path.stem}_cutof_{pretrain_cutoff.as_py().year}f{source_path.suffix}"
        )

        # Open the Parquet file as a dataset and apply filtering using a lazy scan
        dataset = ds.dataset(source_path, format="parquet")

        # Stream and filter dataset in chunks to handle larger-than-memory data
        scanner = ds.Scanner.from_dataset(
            dataset,
            filter=pc.less(
                ds.field("date_col"), pretrain_cutoff
            ),  # Use ds.field for the column
            batch_size=1_000_000,
        )

        # Write filtered data to a new parquet file
        with pq.ParquetWriter(new_path, schema=dataset.schema) as writer:
            for batch in scanner.to_batches():
                writer.write_batch(batch)

        # Add new path to the list
        new_paths.append(new_path)

    return new_paths


import time


def load_pipeline_with_retry(dir_path: Path, max_attempts: int = 5):
    """
    Tries to load the pipeline up to max_attempts times, waiting a random amount of time
    between attempts if it fails.

    Args:
        dir_path (Path): The directory path where the pipeline is located.
        max_attempts (int): The maximum number of attempts to load the pipeline.

    Returns:
        pipeline: The loaded pipeline object.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            pipeline = torch.load(dir_path / "pipeline.pt")
            return pipeline
        except Exception as e:
            if attempt == max_attempts:
                raise e  # Raise the exception if it's the last attempt
            wait_time = random.uniform(1, 5)  # Wait between 1 and 5 seconds
            print(f"Attempt {attempt} failed. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)


def filter_parquet_by_person_ids_to_dataset(
    file_path: str, person_ids: set[str]
) -> pa.Table:
    """
    Filters a Parquet dataset by a given set of person_ids and returns the filtered data.

    Args:
        file_path (str): The path to the Parquet file.
        person_ids (set[str]): A set of person IDs to filter by.

    Returns:
        pa.Table: A filtered PyArrow table containing rows matching the person_ids.
    """
    # Create a dataset object
    dataset = ds.dataset(file_path, format="parquet")

    return dataset.filter(ds.field("person_id").isin(person_ids))


def sample_the_goods(parquet_file_path: Path, sample_size: int = 10):
    """Prints a random sample of rows from a Parquet file using Polars without loading the entire dataset.

    Args:
        parquet_file_path (Path): The path to the Parquet file.
        sample_size (int): The number of random rows to print.
    """
    parquet_file = pq.ParquetFile(parquet_file_path)

    # Get the total number of rows in the file
    total_rows = parquet_file.metadata.num_rows

    # Generate random row indices to sample
    sampled_indices = sorted(random.sample(range(total_rows), sample_size))

    collected_rows = []

    # Iterate over row groups and collect sampled rows
    current_row_idx = 0
    for row_group_index in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(row_group_index)
        batch_df = pl.from_arrow(batch)
        num_rows_in_batch = batch_df.height

        # Collect rows from the current batch based on sampled indices
        for index in sampled_indices:
            if current_row_idx <= index < current_row_idx + num_rows_in_batch:
                collected_rows.append(batch_df[index - current_row_idx])

        current_row_idx += num_rows_in_batch

        # If enough rows are collected, stop
        if len(collected_rows) >= sample_size:
            break

    # Convert the collected rows to a Polars DataFrame and print
    if collected_rows:
        sample_df = pl.concat(collected_rows)
        with pl.Config() as cfg:
            cfg.set_tbl_cols(-1)  # Set to -1 to print all columns
            cfg.set_tbl_width_chars(300)  # Increase the width
            print(sample_df)
    else:
        print("No rows sampled.")


@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def stratified_deterministic_indices(
    y: np.ndarray, subset_size: int, seed: int = 73
) -> np.ndarray:
    """
    Returns stratified and deterministic indices for a subset of given size.

    Args:
        y (np.ndarray): Array of binary labels.
        subset_size (int): Number of samples to select.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Selected row indices.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(y))
    class_0 = indices[y == 0]
    class_1 = indices[y == 1]

    rng.shuffle(class_0)
    rng.shuffle(class_1)

    total = len(y)
    p0 = len(class_0) / total

    n0 = math.floor(p0 * subset_size)
    n1 = subset_size - n0

    selected = np.concatenate([class_0[:n0], class_1[:n1]])

    return selected


def stratified_sample(
    df: pl.DataFrame, label_col: str, total_amount: int, seed: Optional[int]
) -> pl.DataFrame:
    """
    Deterministically sample `total_amount` rows stratified on a binary label column.

    Args:
        df (pl.DataFrame): Input dataframe.
        label_col (str): Name of the binary column (0 and 1).
        total_amount (int): Desired total sample size.
        seed (int): Seed for Polars’ `.sample()`.

    Returns:
        pl.DataFrame: Stratified sample of size `total_amount`.
    """
    # split classes
    pos_df = df.filter(pl.col(label_col) == 1)
    neg_df = df.filter(pl.col(label_col) == 0)

    # compute ratios and counts
    total = df.height
    pos_ratio = pos_df.height / total if total else 0.0

    # allocate samples: ceil for positives, floor via subtraction for negatives
    n_pos = math.ceil(total_amount * pos_ratio)
    n_neg = total_amount - n_pos

    # sample
    pos_samp = pos_df.sample(n=n_pos, seed=seed)
    neg_samp = neg_df.sample(n=n_neg, seed=seed)

    return pl.concat([pos_samp, neg_samp])


def save_latex_table(table: str, file_path: str) -> None:
    """
    Save a LaTeX table string to a .tex file.

    Args:
        table (str): LaTeX table content.
        file_path (str): Path to the output .tex file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(table)
