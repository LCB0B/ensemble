""" Misc files for utils """

import random
from pathlib import Path
from typing import Union, List
from collections import Counter
import torch
import polars as pl
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.compute as pc


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

def mask_inputs_for_next_token_prediction(
    inputs: torch.Tensor,padding=0
):
    """
    Prepares inputs and targets for causal next-token prediction.

    Args:
        inputs (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of:
            - inputs (torch.Tensor): Input tokens (unchanged for causal prediction).
            - targets (torch.Tensor): Target tokens (next token for each position).
    """
    # Clone inputs to create targets
    targets = inputs.clone().long()

    # Shift targets left by one position to align with the causal next-token task
    # The last position in the sequence cannot predict a next token, so set to -100
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = padding  # No loss computation for the last position

    return inputs, targets


# @torch.compiler.disable()
def flex_attn_causal_event(causal_mapping):
    def causal_event(b, h, q_idx, kv_idx):
        q_mapped = causal_mapping[b[None]][0][q_idx[None]][0]
        kv_mapped = causal_mapping[b[None]][0][kv_idx[None]][0]
        return q_mapped >= kv_mapped

    return causal_event


def flex_attn_padding(sequence_lens):
    def padded(b, h, q_idx, kv_idx):
        return kv_idx < sequence_lens[b[None]][0]

    return padded


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


def split_dict(data: dict, split_ratio: float):
    """Splits a dictionary using one or more split ratios"""
    first_key = next(iter(data))
    N = len(data[first_key])
    idxs = list(range(N))

    random.shuffle(idxs)

    split_idx = int(N * split_ratio)
    first_split = {
        key: [values[i] for i in idxs[:split_idx]] for key, values in data.items()
    }
    second_split = {
        key: [values[i] for i in idxs[split_idx:]] for key, values in data.items()
    }
    return first_split, second_split


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
            cfg.set_tbl_width_chars(150)  # Increase the width
            print(sample_df)
    else:
        print("No rows sampled.")


def decode_tokens(tokens, vocab, max_tokens=None, remove_special=False):
    """
    Decode array-like structure of token IDs into human-readable strings using a vocabulary.
    
    Args:
        tokens: Array-like structure of token IDs (tensor, list, nested list, etc.)
        vocab: Dictionary mapping token strings to token IDs or reverse mapping
        max_tokens: Maximum number of tokens to decode (None for all)
        remove_special: Whether to remove special tokens like padding, etc.
        
    Returns:
        Same structure as input, but with token IDs replaced by their string representations
    """
    import torch
    import numpy as np
    
    # Convert vocabulary if it's in the wrong direction (string->id)
    id_to_str = {}
    if isinstance(list(vocab.keys())[0], str):
        # Vocab is in format {string: id}
        id_to_str = {v: k for k, v in vocab.items()}
    else:
        # Vocab is already in format {id: string}
        id_to_str = vocab
    
    # Special tokens to remove if requested
    special_tokens = set([0]) if remove_special else set()  # Add other special token IDs as needed
    
    def decode_item(item):
        if isinstance(item, (torch.Tensor, np.ndarray)):
            # Convert to list for processing
            return decode_item(item.tolist())
        elif isinstance(item, list):
            if all(isinstance(x, (int, float, bool, np.integer)) for x in item):
                # This is a list of token IDs, decode it
                tokens_to_decode = item[:max_tokens] if max_tokens else item
                if remove_special:
                    tokens_to_decode = [t for t in tokens_to_decode if t not in special_tokens]
                return [id_to_str.get(int(t), f"<UNK:{int(t)}>") for t in tokens_to_decode]
            else:
                # This is a list of lists or other objects, recurse
                return [decode_item(x) for x in item]
        else:
            # Single token
            if item in id_to_str:
                return id_to_str[item]
            else:
                return f"<UNK:{item}>"
    
    return decode_item(tokens)