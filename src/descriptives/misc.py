# %%
import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT

from typing import List, Tuple  # noqa: E402

import polars as pl  # noqa: E402


def unique_courses_vs_min_count(df: pl.DataFrame) -> Tuple[List[int], List[int]]:
    """
    Compute the number of unique courses as a function of minimum row count per course.

    For each threshold from 1 up to the maximum count in the dataframe,
    the function counts how many courses have at least that many rows.

    Args:
        df (pl.DataFrame): DataFrame containing a 'course' column.

    Returns:
        Tuple[List[int], List[int]]: A tuple of two lists:
            - thresholds: List of minimum row count thresholds.
            - unique_counts: List of unique course counts for each threshold.
    """
    # Group by 'course' and count occurrences
    df_counts = df.group_by("course").agg(pl.len())
    counts = df_counts["len"].to_list()
    max_count = max(counts) if counts else 0

    thresholds = list(range(1, max_count + 1))
    unique_counts = [sum(1 for c in counts if c >= t) for t in thresholds]
    return thresholds, unique_counts


def unique_courses_at_cutoff(df: pl.DataFrame, cutoff: int) -> int:
    """
    Calculate the number of unique courses that have at least 'cutoff' occurrences.

    Args:
        df (pl.DataFrame): DataFrame containing a 'course' column.
        cutoff (int): Minimum number of occurrences required.

    Returns:
        int: Count of unique courses with occurrences >= cutoff.
    """
    df_counts = df.group_by("course").agg(pl.len().alias("count"))
    return df_counts.filter(pl.col("count") >= cutoff).height
