import json  # noqa: E402
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Literal, Union

import joblib  # noqa: E402
import numpy as np
import pandas as pd
import polars as pl
from loguru import _Logger
from scipy.sparse import save_npz  # noqa: E402

from src.paths import FPATH, check_and_copy_file_or_dir, copy_file_or_dir  # noqa: E402
from src.tabular_models import create_sparse_pipeline  # noqa: E402


def group_values_multiple(
    df: pl.DataFrame, group_col: str, value_cols: Union[str, list[str]]
) -> pl.DataFrame:
    """
    Group values in a dataframe by a specific column for one or more value columns and concatenate them into space-separated strings.
    The resulting dataframes are joined together using an inner join.

    Args:
        df (pl.DataFrame): The input dataframe.
        group_col (str): The name of the column to group by.
        value_cols (Union[str, list[str]]): A single value column or a list of value columns to concatenate.

    Returns:
        pl.DataFrame: Dataframe with grouped and concatenated values, joined by the group column.
    """
    # If value_cols is a string, convert it to a list
    if isinstance(value_cols, str):
        value_cols = [value_cols]

    # List to store individual grouped dataframes
    grouped_dfs = []

    # Loop over each column and group by group_col, then concatenate values
    for value_col in value_cols:
        grouped_df = (
            df.group_by(group_col)
            .agg(pl.col(value_col).str.concat(" "))
            .rename({value_col: f"{value_col}"})
        )
        grouped_dfs.append(grouped_df)

    # Perform inner join on all grouped dataframes
    result_df = grouped_dfs[0]
    for grouped_df in grouped_dfs[1:]:
        result_df = result_df.join(grouped_df, on=group_col, how="inner")

    return result_df


from datetime import datetime
from typing import List, Optional, Union

import polars as pl


def calculate_means_modes_sums(
    df: pl.DataFrame,
    person_id: Union[int, List[int]],
    reference_year: int,
    mean_columns: Optional[List[str]] = None,
    mode_columns: Optional[List[str]] = None,
    sum_columns: Optional[List[str]] = None,
    previous_years: int = 1,
    date_col: str = "date_to",
) -> pl.DataFrame:
    """
    Calculate means, modes, and/or sums for specified columns within the given number
    of previous years for one or more person_id(s), based on a reference year.

    Args:
        df (pl.DataFrame): DataFrame containing the data with person_id and datetime columns.
        person_id (Union[int, List[int]]): The person_id(s) to filter.
        reference_year (int): The year used to calculate the time window.
        mean_columns (List[str], optional): List of column names for mean calculation.
        mode_columns (List[str], optional): List of column names for mode calculation.
        sum_columns (List[str], optional): List of column names for sum calculation.
        previous_years (int): Number of previous years to consider for the calculation.
        date_col (str): Name of the date column to filter on.

    Returns:
        pl.DataFrame: DataFrame with calculated means, modes, and/or sums for the given person_id(s).
    """
    # Define the time range
    start_date = datetime(reference_year - previous_years, 1, 1)
    end_date = datetime(reference_year, 1, 1)

    # Filter the dataframe
    filtered_df = df.filter(
        (pl.col("person_id").is_in(person_id))
        & (pl.col(date_col).is_between(start_date, end_date))
    )

    # Group by person_id
    grouped = filtered_df.group_by("person_id")

    # Prepare aggregations
    agg_list = []

    if mean_columns:
        agg_list += [pl.col(c).mean().alias(f"{c}_mean") for c in mean_columns]

    if mode_columns:
        agg_list += [
            pl.col(c).mode().first().cast(pl.Utf8).alias(f"{c}_mode")
            for c in mode_columns
        ]

    if sum_columns:
        agg_list += [pl.col(c).sum().alias(f"{c}_sum") for c in sum_columns]

    if not agg_list:
        raise ValueError(
            "Provide at least one of 'mean_columns', 'mode_columns', or 'sum_columns'."
        )

    return grouped.agg(agg_list)


def calculate_mean_grade(
    df: pl.DataFrame,
    personid_col: str,
    course_col: str,
    grade_col: str,
    alias_prefix: str = "mean_",
) -> pl.DataFrame:
    """
    Group the dataframe by specified 'personid' and 'course' columns, then calculate the mean of the 'grade' column.
    Optionally, add a prefix to the alias for the mean column.

    Args:
        df (pl.DataFrame): The Polars DataFrame.
        person_id_col (str): The name of the person ID column.
        course_col (str): The name of the course column.
        grade_col (str): The name of the grade column.
        alias_prefix (str): The prefix to add to the alias of the mean column. Default is "mean_".

    Returns:
        pl.DataFrame: A DataFrame with the mean 'grade' for each group of 'personid' and 'course'.
    """
    grouped_df = df.group_by([personid_col, course_col]).agg(
        pl.col(grade_col).mean().alias(f"{grade_col}")
    )
    # Pivot the data so each course becomes a column
    grouped_df = grouped_df.pivot(
        values=f"{grade_col}", index=personid_col, on=course_col
    )

    grouped_df = grouped_df.rename(
        lambda column_name: (
            f"{alias_prefix}_{column_name}"
            if column_name != "person_id"
            else column_name
        )
    )
    return grouped_df


def calculate_mean_grade_pool_infrequent(
    df: pl.DataFrame,
    personid_col: str,
    course_col: str,
    grade_col: str,
    alias_prefix: str = "mean_",
    frequent_courses: list[str] = None,
) -> pl.DataFrame:
    """
    Calculate the mean grade for each person, splitting courses into frequent and infrequent groups.

    For courses in `frequent_courses`, compute separate mean columns per course.
    For courses not in `frequent_courses`, compute a single overall mean column.

    Args:
        df (pl.DataFrame): (Polars DataFrame) The input DataFrame.
        personid_col (str): (str) Name of the person ID column.
        course_col (str): (str) Name of the course column.
        grade_col (str): (str) Name of the grade column.
        alias_prefix (str): (str) Prefix for the output mean columns. Default is "mean_".
        frequent_courses (list[str]): (list[str]) List of courses that occur at least a specified number of times.

    Returns:
        pl.DataFrame: DataFrame with separate mean columns for frequent courses and one overall mean column for infrequent courses.
    """
    if frequent_courses is None:
        frequent_courses = []

    # Mean for frequent courses (one column per course)
    frequent_df = (
        df.filter(pl.col(course_col).is_in(frequent_courses))
        .group_by([personid_col, course_col])
        .agg(pl.col(grade_col).mean().alias(grade_col))
        .pivot(values=grade_col, index=personid_col, columns=course_col)
    )
    if frequent_df.columns:
        frequent_df = frequent_df.rename(
            lambda col: f"{alias_prefix}{col}" if col != personid_col else col
        )

    # Overall mean for infrequent courses
    infrequent_df = (
        df.filter(~pl.col(course_col).is_in(frequent_courses))
        .group_by(personid_col)
        .agg(pl.col(grade_col).mean().alias("others"))
        .rename({"others": f"{alias_prefix}others"})
    )

    return frequent_df, infrequent_df


def calculate_weighted_mean_grade_pool_infrequent(
    df: pl.DataFrame,
    personid_col: str,
    course_col: str,
    grade_col: str,
    weight_col: str,
    alias_prefix: str = "he_mean",
    frequent_courses: list[str] = None,
) -> pl.DataFrame:
    """
    Calculate the weighted mean grade for each person, computing separate columns for frequent courses
    and a pooled column for infrequent courses.

    Args:
        df (pl.DataFrame): (Polars DataFrame) The input DataFrame.
        personid_col (str): (str) Name of the person ID column.
        course_col (str): (str) Name of the course column.
        grade_col (str): (str) Name of the grade column.
        weight_col (str): (str) Name of the column representing weights (e.g., ects).
        alias_prefix (str): (str) Prefix for the output weighted mean columns.
        frequent_courses (list[str]): (list[str]) List of courses that occur at least min_occurrence_time.

    Returns:
        pl.DataFrame: DataFrame with separate weighted mean columns for frequent courses and a pooled column for infrequent courses.
    """
    if frequent_courses is None:
        frequent_courses = []

    # Compute weighted mean for frequent courses (separate column per course).
    frequent_df = (
        df.filter(pl.col(course_col).is_in(frequent_courses))
        .group_by([personid_col, course_col])
        .agg(
            (pl.col(grade_col).cast(pl.Float64, strict=False) * pl.col(weight_col))
            .sum()
            .alias("num"),
            pl.col(weight_col).sum().alias("denom"),
        )
        .with_columns((pl.col("num") / pl.col("denom")).alias(grade_col))
        .select([personid_col, course_col, grade_col])
        .pivot(values=grade_col, index=personid_col, columns=course_col)
    )
    if frequent_df.columns:
        frequent_df = frequent_df.rename(
            lambda col: f"{alias_prefix}{col}" if col != personid_col else col
        )

    # Compute weighted mean for infrequent courses (pooled into one column).
    infrequent_df = (
        df.filter(~pl.col(course_col).is_in(frequent_courses))
        .group_by(personid_col)
        .agg(
            (pl.col(grade_col).cast(pl.Float64, strict=False) * pl.col(weight_col))
            .sum()
            .alias("num"),
            pl.col(weight_col).sum().alias("denom"),
        )
        .with_columns((pl.col("num") / pl.col("denom")).alias("others"))
        .select([personid_col, "others"])
        .rename({"others": f"{alias_prefix}others"})
    )

    return frequent_df, infrequent_df


def calculate_mode_by_personid(
    df: pl.DataFrame, mode_columns: Union[str, List[str]], prefix: str = "mode_"
) -> pl.DataFrame:
    """
    Group the dataframe by 'personid' and calculate the mode for the specified columns.
    Returns the first mode for each column after grouping and adds an optional prefix to the column names.

    Args:
        df (pl.DataFrame): The Polars DataFrame.
        mode_columns (Union[str, List[str]]): Column or list of columns to calculate the mode.
        prefix (str, optional): Optional prefix to add to the alias of the mode columns. Default is "mode_".

    Returns:
        pl.DataFrame: A DataFrame with the first mode for each column grouped by 'personid'.
    """
    # Ensure mode_columns is a list
    if isinstance(mode_columns, str):
        mode_columns = [mode_columns]

    # Group by person_id and calculate mode, adding the prefix to the alias
    return df.group_by("person_id").agg(
        [pl.col(col).mode().first().alias(f"{prefix}{col}") for col in mode_columns]
    )


def drop_columns_that_are_all_null(df: pl.DataFrame) -> pl.DataFrame:
    return df[[s.name for s in df if not (s.null_count() == df.height)]]


# def collect_filtered_parquet(
#     path: Path, person_ids: List[int], force_copy: bool = False
# ) -> pl.LazyFrame:
#     """
#     Loads a Parquet file and filters rows based on person_ids.

#     Args:
#         path (Path): The path to the Parquet file.
#         person_ids (List[int]): List of person IDs to filter by.
#         force_copy (bool): Force copying from opposite drive

#     Returns:
#         pl.LazyFrame: A LazyFrame with filtered rows.
#     """
#     if force_copy:
#         copy_file_or_dir(path)
#     else:
#         check_and_copy_file_or_dir(path)
#     return pl.scan_parquet(path).filter(pl.col("person_id").is_in(person_ids)).collect()

from pathlib import Path
from typing import List

import polars as pl
import pyarrow.dataset as ds


def collect_filtered_parquet(
    path: Path, person_ids: list[int], force_copy: bool = False
) -> pl.DataFrame:
    """
    Load a Parquet file, apply predicate pushdown via PyArrow, and return a filtered Polars DataFrame.

    Args:
        path (Path): Path to the Parquet file or directory.
        person_ids (list[int]): IDs of persons to include.
        force_copy (bool): If True, unconditionally copy from the opposite drive before scanning.
    """
    if force_copy:
        copy_file_or_dir(path)
    else:
        check_and_copy_file_or_dir(path)

    # Open dataset and push down the filter to avoid loading all data
    dataset = ds.dataset(str(path), format="parquet")
    predicate = ds.field("person_id").isin(person_ids)
    table = dataset.to_table(filter=predicate)

    # Convert to Polars DataFrame
    return pl.from_arrow(table)


from pathlib import Path
from typing import List

import polars as pl
import pyarrow.dataset as ds


def collect_filtered_parquet_by_year(
    path: Path, years: List[int], force_copy: bool = False
) -> pl.DataFrame:
    """
    Load a Parquet file, apply predicate pushdown via PyArrow, and return a Polars DataFrame filtered by years.

    Args:
        path (Path): Path to the Parquet file or directory.
        years (List[int]): Years to include.
        force_copy (bool): If True, unconditionally copy from the opposite drive before scanning.
    """
    if force_copy:
        copy_file_or_dir(path)
    else:
        check_and_copy_file_or_dir(path)

    dataset = ds.dataset(str(path), format="parquet")
    predicate = ds.field("year").isin(years)
    table = dataset.to_table(filter=predicate)

    return pl.from_arrow(table)


import time
from pathlib import Path
from typing import List, Optional

import polars as pl
import pyarrow.dataset as ds

# def collect_filtered_parquet_by_person_and_year(
#     path: Path,
#     person_ids: Optional[List[int]] = None,
#     years: Optional[List[int]] = None,
#     force_copy: bool = False,
# ) -> pl.DataFrame:
#     """
#     Load a Parquet file, apply predicate pushdown via PyArrow, and return a Polars DataFrame
#     filtered by person_id and year.

#     Args:
#         path (Path): Path to the Parquet file or directory.
#         person_ids (List[int] | None): IDs of persons to include. If None, no filtering on person_id.
#         years (List[int] | None): Years to include. If None, no filtering on year.
#         force_copy (bool): If True, unconditionally copy from the opposite drive before scanning.
#     """
#     start = time.perf_counter()
#     print(f"[{path.name}] Start loading at {time.strftime('%H:%M:%S')}")

#     if force_copy:
#         copy_file_or_dir(path)
#     else:
#         check_and_copy_file_or_dir(path)

#     print(f"[{path.name}] Dataset creation {time.strftime('%H:%M:%S')}")
#     dataset = ds.dataset(str(path), format="parquet")
#     print(f"[{path.name}] Predicate creation {time.strftime('%H:%M:%S')}")

#     predicates = []
#     if person_ids is not None:
#         predicates.append(ds.field("person_id").isin(person_ids))
#     if years is not None:
#         predicates.append(ds.field("year").isin(years))

#     predicate = None
#     if predicates:
#         import operator
#         from functools import reduce

#         predicate = reduce(operator.and_, predicates)

#     print(f"[{path.name}] Table creation {time.strftime('%H:%M:%S')}")

#     table = (
#         dataset.to_table(filter=predicate)
#         if predicate is not None
#         else dataset.to_table()
#     )
#     print(f"[{path.name}] Conversion to polars {time.strftime('%H:%M:%S')}")

#     df = pl.from_arrow(table)

#     elapsed = time.perf_counter() - start
#     print(f"[{path.name}] Finished loading in {elapsed:.2f} seconds")
#     return df


def collect_filtered_parquet_by_person_and_year(
    path: Path,
    person_ids: Optional[List[int]] = None,
    years: Optional[List[int]] = None,
    force_copy: bool = False,
) -> pl.DataFrame:
    """
    Efficiently load and filter a Parquet file on year (PyArrow) and person_id (Polars).

    Args:
        path (Path): Path to the Parquet file or directory.
        person_ids (List[int] | None): IDs of persons to include. If None, no filtering on person_id.
        years (List[int] | None): Years to include. If None, no filtering on year.
        force_copy (bool): If True, unconditionally copy from the opposite drive before scanning.
    """
    start = time.perf_counter()
    print(f"[{path.name}] Start loading at {time.strftime('%H:%M:%S')}")

    if force_copy:
        copy_file_or_dir(path)
    else:
        check_and_copy_file_or_dir(path)

    dataset = ds.dataset(str(path), format="parquet")

    # Filter only by year in PyArrow
    predicate = ds.field("year").isin(years) if years else None
    table = (
        dataset.to_table(filter=predicate)
        if predicate is not None
        else dataset.to_table()
    )

    df = pl.from_arrow(table)

    # Then filter by person_id in Polars (better for large lists)
    if person_ids is not None:
        df = df.filter(pl.col("person_id").is_in(person_ids))

    elapsed = time.perf_counter() - start
    print(f"[{path.name}] Finished in {elapsed:.2f} seconds")
    return df


def filter_and_flag_events(
    df: pl.DataFrame,
    target_date: datetime,
    date_col: str,
    days_col: str,
    flag_col: str,
    event_col: str,
) -> pl.DataFrame:
    """
    Adds columns indicating
    whether the target date is before `date_col + days_col` for each person, and an indicator column.

    Args:
        df (pl.DataFrame): DataFrame with person_id, a date column, and a days column.
        target_date (datetime): The date to filter events and check against.
        date_col (str): Name of the date column in the DataFrame.
        days_col (str): Name of the days column in the DataFrame.
        flag_col (str): Name of the output column indicating if target date is before `date_col + days_col`.
        event_col (str): Name of the output column to indicate the presence of an event.

    Returns:
        pl.DataFrame: A DataFrame with 'person_id', `flag_col` as int, and `event_col` as 1.
    """
    result_df = (
        df.with_columns(
            [
                (pl.col(date_col) + pl.duration(days=pl.col(days_col)) > target_date)
                .cast(pl.Int8)  # Convert bool to int
                .alias(flag_col),
                pl.lit(1).alias(event_col),  # Add event column with value 1
            ]
        )
        .sort(date_col)
        .group_by("person_id")
        .last()
        .select(["person_id", flag_col, event_col])
    )

    return result_df


def filter_and_flag_events_with_binary(
    df: pl.DataFrame,
    target_date: datetime,
    date_col: str,
    days_col: str,
    binary_col: str,
    flag_col: str,
    event_col: str,
    combined_event_col: str,
) -> pl.DataFrame:
    """
    Adds columns indicating
    whether the target date is before `date_col + days_col` for each person, and an indicator column.
    Adds a third column that is 1 if both `flag_col` and the binary column are 1.

    Args:
        df (pl.DataFrame): DataFrame with person_id, a date column, a days column, and a binary column.
        target_date (datetime): The date to filter events and check against.
        date_col (str): Name of the date column in the DataFrame.
        days_col (str): Name of the days column in the DataFrame.
        binary_col (str): Name of the binary column in the DataFrame.
        flag_col (str): Name of the output column indicating if target date is before `date_col + days_col`.
        event_col (str): Name of the output column to indicate the presence of an event.
        combined_event_col (str): Name of the output column to indicate when both flag and binary are 1.

    Returns:
        pl.DataFrame: A DataFrame with 'person_id', `flag_col` as int, `event_col` as 1, and `combined_event_col` as 1.
    """
    result_df = (
        df.with_columns(
            [
                (pl.col(date_col) + pl.duration(days=pl.col(days_col)) > target_date)
                .cast(pl.Int8)  # Convert bool to int
                .alias(flag_col),
                pl.lit(1).alias(event_col),  # Add event column with value 1
                (pl.col(binary_col) == 1)
                .cast(pl.Int8)
                .alias(
                    combined_event_col
                ),  # Add combined_event_col if both conditions are met
            ]
        )
        .sort(date_col)
        .group_by("person_id")
        .last()
        .select(["person_id", flag_col, event_col, combined_event_col])
    )

    return result_df


def filter_and_flag_events_with_end_date(
    df: pl.DataFrame,
    target_date: datetime,
    start_date_col: str,
    end_date_col: str,
    flag_col: str,
    event_col: str,
) -> pl.DataFrame:
    """
    Adds columns indicating whether the target date is between `start_date_col` and `end_date_col`
    for each person, flags the presence of an event, and groups by 'person_id' to retain only the maximum flag values.

    Args:
        df (pl.DataFrame): DataFrame with person_id, start date column, and end date column.
        target_date (datetime): The date to check if it is within the range.
        start_date_col (str): Name of the start date column in the DataFrame.
        end_date_col (str): Name of the end date column in the DataFrame.
        flag_col (str): Name of the output column indicating if target date is within the date range.
        event_col (str): Name of the output column to indicate the presence of an event.

    Returns:
        pl.DataFrame: A grouped DataFrame with 'person_id', maximum `flag_col`, and `event_col`.
    """
    df_with_flags = df.with_columns(
        [
            (
                (pl.col(start_date_col) <= target_date)
                & (pl.col(end_date_col) > target_date)
            )
            .cast(pl.Int8)  # Convert bool to int
            .alias(flag_col),
            pl.lit(1).alias(event_col),  # Add event column with value 1
        ]
    ).select(["person_id", flag_col, event_col])

    result_df = df_with_flags.group_by("person_id").agg(
        [
            pl.col(flag_col).max().alias(flag_col),
            pl.col(event_col).max().alias(event_col),
        ]
    )

    return result_df


def prepare_dataframe(
    df: pl.DataFrame,
    cutoff_year: int,
    nominal_cols: List[str],
    count_cols: List[str],
    is_test: bool,
) -> pd.DataFrame:
    """
    Filters, converts, and fills specified columns in the DataFrame.

    Args:
        df (pl.DataFrame): Original dataframe to filter and process.
        cutoff_year (int): Year value to use as cutoff for filtering.
        nominal_cols (List[str]): Columns to convert to category with missing filled as 'None'.
        count_cols (List[str]): Columns to convert to string with missing filled as an empty string.
        is_test (bool): If True, filters rows where year == cutoff_year, otherwise where year < cutoff_year.

    Returns:
        pd.DataFrame: Processed and filtered DataFrame.
    """
    filter_expr = (
        pl.col("year") == cutoff_year if is_test else pl.col("year") < cutoff_year
    )
    df_filtered = df.filter(filter_expr)

    # Fill nulls for nominal columns and cast to categorical-compatible string
    for col in nominal_cols:
        df_filtered = df_filtered.with_columns(
            pl.col(col).fill_null("None").cast(pl.Utf8)
        )

    # Fill nulls for count columns and cast to string
    for col in count_cols:
        df_filtered = df_filtered.with_columns(pl.col(col).fill_null("").cast(pl.Utf8))

    return df_filtered.to_pandas()


def prepare_dataframe_no_temporal_splitting(
    df: pl.DataFrame,
    # cutoff_year: int,
    nominal_cols: List[str],
    count_cols: List[str],
) -> pd.DataFrame:
    """
    Filters, converts, and fills specified columns in the DataFrame.

    Args:
        df (pl.DataFrame): Original dataframe to filter and process.
        cutoff_year (int): Max year to create data for.
        nominal_cols (List[str]): Columns to convert to category with missing filled as 'None'.
        count_cols (List[str]): Columns to convert to string with missing filled as an empty string.

    Returns:
        pd.DataFrame: Processed and filtered DataFrame.
    """
    # filter_expr = pl.col("year") <= cutoff_year
    df_filtered = df  # .filter(filter_expr)

    # Fill nulls for nominal columns and cast to categorical-compatible string
    for col in nominal_cols:
        df_filtered = df_filtered.with_columns(
            pl.col(col).fill_null("None").cast(pl.Utf8)
        )

    # Fill nulls for count columns and cast to string
    for col in count_cols:
        df_filtered = df_filtered.with_columns(pl.col(col).fill_null("").cast(pl.Utf8))

    return df_filtered.to_pandas()


def describe_parquet_files(logger, folder: str, missing_threshold: float) -> None:
    """
    Load all parquet files in a folder and print descriptive information for each file,
    including a sample of up to 10 rows, unique counts for 'person_id' and 'year', shape,
    columns and types, missing values, descriptive statistics, a cumulative count of columns
    across all files (excluding 'person_id' and 'year'), and counts of columns with more than
    and less than a given percentage of missing values.

    Args:
        logger (Logger): Loguru logger instance for logging.
        folder (str): Path to the folder containing parquet files.
        missing_threshold (float): Threshold (as a fraction, e.g. 0.2 for 20%) for missing values.
    """
    folder_path = Path(folder)
    parquet_files = list(folder_path.glob("*.parquet"))
    parquet_files = [x for x in parquet_files if x.name != "combined_dataframe.parquet"]

    if not parquet_files:
        logger.info(f"No parquet files found in {folder}.")
        return

    pl.Config.set_tbl_cols(15)
    pl.Config.set_fmt_str_lengths = 50

    cumulative_columns = 0
    cumulative_missing_cols = 0

    for file_path in parquet_files:
        logger.info(
            "\n\n----------------------------------------------------------------------------------------------------\n"
        )
        logger.info(f"File: {file_path.name}")
        try:
            df = pl.read_parquet(file_path)
        except Exception as e:
            logger.info(f"Error reading {file_path.name}: {e}")
            continue

        # Exclude 'person_id' and 'year' from the column counts for missing value reporting.
        cols_to_check = [col for col in df.columns if col not in {"person_id", "year"}]

        # Update cumulative columns count (excluding 'person_id' and 'year')
        n_rows, _ = df.shape
        n_cols = len(cols_to_check)
        cumulative_columns += n_cols

        # Print shape
        logger.info(
            f"Shape: {n_rows} rows x {len(df.columns)} columns (excluding 'person_id' and 'year': {n_cols} columns)"
        )

        # Print columns and their data types
        logger.info("Columns and Types:")
        for col, dtype in zip(df.columns, df.dtypes):
            logger.info(f"  {col}: {dtype}")

        # Print missing values per column and count columns with > K% missing (only for cols_to_check)
        logger.info("Missing values per column (excluding 'person_id' and 'year'):")
        missing_exprs = [
            pl.col(col).is_null().sum().alias(col) for col in cols_to_check
        ]
        missing_counts = df.select(missing_exprs)
        missing_count_file = 0
        below_threshold_count_file = 0
        for col in missing_counts.columns:
            missing_ratio = missing_counts[col][0] / n_rows
            logger.info(f"  {col}: {missing_ratio:.2%}")
            if missing_ratio > missing_threshold:
                missing_count_file += 1
            else:
                below_threshold_count_file += 1
        logger.info(
            f"Columns with more than {missing_threshold:.0%} missing: {missing_count_file}"
        )
        logger.info(
            f"Columns with less than {missing_threshold:.0%} missing: {below_threshold_count_file}"
        )
        cumulative_missing_cols += missing_count_file

        # Print a sample of 10 rows with a wider display (using glimpse for compact display)
        logger.info("Sample 10 rows:")
        df.sample(n=10).glimpse()

        # Unique counts for 'person_id' and 'year' if they exist
        if "person_id" in df.columns:
            unique_person_ids = df.select(
                pl.col("person_id").unique().count()
            ).to_series()[0]
            logger.info(f"Unique 'person_id' count: {unique_person_ids}")
        else:
            logger.info("Column 'person_id' not found.")

        if "year" in df.columns:
            unique_years = df.select(pl.col("year").unique().count()).to_series()[0]
            logger.info(f"Unique 'year' count: {unique_years}")
        else:
            logger.info("Column 'year' not found.")

        # Print descriptive statistics
        logger.info("Descriptive statistics:")
        try:
            description = df.describe()
            logger.info(description)
        except Exception as e:
            logger.info(f"  Could not generate descriptive statistics: {e}")

    # Print cumulative counts across all files
    logger.info(
        f"Cumulative count of columns (excluding 'person_id' and 'year') across all files: {cumulative_columns}"
    )
    logger.info(
        f"Cumulative count of columns with more than {missing_threshold:.0%} missing: {cumulative_missing_cols}"
    )
    logger.info(
        f"Cumulative count of columns with less than or equal to {missing_threshold:.0%} missing: {cumulative_columns - cumulative_missing_cols}"
    )


def normalize_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """
    Normalize specified columns across the full DataFrame using z-score.

    Args:
        df (pl.DataFrame): Input DataFrame.
        columns (list[str]): List of column names (str) to normalize.
    """
    return df.with_columns(
        [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in columns]
    )


def winsorize(
    df: pd.DataFrame, col_bounds: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """
    Applies winsorization to numerical columns based on provided bounds.

    Args:
        df (pd.DataFrame): DataFrame to transform.
        col_bounds (dict[str, tuple[float, float]]): Lower and upper bounds for each column.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    df = df.copy()
    for col, (lo, hi) in col_bounds.items():
        df.loc[:, col] = df[col].clip(lower=lo, upper=hi)
    return df


def process_tabular_data(
    fname_prefix: str,
    sample_folder: str,
    data_folder: str,
    cutoff_year_dev: int,
    test_year: int,
    included_cols: list[str],
    numerical_cols: list[str],
    nominal_cols: list[str],
    ordinal_cols: list[str],
    count_cols: list[str],
    min_df: float,
    max_count: int,
    loguru_logger: _Logger,
) -> None:
    """
    Processes one fname_prefix by preparing datasets, saving dense and sparse features, and outcomes.

    Args:
        fname_prefix (str): Prefix identifying the outcome (e.g., "neet", "dropout").
    """
    (FPATH.DATA / f"{data_folder}").mkdir(exist_ok=True)
    # check_and_copy_file_or_dir(FPATH.DATA / f"{data_folder}" / f"{fname_prefix}_targets_tabular.parquet")
    df_combined = pl.read_parquet(
        FPATH.NETWORK_DATA
        / f"{data_folder}"
        / f"{fname_prefix}_targets_tabular.parquet"
    )

    loguru_logger.info("Copying train/val/test person ID parquet files")

    loguru_logger.info("Reading train/val/test person IDs")
    train_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{fname_prefix}_train_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()
    val_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{fname_prefix}_val_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()
    test_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{fname_prefix}_test_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()

    loguru_logger.info("Preparing test and development datasets")
    df_test = prepare_dataframe(
        df_combined,
        cutoff_year=test_year,
        nominal_cols=nominal_cols,
        count_cols=count_cols,
        is_test=True,
    )

    loguru_logger.info("Subsetting dev/test data by person_id")
    df_dev = prepare_dataframe(
        df_combined,
        cutoff_year=cutoff_year_dev,
        nominal_cols=nominal_cols,
        count_cols=count_cols,
        is_test=False,
    )

    loguru_logger.info("Saving all-info parquet files")
    df_test = df_test[df_test["person_id"].isin(test_person_ids)]
    df_train = df_dev[df_dev["person_id"].isin(train_person_ids)]
    df_val = df_dev[df_dev["person_id"].isin(val_person_ids)]

    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_test_all_info.parquet",
        lambda p: df_test.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_train_all_info.parquet",
        lambda p: df_train.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_val_all_info.parquet",
        lambda p: df_val.to_parquet(p),
    )
    loguru_logger.info("Winsorizing numerical columns")

    percentiles = {
        col: (
            df_train[col].quantile(0.05),
            df_train[col].quantile(0.95),
        )
        for col in numerical_cols
    }

    df_train = winsorize(df_train, percentiles)
    df_val = winsorize(df_val, percentiles)
    df_test = winsorize(df_test, percentiles)

    loguru_logger.info("Extracting feature matrices and targets")
    X_test, y_test = df_test[included_cols], df_test["target"]
    X_train, y_train = df_train[included_cols], df_train["target"]
    X_val, y_val = df_val[included_cols], df_val["target"]

    loguru_logger.info("Saving dense feature matrices")
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_train_dense.parquet",
        lambda p: X_train.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_val_dense.parquet",
        lambda p: X_val.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_test_dense.parquet",
        lambda p: X_test.to_parquet(p),
    )

    loguru_logger.info("Fitting sparse pipeline and transforming data")

    onehot_cols = sorted(nominal_cols + ordinal_cols)

    pipeline = create_sparse_pipeline(
        numerical_cols, onehot_cols, count_cols, min_df, max_count
    )
    train_arr = pipeline.fit_transform(X_train)
    val_arr = pipeline.transform(X_val)
    test_arr = pipeline.transform(X_test)

    loguru_logger.info("Saving sparse matrices and pipeline")
    joblib.dump(
        pipeline, FPATH.DATA / data_folder / f"{fname_prefix}_tabular_pipeline.pkl"
    )
    FPATH.alternative_copy_to_opposite_drive(
        FPATH.DATA / data_folder / f"{fname_prefix}_tabular_pipeline.pkl"
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_train_sparse_matrix.npz",
        lambda p: save_npz(p, train_arr),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_val_sparse_matrix.npz",
        lambda p: save_npz(p, val_arr),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_test_sparse_matrix.npz",
        lambda p: save_npz(p, test_arr),
    )

    loguru_logger.info("Saving target arrays")
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_train_y.npy",
        lambda p: np.save(p, y_train.to_numpy()),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_val_y.npy",
        lambda p: np.save(p, y_val.to_numpy()),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_test_y.npy",
        lambda p: np.save(p, y_test.to_numpy()),
    )
    loguru_logger.success(f"Finished processing: {fname_prefix}")


def process_tabular_data_no_temporal_splitting_censoring(
    fname_prefix: str,
    sample_folder: str,
    data_folder: str,
    cutoff_year: int,
    included_cols: list[str],
    numerical_cols: list[str],
    nominal_cols: list[str],
    ordinal_cols: list[str],
    count_cols: list[str],
    min_df: float,
    max_count: int,
    loguru_logger: _Logger,
    negative_censor: int,
) -> None:
    """
    Processes one fname_prefix by preparing datasets, saving dense and sparse features, and outcomes.

    Args:
        fname_prefix (str): Prefix identifying the outcome (e.g., "neet", "dropout").
    """
    (FPATH.DATA / f"{data_folder}").mkdir(exist_ok=True)

    df_combined = pl.read_parquet(
        FPATH.NETWORK_DATA
        / f"{data_folder}"
        / f"{fname_prefix}_targets_tabular_censor_{negative_censor}.parquet"
    )

    loguru_logger.info("Copying train/val/test person ID parquet files")

    loguru_logger.info("Reading train/val/test person IDs")
    train_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{fname_prefix}_train_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()
    val_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{fname_prefix}_val_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()
    test_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{fname_prefix}_test_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()

    loguru_logger.info("Preparing test and development datasets")
    df_prepared = prepare_dataframe_no_temporal_splitting(
        df_combined,
        cutoff_year=cutoff_year,
        nominal_cols=nominal_cols,
        count_cols=count_cols,
    )

    loguru_logger.info("Saving all-info parquet files")
    df_test = df_prepared[df_prepared["person_id"].isin(test_person_ids)]
    df_train = df_prepared[df_prepared["person_id"].isin(train_person_ids)]
    df_val = df_prepared[df_prepared["person_id"].isin(val_person_ids)]

    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_test_all_info_censor_{negative_censor}.parquet",
        lambda p: df_test.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_train_all_info_censor_{negative_censor}.parquet",
        lambda p: df_train.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_val_all_info_censor_{negative_censor}.parquet",
        lambda p: df_val.to_parquet(p),
    )
    loguru_logger.info("Winsorizing numerical columns")

    percentiles = {
        col: (
            df_train[col].quantile(0.05),
            df_train[col].quantile(0.95),
        )
        for col in numerical_cols
    }

    df_train = winsorize(df_train, percentiles)
    df_val = winsorize(df_val, percentiles)
    df_test = winsorize(df_test, percentiles)

    loguru_logger.info("Extracting feature matrices and targets")
    X_test, y_test = df_test[included_cols], df_test["target"]
    X_train, y_train = df_train[included_cols], df_train["target"]
    X_val, y_val = df_val[included_cols], df_val["target"]

    loguru_logger.info("Saving dense feature matrices")
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_train_dense_censor_{negative_censor}.parquet",
        lambda p: X_train.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_val_dense_censor_{negative_censor}.parquet",
        lambda p: X_val.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_test_dense_censor_{negative_censor}.parquet",
        lambda p: X_test.to_parquet(p),
    )

    loguru_logger.info("Fitting sparse pipeline and transforming data")

    onehot_cols = sorted(nominal_cols + ordinal_cols)

    pipeline = create_sparse_pipeline(
        numerical_cols, onehot_cols, count_cols, min_df, max_count
    )
    train_arr = pipeline.fit_transform(X_train)
    val_arr = pipeline.transform(X_val)
    test_arr = pipeline.transform(X_test)

    loguru_logger.info("Saving sparse matrices and pipeline")
    joblib.dump(
        pipeline,
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_tabular_pipeline_censor_{negative_censor}.pkl",
    )
    FPATH.alternative_copy_to_opposite_drive(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_tabular_pipeline_censor_{negative_censor}.pkl"
    )
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_train_sparse_matrix_censor_{negative_censor}.npz",
        lambda p: save_npz(p, train_arr),
    )
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_val_sparse_matrix_censor_{negative_censor}.npz",
        lambda p: save_npz(p, val_arr),
    )
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_test_sparse_matrix_censor_{negative_censor}.npz",
        lambda p: save_npz(p, test_arr),
    )

    loguru_logger.info("Saving target arrays")
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_train_y_censor_{negative_censor}.npy",
        lambda p: np.save(p, y_train.to_numpy()),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{fname_prefix}_val_y_censor_{negative_censor}.npy",
        lambda p: np.save(p, y_val.to_numpy()),
    )
    save_and_copy(
        FPATH.DATA
        / data_folder
        / f"{fname_prefix}_test_y_censor_{negative_censor}.npy",
        lambda p: np.save(p, y_test.to_numpy()),
    )
    loguru_logger.success(f"Finished processing: {fname_prefix}")


from scipy.sparse import csr_matrix, issparse


def ensure_csr(X):
    return X if issparse(X) else csr_matrix(X)


def process_tabular_data_random_splitting(
    combined_prefix: str,
    person_id_prefix: str,
    output_prefix: str,
    sample_folder: str,
    data_folder: str,
    cutoff_year: int,
    included_cols: list[str],
    numerical_cols: list[str],
    nominal_cols: list[str],
    ordinal_cols: list[str],
    count_cols: list[str],
    min_df: float,
    max_count: int,
    loguru_logger: _Logger,
) -> None:
    """
    Processes one fname_prefix by preparing datasets, saving dense and sparse features, and outcomes.

    Args:
        fname_prefix (str): Prefix identifying the outcome (e.g., "neet", "dropout").
    """
    (FPATH.DATA / f"{data_folder}").mkdir(exist_ok=True)

    df_combined = pl.read_parquet(
        FPATH.NETWORK_DATA
        / f"{data_folder}"
        / f"{combined_prefix}_targets_tabular.parquet"
    )

    loguru_logger.info("Copying train/val/test person ID parquet files")

    loguru_logger.info("Reading train/val/test person IDs")
    train_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{person_id_prefix}_train_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()
    val_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{person_id_prefix}_val_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()
    test_person_ids = pl.read_parquet(
        FPATH.NETWORK_DATA / sample_folder / f"{person_id_prefix}_test_pids.parquet",
        columns=["person_id"],
    )["person_id"].to_list()

    loguru_logger.info("Preparing test and development datasets")
    df_prepared = prepare_dataframe_no_temporal_splitting(
        df_combined,
        nominal_cols=nominal_cols,
        count_cols=count_cols,
    )

    del df_combined

    loguru_logger.info("Saving all-info parquet files")
    df_test = df_prepared[df_prepared["person_id"].isin(test_person_ids)]
    df_train = df_prepared[df_prepared["person_id"].isin(train_person_ids)]
    df_val = df_prepared[df_prepared["person_id"].isin(val_person_ids)]

    del df_prepared

    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_test_all_info.parquet",
        lambda p: df_test.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_train_all_info.parquet",
        lambda p: df_train.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_val_all_info.parquet",
        lambda p: df_val.to_parquet(p),
    )

    loguru_logger.info("Winsorizing numerical columns")

    percentiles = {
        col: (
            df_train[col].quantile(0.05),
            df_train[col].quantile(0.95),
        )
        for col in numerical_cols
    }

    df_train = winsorize(df_train, percentiles)
    df_val = winsorize(df_val, percentiles)
    df_test = winsorize(df_test, percentiles)

    loguru_logger.info("Extracting feature matrices and targets")
    X_test, y_test = df_test[included_cols], df_test["target"]
    X_train, y_train = df_train[included_cols], df_train["target"]
    X_val, y_val = df_val[included_cols], df_val["target"]

    loguru_logger.info("Saving dense feature matrices")
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_train_dense.parquet",
        lambda p: X_train.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_val_dense.parquet",
        lambda p: X_val.to_parquet(p),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_test_dense.parquet",
        lambda p: X_test.to_parquet(p),
    )

    loguru_logger.info("Fitting sparse pipeline and transforming data")

    onehot_cols = sorted(nominal_cols + ordinal_cols)

    pipeline = create_sparse_pipeline(
        numerical_cols, onehot_cols, count_cols, min_df, max_count
    )
    train_arr = ensure_csr(pipeline.fit_transform(X_train))
    del X_train
    val_arr = ensure_csr(pipeline.transform(X_val))
    del X_val
    test_arr = ensure_csr(pipeline.transform(X_test))
    del X_test

    loguru_logger.info("Saving sparse matrices and pipeline")

    loguru_logger.info(f"Train array type: {type(train_arr)}, shape {train_arr.shape}")
    joblib.dump(
        pipeline,
        FPATH.DATA / data_folder / f"{output_prefix}_tabular_pipeline.pkl",
    )
    FPATH.alternative_copy_to_opposite_drive(
        FPATH.DATA / data_folder / f"{output_prefix}_tabular_pipeline.pkl"
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_train_sparse_matrix.npz",
        lambda p: save_npz(p, train_arr),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_val_sparse_matrix.npz",
        lambda p: save_npz(p, val_arr),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_test_sparse_matrix.npz",
        lambda p: save_npz(p, test_arr),
    )

    loguru_logger.info("Saving target arrays")
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_train_y.npy",
        lambda p: np.save(p, y_train.to_numpy()),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_val_y.npy",
        lambda p: np.save(p, y_val.to_numpy()),
    )
    save_and_copy(
        FPATH.DATA / data_folder / f"{output_prefix}_test_y.npy",
        lambda p: np.save(p, y_test.to_numpy()),
    )
    loguru_logger.success(f"Finished processing: {output_prefix}")


def save_column_types(
    prefix: str,
    data_folder: str,
    numerical: list[str],
    ordinal: list[str],
    nominal: list[str],
    count: list[str],
) -> None:
    """
    Saves column type lists to a JSON file.

    Args:
        prefix (str): Prefix which denotes the target.
        data_folder (str): Path to the folder to save the JSON file.
        numerical (list[str]): List of numerical columns.
        ordinal (list[str]): List of ordinal columns.
        nominal (list[str]): List of nominal columns.
        count (list[str]): List of count columns.
    """
    col_types = {
        "numerical": numerical,
        "ordinal": ordinal,
        "nominal": nominal,
        "count": count,
    }

    fpath = FPATH.DATA / data_folder / f"{prefix}_column_types.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(col_types, f, ensure_ascii=False, indent=2)

    FPATH.alternative_copy_to_opposite_drive(fpath)


def save_and_copy(file_path: Path, save_func: Callable[[Path], None]) -> None:
    """
    Saves a file using the provided save function and performs an alternative copy.

    Args:
        file_path (Path): The full path to the file to be saved.
        save_func (Callable[[Path], None]): A function that saves content to the given file path.
    """
    save_func(file_path)
    FPATH.alternative_copy_to_opposite_drive(file_path)


def compute_weighted_past_average(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each (person_id, year), compute a weighted average of all previous
    at_risk_above_threshold values, weighted by 1 / (year_diff + 1).

    Returns:
        pl.DataFrame with ['person_id', 'year', 'average_at_risk']
    """
    df = df.sort(["person_id", "year"])
    all_years = df["year"].unique().to_list()
    results = []

    for year in all_years:
        df_current = df.filter(pl.col("year") == year).select(["person_id"]).unique()
        df_history = df.filter(
            (pl.col("year") < year)
            & (pl.col("person_id").is_in(df_current["person_id"]))
        )

        if df_history.is_empty():
            continue

        df_weighted = (
            df_history.with_columns((pl.lit(year) - pl.col("year")).alias("year_diff"))
            .with_columns((1 / (pl.col("year_diff"))).alias("weight"))
            .with_columns(
                (pl.col("at_risk_above_threshold") * pl.col("weight")).alias(
                    "weighted_risk"
                )
            )
        )

        df_avg = (
            df_weighted.group_by("person_id")
            .agg(
                (pl.col("weighted_risk").sum() / pl.col("weight").sum()).alias(
                    "average_at_risk"
                )
            )
            .with_columns(pl.lit(year).alias("year"))
        )

        results.append(df_avg)

    return pl.concat(results)


def filter_feature_columns(
    cols: List[str],
    flag: Literal["own", "parents", "both"],
    mom_prefix: str = "mom_",
    dad_prefix: str = "dad_",
) -> List[str]:
    """
    Filter base feature columns to include only own, only parents, or both.

    Args:
        cols (List[str]):
            List of base (own) column names.
        flag (Literal["own", "parents", "both"]):
            Which set to include:
            - "own": only `cols`
            - "parents": only prefixed columns
            - "both": both own and prefixed
        mom_prefix (str):
            Prefix to apply for mother’s columns.
        dad_prefix (str):
            Prefix to apply for father’s columns.

    Returns:
        List[str]:
            Filtered list of column names.
    """
    own = cols
    parents = [f"{mom_prefix}{c}" for c in cols] + [f"{dad_prefix}{c}" for c in cols]
    if flag == "own":
        return own
    if flag == "parents":
        return parents
    return own + parents


def filter_existing_columns(
    available: List[str], desired: List[str], logger: _Logger
) -> List[str]:
    """
    Filter a list of desired column names against what’s actually available,
    logging a warning for each missing column.

    Args:
        available (List[str]):
            Column names present in the loaded data (e.g. schema.keys()).
        desired (List[str]):
            Column names you want to include.
        logger (Logger):
            Loguru logger instance for warnings.

    Returns:
        List[str]:
            Subset of `desired` that are found in `available`.
    """
    missing = set(desired) - set(available)
    for col in missing:
        logger.warning(f"Column '{col}' not found in data; excluding it.")
    return [c for c in desired if c in available]


def filter_out_parent_prefixed(
    cols: List[str],
    mom_prefix: str = "mom_",
    dad_prefix: str = "dad_",
) -> List[str]:
    """
    Remove columns that start with the mother or father prefix.

    Args:
        cols (List[str]):
            List of column names to filter.
        mom_prefix (str):
            Prefix used for mother’s columns.
        dad_prefix (str):
            Prefix used for father’s columns.

    Returns:
        List[str]:
            Filtered list excluding any names that start with the given prefixes.
    """
    return [
        c for c in cols if not (c.startswith(mom_prefix) or c.startswith(dad_prefix))
    ]
