import polars as pl
from datetime import datetime
from typing import Union, List
from pathlib import Path
from src.paths import check_and_copy_file_or_dir

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


def calculate_means_modes(
    df: pl.DataFrame,
    person_id: Union[int, List[int]],
    reference_year: int,
    mean_columns: List[str] = None,
    mode_columns: List[str] = None,
    previous_years: int = 1,
) -> pl.DataFrame:
    """
    Calculate means and/or modes for specific columns within the given number of previous years
    for a person_id, based on a reference year. Either means, modes, or both can be calculated.

    Args:
        df (pl.DataFrame): DataFrame containing the data with person_id and datetime columns.
        person_id (Union[int, List[int]]): The person_id(s) to filter.
        reference_year (int): The year used to calculate the time window.
        mean_columns (List[str], optional): List of column names for mean calculation.
        mode_columns (List[str], optional): List of column names for mode calculation.
        previous_years (int): Number of previous years to consider for the calculation. Default is 1.

    Returns:
        pl.DataFrame: DataFrame with calculated means and/or modes for the given person_id.
    """
    # Define the time range (last 'previous_years' from the reference_year)
    start_date = datetime(reference_year - previous_years, 1, 1)
    end_date = datetime(reference_year, 1, 1)

    # Filter the dataframe for the given person_id(s) and the time window
    filtered_df = df.filter(
        (pl.col("person_id").is_in(person_id))
        & (pl.col("date_to").is_between(start_date, end_date))
    )

    # Group by person_id
    grouped = filtered_df.group_by("person_id")

    # List to store aggregations
    agg_list = []

    # Calculate means if mean_columns is provided
    if mean_columns:
        agg_list.extend(
            [pl.col(col).mean().alias(f"{col}_mean") for col in mean_columns]
        )

    # Calculate modes if mode_columns is provided
    if mode_columns:
        agg_list.extend(
            [
                pl.col(col).mode().first().cast(pl.Utf8).alias(f"{col}_mode")
                for col in mode_columns
            ]
        )

    # Aggregate the results based on the specified columns
    if agg_list:
        result = grouped.agg(agg_list)
    else:
        raise ValueError("Either 'mean_columns' or 'mode_columns' must be provided.")

    return result


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



def collect_filtered_parquet(path: Path, person_ids: List[int]) -> pl.LazyFrame:
    """
    Loads a Parquet file and filters rows based on person_ids.

    Args:
        path (Path): The path to the Parquet file.
        person_ids (List[int]): List of person IDs to filter by.

    Returns:
        pl.LazyFrame: A LazyFrame with filtered rows.
    """
    check_and_copy_file_or_dir(path)
    return pl.scan_parquet(path).filter(pl.col('person_id').is_in(person_ids)).collect()


def filter_and_flag_events(
    df: pl.DataFrame,
    target_date: datetime,
    date_col: str,
    days_col: str,
    flag_col: str,
    event_col: str
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
    result_df = df.with_columns([
        (pl.col(date_col) + pl.duration(days=pl.col(days_col)) > target_date)
        .cast(pl.Int8)  # Convert bool to int
        .alias(flag_col),
        pl.lit(1).alias(event_col)  # Add event column with value 1
    ]).select(["person_id", flag_col, event_col])

    return result_df


def filter_and_flag_events_with_binary(
    df: pl.DataFrame,
    target_date: datetime,
    date_col: str,
    days_col: str,
    binary_col: str,
    flag_col: str,
    event_col: str,
    combined_event_col: str
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
    result_df = df.with_columns([
        (pl.col(date_col) + pl.duration(days=pl.col(days_col)) > target_date)
        .cast(pl.Int8)  # Convert bool to int
        .alias(flag_col),
        pl.lit(1).alias(event_col),  # Add event column with value 1
        (pl.col(binary_col) == 1)
        .cast(pl.Int8)
        .alias(combined_event_col)  # Add combined_event_col if both conditions are met
    ]).select(["person_id", flag_col, event_col, combined_event_col])

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
    df_with_flags  = df.with_columns(
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

    result_df = (
        df_with_flags  
        .group_by("person_id")
        .agg([
            pl.col(flag_col).max().alias(flag_col),
            pl.col(event_col).max().alias(event_col)
        ])
    )

    return result_df

import polars as pl
import pandas as pd
from typing import List


def prepare_dataframe(
    df: pl.DataFrame,
    cutoff_year: int,
    onehot_cols: List[str],
    count_cols: List[str],
    is_test: bool,
) -> pd.DataFrame:
    """
    Filters, converts, and fills specified columns in the DataFrame.

    Args:
        df (pl.DataFrame): Original dataframe to filter and process.
        cutoff_year (int): Year value to use as cutoff for filtering.
        onehot_cols (List[str]): Columns to convert to category with missing filled as 'None'.
        count_cols (List[str]): Columns to convert to string with missing filled as an empty string.
        is_test (bool): If True, filters rows where year == cutoff_year, otherwise where year < cutoff_year.

    Returns:
        pd.DataFrame: Processed and filtered DataFrame.
    """
    if is_test:
        df_filtered = df.filter(pl.col("year") == cutoff_year).to_pandas()
    else:
        df_filtered = df.filter(pl.col("year") < cutoff_year).to_pandas()

    for col in onehot_cols:
        df_filtered[col] = (
            df_filtered[col].astype("string").fillna("None").astype("category")
        )
    for col in count_cols:
        df_filtered[col] = df_filtered[col].astype("string").fillna("")

    return df_filtered