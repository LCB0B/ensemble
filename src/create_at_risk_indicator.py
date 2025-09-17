# %%
import os

os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["RAYON_NUM_THREADS"] = "8"

from datetime import date, datetime, timedelta
from typing import List, Tuple

import polars as pl  # noqa: E402


def check_overlaps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Check for overlaps between consecutive date ranges per person.

    For each person (grouped by 'person_id'), this function sorts the records by 'date_from'
    and compares each row with the previous row. It flags:
      - 'overlap' if the current 'date_from' is before the previous 'date_to'

    Args:
        df (pl.DataFrame):
            - (pl.DataFrame) A Polars DataFrame with 'person_id', 'date_from', and 'date_to' columns
              where 'date_from' and 'date_to' are of datetime type.

    Returns:
        pl.DataFrame: A DataFrame with an additional boolean columns: 'overlap'
    """
    # Ensure data is sorted per person and by start date
    df_sorted = df.sort(["person_id", "date_from"])

    # Get the previous 'date_to' per person
    df_sorted = df_sorted.with_columns(
        pl.col("date_to").shift(1).over("person_id").alias("prev_date_to")
    )

    # Check for overlap
    df_checked = df_sorted.with_columns(
        [
            # True if current date_from is before previous date_to
            pl.col("date_from")
            .lt(pl.col("prev_date_to"))
            .fill_null(False)
            .alias("overlap")
        ]
    )

    return df_checked


def calculate_yearly_durations(df: pl.DataFrame, at_risk_col: str) -> pl.DataFrame:
    """
    For each person_id and each year (extracted from date_from), calculate the total duration (in days)
    during which the person is at-risk and not at-risk. Each interval is assumed to lie within a single calendar year.
    The returned DataFrame has one row per person_id and year with separate columns for at-risk and not-at-risk durations.

    Args:
        df (pl.DataFrame):
            A Polars DataFrame with the following columns:
                - person_id (any): Identifier for the person.
                - date_from (datetime or date): Start date of the interval.
                - date_to (datetime or date): End date of the interval.
                - at_risk_col (str): Column name indicating whether the person is at risk during the interval.

    Returns:
        pl.DataFrame: A DataFrame with columns:
                - person_id: Identifier of the person.
                - year (int): Year extracted from date_from.
                - {at_risk_col}_duration (int): Total duration (in days, including both endpoints) for at-risk periods.
                - not_{at_risk_col}_duration (int): Total duration (in days, including both endpoints) for non-at-risk periods.
                - observed_duration (int): Sum of observed durations.
    """

    yearly_durations = (
        df.with_columns([pl.col("date_from").dt.year().alias("year")])
        .group_by(["person_id", "year", at_risk_col])
        .agg(pl.col("duration").sum().alias("duration"))
    )

    # Group by person_id and year, then sum durations conditionally based on risk status.
    result = yearly_durations.group_by(["person_id", "year"]).agg(
        [
            pl.when(pl.col(at_risk_col) == 1)
            .then(pl.col("duration"))
            .otherwise(0)
            .sum()
            .alias(f"{at_risk_col}_duration"),
            pl.when(pl.col(at_risk_col) == 0)
            .then(pl.col("duration"))
            .otherwise(0)
            .sum()
            .alias(f"not_{at_risk_col}_duration"),
        ]
    )
    # Add observed duration
    result = result.with_columns(
        (
            pl.col(f"{at_risk_col}_duration") + pl.col(f"not_{at_risk_col}_duration")
        ).alias("observed_duration")
    )
    return result


def assert_correct_duration(df: pl.DataFrame) -> None:
    """
    Assert that 'observed_duration' matches the expected number of days in the year.

    Args:
        df (pl.DataFrame):
            - 'year' (int): The year.
            - 'observed_duration' (int): Number of days observed in the year.
    """
    df_expected = df.with_columns(
        pl.when(
            (pl.col("year") % 4 == 0)
            & ((pl.col("year") % 100 != 0) | (pl.col("year") % 400 == 0))
        )
        .then(366)
        .otherwise(365)
        .alias("expected_duration")
    )
    if not df_expected.filter(
        pl.col("observed_duration") != pl.col("expected_duration")
    ).is_empty():
        raise AssertionError("Some rows have incorrect observed_duration.")


def compute_consecutive_at_risk_indicators(
    df: pl.DataFrame, at_risk_col: str, max_years: int, thresholds: List[int]
) -> pl.DataFrame:
    """
    Compute at-risk status by day thresholds and identify consecutive at-risk years for each person.

    Args:
        df (pl.DataFrame): The input DataFrame containing:
            - at_risk_col (str): Column name indicating the at-risk duration.
            - "person_id" (str or int): Unique identifier for each person.
        at_risk_col (str): Name of the column indicating at-risk duration.
        max_years (int): Maximum number of consecutive years to evaluate.
        thresholds (List[int]): List of day thresholds to define at-risk status.

    Returns:
        pl.DataFrame: DataFrame with new columns named "at_risk_{n}_{threshold}_days" indicating whether a person
        has been at risk (i.e. at_risk_col >= threshold) for n consecutive years.
    """
    for threshold in thresholds:
        base_col = f"at_risk_{threshold}_days"
        df = df.with_columns(
            (pl.col(at_risk_col) >= threshold).cast(pl.Int64).alias(base_col)
        )
        for n in range(1, max_years + 1):
            df = df.with_columns(
                pl.col(base_col)
                .rolling_min(n)
                .over("person_id")
                .alias(f"at_risk_{n}_{threshold}_days")
            )
    return df


def compute_ever_at_risk_indicators(
    df: pl.DataFrame, max_years: int, thresholds: List[int]
) -> pl.DataFrame:
    """
    Compute a per-person indicator for each at-risk type showing whether they have ever been classified as at risk.

    Args:
        df (pl.DataFrame): Input DataFrame containing binary at-risk indicator columns named as
            "at_risk_{n}_{threshold}_days", where n is the number of consecutive years and threshold is in days.
        max_years (int): Maximum number of consecutive years considered.
        thresholds (List[int]): List of at-risk duration thresholds (in days).

    Returns:
        pl.DataFrame: A new DataFrame with one row per person, including 'person_id' and indicator columns
        named as "ever_at_risk_{n}_{threshold}_days", indicating whether the person was ever classified as at risk.
    """
    ever_exprs = []
    for n in range(1, max_years + 1):
        for threshold in thresholds:
            col_name = f"at_risk_{n}_{threshold}_days"
            ever_col_name = f"ever_{col_name}"
            ever_exprs.append(pl.col(col_name).max().alias(ever_col_name))

    return df.group_by("person_id").agg(ever_exprs)


def compute_consecutive_at_risk_indicators_no_threshold(
    df: pl.DataFrame, at_risk_col: str, max_years: int
) -> pl.DataFrame:
    """
    Compute consecutive at-risk years for each person based on a binary at-risk column.

    Args:
        df (pl.DataFrame): The input DataFrame containing:
            - at_risk_col (str): Column name indicating binary at-risk status (0 or 1).
            - "person_id" (str or int): Unique identifier for each person.
        at_risk_col (str): Name of the column indicating binary at-risk status.
        max_years (int): Maximum number of consecutive years to evaluate.

    Returns:
        pl.DataFrame: DataFrame with new columns named "at_risk_{n}_years" indicating whether a person
        has been at risk for n consecutive years.
    """
    for n in range(1, max_years + 1):
        df = df.with_columns(
            pl.col(at_risk_col)
            .rolling_min(n)
            .over("person_id")
            .alias(f"at_risk_{n}_years")
        )
    return df


def compute_ever_at_risk_indicators_no_threshold(
    df: pl.DataFrame, max_years: int
) -> pl.DataFrame:
    """
    Compute a per-person indicator showing whether they have ever been classified as at risk.

    Args:
        df (pl.DataFrame): Input DataFrame containing binary at-risk indicator columns named as
            "at_risk_{n}_years", where n is the number of consecutive years.
        max_years (int): Maximum number of consecutive years considered.

    Returns:
        pl.DataFrame: A new DataFrame with one row per person, including 'person_id' and indicator columns
        named as "ever_at_risk_{n}_years", indicating whether the person was ever classified as at risk.
    """
    ever_exprs = []
    for n in range(1, max_years + 1):
        col_name = f"at_risk_{n}_years"
        ever_col_name = f"ever_{col_name}"
        ever_exprs.append(pl.col(col_name).max().alias(ever_col_name))

    return df.group_by("person_id").agg(ever_exprs)


def join_at_risk_status(
    df_present: pl.DataFrame, yearly_at_risk_status: pl.DataFrame
) -> pl.DataFrame:
    """
    Join df_present with yearly_at_risk_status. For each row in df_present (with person_id and year),
    merge the at_risk_above_threshold value from yearly_at_risk_status for offsets:
      - offset +0: where yearly_at_risk_status.year equals df_present.year + 0.
      - offset +1: where yearly_at_risk_status.year equals df_present.year + 1,

    Args:
        df_present (pl.DataFrame):
            - DataFrame with columns:
                * person_id (any type)
                * year (int)
                * ... (other columns)
        yearly_at_risk_status (pl.DataFrame):
            - DataFrame with columns:
                * person_id (any type)
                * year (int)
                * at_risk_above_threshold (bool)

    Returns:
        pl.DataFrame:
            - DataFrame with the at_risk_above_threshold status from yearly_at_risk_status joined
              on the offsets. New columns:
                * at_risk_above_threshold_0.
                * at_risk_above_threshold_1,
    """
    # Ensure year columns have the same type.
    df_present = df_present.with_columns(pl.col("year").cast(pl.Int32))
    yearly_at_risk_status = yearly_at_risk_status.with_columns(
        pl.col("year").cast(pl.Int32)
    )

    # Prepare copies of yearly_at_risk_status with renamed year and status columns for each offset.

    yars0 = yearly_at_risk_status.rename(
        {"at_risk_above_threshold": "at_risk_above_threshold_0"}
    )
    yars1 = yearly_at_risk_status.rename(
        {"year": "year_plus_1", "at_risk_above_threshold": "at_risk_above_threshold_1"}
    )
    # Add shifted year columns to df_present.
    df_present = df_present.with_columns(
        [
            (pl.col("year") + 1).alias("year_plus_1"),
        ]
    )

    # Perform the joins on person_id and the shifted year columns.
    joined = df_present.join(yars0, on=["person_id", "year"], how="left").join(
        yars1, on=["person_id", "year_plus_1"], how="left"
    )

    # Drop the temporary shifted columns.
    result = joined.drop(["year_plus_1"])
    return result


def merge_intervals(intervals: List[Tuple[date, date]]) -> List[Tuple[date, date]]:
    """
    Merge overlapping or adjacent date intervals.

    Args:
        intervals (List[Tuple[date, date]]):
            - (date, date) tuple list representing intervals.

    Returns:
        List[Tuple[date, date]]:
            - List of merged intervals.
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last_start, last_end = merged[-1]
        current_start, current_end = current
        if current_start <= last_end + timedelta(days=1):
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append(current)
    return merged


def covers_full_year(intervals: List[Tuple[date, date]], year: int) -> bool:
    """
    Check if the merged intervals cover the full calendar year.

    Args:
        intervals (List[Tuple[date, date]]):
            - List of merged intervals.
        year (int):
            - Year to check.

    Returns:
        bool:
            - True if the intervals cover from January 1 to December 31 of the year.
    """
    start_year = date(year, 1, 1)
    end_year = date(year, 12, 31)
    # Check if any interval covers the whole year.
    for int_start, int_end in intervals:
        if int_start <= start_year and int_end >= end_year:
            return True
    return False


def to_date(dt: object) -> date:
    """
    Convert a datetime object to a date object.

    Args:
        dt (object):
            - A date or datetime object.

    Returns:
        date:
            - Date object extracted from dt.
    """
    return dt.date() if isinstance(dt, datetime) else dt


def process_person_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Process event data to compute for each person, for ages 16 to 26,
    the year they turn that age and an indicator for whether they were
    present for the full calendar year.

    Args:
        df (pl.DataFrame):
            - DataFrame with columns:
                * person_id (any type)
                * birthday (date)
                * event_start_date (date)
                * event_final_date (date)

    Returns:
        pl.DataFrame:
            - DataFrame with columns:
                * person_id
                * year (int): Year when the person turns a given age.
                * age (int)
                * full_year (bool): True if present for full calendar year.
    """
    # Group by person_id and aggregate event intervals and birthday.
    grouped = df.group_by("person_id").agg(
        [
            pl.col("birthday").first().alias("birthday"),
            pl.col("event_start_date").explode().alias("start_dates"),
            pl.col("event_final_date").explode().alias("final_dates"),
        ]
    )

    rows = []
    for row in grouped.iter_rows(named=True):
        pid = row["person_id"]
        bday = row["birthday"]
        start_dates = [to_date(s) for s in row["start_dates"]]
        final_dates = [to_date(f) for f in row["final_dates"]]

        # Create list of event intervals
        intervals = [(s, f) for s, f in zip(start_dates, final_dates)]
        merged_intervals = merge_intervals(intervals)

        # Process ages 16 to 25
        # Need 25 due to two consecutive years
        for age in range(16, 26):
            turning_year = bday.year + age
            if turning_year < 2008 or turning_year > 2019:
                continue
            else:
                full_year = covers_full_year(merged_intervals, turning_year)
                rows.append(
                    {
                        "person_id": pid,
                        "year": turning_year,
                        "age": age,
                        "full_year": full_year,
                    }
                )

    return pl.DataFrame(rows)


def mark_two_consecutive_presence(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each person, mark the row as True if the person is present for the given
    year and the next two consecutive years using polars shift.

    Args:
        df (pl.DataFrame):
            - DataFrame with columns:
                * person_id (any type)
                * year (int)
                * full_year (bool)

    Returns:
        pl.DataFrame:
            - DataFrame with an additional column 'two_consecutive' (bool),
              True if the person is present (full_year True) for two consecutive years.
    """
    # For each person, sort by year and compute shifted columns.
    return (
        df.sort("year")
        .with_columns(
            [
                pl.col("year").shift(-1).over("person_id").alias("year_next"),
                # pl.col("year").shift(-2).over("person_id").alias("year_next2"),
                # pl.col("year").shift(-3).over("person_id").alias("year_next3"),
                pl.col("full_year").shift(-1).over("person_id").alias("full_next"),
                # pl.col("full_year").shift(-2).over("person_id").alias("full_next2"),
                # pl.col("full_year").shift(-3).over("person_id").alias("full_next3"),
            ]
        )
        .with_columns(
            (
                (pl.col("year_next") == pl.col("year") + 1)
                # & (pl.col("year_next2") == pl.col("year") + 2)
                # & (pl.col("year_next3") == pl.col("year") + 3)
                & pl.col("full_year")
                & pl.col("full_next")
                # & pl.col("full_next2")
                # & pl.col("full_next3")
            ).alias("two_consecutive")
        )
        .drop(
            [
                "year_next",
                # "year_next2",
                # "year_next3",
                "full_next",
                # "full_next2",
                # "full_next3",
            ]
        )
    )
