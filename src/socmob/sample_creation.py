import math
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import polars as pl
import seaborn as sns

from src.paths import FPATH


def split_by_cohort(
    df: pl.DataFrame,
    cohort_col: str,
    person_id_col: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split a Polars DataFrame into train/validation/test sets by assigning whole persons
    within each cohort.

    Args:
        df (pl.DataFrame): DataFrame containing person IDs and cohort labels.
        cohort_col (str): Name of the cohort column.
        person_id_col (str): Name of the person‐ID column.
        train_frac (float): Fraction of persons per cohort for the training split.
        val_frac (float): Fraction of persons per cohort for the validation split.
        test_frac (float): Fraction of persons per cohort for the test split.
        seed (int): Random seed for reproducible shuffling.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
            Train, validation, and test DataFrames.
    """
    assert train_frac + val_frac + test_frac == 1, "Fractions should sum to one"
    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.group_by([cohort_col], maintain_order=True):
        # get unique persons in this cohort
        persons = (
            group.select(person_id_col)
            .unique()
            .sample(seed=seed, fraction=1, shuffle=True)
            .to_series()
            .to_list()
        )
        n = len(persons)
        n_train = math.ceil(train_frac * n)
        n_val = math.ceil(val_frac * n)

        train_ids = set(persons[:n_train])
        val_ids = set(persons[n_train : n_train + n_val])
        test_ids = set(persons[n_train + n_val :])

        # filter entire group by assigned person‐IDs
        train_parts.append(group.filter(pl.col(person_id_col).is_in(train_ids)))
        val_parts.append(group.filter(pl.col(person_id_col).is_in(val_ids)))
        test_parts.append(group.filter(pl.col(person_id_col).is_in(test_ids)))

    train_df = pl.concat(train_parts, how="vertical", rechunk=True).drop(cohort_col)
    val_df = pl.concat(val_parts, how="vertical", rechunk=True).drop(cohort_col)
    test_df = pl.concat(test_parts, how="vertical", rechunk=True).drop(cohort_col)

    return train_df, val_df, test_df


def apply_ranks_by_nearest_train(
    df: pl.DataFrame,
    train_pids: pl.Series,
    val_pids: pl.Series,
    test_pids: pl.Series,
    person_id_col: str,
    cohort_col: str,
    value_col: str,
    output_rank_col: str,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Compute within-cohort ranks using training set and assign ranks to validation/test
    by matching on cohort and nearest value (e.g. income or GPA).

    Args:
        df (pl.DataFrame): Data with value and cohort columns.
        train_pids (pl.Series): person_ids for training set.
        val_pids (pl.Series): person_ids for validation set.
        test_pids (pl.Series): person_ids for test set.
        person_id_col (str): Name of person_id column.
        cohort_col (str): Name of cohort column.
        value_col (str): Name of the value column (e.g. "mean_income" or "gpa").
        output_rank_col (str): Name of output rank column.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: train/val/test with rank column.
    """
    # Split
    train = df.filter(pl.col(person_id_col).is_in(train_pids))
    val = df.filter(pl.col(person_id_col).is_in(val_pids))
    test = df.filter(pl.col(person_id_col).is_in(test_pids))

    # Compute cohort-based ranks in train
    train = (
        train.with_columns(
            pl.col(value_col).rank(method="average").over(cohort_col).alias("rank_raw")
        )
        .with_columns(
            (pl.col("rank_raw") / pl.len().over(cohort_col)).alias(output_rank_col)
        )
        .drop("rank_raw")
    )

    train_ranks = train.select([cohort_col, value_col, output_rank_col]).sort(
        [cohort_col, value_col]
    )

    val = val.sort([cohort_col, value_col])
    test = test.sort([cohort_col, value_col])

    def map_nearest_rank(df_split: pl.DataFrame) -> pl.DataFrame:
        return df_split.join_asof(
            train_ranks,
            on=value_col,
            by=cohort_col,
            strategy="backward",
        )

    return (
        train,
        map_nearest_rank(val),
        map_nearest_rank(test),
    )


def load_and_prepare_data() -> pl.DataFrame:
    """
    Load and preprocess UDFK 9th grade exam data with age and course filters.

    Returns:
        df_9th (pl.DataFrame): Preprocessed DataFrame.
    """
    courses = ["Fysik/kemi", "Engelsk", "Matematik", "Dansk"]

    df_udfk = pl.read_parquet(
        FPATH.NETWORK_DUMP_DIR / "udfk_DUMP_AUGMENTED.parquet"
    ).cast({"year": int})
    df_background = pl.read_parquet(
        FPATH.NETWORK_DUMP_DIR / "all_background_integers.parquet"
    ).select(["person_id", "birthday"])

    df_9th = (
        df_udfk.filter(
            (pl.col("class_level") == "09")
            & (pl.col("test_type") == "Afgangsprøve")
            & (pl.col("course").is_in(courses) & (pl.col("test_discipline") != "Orden"))
        )
        .join(df_background, on="person_id", how="left")
        .with_columns(
            [
                pl.col("birthday").dt.year().alias("birthyear"),
                (pl.col("year") - pl.col("birthday").dt.year()).alias("age"),
                pl.datetime(pl.col("year") - 1, 8, 1).alias("school_start_date"),
                (pl.datetime(pl.col("year"), 1, 5) - pl.col("birthday"))
                .dt.total_days()
                .alias("days_may"),
            ]
        )
        .with_columns((pl.col("days_may") / 365.25).alias("years_may"))
        .filter(pl.col("birthday").is_not_null())
    )
    return df_9th


def plot_summary(
    df_9th: pl.DataFrame,
    fig: plt.Figure,
    axs: list[plt.Axes],
    course_labels: dict[str, str],
) -> None:
    """
    Create a 1x3 subplot summary on externally provided axes:
    1. Age distribution (binned)
    2. Barplot of people with grades in each course
    3. GPA distplot over selected courses, with count >= 5 in each bin

    Args:
        df_9th (pl.DataFrame): Preprocessed exam dataset.
        fig (plt.Figure): Matplotlib figure object.
        axs (list[Axes]): List of 3 Matplotlib axes.
        course_labels: dict[str, str]: Dict mapping labels of courses
    """
    df_pd = df_9th.to_pandas()

    # 1. years_may bin plot
    df_pd["age_group"] = pd.cut(
        df_pd["years_may"],
        bins=[-float("inf"), 13, 14, 15, 16, 17, 18, float("inf")],
        labels=["≤13", "14", "15", "16", "17", "18", "19≤"],
        right=True,
    )

    # Assert all bins have count ≥ 5
    age_bin_counts = df_pd["age_group"].value_counts().sort_index()
    assert (
        age_bin_counts >= 5
    ).all(), f"Bins with <5: {age_bin_counts[age_bin_counts < 5].to_dict()}"

    sns.countplot(
        data=df_pd,
        x="age_group",
        ax=axs[0],
        edgecolor="black",
        linewidth=0.8,
    )
    axs[0].set_title("Age at start of school year of received grade")
    axs[0].set_xlabel("Age group")
    axs[0].set_ylabel("Count")
    axs[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x*1e-6:.0f}M"))

    course_labels

    # 2. Count of students per course
    course_counts = (
        df_9th.group_by(["person_id", "course"])
        .first()
        .group_by("course")
        .len()
        .sort("len")
        .with_columns(pl.col("course").replace_strict(course_labels).alias("course"))
        .to_pandas()
    )
    sns.barplot(
        data=course_counts,
        x="course",
        y="len",
        ax=axs[1],
        edgecolor="black",
        linewidth=0.8,
    )
    axs[1].set_title("Grade occurrence across students")
    axs[1].set_xlabel("Course")
    axs[1].set_ylabel("Number of students")
    axs[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x*1e-6:.1f}M"))

    # # Grey out unused phys/chem grade
    # first_bar = axs[1].patches[0]
    # first_bar.set_facecolor("grey")
    # first_bar.set_alpha(0.9)
    # first_bar.set_edgecolor("none")
    # first_bar.set_linewidth(0)

    # 3. GPA distribution (excluding fysik/kemi)
    gpa_df = (
        df_9th.filter(~pl.col("course").str.contains("Fysik/kemi"))
        .group_by(["person_id", "course"])
        .agg(pl.col("grade").mean().alias("within_course_gpa"))
        .group_by(["person_id"])
        .agg(pl.col("within_course_gpa").mean().alias("gpa"))
        .to_pandas()
    )
    sns.histplot(
        data=gpa_df,
        x="gpa",
        kde=False,
        ax=axs[2],
        binwidth=1,
        binrange=(-3, 12),
        thresh=5,
        edgecolor="black",
        linewidth=0.8,
    )
    axs[2].set_xlim(-4, 13)
    x_ticks = list(range(-3, 13))
    axs[2].set_xticks(x_ticks)
    axs[2].set_title("Distribution of Danish, Math and English GPA")
    axs[2].set_xlabel("Grade point aveage")
    axs[2].set_ylabel("Count")
    axs[2].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x*1e-3:.0f}K"))

    # Extract and assert bin counts
    counts = [patch.get_height() for patch in axs[2].patches]
    assert all(c >= 5 for c in counts), "Found GPA bins with count < 5"

    # Remove spines
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
