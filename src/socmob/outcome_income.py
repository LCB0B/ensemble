from functools import reduce
from itertools import product
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from tqdm.auto import tqdm


def deflate_income(
    df: pl.DataFrame, price_df: pl.DataFrame, income_col: str, base_year: int
) -> pl.DataFrame:
    """
    Deflate a nominal income column by a price index, anchoring to a specified base year.

    Args:
        df (pl.DataFrame): DataFrame with columns "year" and the nominal income column.
        price_df (pl.DataFrame): Single-row DataFrame of price indices, years as columns.
        income_col (str): Name of the nominal income column in df.
        base_year (int): Year to use as the base (index = 100).

    Returns:
        pl.DataFrame: Input DataFrame with an added "real_<income_col>" column.
    """
    # Melt price index into long form and cast year
    price_long = price_df.unpivot(
        variable_name="year", value_name="price_index"
    ).with_columns(pl.col("year").cast(pl.Int32))
    # Extract the index value for the base year
    base_index = price_long.filter(pl.col("year") == base_year).select("price_index")[
        0, 0
    ]
    # Join, deflate by (price_index / base_index * 100), and clean up
    return (
        df.join(price_long, on="year")
        .with_columns(
            (pl.col(income_col) / (pl.col("price_index") / base_index)).alias(
                f"real_{income_col}"
            )
        )
        .drop("price_index")
    ), f"real_{income_col}"


def total_income_by_age_range(
    df: pl.DataFrame,
    min_age: int,
    max_age: int,
    max_missing_ages: int,
    income_col: str = "brutto_income",
) -> pl.DataFrame:
    """
    Calculate total income per person over a given age range,
    allowing for a specified number of missing years.

    Args:
        df (pl.DataFrame): DataFrame with columns 'person_id', 'age' and income column.
        min_age (int): Minimum age (inclusive).
        max_age (int): Maximum age (inclusive).
        max_missing_ages (int): Maximum number of ages without income data allowed.
        income_col (str): Name of the income column (default 'brutto_income').

    Returns:
        pl.DataFrame: DataFrame with 'person_id' and 'total_income' for persons
        whose missing-year count ≤ max_missing_years.
    """
    # filter by age range
    df = df.filter(pl.col("age").is_between(min_age, max_age))
    expected_ages = max_age - min_age + 1

    # group, count years, sum income, compute missing and filter
    return (
        df.group_by("person_id", maintain_order=True)
        .agg(
            [
                pl.col(income_col).sum().alias("total_income"),
                pl.col("age").n_unique().alias("years_present"),
            ]
        )
        .with_columns(
            [(pl.lit(expected_ages) - pl.col("years_present")).alias("missing_ages")]
        )
        .filter(pl.col("missing_ages") <= max_missing_ages)
        .select(["person_id", "total_income"])
    )


def add_total_income_rank(
    df_total: pl.DataFrame,
    df_bday: pl.DataFrame,
    income_col: str = "total_income",
    out_col: str = "total_income_pct",
    cohort_col: str = "birth_year",
) -> pl.DataFrame:
    """
    Add a 0-1 percentile rank of total income within each birth cohort.

    Args:
        df_total (pl.DataFrame): ['person_id', income_col] with total income.
        df_bday (pl.DataFrame): ['person_id', 'birthday'] (Date).
        income_col (str): Column holding total income.
        out_col (str): Name of the resulting percentile column.
        cohort_col (str): Name for the cohort column to create.

    Returns:
        pl.DataFrame: Original columns plus `out_col`.
    """
    # attach cohort (birth year)
    df = df_total.join(
        df_bday.with_columns(pl.col("birthday").dt.year().alias(cohort_col)).select(
            ["person_id", cohort_col]
        ),
        on="person_id",
        how="left",
    )

    # percentile rank within cohort
    df = (
        df.with_columns(
            [
                pl.col(income_col).rank("average").over(cohort_col).alias("_r"),
                pl.count().over(cohort_col).alias("_n"),
            ]
        )
        .with_columns(((pl.col("_r") - 1) / (pl.col("_n") - 1)).alias(out_col))
        .drop(["_r", "_n"])
    )

    return df


def mean_income_forward_by_age(
    df: pl.DataFrame,
    target_age: int,
    window_size: int,
    max_missing_ages: int,
    income_col: str = "brutto_income",
    age_col: str = "age",
) -> pl.DataFrame:
    """
    Compute each person’s average income over ages
    [target_age, target_age + window_size), excluding those with too many missing ages.

    Args:
        df (pl.DataFrame): DataFrame with 'person_id', age column, and income column.
        target_age (int): Starting age of the forward window.
        window_size (int): Number of ages in the window.
        max_missing_ages (int): Maximum allowed missing age observations.
        income_col (str): Name of the income column (default "brutto_income").
        age_col (str): Name of the age column (default "age").

    Returns:
        pl.DataFrame: DataFrame with columns:
            - person_id
            - target_age
            - mean_income
    """
    summary = (
        df.filter(
            (pl.col(age_col) >= target_age)
            & (pl.col(age_col) < target_age + window_size)
        )
        .group_by("person_id")
        .agg(
            [
                pl.col(income_col).mean().alias("mean_income"),
                pl.col(age_col).n_unique().alias("n_ages"),
            ]
        )
        .filter((window_size - pl.col("n_ages")) <= max_missing_ages)
        .with_columns(pl.lit(target_age).alias("target_age"))
        .select(["person_id", "target_age", "mean_income"])
    )

    return summary


def compute_mean_income_windows(
    df: pl.DataFrame,
    target_ages: List[int],
    window_sizes: List[int],
    max_missing_ages: int,
    income_col: str = "brutto_income",
    age_col: str = "age",
) -> pl.DataFrame:
    """
    Compute mean income for multiple forward age‐windows and assemble into a wide DataFrame.

    Args:
        df (pl.DataFrame): DataFrame with 'person_id', age and income columns.
        target_ages (List[int]): Starting ages of the forward windows.
        window_sizes (List[int]): Sizes of each forward window.
        max_missing_ages (int): Maximum allowed missing age entries in each window.
        income_col (str): Name of the income column.
        age_col (str): Name of the age column.

    Returns:
        pl.DataFrame: Wide DataFrame with 'person_id' and one column per
                      (target_age, window_size) named
                      f"mean_{income_col}_age{target_age}_w{window_size}".
    """
    results: List[pl.DataFrame] = []
    combos = list(product(target_ages, window_sizes))
    for target_age, window_size in tqdm(combos, total=len(combos)):
        alias = f"mean_{income_col}_age{target_age}_w{window_size}"
        df_window = mean_income_forward_by_age(
            df,
            target_age=target_age,
            window_size=window_size,
            max_missing_ages=max_missing_ages,
            income_col=income_col,
            age_col=age_col,
        ).rename({"mean_income": alias})
        results.append(df_window.select(["person_id", alias]))

    return reduce(
        lambda left, right: left.join(right, on="person_id", how="inner"), results
    )


def rank_within_cohort(
    df_income: pl.DataFrame,
    df_bday: pl.DataFrame,
    metric_cols: List[str] | None = None,
    cohort_col: str = "birth_year",
    out_suffix: str = "_pct",
) -> pl.DataFrame:
    """
    Add 0–1 percentile ranks for each income metric within birth cohorts.

    Args:
        df_income (pl.DataFrame): Wide table, must include 'person_id'.
        df_bday (pl.DataFrame): ['person_id', 'birthday'] (Date).
        metric_cols (List[str] | None): Income columns to rank.
            If None, use all non-id columns.
        cohort_col (str): Column name for cohort (defaults to birth_year).
        out_suffix (str): Suffix for rank columns when re-pivoted.

    Returns:
        pl.DataFrame: Same dimensions as input + rank columns.
    """
    # 1. attach cohort
    df_income = df_income.join(
        df_bday.with_columns(pl.col("birthday").dt.year().alias(cohort_col)).select(
            ["person_id", cohort_col]
        ),
        on="person_id",
        how="left",
    )

    # 2. choose metrics
    if metric_cols is None:
        metric_cols = [
            c for c in df_income.columns if c not in ("person_id", cohort_col)
        ]

    # 3. long format
    df_long = df_income.melt(
        id_vars=["person_id", cohort_col],
        value_vars=metric_cols,
        variable_name="metric",
        value_name="value",
    )

    # 4. percentile rank within cohort × metric
    df_rank = (
        df_long.with_columns(
            [
                pl.col("value")
                .rank("average")
                .over([cohort_col, "metric"])
                .alias("_r"),
                pl.count().over([cohort_col, "metric"]).alias("_n"),
            ]
        )
        .with_columns(((pl.col("_r") - 1) / (pl.col("_n") - 1)).alias("pct_rank"))
        .select(["person_id", "metric", "pct_rank"])
    )

    # 5. back to wide
    df_wide = df_rank.pivot(
        index="person_id", columns="metric", values="pct_rank"
    ).rename({m: f"{m}{out_suffix}" for m in metric_cols})

    # 6. merge ranks to original (keeps original order/columns)
    return df_wide


def corr_by_age_window(
    df_ranked: pl.DataFrame,
    df_total_rank: pl.DataFrame,
    income_col: str,
    target_ages: Iterable[int],
    window_sizes: Iterable[int],
    lifetime_col: str = "total_income_pct",
) -> pd.DataFrame:
    """
    Correlate lifetime rank with early-career rank columns named
    f"mean_{income_col}_age{age}_w{window}{suffix}".

    Args:
        df_ranked (pl.DataFrame): Wide table with rank columns + person_id.
        df_total_rank (pl.DataFrame): ['person_id', lifetime_col].
        income_col (str): Core income identifier (e.g. 'real_brutto_income').
        target_ages (Iterable[int]): Ages to test.
        window_sizes (Iterable[int]): Window sizes to test.
        lifetime_col (str): Lifetime rank column.

    Returns:
        pd.DataFrame: ['age', 'window_size', 'corr'].
    """
    df = df_ranked.join(
        df_total_rank.select(["person_id", lifetime_col]),
        on="person_id",
        how="inner",
    )

    rows: List[dict] = []
    for age in target_ages:
        for w in window_sizes:
            col = f"mean_{income_col}_age{age}_w{w}"
            if col not in df.columns:
                continue
            rho = df.select(pl.corr(col, lifetime_col)).item()
            rows.append({"age": age, "window_size": w, "corr": rho})

    return pl.DataFrame(rows)


def plot_age_window_corr(
    corr_df: pl.DataFrame,
    ax: Optional[Axes] = None,
) -> None:
    """
    Draw lines of ρ(age, window) vs. age on a given Matplotlib axis.

    Args:
        corr_df (pl.DataFrame): Output from `corr_by_age_window`.
        ax (Axes | None): Axis to plot on. Creates a new figure if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.lineplot(
        data=corr_df.to_pandas(),
        x="age",
        y="corr",
        hue="window_size",
        marker="o",
        palette="Set2",
        ax=ax,
    )


def plot_age_window_heatmap(
    corr_df: pl.DataFrame,
    ax: Optional[Axes] = None,
    cmap: str = "Greens",
) -> None:
    """
    Heat-map of rank correlations with
    y-axis = age and x-axis = window size.

    Args:
        corr_df (pl.DataFrame): Must have ['age','window_size','corr'].
        ax (Axes | None): Axis to draw on (new fig if None).
        cmap (str): Matplotlib colormap name.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    heat = (
        corr_df.to_pandas()
        .pivot(index="age", columns="window_size", values="corr")
        .sort_index()
        .sort_index(axis=1)
    )

    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"label": "ρ (early rank, lifetime rank)"},
        ax=ax,
    )
    ax.set_xlabel("Window size (years)")
    ax.set_ylabel("Age at measurement")
    ax.set_title("Rank Correlation Heat-map")


def cohort_rank_corr_mae(
    df_ranked: pl.DataFrame,
    df_total_rank: pl.DataFrame,
    df_bday: pl.DataFrame,
    income_col: str,
    target_ages: Iterable[int],
    window_sizes: Iterable[int],
    suffix: str = "_pct",
    lifetime_col: str = "total_income_pct",
) -> pd.DataFrame:
    """
    For each (age, window) compute:
      1. Correlation of early rank with lifetime rank **within every birth cohort**.
      2. mae of these cohort-level correlations relative to their own mean.

    Args:
        df_ranked (pl.DataFrame): Wide table with rank columns + person_id.
        df_total_rank (pl.DataFrame): Lifetime rank for each person.
        df_bday (pl.DataFrame): ['person_id', 'birthday'] (Date) for cohorts.
        income_col (str): Core string of the early-rank column names.
        target_ages, window_sizes: Ages & windows to evaluate.
        suffix (str): Suffix of rank cols (default '_pct').
        lifetime_col (str): Column holding lifetime rank.

    Returns:
        pl.DataFrame: ['age', 'window_size', 'mae'].
    """
    # ── merge once ───────────────────────────────────────────────────────────
    df = df_ranked.join(
        df_total_rank.select(["person_id", lifetime_col]),
        on="person_id",
        how="inner",
    ).join(
        df_bday.with_columns(pl.col("birthday").dt.year().alias("birth_year")).select(
            ["person_id", "birth_year"]
        ),
        on="person_id",
        how="left",
    )
    rows: List[dict] = []
    for age in target_ages:
        for w in window_sizes:
            col = f"mean_{income_col}_age{age}_w{w}"
            if col not in df.columns:
                continue

            # correlation per cohort
            corr_by_cohort = (
                df.select(["birth_year", col, lifetime_col])
                .group_by("birth_year")
                .agg(pl.corr(col, lifetime_col).alias("corr"))
            )

            if corr_by_cohort.is_empty():
                continue

            # Polars → pandas (tiny)
            c = corr_by_cohort.to_pandas()["corr"]
            mae = float(np.mean(np.abs(c - c.mean())))
            rows.append({"age": age, "window_size": w, "mae": mae})

    return pl.DataFrame(rows)


def plot_mae_heatmap(
    mae_df: pd.DataFrame,
    ax: Optional[Axes] = None,
    cmap: str = "Reds",
) -> None:
    """
    Heatmap of mae(ρ) with y = age, x = window size.

    Args:
        mae_df (pd.DataFrame): Output of `cohort_rank_corr_mae`.
        ax (Axes | None): Axis to plot on. Creates new figure if None.
        cmap (str): Matplotlib colormap.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    heat = (
        mae_df.to_pandas()
        .pivot(index="age", columns="window_size", values="mae")
        .sort_index()
        .sort_index(axis=1)
    )

    sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar_kws={"label": "MAE of Cohort Correlations"},
        ax=ax,
    )
    ax.set_xlabel("Window size (years)")
    ax.set_ylabel("Age at measurement")
    ax.set_title("Heterogeneity of Correlations Across Cohorts")

def cohort_correlations(
    df_ranked: pl.DataFrame,
    df_total_rank: pl.DataFrame,
    df_bday: pl.DataFrame,
    income_col: str,
    age: int,
    window_size: int,
    lifetime_col: str = "total_income_pct",
    n_boot: int = 1_000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Pearson correlation between early-rank and lifetime-rank within each birth
    cohort, plus bootstrap 95 % CI.

    Args:
        … same as before …
        n_boot (int): Bootstrap draws per cohort.
        random_state (int | None): Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        columns ['birth_year','corr','ci_low','ci_high'].
    """
    rng = np.random.default_rng(random_state)
    col = f"mean_{income_col}_age{age}_w{window_size}"
    if col not in df_ranked.columns:
        raise ValueError(f"Column '{col}' not found.")

    df = (
        df_ranked.select(["person_id", col])
        .join(df_total_rank.select(["person_id", lifetime_col]), on="person_id")
        .join(
            df_bday.with_columns(
                pl.col("birthday").dt.year().alias("birth_year")
            ).select(["person_id", "birth_year"]),
            on="person_id",
        )
    )

    records: list[dict] = []
    for yr, grp in df.group_by("birth_year"):
        x, y = grp[col].to_numpy(), grp[lifetime_col].to_numpy()
        base_corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan

        # bootstrap
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(x), len(x))
            boots.append(np.corrcoef(x[idx], y[idx])[0, 1])
        ci_low, ci_high = np.nanpercentile(boots, [2.5, 97.5])
        records.append(
            {
                "birth_year": int(yr[0]),
                "corr": base_corr,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pl.DataFrame(records).sort("birth_year")


def plot_cohort_correlations(
    corr_df: pl.DataFrame,
    ax: Optional[Axes] = None,
) -> None:
    """
    Line plot of cohort correlations with 95 % CI ribbons.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    corr_df = corr_df.to_pandas()

    sns.lineplot(data=corr_df, x="birth_year", y="corr", marker="o", ax=ax)
    ax.fill_between(
        corr_df["birth_year"],
        corr_df["ci_low"],
        corr_df["ci_high"],
        alpha=0.25,
        color=ax.lines[0].get_color(),
    )
    ax.set(
        ylim=(0, 1),
        xlabel="Birth Year",
        ylabel="ρ (early rank, lifetime rank)",
    )
