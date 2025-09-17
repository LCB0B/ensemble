# %%
import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import polars as pl  # noqa: E402
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from polars import DataFrame

from src.chunking import yield_chunks_given_pnrs
from src.paths import FPATH, check_and_copy_file_or_dir


def load_event_counts_with_major_group(
    config: Dict[str, Dict[str, Any]], output_folder: Path
) -> pl.DataFrame:
    """
    Loads event count files and appends a major group column based on config keys.

    Args:
        config (Dict): Dictionary with dataset configs.
        output_folder (Path): Folder where event_counts_<key>.parquet files are stored.

    Returns:
        pl.DataFrame: Combined dataframe with 'person_id', 'year', 'count', 'major_group'.
    """
    dfs = []
    for key in config.keys():
        file_path = output_folder / f"event_counts_{key}.parquet"
        if file_path.exists():
            df = pl.read_parquet(file_path)
            df = df.with_columns(
                # pl.lit(config[key]["desc_name"]).alias("minor_group"),
                pl.lit(config[key]["overall_group_name"]).alias("major_group"),
            )
            dfs.append(df)

    if dfs:
        return pl.concat(dfs)
    else:
        return pl.DataFrame()


def plot_overall_and_group_fraction(
    lens_overall: pl.DataFrame,
    lens_major_groups: pl.DataFrame,
    x_max: int,
    colors: Dict[str, str],
    fig: plt.Figure,
    axes: Tuple[Axes, Axes],
) -> None:
    """
    Renders a 1×2 plot on the given Figure and Axes:
      - Left: barplot of overall total_count frequencies.
      - Right: stacked barplot of major_group fractions per total_count (bar height = 1).
    Saves the figure to `save_path`.

    Args:
        lens_overall (pl.DataFrame): DataFrame with ['person_id','total_count'].
        lens_major_groups (pl.DataFrame): DataFrame with ['person_id','major_group'].
        x_max (int): Maximum total_count to include (inclusive).
        colors (Dict[str, str]): Mapping from major_group to color.
        fig (plt.Figure): Matplotlib Figure to draw on.
        axes (Tuple[Axes, Axes]): Tuple of two Axes: (ax_overall, ax_fraction).
    """
    ax1, ax2 = axes

    # --- overall freq ---
    overall = (
        lens_overall.filter(pl.col("total_count") <= x_max)
        .group_by("total_count")
        .agg(pl.len().alias("freq"))
        .sort("total_count")
        .to_pandas()
    )

    # --- fractions per bin ---
    records = []
    for k in overall["total_count"]:
        pids = lens_overall.filter(pl.col("total_count") == k)["person_id"].to_list()
        sub = lens_major_groups.filter(pl.col("person_id").is_in(pids))
        cnt = sub.group_by("major_group").agg(pl.len().alias("n")).to_pandas()
        cnt["total_count"] = k
        cnt["fraction"] = cnt["n"] / cnt["n"].sum()
        records.append(cnt[["total_count", "major_group", "fraction"]])
    frac_df = pd.concat(records, ignore_index=True)

    pivot = (
        frac_df.pivot(index="total_count", columns="major_group", values="fraction")
        .fillna(0)
        .loc[overall["total_count"]]
    )

    # --- plotting ---
    ax1.bar(overall["total_count"], overall["freq"], color="grey", edgecolor="black")
    ax1.set_xlabel("Total events")
    ax1.set_ylabel("Number of people")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    ax1.set_title("Overall frequency")
    ax1.set_xticks(range(0, x_max + 1))

    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax2,
        color=[colors.get(g) for g in pivot.columns],
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )
    ax2.set_xlabel("Total events")
    ax2.set_ylabel("Fraction")
    ax2.set_ylim(0, 1)
    ax2.set_title("Group distribution (fraction)")
    ax2.set_xticklabels(pivot.index, rotation=0)

    # legend below
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False, title=""
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])


def get_chunk(source: ds.Dataset, chunk_pnrs: List) -> DataFrame:
    table = source.to_table(
        filter=pc.is_in(pc.field("person_id"), pa.array(chunk_pnrs))
    )
    return pl.from_arrow(table)


def filter_dataset_to_dataframe(source, person_ids):
    table = source.to_table(
        filter=pc.is_in(pc.field("person_id"), pa.array(person_ids))
    )
    return pl.from_arrow(table)


def compute_population_by_age(
    events_parquet_path: Union[str, Path],
    person_ids: List[int],
    chunk_size: int = 200_000,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
) -> pl.DataFrame:
    """
    Stream‐load event spans, explode into ages, and count distinct
    persons at each integer age.

    Returns pl.DataFrame[age: i64, n_people: i64].
    """
    dataset = ds.dataset(events_parquet_path, format="parquet")
    age_counts: Dict[int, int] = {}

    for df in yield_chunks_given_pnrs(dataset, person_ids, chunk_size):
        # cast to dates & compute years
        df = df.with_columns(
            [
                pl.col("event_start_date").cast(pl.Date).alias("start"),
                pl.col("event_final_date").cast(pl.Date).alias("end"),
                pl.col("birthday").cast(pl.Date).alias("birth"),
            ]
        ).with_columns(
            [
                (pl.col("start").dt.year() - pl.col("birth").dt.year()).alias(
                    "start_age"
                ),
                (pl.col("end").dt.year() - pl.col("birth").dt.year()).alias("end_age"),
            ]
        )

        # filter out spans completely outside [min_age, max_age]
        if min_age is not None:
            df = df.filter(pl.col("end_age") >= min_age)
        if max_age is not None:
            df = df.filter(pl.col("start_age") <= max_age)

        # explode each row into all integer ages in [start_age..end_age]
        df = (
            df.with_columns([(pl.col("end_age") + 1).alias("end_age_plus")])
            .with_columns(
                [pl.int_ranges("start_age", "end_age_plus", 1).alias("age_list")]
            )
            .explode("age_list")
            .rename({"age_list": "age"})
        )

        # dedupe per-person per-age
        df = df.select(["person_id", "age"]).unique(subset=["person_id", "age"])

        # count distinct persons at each age in this chunk
        counts = (
            df.group_by("age")
            .agg(pl.count("person_id").alias("cnt"))
            .filter(pl.col("age").is_not_null())
        )
        for age, cnt in counts.rows():
            age_counts[age] = age_counts.get(age, 0) + cnt

    return pl.DataFrame(
        [
            (age, cnt)
            for age, cnt in sorted(age_counts.items())
            if (min_age is None or age >= min_age)
            and (max_age is None or age <= max_age)
        ],
        schema=["age", "n_people"],
    )


def compute_age_group_averages(
    counts_parquet_path: Union[str, Path],
    background_df: pl.DataFrame,
    pop_df: pl.DataFrame,
    person_ids: List[int],
    value_col: str,
    chunk_size: int = 200_000,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
) -> pl.DataFrame:
    """
    Stream‐load the counts Parquet in person_id‐filtered chunks, join the in‐memory
    `background_df` to compute age, then for each (age, major_group) compute the
    mean of `value_col` by dividing the summed values by the population size in
    `pop_df`.

    Args:
        counts_parquet_path (str | Path): Path to the Parquet file containing
            columns ["person_id","year","major_group", <value_col>].
        background_df (pl.DataFrame): In‐memory table with columns
            ["person_id","birthyear"].
        pop_df (pl.DataFrame): In‐memory table with columns ["age","n_people"].
        person_ids (List[int]): List of person_id values to include.
        value_col (str): Name of the column to sum & average (e.g.
            "row_count_sum" or "non_null_count_sum").
        chunk_size (int): Number of person_ids per chunk load.
        min_age (int | None): If set, only include ages ≥ this value.
        max_age (int | None): If set, only include ages ≤ this value.

    Returns:
        pl.DataFrame: Columns ["age","major_group","mean_value"], where
            `mean_value` = sum of `value_col` for that (age, major_group)
            divided by `n_people` for that age.
    """
    # slim down background to just person_id→birthyear
    bg = background_df.select(["person_id", "birthyear"]).unique()

    # build age→population map
    pop_map: Dict[int, int] = {
        age: count for age, count in pop_df.select(["age", "n_people"]).iter_rows()
    }

    # prepare to accumulate sums by (age, major_group)
    sum_by_age_group: Dict[Tuple[int, str], float] = {}

    dataset = ds.dataset(counts_parquet_path, format="parquet")

    # stream through the large counts table in chunks
    for df in yield_chunks_given_pnrs(dataset, person_ids, chunk_size):
        # join birthyear
        df = df.join(bg, on="person_id", how="left")
        # compute age = calendar year − birthyear
        df = df.with_columns((pl.col("year") - pl.col("birthyear")).alias("age"))

        # apply age filters
        if min_age is not None:
            df = df.filter(pl.col("age") >= min_age)
        if max_age is not None:
            df = df.filter(pl.col("age") <= max_age)

        # sum value_col by (age, major_group)
        chunk_agg = df.group_by(["age", "major_group"]).agg(
            pl.col(value_col).sum().alias("sum_value")
        )

        # accumulate into the global dict
        for age, grp, total in chunk_agg.rows():
            key = (age, grp)
            sum_by_age_group[key] = sum_by_age_group.get(key, 0.0) + total

    # build final DataFrame of means
    rows: List[Tuple[int, str, float]] = []
    for (age, grp), total_sum in sum_by_age_group.items():
        n_people = pop_map.get(age, 0)
        if n_people > 0:
            mean_val = total_sum / n_people
            rows.append((age, grp, mean_val))

    return pl.DataFrame(rows, schema=["age", "major_group", "mean_value"])


def plot_age_grouped_mean_stacked(
    ax,
    df: pl.DataFrame,
    color_mapping: Dict[str, str],
    label_mapping: Dict[str, str],
    linewidth: float = 0.5,
) -> None:
    """
    df must have columns ["age","major_group","mean_value"].
    """
    # pivot to wide form
    pdf = df.to_pandas()
    pivot = pdf.pivot(index="age", columns="major_group", values="mean_value").fillna(0)

    # colors + labels
    cols = pivot.columns
    colors = [color_mapping.get(c, c) for c in cols]
    labels = [label_mapping.get(c, c) for c in cols]

    # stacked bar plot
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        linewidth=linewidth,
        edgecolor="black",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean amount within age")
    ax.set_xticklabels(pivot.index.astype(str), rotation=45, ha="right")
    ax.legend(labels, title=None)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_age_ticks(ax: Axes, min_age: int, max_age: int) -> None:
    """
    Place minor ticks at every age and major ticks (longer) at ages divisible by 5.

    Args:
        ax (Axes): Matplotlib axis.
        min_age (int): first age.
        max_age (int): last age.
    """
    ages: List[int] = list(range(min_age, max_age + 1))
    pos = [a - min_age for a in ages]

    # major ticks at ages % 5 == 0
    major_ages = [a for a in ages if a % 5 == 0]
    major_pos = [a - min_age for a in major_ages]

    # set major ticks + labels
    ax.set_xticks(major_pos, minor=False)
    ax.set_xticklabels([str(a) for a in major_ages], rotation=45, ha="center")

    # set minor ticks (no labels)
    ax.set_xticks(pos, minor=True)

    # styling: longer major, shorter minor
    ax.tick_params(axis="x", which="major", length=6, width=1)
    ax.tick_params(axis="x", which="minor", length=4, width=0.5)


def plot_binned_group_average_lineplot(
    ax: Axes,
    df: pl.DataFrame,
    count_column: str,
    binwidth: int = 100,
    color_mapping: Dict[str, str] = None,
    label_mapping: Dict[str, str] = None,
    min_bin_count: int = 5,
    linewidth=1.5,
) -> None:
    """
    Plot a lineplot where each point is the average count_column per group in bins of total counts per person.
    Persons are binned by their total count_column, and for each bin, the average count_column is computed per group.

    Args:
        ax (Axes): (Axes) Matplotlib axis to plot on.
        df (pl.DataFrame): (pl.DataFrame) DataFrame with columns 'person_id', 'overall_group', and the count_column.
        count_column (str): (str) Column name to aggregate and plot.
        binwidth (int): (int) Width of bins when grouping total values per person.
        color_mapping (Dict[str, str]): (dict) Optional color mapping for each group.
        label_mapping (Dict[str, str]): (dict) Optional label mapping for each group.
        min_bin_count (int): (int) Minimum number of persons in a bin for it to be included.
        linewidth (float): Width of lines.
    """

    # Convert to pandas for convenience
    pdf = df.to_pandas()

    # Step 1: Total per person
    totals = (
        pdf.groupby("person_id", observed=False)[count_column]
        .sum()
        .reset_index(name="total")
    )
    merged = pdf.merge(totals, on="person_id")

    # Step 2: Assign bin based on total
    max_total = merged["total"].max()
    bins = np.arange(0, max_total + binwidth, binwidth)
    merged["bin"] = pd.cut(merged["total"], bins=bins, right=False)

    # Step 3: Group by bin and overall_group, then compute mean count_column
    grouped = (
        merged.groupby(["bin", "overall_group"], observed=False)[count_column]
        .mean()
        .reset_index(name="mean_value")
    )

    # Step 4: Only keep bins with enough total people
    bin_counts = merged.groupby("bin", observed=False)["person_id"].nunique()
    valid_bins = bin_counts[bin_counts >= min_bin_count].index
    grouped = grouped[grouped["bin"].isin(valid_bins)]

    # Step 5: Prepare data for plotting
    bin_centers = [interval.left + binwidth / 2 for interval in grouped["bin"]]
    grouped["bin_center"] = bin_centers

    for group in sorted(grouped["overall_group"].unique()):
        group_df = grouped[grouped["overall_group"] == group]
        color = color_mapping.get(group, None) if color_mapping else None
        label = label_mapping.get(group, group) if label_mapping else group
        ax.plot(
            group_df["bin_center"],
            group_df["mean_value"],
            # marker="o",
            label=label,
            color=color,
            linewidth=linewidth,
        )

    ax.set_yscale("log")
    ax.legend(title="Group")


def plot_yearly_grouped_mean_metric(
    ax: Axes,
    df: pl.DataFrame,
    column_type: str,
    start_year: int = 1990,
    end_year: int = 2023,
    color_mapping: Dict[str, str] = None,
    label_mapping: Dict[str, str] = None,
    linewidth=0.5,
) -> None:
    """
    Plot a stacked barplot of mean values per year grouped by overall_group.
    Uses provided color and label mappings for overall groups.

    Args:
        ax (Axes): (Axes) The matplotlib axis to plot on.
        df (pl.DataFrame): (pl.DataFrame) DataFrame with person-level per-year data and an 'overall_group' column.
        column_type (str): (str) Column suffix to plot, e.g., "row_count" or "non_null_values".
        start_year (int): (int) Starting year for the plot (inclusive).
        color_mapping (Dict[str, str]): (dict) Mapping of overall_group to color.
        label_mapping (Dict[str, str]): (dict) Mapping of overall_group to new label.
    """

    # Identify all relevant years
    year_cols = [
        col
        for col in df.columns
        if col.endswith(f"_{column_type}")
        if not col.startswith("total")
    ]

    years = [int(col.split("_")[0]) for col in year_cols]
    years = sorted(
        [year for year in years if (year >= start_year) & (year <= end_year)]
    )

    # Get unique overall groups in the DataFrame
    groups = sorted(df["overall_group"].unique().to_list())

    # Prepare data: group by overall_group and compute means per year
    group_means = {}
    for group in groups:
        group_df = df.filter(pl.col("overall_group") == group)
        means = []
        for year in years:
            col = f"{year}_{column_type}"
            means.append(group_df[col].mean() if col in group_df.columns else 0)
        group_means[group] = means

    # Create stacked barplot using mappings if provided.
    bottom = np.zeros(len(years))
    for group in groups:
        means = group_means[group]
        color = color_mapping.get(group, None) if color_mapping else None
        label = label_mapping.get(group, group) if label_mapping else group
        ax.bar(
            years,
            means,
            bottom=bottom,
            label=label,
            color=color,
            linewidth=linewidth,
            edgecolor="black",
        )
        bottom += np.array(means)

    ax.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")


def create_person_group_dataframe(
    person_ids: List[Any],
    overall_group: str,
    aggregated: Dict[Any, Dict[str, Dict[str, Any]]],
) -> pl.DataFrame:
    """
    Create a Polars DataFrame for a list of person_ids and a given overall group.
    The DataFrame will have a 'person_id' column, an 'overall_group' column, a 'total_rows'
    column, a 'total_non_null_values' column (summing non-null values across all years),
    and additional columns for each year in the aggregated data.
    For each year, two columns will be created:
      - '{year}_row_count': the number of rows for that year.
      - '{year}_non_null_values': the sum of non-null values for that year.
    If a person does not have data for a given year, 0 is used.

    Args:
        person_ids (List[Any]): (list) List of person_ids to include in the DataFrame.
        overall_group (str): (str) The overall group name to filter data.
        aggregated (Dict[Any, Dict[str, Dict[str, Any]]]): (dict) Aggregated data
            structured as {person_id: {overall_group: {"total_rows": int, "years": {year: {"row_count": int, "non_null_values": int}}}}}

    Returns:
        pl.DataFrame: A Polars DataFrame with the aggregated information, including total non-null values.
    """
    # Determine all years present across the provided persons for the given overall_group.
    all_years = set()
    for person_id in person_ids:
        group_data = aggregated.get(person_id, {}).get(overall_group, {})
        years = group_data.get("years", {})
        all_years.update(years.keys())
    all_years = sorted(all_years)

    # Build list of rows as dicts.
    rows = []
    for person_id in person_ids:
        # Default values if no data exists for the person and group.
        person_data = aggregated.get(person_id, {}).get(overall_group, {})
        total_rows = person_data.get("total_rows", 0)
        years_data = person_data.get("years", {})

        # Create a row dictionary with basic columns.
        row = {
            "person_id": person_id,
            "overall_group": overall_group,
            "total_rows": total_rows,
        }

        total_non_null = 0
        # For each year, add row_count and non_null_values columns.
        for year in all_years:
            year_info = years_data.get(year, {})
            row_count = year_info.get("row_count", 0)
            non_null_values = year_info.get("non_null_values", 0)
            row[f"{year}_row_count"] = row_count
            row[f"{year}_non_null_values"] = non_null_values
            total_non_null += non_null_values

        row["total_non_null_values"] = total_non_null
        rows.append(row)

    return pl.DataFrame(rows)


def filtered_histplot(
    ax: Axes,
    df: pl.DataFrame,
    column: str,
    binwidth: int = 100,
    color: str = "skyblue",
    edgecolor: str = "black",
) -> None:
    """
    Plot histogram with fixed binwidth, only showing bins with at least 5 observations.

    Args:
        ax (Axes): Matplotlib axis.
        df (pl.DataFrame): DataFrame containing data.
        column (str): Column to plot.
        binwidth (int): Width of histogram bins.
        color (str): Bar color.
        edgecolor (str): Edge color of bars.
    """
    data = df[column].to_numpy()

    # Define bins
    bins = np.arange(0, data.max() + binwidth, binwidth)

    # Calculate counts per bin
    counts, bin_edges = np.histogram(data, bins=bins)

    # Filter bins with at least 5 counts
    valid_bins = counts >= 5

    if not valid_bins.any():
        ax.text(0.5, 0.5, "No bins with ≥5 observations", ha="center", va="center")
        return

    # Plot only valid bins
    filtered_counts = counts[valid_bins]
    filtered_counts = filtered_counts * 5_855_898 / 25_000
    filtered_edges = bin_edges[:-1][valid_bins]

    ax.bar(
        filtered_edges,
        filtered_counts,
        width=binwidth,
        align="edge",
        color=color,
        edgecolor=edgecolor,
        linewidth=0.5,
    )

    ax.set_yscale("log")
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Frequency")


def plot_yearly_grouped_mean_value(
    ax: Axes,
    df: pl.DataFrame,
    column_type: str,
    start_year: int = 1990,
    end_year: int = 2023,
    color_mapping: Optional[Dict[str, str]] = None,
    label_mapping: Optional[Dict[str, str]] = None,
    linewidth: float = 0.5,
) -> None:
    """
    Plot a stacked barplot of mean values per year grouped by major_group.

    Args:
        ax (Axes): Matplotlib axis to plot on.
        df (pl.DataFrame): DataFrame containing at least columns
            ["person_id", "year", "major_group", column_type].
        column_type (str): Name of the column to average, e.g.
            "row_count_sum" or "non_null_count_sum".
        start_year (int): First year (inclusive) to include.
        end_year (int): Last year (inclusive) to include.
        color_mapping (Dict[str, str], optional): Map from major_group
            to a matplotlib color string.
        label_mapping (Dict[str, str], optional): Map from major_group
            to legend label.
        linewidth (float): Width of bar edges.
    """
    # 1. filter to desired year range
    df_filtered = df.filter(
        (pl.col("year") >= start_year) & (pl.col("year") <= end_year)
    )

    # 2. group by year and major_group, compute mean
    df_mean = (
        df_filtered.group_by(["year", "major_group"])
        .agg(pl.mean(column_type).alias("mean_value"))
        .sort("year")
    )

    # 3. convert to pandas and pivot for stacked bars
    pdf = df_mean.to_pandas()
    pivot = pdf.pivot(index="year", columns="major_group", values="mean_value").fillna(
        0
    )

    # 3a. ensure every year shows up
    full_years = list(range(start_year, end_year + 1))
    pivot = pivot.reindex(index=full_years, fill_value=0)

    # 5. build color list in column order
    colors = None
    if color_mapping:
        colors = [color_mapping.get(col, col) for col in pivot.columns]

    # 4. apply label mapping if provided
    if label_mapping:
        pivot = pivot.rename(columns=label_mapping)

    # 6. plot
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        linewidth=linewidth,
        edgecolor="black",
    )
    ax.set_xticklabels(pivot.index.astype(str), rotation=45, ha="right")
    ax.legend(title=None)


def set_yearly_ticks(ax: Axes, start_year: int, end_year: int) -> None:
    """
    Place minor ticks at every year and major ticks (longer) at years divisible by 5.

    Args:
        ax (Axes): Matplotlib axis.
        start_year (int): first year.
        end_year (int): last year.
    """
    # full range and positions
    years = list(range(start_year, end_year + 1))
    pos = [y - start_year for y in years]

    # major ticks at multiples of 5
    major_years = [y for y in years if y % 5 == 0]
    major_pos = [y - start_year for y in major_years]

    # set major ticks + labels
    ax.set_xticks(major_pos, minor=False)
    ax.set_xticklabels([str(y) for y in major_years], rotation=45, ha="center")

    # set minor ticks (no labels)
    ax.set_xticks(pos, minor=True)

    # styling: longer major, shorter minor
    ax.tick_params(axis="x", which="major", length=6, width=1)
    ax.tick_params(axis="x", which="minor", length=4, width=0.5)


def binned_group_stats_from_dataset(
    parquet_path: Union[str, Path],
    person_ids: List[int],
    value_col: str,
    binwidth: int,
    chunk_size: int = 200_000,
    max_year: Optional[int] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute per‐bin means and counts of persons per bin.

    Args:
        parquet_path: Path to your Parquet with columns
          ["person_id","year","major_group",…].
        person_ids: List of all person_ids.
        value_col: "row_count_sum" or "non_null_count_sum".
        binwidth: Width of each bin.
        chunk_size: How many person_ids per chunk.
        max_year: If set, filter out rows where year > max_year.

    Returns:
        means_df (pl.DataFrame): ["bin","bin_center","major_group","mean_total"]
        counts_df (pl.DataFrame): ["bin","bin_center","count_persons"]
    """
    dataset = ds.dataset(parquet_path, format="parquet")
    bin_counts: Dict[int, int] = {}
    group_sums: Dict[Tuple[int, str], float] = {}

    for df in yield_chunks_given_pnrs(dataset, person_ids, chunk_size):
        if max_year is not None:
            df = df.filter(pl.col("year") <= max_year)

        # per-person total → bin
        per_total = (
            df.group_by("person_id")
            .agg(pl.col(value_col).sum().alias("person_total"))
            .with_columns((pl.col("person_total") // binwidth * binwidth).alias("bin"))
            .select(["person_id", "bin"])
        )
        # count persons per bin
        for bin_val, cnt in (
            per_total.group_by("bin").agg(pl.count().alias("cnt")).rows()
        ):
            bin_counts[bin_val] = bin_counts.get(bin_val, 0) + cnt

        # per-person per-group total
        per_group = df.group_by(["person_id", "major_group"]).agg(
            pl.col(value_col).sum().alias("group_total")
        )
        # join bin info
        joined = per_group.join(per_total, on="person_id", how="inner")
        # sum group_total per (bin,group)
        for bin_val, grp, sum_val in (
            joined.group_by(["bin", "major_group"])
            .agg(pl.col("group_total").sum().alias("sum_total"))
            .rows()
        ):
            key = (bin_val, grp)
            group_sums[key] = group_sums.get(key, 0.0) + sum_val

    # build outputs
    means_rows = []
    for (bin_val, grp), total_sum in group_sums.items():
        count = bin_counts.get(bin_val, 0)
        if count:
            means_rows.append((bin_val, bin_val + binwidth / 2, grp, total_sum / count))

    counts_rows = [(b, b + binwidth / 2, c) for b, c in sorted(bin_counts.items())]

    means_df = pl.DataFrame(
        means_rows, schema=["bin", "bin_center", "major_group", "mean_total"]
    )
    counts_df = pl.DataFrame(counts_rows, schema=["bin", "bin_center", "count_persons"])
    return means_df, counts_df


def _log_floor(val: float) -> float:
    """
    Return the largest power-of-10 less than or equal to val.
    E.g. _log_floor(37) -> 10, _log_floor(0.3) -> 0.1.
    """
    exp = math.floor(math.log10(val))
    return 10**exp


def plot_2x2_binned_stats(
    axes: Sequence[Sequence[Axes]],
    events_means: pl.DataFrame,
    events_counts: pl.DataFrame,
    tokens_means: pl.DataFrame,
    tokens_counts: pl.DataFrame,
    event_binwidth: int,
    token_binwidth: int,
    x_lim_events: int,
    x_lim_tokens: int,
    color_mapping: Dict[str, str],
    label_mapping: Dict[str, str],
    linewidth: float = 2.0,
) -> None:
    """
    Populate a 2×2 grid of pre-created axes:
      Top row: barplots of person-counts per bin (log-y, auto-bottom)
      Bottom row: lineplots of mean_total per major_group (log-y, auto-bottom)

    Args:
        fig: The matplotlib Figure object.
        axes: 2×2 array of Axes:
            axes[0][0]=events counts, [0][1]=tokens counts,
            axes[1][0]=events mean,   [1][1]=tokens mean.
        events_means: DataFrame with ["bin","bin_center","major_group","mean_total"] for events.
        events_counts: DataFrame with ["bin","bin_center","count_persons"] for events.
        tokens_means: same as events_means but for tokens.
        tokens_counts: same as events_counts but for tokens.
        event_binwidth: width of event bins.
        token_binwidth: width of token bins.
        x_lim_events: max x-axis for event plots.
        x_lim_tokens: max x-axis for token plots.
        color_mapping: major_group → color.
        label_mapping: major_group → legend label.
        linewidth: width of mean lines.
    """
    # Top row: counts barplots
    for ax, df_cnt, xlim, title, bw in zip(
        axes[0],
        [events_counts, tokens_counts],
        [x_lim_events, x_lim_tokens],
        ["Events (count)", "Tokens (count)"],
        [event_binwidth, token_binwidth],
    ):
        pdf = df_cnt.filter(pl.col("bin_center") <= xlim).sort("bin_center").to_pandas()
        ax.bar(
            pdf["bin_center"],
            pdf["count_persons"],
            width=bw * 0.9,
            color="lightgray",
            edgecolor="black",
        )
        # compute bottom based on minimal positive count
        pos_counts = pdf["count_persons"][pdf["count_persons"] > 0]
        bottom = _log_floor(pos_counts.min()) if not pos_counts.empty else 1
        ax.set_yscale("log")
        ax.set_ylim(bottom=bottom)
        ax.set_xlim(0, xlim)
        ax.set_ylabel("Person count (log scale)")
        ax.set_title(title)
        ax.grid(axis="y", ls="--", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # annotate
        for x, y in zip(pdf["bin_center"], pdf["count_persons"]):
            label = f"{y/1000:.1f}k"
            ax.text(x, y * 1.05, label, ha="center", va="bottom", fontsize=8)

    # Bottom row: mean lineplots
    for ax, df_mean, xlim, title in zip(
        axes[1],
        [events_means, tokens_means],
        [x_lim_events, x_lim_tokens],
        ["Events (mean)", "Tokens (mean)"],
    ):
        pdf = df_mean.sort("bin_center").to_pandas()
        pivot = pdf.pivot(
            index="bin_center",
            columns="major_group",
            values="mean_total",
        ).fillna(0)
        # get minimal positive mean for bottom
        arr = pivot.values.flatten()
        pos_vals = arr[arr > 0]
        bottom = _log_floor(pos_vals.min()) if pos_vals.size else 1e-3
        for grp in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[grp].replace(0, np.nan),
                label=label_mapping.get(grp, grp),
                color=color_mapping.get(grp),
                linewidth=linewidth,
            )
        ax.set_yscale("log")
        ax.set_ylim(bottom=bottom)
        ax.set_xlim(0, xlim)
        ax.set_ylabel("Mean per group (log scale)")
        ax.set_title(title)
        ax.grid(True, ls="--", lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def aggregate_prepped_event_counts(
    configs: Dict[str, Dict[str, str]],
    prep_folder: Path,
    output_folder: Path,
    person_ids: List[int],
    chunk_size: int = 1_000_000,
) -> None:
    """
    Aggregate per-key event counts and non-null counts by person and year,
    streaming in chunks to avoid OOM, and write two Parquet files per key.

    Args:
        configs (Dict[str, Dict[str, str]]):
            Mapping from dataset key to its config dict (must match filenames).
        prep_folder (Path):
            Directory under `FPATH.DATA` containing `{key}.parquet` files.
        output_folder (Path):
            Directory under `FPATH.DATA` where `{key}_row_counts.parquet` and
            `{key}_non_null_counts.parquet` will be written.
        person_ids (List[int]):
            List of person_id values to include in the aggregation.
        chunk_size (int):
            Number of person_ids to process per chunk.
    """
    for key in configs:
        # ensure input exists
        in_path = FPATH.DATA / prep_folder / f"{key}.parquet"
        check_and_copy_file_or_dir(in_path)
        source = ds.dataset(str(in_path), format="parquet")

        writer_rows: Optional[pq.ParquetWriter] = None
        writer_nonnull: Optional[pq.ParquetWriter] = None

        # stream in chunks of person_ids
        for chunk in yield_chunks_given_pnrs(source, person_ids, chunk_size):
            # extract year and drop the raw date
            df = chunk.with_columns(pl.col("date_col").dt.year().alias("year")).drop(
                "date_col"
            )

            # 1) row counts per person/year
            row_counts = df.group_by(["person_id", "year"]).agg(
                pl.count().alias("row_count")
            )
            table_rows = row_counts.to_arrow()
            if writer_rows is None:
                writer_rows = pq.ParquetWriter(
                    FPATH.DATA / output_folder / f"{key}_row_counts.parquet",
                    schema=table_rows.schema,
                )
            writer_rows.write_table(table_rows)

            # 2) non-null counts per person/year
            other_cols = [c for c in df.columns if c not in ("person_id", "year")]
            melted = df.unpivot(
                index=["person_id", "year"],
                on=other_cols,
                variable_name="column",
                value_name="value",
            ).filter(pl.col("value").is_not_null())
            nonnull_counts = melted.group_by(["person_id", "year"]).agg(
                pl.count("value").alias("non_null_count")
            )
            table_nonnull = nonnull_counts.to_arrow()
            if writer_nonnull is None:
                writer_nonnull = pq.ParquetWriter(
                    FPATH.DATA / output_folder / f"{key}_non_null_counts.parquet",
                    schema=table_nonnull.schema,
                )
            writer_nonnull.write_table(table_nonnull)

        # close writers and copy off drive
        if writer_rows:
            writer_rows.close()
            FPATH.alternative_copy_to_opposite_drive(
                FPATH.DATA / output_folder / f"{key}_row_counts.parquet"
            )
        if writer_nonnull:
            writer_nonnull.close()
            FPATH.alternative_copy_to_opposite_drive(
                FPATH.DATA / output_folder / f"{key}_non_null_counts.parquet"
            )
        print(f"Finished {key}")


def aggregate_by_minor_group_into_single_file(
    input_folder: Path,
    configs: Dict[str, Dict[str, str]],
    person_ids: List[int],
    output_path: Path,
    chunk_size: int = 200_000,
) -> None:
    """
    Chunked aggregation of per-key row_counts & non_null_counts into one Parquet,
    tagging each record with both minor and overall group names.

    Args:
        input_folder (Path): directory containing `{key}_row_counts.parquet`
            and `{key}_non_null_counts.parquet` for each key in configs.
        configs (Dict[str, Dict[str, str]]): mapping key →
            {"desc_name": str, "overall_group_name": str}.
        person_ids (List[int]): list of all person_ids to include.
        output_path (Path): where to write the final aggregated Parquet.
        chunk_size (int): max person_ids per chunk.
    """
    # prepare dataset sources & parallel group lists
    sources: List[ds.Dataset] = []
    minor_groups: List[str] = []
    major_groups: List[str] = []
    for key, cfg in configs.items():
        rows_ds = ds.dataset(
            input_folder / f"{key}_row_counts.parquet", format="parquet"
        )
        nonnull_ds = ds.dataset(
            input_folder / f"{key}_non_null_counts.parquet", format="parquet"
        )
        sources.extend([rows_ds, nonnull_ds])
        minor_groups.extend([cfg["desc_name"], cfg["desc_name"]])
        major_groups.extend([cfg["overall_group_name"], cfg["overall_group_name"]])

    writer: pq.ParquetWriter | None = None

    # iterate chunks (each yield is a list of pl.DataFrames)
    for chunk_dfs in yield_chunks_given_pnrs(sources, person_ids, chunk_size):
        chunk_tables: List[pa.Table] = []

        # process each pair (rows_df, nonnull_df)
        for idx in range(0, len(chunk_dfs), 2):
            df_rows, df_nonnull = chunk_dfs[idx], chunk_dfs[idx + 1]
            # join in Polars
            joined_df = df_rows.join(df_nonnull, on=["person_id", "year"], how="left")
            # to Arrow
            tbl = joined_df.to_arrow()

            # add minor_group and overall_group_name columns
            mg = minor_groups[idx]
            maj = major_groups[idx]
            mg_col = pa.array([mg] * tbl.num_rows, pa.string())
            maj_col = pa.array([maj] * tbl.num_rows, pa.string())
            tbl = tbl.append_column("minor_group", mg_col)
            tbl = tbl.append_column("major_group", maj_col)

            chunk_tables.append(tbl)

        if not chunk_tables:
            continue

        # concat all keys for this chunk
        chunk_all = pa.concat_tables(chunk_tables, promote=True)

        # aggregate by person_id, year, minor_group, overall_group_name
        grouped = chunk_all.group_by(
            keys=["person_id", "year", "major_group", "minor_group"]
        ).aggregate(
            [
                ("row_count", "sum"),
                ("non_null_count", "sum"),
            ]
        )

        # write (or append) to output
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema=grouped.schema)
        writer.write_table(grouped)

    if writer:
        writer.close()


def aggregate_by_cols(
    cols,
    input_path: Path,
    person_ids: List[int],
    output_path: Path,
    chunk_size: int = 200_000,
) -> None:
    """
    Stream‐load the overall‐grouped Parquet in chunks via `yield_chunks_given_pnrs`,
    aggregate each chunk by person_id, and append to one Parquet file.

    Args:
        cols (union[list(str), str]): Columns to aggregate by
        input_path (Path): Path to the `agg_by_group.parquet` file.
        person_ids (List[int]): List of all person_ids to include.
        output_path (Path): File path for the person_id aggregated Parquet.
        chunk_size (int): Number of person_ids per chunk.
    """
    ds_src = ds.dataset(input_path, format="parquet")
    writer = None

    # iterate over person_id‐filtered chunks
    for df in yield_chunks_given_pnrs(ds_src, person_ids, chunk_size):
        # aggregate this chunk by year
        yearly: pl.DataFrame = df.group_by(cols).agg(
            [
                pl.sum("row_count_sum").alias("row_count_sum"),
                pl.sum("non_null_count_sum").alias("non_null_count_sum"),
            ]
        )

        # convert to Arrow and write/appended to Parquet
        table = yearly.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema=table.schema)
        writer.write_table(table)

    if writer:
        writer.close()
    output_path.parent.mkdir(exist_ok=True, parents=True)
    FPATH.alternative_copy_to_opposite_drive(output_path)


def create_barplot_total_counts_df(
    df: pl.DataFrame,
    group_col: str,
    count_field: str,
    ax: Axes,
    color_mapping: Optional[Dict[str, str]] = None,
    label_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """
    Create a barplot of counts (total rows or non-null values) for each group sorted by count.

    Args:
        df (pl.DataFrame):
            DataFrame containing grouping and counts.
        group_col (str):
            Column name for group labels.
        count_field (str):
            Column name for count values.
        ax (Axes):
            Matplotlib Axes for plotting.
        color_mapping (Optional[Dict[str, str]]):
            Mapping of group names to colors.
        label_mapping (Optional[Dict[str, str]]):
            Mapping of group names to display names.
    """
    # aggregate if necessary
    agg = (
        df.group_by(group_col)
        .agg(pl.col(count_field).sum().alias(count_field))
        .sort(count_field)
    )

    groups = agg.get_column(group_col).cast(str).to_list()
    counts = agg.get_column(count_field).to_list()

    # Display labels
    display_labels = (
        [label_mapping.get(g, g) for g in groups] if label_mapping else groups
    )

    # Colors
    if color_mapping:
        bar_colors = [color_mapping.get(g, "gray") for g in groups]
    else:
        bar_colors = ["gray"] * len(groups)

    # Plot
    x = np.arange(len(groups))
    ax.bar(groups, counts, color=bar_colors, align="center", edgecolor="black")
    ax.set_xlim(-0.75, len(groups) - 0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, ha="right", rotation=45, rotation_mode="anchor")
    ax.set_yscale("log")

    # Nudge labels
    dx = 5 / 72.0
    offset = mtransforms.ScaledTranslation(dx, 0, ax.figure.dpi_scale_trans)
    for lbl in ax.xaxis.get_majorticklabels():
        lbl.set_transform(lbl.get_transform() + offset)
