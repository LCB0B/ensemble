from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


def compute_at_risk_min_max(df: pl.DataFrame) -> Tuple[float, float]:
    """
    Compute the mean for each column containing 'at_risk' in its name and return the min and max of these means.

    Args:
        df (pl.DataFrame): Input DataFrame with numerical columns containing 'at_risk' in their names.

    Returns:
        Tuple[float, float]: A tuple with the minimum mean and maximum mean.
    """
    means: List[float] = []
    for col in df.columns:
        if "at_risk" in col:
            if "at_risk_duration" not in col:
                means.append(df[col].mean())
    return min(means), max(means)


def plot_at_risk_heatmap_on_ax(
    df: pl.DataFrame,
    max_years: int,
    thresholds: List[int],
    ax: Axes,
    vmin: float,
    vmax: float,
    color_palette: Optional[str] = "Reds",
    fmt: Optional[str] = ".3f",
    convert_to_float_percent: bool = False,
) -> None:
    """
    Plot a heatmap of at-risk rates from indicator columns with thresholds on a given axis.

    Args:
        df (pl.DataFrame): DataFrame with columns "at_risk_{n}_{threshold}_days".
        max_years (int): Maximum number of consecutive years to consider.
        thresholds (List[int]): List of day thresholds used in the column names.
        ax (Axes): Matplotlib Axes object to plot the heatmap on.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        color_palette (Optional[str]): color_palette for the plot (default "Reds")
        fmt (Optional[str]): Formatting of numbers in heatmap.
        convert_to_float_percent (bool): Multiply by a 100 for percent as float.
    """
    heatmap_data = {}
    for n in range(1, max_years + 1):
        row = {}
        for threshold in thresholds:
            col_name = f"at_risk_{n}_{threshold}_days"
            row[threshold] = df[col_name].mean()
        heatmap_data[n] = row

    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    heatmap_df.index.name = "Consecutive years"
    heatmap_df.columns.name = "Threshold (days)"

    if convert_to_float_percent:
        heatmap_df = heatmap_df.map(lambda x: x * 100)
        vmin = vmin * 100
        vmax = vmax * 100

    cmap = sns.color_palette(color_palette, as_cmap=True)
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )


def plot_at_risk_heatmap_single_on_ax(
    df: pl.DataFrame,
    max_years: int,
    ax: Axes,
    vmin: float,
    vmax: float,
    color_palette: Optional[str] = "Reds",
    fmt: Optional[str] = ".3f",
    convert_to_float_percent: bool = False,
) -> None:
    """
    Plot a heatmap of at-risk rates from indicator columns without thresholds on a given axis.

    Args:
        df (pl.DataFrame): DataFrame with columns "at_risk_{n}_years".
        max_years (int): Maximum number of consecutive years to consider.
        ax (Axes): Matplotlib Axes object to plot the heatmap on.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        color_palette (Optional[str]): color_palette for the plot (default "Reds")
        fmt (Optional[str]): Formatting of numbers in heatmap.
        convert_to_float_percent (bool): Multiply by a 100 for percent as float.
    """
    heatmap_data = {}
    for n in range(1, max_years + 1):
        col_name = f"at_risk_{n}_years"
        heatmap_data[n] = {"Cross\nsection": df[col_name].mean()}

    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    heatmap_df.index.name = "Consecutive years"
    heatmap_df.columns.name = " "

    if convert_to_float_percent:
        heatmap_df = heatmap_df.map(lambda x: x * 100)
        vmin = vmin * 100
        vmax = vmax * 100

    cmap = sns.color_palette(color_palette, as_cmap=True)
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.xaxis.set_ticks_position("none")


def compute_indicator_heatmap_min_max(
    ever_df: pl.DataFrame,
    indicator_df: pl.DataFrame,
    indicator_col: str,
    max_years: int,
    thresholds: Optional[List[int]] = None,
) -> Tuple[float, float]:
    """
    Merge ever at-risk indicators with indicator data, compute indicator rates for each group,
    and return the minimum and maximum rate among all groups.

    Args:
        ever_df (pl.DataFrame): DataFrame with one row per person containing 'person_id' and ever at-risk indicator columns.
        indicator_df (pl.DataFrame): DataFrame with 'person_id' and a binary indicator column.
        indicator_col (str): Name of the indicator column in indicator_df.
        max_years (int): Maximum number of consecutive years considered in the indicators.
        thresholds (List[int], optional): List of at-risk duration thresholds (in days) used in the column names.
            If None, uses columns formatted as "ever_at_risk_{n}_years".

    Returns:
        Tuple[float, float]: A tuple with the minimum rate and maximum rate computed.
    """
    merged_df = ever_df.join(indicator_df, on="person_id", how="left")
    merged_df = merged_df.with_columns(pl.col(indicator_col).fill_null(0))
    rates: List[float] = []
    if thresholds is None:
        for n in range(1, max_years + 1):
            col_name = f"ever_at_risk_{n}_years"
            filtered = merged_df.filter(pl.col(col_name) == 1)
            if filtered.height > 0:
                rate = filtered[indicator_col].mean()
                rates.append(rate)
    else:
        for n in range(1, max_years + 1):
            for threshold in thresholds:
                col_name = f"ever_at_risk_{n}_{threshold}_days"
                filtered = merged_df.filter(pl.col(col_name) == 1)
                if filtered.height > 0:
                    rate = filtered[indicator_col].mean()
                    rates.append(rate)
    return min(rates), max(rates)


def plot_indicator_heatmap_single_on_ax(
    ever_df: pl.DataFrame,
    indicator_df: pl.DataFrame,
    indicator_col: str,
    max_years: int,
    ax: Axes,
    vmin: float,
    vmax: float,
    color_palette: Optional[str] = "Reds",
    fmt: Optional[str] = ".3f",
    convert_to_float_percent: bool = False,
) -> None:
    """
    Merge ever at-risk indicators with indicator data and plot a heatmap of indicator rates.
    This version does not use thresholds and creates a single-column heatmap.

    Args:
        ever_df (pl.DataFrame): DataFrame with one row per person containing 'person_id' and
            ever at-risk indicator columns formatted as "ever_at_risk_{n}_years".
        indicator_df (pl.DataFrame): DataFrame with 'person_id' and a binary indicator specified by indicator_col.
        indicator_col (str): Name of the column in indicator_df indicating the indicator status.
        max_years (int): Maximum number of consecutive years considered in the indicators.
        ax (Axes): Matplotlib Axes object to plot the heatmap on.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        color_palette (Optional[str]): color_palette for the plot (default "Reds")
        fmt (Optional[str]): Formatting of numbers in heatmap.
        convert_to_float_percent (bool): Multiply by a 100 for percent as float.
    """
    merged_df = ever_df.join(indicator_df, on="person_id", how="left")
    merged_df = merged_df.with_columns(pl.col(indicator_col).fill_null(0))

    heatmap_data = {}
    for n in range(1, max_years + 1):
        col_name = f"ever_at_risk_{n}_years"
        filtered = merged_df.filter(pl.col(col_name) == 1)
        rate = filtered[indicator_col].mean() if filtered.height > 0 else None
        heatmap_data[n] = {"Cross\nsection": rate}

    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    heatmap_df.index.name = "Consecutive years"
    heatmap_df.columns.name = " "

    if convert_to_float_percent:
        heatmap_df = heatmap_df.map(lambda x: x * 100)
        vmin = vmin * 100
        vmax = vmax * 100

    cmap = sns.color_palette(color_palette, as_cmap=True)
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )


def plot_indicator_heatmap_on_ax(
    ever_df: pl.DataFrame,
    indicator_df: pl.DataFrame,
    indicator_col: str,
    max_years: int,
    thresholds: List[int],
    ax: Axes,
    vmin: float,
    vmax: float,
    color_palette: Optional[str] = "Reds",
    fmt: Optional[str] = ".3f",
    convert_to_float_percent: bool = False,
) -> None:
    """
    Merge ever at-risk indicators with indicator data and plot a heatmap of indicator rates.
    Persons not present in indicator_df are assumed not to have the indicator.

    Args:
        ever_df (pl.DataFrame): DataFrame with one row per person containing 'person_id' and
            ever at-risk indicator columns formatted as "ever_at_risk_{n}_{threshold}_days".
        indicator_df (pl.DataFrame): DataFrame with 'person_id' and a binary indicator specified by indicator_col.
        indicator_col (str): Name of the column in indicator_df indicating the indicator status.
        max_years (int): Maximum number of consecutive years considered in the indicators.
        thresholds (List[int]): List of at-risk duration thresholds (in days) used in the indicator column names.
        ax (Axes): Matplotlib Axes object to plot the heatmap on.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        color_palette (Optional[str]): color_palette for the plot (default "Reds")
        fmt (Optional[str]): Formatting of numbers in heatmap.
        convert_to_float_percent (bool): Multiply by a 100 for percent as float.
    """
    merged_df = ever_df.join(indicator_df, on="person_id", how="left")
    merged_df = merged_df.with_columns(pl.col(indicator_col).fill_null(0))

    heatmap_data = {}
    for n in range(1, max_years + 1):
        row = {}
        for threshold in thresholds:
            col_name = f"ever_at_risk_{n}_{threshold}_days"
            filtered = merged_df.filter(pl.col(col_name) == 1)
            rate = filtered[indicator_col].mean() if filtered.height > 0 else None
            row[threshold] = rate
        heatmap_data[n] = row

    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    heatmap_df.index.name = "Consecutive years"
    heatmap_df.columns.name = "Threshold (days)"

    if convert_to_float_percent:
        heatmap_df = heatmap_df.map(lambda x: x * 100)
        vmin = vmin * 100
        vmax = vmax * 100

    cmap = sns.color_palette(color_palette, as_cmap=True)
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )


def compute_value_heatmap_min_max(
    ever_df: pl.DataFrame,
    value_df: pl.DataFrame,
    value_col: str,
    max_years: int,
    thresholds: Optional[List[int]] = None,
) -> Tuple[float, float]:
    """
    Merge ever at-risk indicators with a value dataframe, compute mean values for each cell,
    and return the minimum and maximum mean value.

    Args:
        ever_df (pl.DataFrame): DataFrame with 'person_id' and ever at-risk indicator columns.
        value_df (pl.DataFrame): DataFrame with 'person_id' and a numerical column specified by value_col.
        value_col (str): (str) Name of the column in value_df containing values.
        max_years (int): (int) Maximum number of consecutive years considered.
        thresholds (List[int], optional): (List[int]) List of at-risk duration thresholds. If None, uses columns
            formatted as "ever_at_risk_{n}_years".

    Returns:
        Tuple[float, float]: A tuple with the minimum and maximum mean values.
    """
    merged_df = ever_df.join(value_df, on="person_id", how="inner")
    means: List[float] = []
    if thresholds is None:
        for n in range(1, max_years + 1):
            col_name = f"ever_at_risk_{n}_years"
            filtered = merged_df.filter(pl.col(col_name) == 1)
            if filtered.height > 0:
                means.append(filtered[value_col].mean())
    else:
        for n in range(1, max_years + 1):
            for threshold in thresholds:
                col_name = f"ever_at_risk_{n}_{threshold}_days"
                filtered = merged_df.filter(pl.col(col_name) == 1)
                if filtered.height > 0:
                    means.append(filtered[value_col].mean())

    return min(means), max(means)


def plot_value_heatmap_on_ax(
    ever_df: pl.DataFrame,
    value_df: pl.DataFrame,
    value_col: str,
    max_years: int,
    thresholds: List[int],
    ax: Axes,
    vmin: float,
    vmax: float,
    title: Optional[str] = None,
    color_palette: Optional[str] = "Greens",
    fmt: Optional[str] = ".0f",
    convert_to_float_percent: bool = False,
) -> None:
    """
    Merge ever at-risk indicators with a value dataframe and plot a heatmap of mean values
    (using thresholds) on the provided axis.

    Args:
        ever_df (pl.DataFrame): DataFrame with 'person_id' and ever at-risk indicator columns
            formatted as "ever_at_risk_{n}_{threshold}_days".
        value_df (pl.DataFrame): DataFrame with 'person_id' and a numerical column specified by value_col.
        value_col (str): Name of the column in value_df containing values.
        max_years (int): Maximum number of consecutive years considered.
        thresholds (List[int]):  List of at-risk duration thresholds (in days).
        ax (Axes): Matplotlib Axes object to plot the heatmap on.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        title (Optional[str]): Title for the heatmap.
        color_palette (Optional[str]): color_palette for the plot (default "Greens")
        fmt (Optional[str]): Formatting of numbers in heatmap.
        convert_to_float_percent (bool): Multiply by a 100 for percent as float.
    """
    merged_df = ever_df.join(value_df, on="person_id", how="inner")
    heatmap_data = {}
    for n in range(1, max_years + 1):
        row = {}
        for threshold in thresholds:
            col_name = f"ever_at_risk_{n}_{threshold}_days"
            filtered = merged_df.filter(pl.col(col_name) == 1)
            mean_val = filtered[value_col].mean() if filtered.height > 0 else None
            row[threshold] = mean_val
        heatmap_data[n] = row

    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    heatmap_df.index.name = "Consecutive years"
    heatmap_df.columns.name = "Threshold (days)"

    if convert_to_float_percent:
        heatmap_df = heatmap_df.map(lambda x: x * 100)
        vmin = vmin * 100
        vmax = vmax * 100

    cmap = sns.color_palette(color_palette, as_cmap=True)
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title if title is not None else "")
    ax.set_xlabel("Threshold (days)")
    ax.set_ylabel("Consecutive years")


def plot_value_heatmap_single_on_ax(
    ever_df: pl.DataFrame,
    value_df: pl.DataFrame,
    value_col: str,
    max_years: int,
    ax: Axes,
    vmin: float,
    vmax: float,
    title: Optional[str] = None,
    color_palette: Optional[str] = "Greens",
    fmt: Optional[str] = ".0f",
    convert_to_float_percent: bool = False,
) -> None:
    """
    Merge ever at-risk indicators with a value dataframe and plot a single-column heatmap
    of mean values on the provided axis (without thresholds).

    Args:
        ever_df (pl.DataFrame): DataFrame with 'person_id' and ever at-risk indicator columns
            formatted as "ever_at_risk_{n}_years".
        value_df (pl.DataFrame): DataFrame with 'person_id' and a numerical column specified by value_col.
        value_col (str): (str) Name of the column in value_df containing values.
        max_years (int): (int) Maximum number of consecutive years considered.
        ax (Axes): Matplotlib Axes object to plot the heatmap on.
        vmin (float): (float) Minimum value for the color scale.
        vmax (float): (float) Maximum value for the color scale.
        title (Optional[str]): (Optional[str]) Title for the heatmap.
        color_palette (Optional[str]): color_palette for the plot (default "Greens")
        fmt (Optional[str]): Formatting of numbers in heatmap.
        convert_to_float_percent (bool): Multiply by a 100 for percent as float.
    """
    merged_df = ever_df.join(value_df, on="person_id", how="inner")
    heatmap_data = {}
    for n in range(1, max_years + 1):
        col_name = f"ever_at_risk_{n}_years"
        filtered = merged_df.filter(pl.col(col_name) == 1)
        mean_val = filtered[value_col].mean() if filtered.height > 0 else None
        heatmap_data[n] = {"Cross\nsection": mean_val}

    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    heatmap_df.index.name = "Consecutive years"
    heatmap_df.columns.name = " "

    if convert_to_float_percent:
        heatmap_df = heatmap_df.map(lambda x: x * 100)
        vmin = vmin * 100
        vmax = vmax * 100

    cmap = sns.color_palette(color_palette, as_cmap=True)
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title if title is not None else "")
    ax.set_xlabel(" ")
    ax.set_ylabel("Consecutive years")


def plot_mean_and_count_heatmaps(df: pl.DataFrame, axes: Sequence[plt.Axes]) -> None:
    """
    Plot heatmaps of mean target (as percent) and subgroup counts (in thousands)
    for 'female', 'native', and 'mom_long_edu', with square cells of equal size
    and reserved space for colorbars on every subplot.

    Args:
        df (pl.DataFrame): Polars DataFrame with columns:
            - 'target' (numeric)
            - 'female' (0/1)
            - 'native' (0/1)
            - 'mom_long_edu' (0/1)
        axes (Sequence[plt.Axes]): Sequence of 4 matplotlib Axes, in order:
            [mean_edu0, mean_edu1, count_edu0, count_edu1]
    """
    # aggregate mean and count
    summary = (
        df.group_by(["female", "native", "mom_long_edu"])
        .agg([pl.col("target").mean().alias("mean_target"), pl.len().alias("count")])
        .to_pandas()
    )

    # compute global scales
    mean_vals = summary["mean_target"] * 100
    mean_vmin, mean_vmax = mean_vals.min(), mean_vals.max()
    count_vals = summary["count"] / 1_000.0
    count_vmin, count_vmax = count_vals.min(), count_vals.max()

    specs = [
        ("mean_target", mean_vmin, mean_vmax, "Mean target (%)"),
        ("mean_target", mean_vmin, mean_vmax, "Mean target (%)"),
        ("count", count_vmin, count_vmax, "Count (thousands)"),
        ("count", count_vmin, count_vmax, "Count (thousands)"),
    ]

    for idx, (metric, vmin, vmax, title) in enumerate(specs):
        ax = axes[idx]
        # reserve cbar space
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        edu = idx % 2
        sub = summary[summary["mom_long_edu"] == edu]
        pivot = sub.pivot(index="female", columns="native", values=metric)

        if metric == "mean_target":
            data = pivot * 100
            annot = data.map(lambda v: f"{v:.1f}%")
            cbar_kwargs = {"label": "Vulnerability rate (%)"}
        else:
            data = pivot / 1_000.0
            annot = data.map(lambda v: f"{v:.1f}")
            cbar_kwargs = {"label": "Count (thousands)"}

        sns.heatmap(
            data,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            cbar_ax=cax,
            cbar_kws=cbar_kwargs,
            square=True,
        )
        # hide cbar for first and third subplot
        if idx in (0, 2):
            cax.set_axis_off()
            cax.remove()

        if idx in (1, 3):
            ax.tick_params(left=False, labelleft=False)

        ax.set_aspect("equal")
        if edu == 1:
            ax.set_title("Mom: Long education")
        if edu == 0:
            ax.set_title("Mom: No long education")
        ax.set_xlabel("Native")
        ax.set_ylabel("Female" if edu == 0 else "")


def compute_split_statistics(
    outcomes: pl.DataFrame,
    train_pids: pl.DataFrame,
    val_pids: pl.DataFrame,
    test_pids: pl.DataFrame,
    dev_year_cutoff: int = 2016,
    test_year: int = 2018,
) -> pl.DataFrame:
    """
    Compute summary statistics for train, validation and development splits.

    Args:
        outcomes (pl.DataFrame): DataFrame with columns ['person_id', 'year', 'target', 'censor', 'birthday', 'age', 'female', 'native', 'mom_long_edu'].
        train_pids (pl.DataFrame): DataFrame with column 'person_id' for the training split.
        val_pids (pl.DataFrame): DataFrame with column 'person_id' for the validation split.
        dev_year_cutoff (int): Maximum year (inclusive) to include in development split. Defaults to 2016.

    Returns:
        pl.DataFrame: Summary table with one row per split and columns:
            - split (str)
            - n_obs (int)
            - n_unique_persons (int)
            - mean_target (float)
            - mean_native (float)
            - mean_female (float)
            - mean_mom_long_edu (float)
    """

    def stats_for(df: pl.DataFrame, name: str) -> dict:
        n_obs = df.height
        n_unique = df["person_id"].n_unique()
        return {
            "Data split": name,
            "# observations": n_obs,
            "# unique people": n_unique,
            "Vulnerability rate": df["target"].mean(),
            "Native (%)": df["native"].mean(),
            "Female (%)": df["female"].mean(),
            "Mom, long education (%)": df["mom_long_edu"].mean(),
        }

    # filter splits
    train_df = outcomes.filter(
        (pl.col("year") <= dev_year_cutoff)
        & (pl.col("person_id").is_in(train_pids["person_id"]))
    )
    val_df = outcomes.filter(
        (pl.col("year") <= dev_year_cutoff)
        & (pl.col("person_id").is_in(val_pids["person_id"]))
    )
    test_df = outcomes.filter(
        (pl.col("year") == test_year)
        & (pl.col("person_id").is_in(test_pids["person_id"]))
    )

    # gather stats
    rows = [
        stats_for(train_df, "Train"),
        stats_for(val_df, "Validation"),
        stats_for(test_df, "Test"),
    ]
    return pl.DataFrame(rows)
