# %%
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import seaborn as sns


def print_summary(lifelines: pl.DataFrame) -> None:
    """
    Print summary statistics from the lifelines DataFrame.

    Args:
        lifelines (pl.DataFrame): The lifelines DataFrame.
    """
    n_unique = lifelines["person_id"].n_unique()
    n_people_1_event = lifelines.filter(pl.col("number_events_person") == 1)[
        "person_id"
    ].n_unique()
    n_immigrants_1_event = lifelines.filter(
        (pl.col("number_events_person") == 1)
        & (pl.col("event_cause_start") == "Indvandret")
    )["person_id"].n_unique()

    print(f"Unique people: {n_unique:,.0f}")
    print(f"Number of people with 1 event: {n_people_1_event:,.0f}")
    print(f"Number of immigrants with one entrance: {n_immigrants_1_event:,.0f}")


def preprocess_lifelines(lifelines: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocess the lifelines DataFrame by casting dates, sorting, and computing re-entry gaps.

    Args:
        lifelines (pl.DataFrame): The raw lifelines DataFrame.

    Returns:
        pl.DataFrame: The processed DataFrame with added columns.
    """
    lifelines = lifelines.with_columns(
        [
            pl.col("event_start_date").cast(pl.Date),
            pl.col("event_final_date").cast(pl.Date),
        ]
    ).sort(["person_id", "event_start_date"])

    lifelines = lifelines.with_columns(
        pl.col("event_start_date").shift(-1).over("person_id").alias("next_entry_date")
    )
    lifelines = lifelines.with_columns(
        (pl.col("next_entry_date") - pl.col("event_final_date")).alias("gap_days"),
        (pl.col("event_final_date") - pl.col("birthday")).alias("age_final"),
        (pl.col("event_start_date") - pl.col("birthday")).alias("age_start"),
    )
    lifelines = lifelines
    return lifelines


def plot_gap_hist(
    ax: plt.Axes, gap_data: List[float], xlim: float, bins_per_year: int
) -> None:
    """
    Plot a histogram of gap durations on the provided axis.

    Args:
        ax (plt.Axes): Axis to plot the histogram on.
        gap_data (List[float]): List of gap durations in years.
        xlim (float): Upper limit for the x-axis.
        bins_per_year (int): Number of bins per year.
    """
    plot_data = [x for x in gap_data if x < xlim]
    bins_count = int((max(plot_data) - min(plot_data)) * bins_per_year)
    sns.histplot(
        plot_data,
        bins=bins_count,
        binwidth=1 / bins_per_year,
        ax=ax,
        edgecolor=None,
        kde=True,
        color="C0",
    )
    # Ensure each bin has at least 5 observations.
    heights = [patch.get_height() for patch in ax.patches]
    assert all(h >= 5 for h in heights), "Some bins have less than 5 observations"
    ax.set_xlim(0, xlim)
    ax.set_xticks(range(0, int(xlim) + 1, 5))
    ax.set_xticks(range(1, int(xlim) + 1), minor=True)
    ax.set_yticks(range(0, int(xlim), 5), minor=True)
    ax.grid(True, which="minor", axis="x", linestyle="--", alpha=0.8)
    ax.grid(True, which="major", axis="x", linestyle="--", alpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:,.0f}"))
    ax.set_xlabel("Gap length (years)")
    ax.set_ylabel("Count (thousands)")


def plot_age_hist_overlay(
    ax: plt.Axes,
    age_data_reentry: List[float],
    age_data_emmig: List[float],
    xlim: float,
    bins_per_year: int,
) -> None:
    """
    Plot an overlaid histogram of age at time of exit for reentry gaps and emmigration data.

    Args:
        ax (plt.Axes): Axis to plot the histogram on.
        age_data_reentry (List[float]): List of ages at exit (years) from reentry gaps.
        age_data_emmig (List[float]): List of ages at exit (years) from emmigration.
        xlim (float): Upper limit for the x-axis.
        bins_per_year (int): Number of bins per year.
    """
    reentry_data = [x for x in age_data_reentry if x < xlim]
    emmig_data = [x for x in age_data_emmig if x < xlim]

    binwidth = 1 / bins_per_year
    lower = min(min(reentry_data), min(emmig_data))
    upper = max(max(reentry_data), max(emmig_data))
    bins = np.arange(lower, upper + binwidth, binwidth)

    # Plot reentry gaps histogram.
    sns.histplot(
        reentry_data,
        bins=bins,
        kde=True,
        ax=ax,
        color="C0",
        label="Reentrants",
        edgecolor=None,
        linewidth=0,
    )
    # Overlay emmigration histogram.
    sns.histplot(
        emmig_data,
        bins=bins,
        kde=True,
        ax=ax,
        color="C1",
        label="All emmigrants",
        edgecolor=None,
        linewidth=0,
    )

    ax.set_xlim(0, xlim)
    ax.set_xticks(range(0, int(xlim) + 1, 10))
    ax.set_xticks(range(0, int(xlim) + 1, 5), minor=True)
    ax.set_yticks(range(0, int(xlim), 5), minor=True)
    ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:,.0f}"))
    ax.set_xlabel("Age at emmigration (years)")
    ax.set_ylabel("Count (thousands)")
    ax.legend()


def plot_hexbin(ax: plt.Axes, df: pl.DataFrame, xlim: int = 70) -> None:
    """
    Plot a hexbin of age_final (years) versus gap (years) on the provided axis.

    Args:
        ax (plt.Axes): Axis to plot the hexbin on.
        df (pl.DataFrame): Pandas-converted DataFrame with 'age_final_years' and 'gap_years' columns.
        xlim (int): Limit for x axis.
    """
    hb = ax.hexbin(
        df["age_final_years"], df["gap_years"], gridsize=30, cmap="Blues", mincnt=5
    )
    hex_data = hb.get_array()
    assert np.all(
        (hex_data >= 5) | (hex_data == 0)
    ), "Some hex bins have counts between 1 and 4!"
    ax.set_xlabel("Age at emmigration (years)")
    ax.set_xticks(range(0, xlim + 1, 10), minor=False)
    ax.set_xticks(range(0, xlim, 5), minor=True)
    ax.set_ylabel("Gap length (years)")
    cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="Count (thousands)")
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:,.0f}")
    )
