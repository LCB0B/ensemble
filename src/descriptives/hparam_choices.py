import re
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import calibration_curve


def create_color_map(subgroups: List[str], ax: plt.Axes) -> Dict[str, str]:
    """
    Create a color map for each unique subgroup based on the current Matplotlib theme.

    Args:
        subgroups (List[str]): List of subgroup names.
        ax (plt.Axes): Axis to get the current color cycle from.

    Returns:
        Dict[str, str]: Mapping of subgroup names to colors.
    """
    unique_subgroups = sorted(set(subgroups))
    color_map = {
        subgroup: ax._get_lines.get_next_color() for subgroup in unique_subgroups
    }
    return color_map


def plot_model_subgroup_metrics(
    result_df: pl.DataFrame,
    prediction_cols: List[str],
    ax: plt.Axes = None,
    col_name_map: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    y_label: str = "Metric Value",
    show_legend: bool = True,
):
    df = result_df.filter(pl.col("col").is_in(prediction_cols)).to_pandas()
    pivot = df.pivot(
        index=["subgroup", "col"], columns="value_type", values="value"
    ).reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if color_map is None:
        color_map = create_color_map(pivot["subgroup"].unique(), ax)
    else:
        for subgroup in pivot["subgroup"].unique():
            assert subgroup in color_map, f"Missing color for {subgroup}"

    for i, col in enumerate(prediction_cols):
        df_col = pivot[pivot["col"] == col]
        for _, row in df_col.iterrows():
            err_low = row["metric"] - row["lb"]
            err_high = row["ub"] - row["metric"]
            ax.errorbar(
                x=i,
                y=row["metric"],
                yerr=[[err_low], [err_high]],
                fmt="o",
                capsize=4,
                color=color_map[row["subgroup"]],
                label=row["subgroup"] if i == 0 and show_legend else None,
            )

    xticklabels = [
        col_name_map.get(c, c) if col_name_map else c for c in prediction_cols
    ]
    ax.set_xticks(np.arange(len(prediction_cols)))
    ax.set_xticklabels(
        xticklabels, rotation=67.5, ha="right", rotation_mode="anchor", va="top"
    )
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.5, len(prediction_cols) - 0.5)

    if show_legend:
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color, markersize=8
            )
            for subgroup, color in color_map.items()
        ]
        ax.legend(
            handles,
            list(color_map.keys()),
            title="Subgroups",
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
        )


def plot_metrics_pointplot(
    results_df: pl.DataFrame,
    ax: plt.Axes = None,
    col_name_map: Optional[Dict[str, str]] = None,
    prediction_cols: Optional[List[str]] = None,
):
    """
    Plots a point plot of metrics with confidence intervals in a consistent model order.

    Args:
        results_df (pl.DataFrame): Polars DataFrame with 'col', 'metric', 'lb', and 'ub' columns.
        ax (plt.Axes, optional): Matplotlib axis to plot on.
        col_name_map (Optional[Dict[str, str]]): Optional mapping for display names.
        prediction_cols (Optional[List[str]]): List of model columns to order.
    """
    df = results_df.to_pandas()

    if prediction_cols is None:
        prediction_cols = df["col"].unique().tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for i, col in enumerate(prediction_cols):
        row = df[df["col"] == col]
        if row.empty:
            continue
        metric = row["metric"].values[0]
        lb = row["lb"].values[0]
        ub = row["ub"].values[0]
        ax.errorbar(
            i,
            metric,
            yerr=[[metric - lb], [ub - metric]],
            fmt="o",
            capsize=5,
            color="black",
        )

    xticklabels = [
        col_name_map.get(c, c) if col_name_map else c for c in prediction_cols
    ]
    ax.set_xticks(np.arange(len(prediction_cols)))
    ax.set_xticklabels(
        xticklabels, rotation=67.5, ha="right", rotation_mode="anchor", va="top"
    )


def plot_comparison_subplots(
    ax: List[plt.Axes],
    level_df: pl.DataFrame,
    gap_df: pl.DataFrame,
    subgroup_df: pl.DataFrame,
    y_label: str,
    model_mapping: Dict[str, str],
    pred_cols: List[str],
    color_map: Dict[str, str],
) -> None:
    """
    Plots a comparison of metrics across level, gap, and subgroup levels.

    Args:
        ax (List[plt.Axes]): List of three subplot axes for plotting.
        level_df (pl.DataFrame): DataFrame for the level plot.
        gap_df (pl.DataFrame): DataFrame for the gap plot.
        subgroup_df (pl.DataFrame): DataFrame for the subgroup plot.
        y_label (str): Label for the y-axis on the left plot.
        recall_subgroup_df (pl.DataFrame): DataFrame with recall subgroup metrics.
        precision_subgroup_df (pl.DataFrame): DataFrame with precision subgroup metrics.
        f1_subgroup_df (pl.DataFrame): DataFrame with F1 subgroup metrics.
        model_mapping (dict): Mapping of model names for column names.
        pred_cols (List[str]): List of prediction columns.
        color_map (Dict[str, str]): Color map for subgroups.

    """

    # Plot Level
    plot_metrics_pointplot(
        level_df, ax=ax[0], col_name_map=model_mapping, prediction_cols=pred_cols
    )
    ax[0].set_title("Level")
    ax[0].set_ylabel(y_label)

    # Plot Gap
    plot_metrics_pointplot(
        gap_df, ax=ax[1], col_name_map=model_mapping, prediction_cols=pred_cols
    )
    ax[1].set_title("Gap")

    # Plot Subgroup Levels
    plot_model_subgroup_metrics(
        subgroup_df,
        prediction_cols=pred_cols,
        ax=ax[2],
        col_name_map=model_mapping,
        color_map=color_map,
        y_label=None,
        show_legend=True,
    )
    ax[2].set_title("Subgroup levels")

    for _ax in ax:
        _ax.spines[["top", "right"]].set_visible(False)


def plot_level_barplot_grouped_by_outcome(
    level_df: pl.DataFrame,
    metric: str,
    pred_cols: List[str],
    model_mapping: Dict[str, str],
    outcome_order: Optional[List[str]] = None,
    metric_label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    models: Optional[pd.Categorical] = None,
):
    """
    Plots a grouped barplot with outcomes on x-axis and bars colored by model,
    including 95% confidence intervals from bootstrap.

    Args:
        level_df (pl.DataFrame): DataFrame with level metric results.
        metric (str): Metric name (e.g. "auc", "recall", etc.).
        pred_cols (List[str]): List of model prediction column names.
        model_mapping (Dict[str, str]): Mapping from prediction col to display name.
        outcome_order (Optional[List[str]]): Order of outcomes on x-axis.
        ax (Optional[plt.Axes]): Axis to plot on. Creates one if None.
    """
    df = (
        level_df.filter(pl.col("metric_type") == metric)
        .filter(pl.col("col").is_in(pred_cols))
        .to_pandas()
    )
    df["Model"] = df["col"].map(model_mapping)
    df["Outcome"] = df["outcome_type"]

    # Enforce consistent ordering
    df["Model"] = pd.Categorical(
        df["Model"], categories=[model_mapping[c] for c in pred_cols], ordered=True
    )
    if outcome_order:
        df["Outcome"] = pd.Categorical(
            df["Outcome"], categories=outcome_order, ordered=True
        )

    # Sort by outcome then model
    df = df.sort_values(["Outcome", "Model"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Bar positions
    outcomes = df["Outcome"].cat.categories
    if models is None:
        models = df["Model"].cat.categories
    bar_width = 0.18
    offsets = np.linspace(-bar_width * 1.5, bar_width * 1.5, len(models))
    x = np.arange(len(outcomes))

    for i, model in enumerate(models):
        df_model = df[df["Model"] == model]
        means = df_model["metric"].values
        lowers = df_model["lb"].values
        uppers = df_model["ub"].values
        yerr = [means - lowers, uppers - means]

        ax.bar(
            x + offsets[i],
            means,
            width=bar_width,
            label=model,
            yerr=yerr,
            capsize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [o.replace("_", " ").title() for o in outcomes], rotation=0, ha="center"
    )
    ax.set_ylabel(metric_label if metric_label else metric.upper())
    ax.set_xlabel("")
    ax.set_title("")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax, models


def plot_subgroup_levels_horizontal_longdf(
    subgroup_df: pl.DataFrame,
    pred_cols: List[str],
    model_mapping: Dict[str, str],
    ax: Optional[plt.Axes] = None,
    jitter: float = 0.15,
    models: Optional[pd.Categorical] = None,
):
    """
    Plots horizontal errorbar chart of subgroup metrics with y-axis jitter by model.

    Args:
        subgroup_df (pl.DataFrame): Long-format df with one outcome and metric_type.
        pred_cols (List[str]): List of prediction column names.
        model_mapping (Dict[str, str]): Mapping of model col -> display name.
        ax (Optional[plt.Axes]): Matplotlib axis. Creates one if None.
        jitter (float): Amount of vertical jitter between models in the same subgroup.
    """
    df = subgroup_df.filter(pl.col("col").is_in(pred_cols)).to_pandas()

    # Pivot to wide format
    df_wide = df.pivot(
        index=["subgroup", "col"], columns="value_type", values="value"
    ).reset_index()
    df_wide["Model"] = df_wide["col"].map(model_mapping)

    # Categorical ordering
    df_wide["Model"] = pd.Categorical(
        df_wide["Model"], categories=[model_mapping[c] for c in pred_cols], ordered=True
    )

    subgroups = sorted(df_wide["subgroup"].unique())
    subgroup_to_y = {sg: i for i, sg in enumerate(subgroups)}
    df_wide["y_base"] = df_wide["subgroup"].map(subgroup_to_y)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(4, len(subgroups) * 0.4)))

    if models is None:
        models = df_wide["Model"].cat.categories

    n_models = len(models)
    model_offsets = np.linspace(-jitter, jitter, n_models)

    for i, model in enumerate(models):
        df_model = df_wide[df_wide["Model"] == model]
        y_jittered = df_model["y_base"] + model_offsets[i]
        ax.errorbar(
            x=df_model["metric"],
            y=y_jittered,
            xerr=[
                df_model["metric"] - df_model["lb"],
                df_model["ub"] - df_model["metric"],
            ],
            fmt="o",
            capsize=4,
            label=model,
        )

    ax.set_yticks(range(len(subgroups)))
    ax.set_yticklabels(subgroups)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Subgroup")
    ax.set_title("Subgroup Performance by Model")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax


def plot_gap_by_sample_size(
    df: pl.DataFrame,
    model_mapping: dict,
    metric_labels: dict,
    outcome_labels: dict,
    figsize: tuple = (12, 8),
    save_path: str = None,
):
    """
    Plots average fairness gap (max - min subgroup metric) across sample sizes,
    grouped by outcome and metric, with CI ribbons and color by model label.

    Args:
        df (pl.DataFrame): Output from `bootstrap_subgroup_metrics_and_gaps()`.
        model_mapping (dict): Maps prediction column names to display labels (can be non-unique).
        metric_labels (dict): Maps metric_type values to display labels (e.g., {"recall": "Recall"}).
        outcome_labels (dict): Maps outcome_type values to display labels (e.g., {"at_risk": "At-risk"}).
        figsize (tuple): Size of the full plot grid.
        save_path (str): Optional path to save the figure.
    """
    # Filter and map labels
    df_gap = df.filter(pl.col("subgroup") == "GAP").to_pandas()
    df_gap["Model"] = df_gap["col"].map(model_mapping)

    # Aggregate per sample size, model, metric, outcome
    df_gap_agg = (
        df_gap.groupby(["outcome_type", "metric_type", "Model", "sample_percent"])
        .agg(
            score=("score", "mean"),
            lb=("score", lambda x: np.percentile(x, 2.5)),
            ub=("score", lambda x: np.percentile(x, 97.5)),
        )
        .reset_index()
    )

    # Apply label mappings
    df_gap_agg["metric_type"] = df_gap_agg["metric_type"].map(metric_labels)
    df_gap_agg["outcome_type"] = df_gap_agg["outcome_type"].map(outcome_labels)

    # Ensure model order
    model_order = list(dict.fromkeys(model_mapping.values()))
    df_gap_agg["Model"] = pd.Categorical(
        df_gap_agg["Model"], categories=model_order, ordered=True
    )

    # Plot
    g = sns.FacetGrid(
        df_gap_agg,
        row="outcome_type",
        col="metric_type",
        hue="Model",
        margin_titles=False,
        sharey=False,
        # palette="tab10",
    )

    for r, label in enumerate(g.row_names):
        g.axes[r][0].set_ylabel(f"{label} gap", fontsize=11)

    g.figure.set_size_inches(*figsize)

    # Draw CI ribbons
    def plot_ribbon(data, **kwargs):
        ax = plt.gca()
        for model in data["Model"].unique():
            d = data[data["Model"] == model]
            ax.fill_between(
                d["sample_percent"],
                d["lb"],
                d["ub"],
                alpha=0.2,
                label=None,
                color=kwargs["color"],
            )

    g.map_dataframe(plot_ribbon)
    g.map(sns.lineplot, "sample_percent", "score")

    # Titles
    # Clear default axis labels and titles
    for ax in g.axes.flat:
        ax.set_xlabel("Sample percent (%)")
        ax.set_ylabel("")
        ax.set_title("")

    # Set column titles (only on first row)
    for c, metric_name in enumerate(g.col_names):
        g.axes[0][c].set_title(metric_name, fontsize=12)

    # Set y-axis label (row titles) on the first column of each row
    for r, outcome_name in enumerate(g.row_names):
        g.axes[r][0].set_ylabel(f"{outcome_name} gap", fontsize=11)

    g.tight_layout()

    # Legend below
    g.figure.subplots_adjust(bottom=0.18)
    g.add_legend(
        title="",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(model_order),
        frameon=False,
    )

    plt.tight_layout()

    if save_path:
        g.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_calibration_overall_1row(
    df: pl.DataFrame,
    pred_cols: list[str],
    target_col: str,
    outcome_order: list[str],
    axs: list,
    calibrated_suffix: str = "_calibrated",
    n_bins: int = 10,
    model_mapping: dict = None,
):
    """
    Plots overall calibration curves into provided axes in a 1×4 layout:
        - Columns: At-risk (uncalibrated, calibrated), Econ-vuln (uncalibrated, calibrated)

    Args:
        df (pl.DataFrame): Polars DataFrame with predictions, target, and outcome_type.
        pred_cols (list[str]): List of base prediction columns.
        target_col (str): Name of the binary target column.
        outcome_order (list[str]): List of two outcome types (e.g., ["at_risk", "econ_vuln"]).
        axs (list): List of 4 matplotlib Axes objects (1×4).
        calibrated_suffix (str): Suffix for calibrated prediction columns.
        n_bins (int): Number of bins for calibration_curve.
        model_mapping (dict): Optional dict mapping prediction col name to display label.
    """
    df_pd = df.to_pandas()

    for i, outcome in enumerate(outcome_order):
        for j, calibrated in enumerate([False, True]):
            col_idx = i * 2 + j
            ax = axs[col_idx]

            pred_set = [
                col + calibrated_suffix if calibrated else col for col in pred_cols
            ]
            df_sub = df_pd[df_pd["outcome_type"] == outcome]

            for col in pred_set:
                if col not in df_sub.columns:
                    continue
                try:
                    prob_true, prob_pred = calibration_curve(
                        df_sub[target_col],
                        df_sub[col],
                        n_bins=n_bins,
                        strategy="uniform",
                    )
                    label = (
                        model_mapping[col.replace(calibrated_suffix, "")]
                        if model_mapping
                        else col
                    )
                    ax.plot(prob_pred, prob_true, marker="o", label=label)
                except ValueError:
                    continue

            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)

            if col_idx == 0:
                ax.set_ylabel("Fraction of positives")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("Predicted probabilities")

            cal = "Calibrated" if calibrated else "Uncalibrated"
            ax.set_title(f"{cal}", fontsize=11)


def plot_calibration_subgroup_grid(
    df: pl.DataFrame,
    pred_cols: list[str],
    target_col: str,
    outcome_order: list[str],
    subgroup_cols: list[str],
    model_mapping: dict,
    axs: list[list],  # 2x8 axes grid externally defined
    fig: plt.Figure,
    calibrated_suffix: str = "_calibrated",
    n_bins: int = 10,
):
    """
    Plots calibration curves in a 2x8 grid layout using externally provided fig and axs.

    Args:
        df (pl.DataFrame): DataFrame with outcome, predictions, calibration, and subgroup info.
        pred_cols (list[str]): List of base prediction columns.
        target_col (str): Name of target column.
        outcome_order (list[str]): List of outcome values (2 elements).
        subgroup_cols (list[str]): List of binary sensitive attribute column names.
        model_mapping (dict): Mapping from pred column names to display labels.
        axs (list[list]): 2x8 list of matplotlib axes.
        fig (plt.Figure): The matplotlib figure object.
        calibrated_suffix (str): Suffix to identify calibrated columns.
        n_bins (int): Number of bins for calibration curve.
    """
    df_pd = df.to_pandas()
    subgroups = sorted(list(product(*[df_pd[c].unique() for c in subgroup_cols])))

    assert len(subgroups) == 4, "Expected exactly 4 subgroup combinations"

    subgroup_names = [
        " & ".join([f"{c[0].upper()}={v}" for c, v in zip(subgroup_cols, sg)])
        for sg in subgroups
    ]

    for row in range(2):
        for col in range(8):
            ax = axs[row, col]

            outcome_idx = col // 4
            cal_idx = (col % 4) // 2
            subgroup_idx = (col % 2) + row * 2

            outcome = outcome_order[outcome_idx]
            calibrated = bool(cal_idx)
            subgroup = subgroups[subgroup_idx]
            subgroup_name = subgroup_names[subgroup_idx]

            df_sub = df_pd[
                (df_pd["outcome_type"] == outcome)
                & np.logical_and.reduce(
                    [df_pd[c] == v for c, v in zip(subgroup_cols, subgroup)]
                )
            ]

            pred_set = [
                col + calibrated_suffix if calibrated else col for col in pred_cols
            ]

            for pred_col in pred_set:
                if pred_col not in df_sub.columns:
                    continue
                try:
                    prob_true, prob_pred = calibration_curve(
                        df_sub[target_col], df_sub[pred_col], n_bins=n_bins
                    )
                    label = model_mapping.get(
                        pred_col.replace(calibrated_suffix, ""), pred_col
                    )
                    ax.plot(prob_pred, prob_true, marker="o", label=label)
                except ValueError:
                    continue

            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(subgroup_name, fontsize=9)
            if row == 1:
                ax.set_xlabel("Predicted probabilities", fontsize=8)
            else:
                ax.set_xlabel("")

            if col % 8 == 0:
                ax.set_ylabel("Fraction of positives", fontsize=9)
            else:
                ax.set_ylabel("")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)

    # Compute label positions for section headers
    fig.tight_layout()

    uncal_center_at_risk = (
        axs[0][0].get_position().x0 + axs[0][1].get_position().x1
    ) / 2
    cal_center_at_risk = (axs[0][2].get_position().x0 + axs[0][3].get_position().x1) / 2
    uncal_center_econ = (axs[0][4].get_position().x0 + axs[0][5].get_position().x1) / 2
    cal_center_econ = (axs[0][6].get_position().x0 + axs[0][7].get_position().x1) / 2

    calibration_text_y = 1.0

    fig.text(
        uncal_center_at_risk,
        calibration_text_y,
        "Uncalibrated",
        ha="center",
        fontsize=11,
    )
    fig.text(
        cal_center_at_risk, calibration_text_y, "Calibrated", ha="center", fontsize=11
    )
    fig.text(
        uncal_center_econ, calibration_text_y, "Uncalibrated", ha="center", fontsize=11
    )
    fig.text(
        cal_center_econ, calibration_text_y, "Calibrated", ha="center", fontsize=11
    )

    at_risk_center = (axs[0][0].get_position().x0 + axs[0][3].get_position().x1) / 2
    econ_vuln_center = (axs[0][4].get_position().x0 + axs[0][7].get_position().x1) / 2
    outcome_text_y = 1.04
    fig.text(
        at_risk_center,
        outcome_text_y,
        "At-risk",
        ha="center",
        fontsize=12,
    )
    fig.text(
        econ_vuln_center,
        outcome_text_y,
        "Economically vulnerable",
        ha="center",
        fontsize=12,
    )

    # Vertical separator between outcome groups
    x_sep = (axs[0][3].get_position().x1 + axs[0][4].get_position().x0) / 2
    fig.add_artist(
        plt.Line2D(
            [x_sep, x_sep],
            [0.1, 0.93],
            transform=fig.transFigure,
            color="gray",
            linestyle="dotted",
            linewidth=1.5,
            alpha=0.8,
        )
    )

    minor_grey_alpha = 0.4

    x_sep = (axs[0][1].get_position().x1 + axs[0][2].get_position().x0) / 2
    fig.add_artist(
        plt.Line2D(
            [x_sep, x_sep],
            [0.1, 0.93],
            transform=fig.transFigure,
            color="gray",
            linestyle="dotted",
            linewidth=1.5,
            alpha=minor_grey_alpha,
        )
    )

    x_sep = (axs[0][5].get_position().x1 + axs[0][6].get_position().x0) / 2
    fig.add_artist(
        plt.Line2D(
            [x_sep, x_sep],
            [0.1, 0.93],
            transform=fig.transFigure,
            color="gray",
            linestyle="dotted",
            linewidth=1.5,
            alpha=minor_grey_alpha,
        )
    )

    # Shared legend
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(labels),
        frameon=False,
    )


def hexbin_lower_triangle_shared_cbar(
    df: pl.DataFrame,
    columns: list[str],
    fpath: Path,
    gridsize: int = 100,
    figsize: tuple[float, float] = (10, 10),
    cmap: str = "Blues",
) -> None:
    """
    Creates a compact grid of hexbin plots for the lower triangular matrix of given columns,
    with a shared color bar. Displays only the leftmost y-axis labels and bottommost x-axis labels.

    Args:
        df (pl.DataFrame): Polars DataFrame containing the data.
        columns (list[str]): List of column names to compare.
        fpath (Path): Path in which to save the figure.
        gridsize (int): Gridsize for the hexbin plots.
        figsize (tuple[float, float]): Size of the Matplotlib figure.
        cmap (str): Colormap for the hexbin plots.
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols - 1, n_cols - 1, figsize=figsize, squeeze=False)

    # Determine the range of counts for consistent color mapping
    global_max = 0
    for i, j in combinations(range(n_cols), 2):
        x = df[columns[i]].to_numpy()
        y = df[columns[j]].to_numpy()
        counts = np.histogram2d(x, y, bins=gridsize)[0]
        global_max = max(global_max, counts.max())

    # Create hexbin plots in the lower triangle
    for j in range(1, n_cols):  # Row index
        for i in range(j):  # Column index
            ax = axes[j - 1, i]  # Adjust axes index for compact layout
            x = df[columns[i]].to_numpy()
            y = df[columns[j]].to_numpy()
            hb = ax.hexbin(
                x, y, gridsize=gridsize, cmap=cmap, vmin=0, vmax=global_max, mincnt=5
            )
            # Set axis labels for the outermost axes
            if i == 0:  # Leftmost column
                ax.set_ylabel(columns[j])
            if j == n_cols - 1:  # Bottommost row
                ax.set_xlabel(columns[i])
            ax.label_outer()

            plt.grid(False)
            ax.set_facecolor("white")

    # Remove unused axes
    for j in range(n_cols - 1):
        for i in range(n_cols - 1):
            if i > j:  # Only use lower triangle
                axes[j, i].axis("off")

    # Add a shared color bar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Position for the color bar
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.set_label("Counts")

    # Add a thousand separator to color bar ticks
    formatter = FuncFormatter(lambda x, _: f"{int(x):,}")
    cbar.formatter = formatter
    cbar.update_ticks()

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout for color bar

    plt.savefig(fpath)
    plt.show()


def rename_terms(term: str) -> str:
    """
    Renames regression terms to a human-readable format, removing 'T.' prefixes.

    Args:
        term (str): The regression term to rename.

    Returns:
        str: Renamed term in the format 'category_col_K=Value' or interaction terms.
    """
    if term == "Intercept":
        return "Intercept"
    parts = term.split(":")
    return ":".join(
        part.split("Treatment")[0]
        .replace("C(", "")
        .replace(")", "")
        .replace("[T.", "=")
        .replace("[", "=")
        .replace("]", "")
        for part in parts
    )


def rename_terms_with_reference_level(term: str) -> str:
    """
    Renames regression terms to a human-readable format, removing 'T.' prefixes.

    Args:
        term (str): The regression term to rename.

    Returns:
        str: Renamed term in the format 'category_col_K=Value' or interaction terms.
    """
    if term == "Intercept":
        return "Intercept"
    parts = term.split(":")
    return ":".join(
        # part.split('Treatment')[0].replace('C(', '').replace(')', '').replace('[T.', '=').replace('[', '=').replace(']', '')
        # "=".join(re.search(r'C\((\w+),\s*Treatment\("(\w+)"\)\)', part).groups()) for part in parts
        "=".join(re.search(r"C\((\w+),\s.*?\)\[T\.(\w+)\]", part).groups())
        for part in parts
    )


def classify_terms(term: str) -> str:
    """
    Classifies terms into groups: Intercept, Main Effect, Two-Way Interaction, or Three-Way Interaction.

    Args:
        term (str): The regression term to classify.

    Returns:
        str: The group name.
    """
    if term == "Intercept":
        return "Intercept"
    elif ":" not in term:
        return "Main Effect"
    elif term.count(":") == 1:
        return "Two-Way Interaction"
    elif term.count(":") == 2:
        return "Three-Way Interaction"
    return "Other"


def stack_columns(
    df: pl.DataFrame, cols_to_stack: list[str], keep_cols: list[str]
) -> pl.DataFrame:
    """
    Stacks specified columns into a single column while retaining the specified columns.

    Args:
        df (pl.DataFrame): The original dataframe.
        cols_to_stack (list[str]): List of column names to stack.
        keep_cols (list[str]): List of column names to retain in the final dataframe.

    Returns:
        pl.DataFrame: The transformed dataframe with stacked columns.
    """
    stacked_dfs = []

    for col in cols_to_stack:
        temp_df = (
            df.select(keep_cols + [col])
            .with_columns(pl.lit(col).alias("stacked_column_name"))
            .rename({col: "stacked_value"})
        )
        stacked_dfs.append(temp_df)

    return pl.concat(stacked_dfs)


def plot_fairness_subgroup_regressions_grid(
    stacked_df: pl.DataFrame,
    main_effect_cols: list[str],
    subgroup_col: str,
    target_col: str,
    outcome_col: str,
    fig: plt.Figure,
    axs: list[list],  # 2x3 grid of axes
    outcome_order: list[str] = ["at_risk", "econ_vuln"],
    outcome_labels: dict[str, str] = None,
):
    """
    Plots subgroup regression results in a 2x3 grid: rows = outcome_type, cols = [all, target==1, target==0].

    Args:
        stacked_df (pl.DataFrame): Long-format DataFrame containing predictions and group labels.
        main_effect_cols (list[str]): Covariates to include in the regression (e.g. ["native", "female"]).
        subgroup_col (str): Column identifying each model.
        target_col (str): Column with prediction error or residual (e.g. absolute residual).
        outcome_col (str): Name of column with outcome types.
        fig (plt.Figure): Matplotlib figure object.
        axs (list[list]): 2x3 axes grid from plt.subplots().
        outcome_order (list[str]): Desired outcome ordering.
    """
    df = stacked_df.to_pandas()
    df[main_effect_cols + [subgroup_col]] = df[
        main_effect_cols + [subgroup_col]
    ].astype("category")

    for i, outcome in enumerate(outcome_order):
        df_outcome = df[df[outcome_col] == outcome]
        for j, subset in enumerate(["all", 1, 0]):
            if subset != "all":
                df_subset = df_outcome[df_outcome["target"] == subset]
            else:
                df_subset = df_outcome

            subgroups = df_subset[subgroup_col].unique()
            results = []
            for subgroup in subgroups:
                sub = df_subset[df_subset[subgroup_col] == subgroup]
                formula = f"{target_col} ~ " + " * ".join(
                    [f'C({col}, Treatment("Y"))' for col in main_effect_cols]
                )
                model = smf.ols(formula=formula, data=sub).fit(cov_type="HC3")

                coefs = model.params
                conf_int = model.conf_int()
                conf_int.columns = ["CI_lower", "CI_upper"]

                coef_df = pd.concat([coefs, conf_int], axis=1).reset_index()
                coef_df.columns = ["Term", "Coefficient", "CI_lower", "CI_upper"]
                coef_df["Subgroup"] = subgroup
                coef_df["Renamed_Term"] = coef_df["Term"].apply(
                    rename_terms_with_reference_level
                )
                coef_df["Group"] = coef_df["Term"].apply(classify_terms)
                results.append(coef_df)

            combined_df = pd.concat(results)

            grouped_terms = []
            for group in [
                "Intercept",
                "Main Effect",
                "Two-Way Interaction",
                "Three-Way Interaction",
            ]:
                # grouped_terms.append(" ")  # blank for spacing
                terms_in_group = combined_df[combined_df["Group"] == group][
                    "Renamed_Term"
                ].unique()
                grouped_terms.extend(terms_in_group)

            spacing = 1.5  # Adjust this value to increase or decrease vertical spacing
            term_to_y = {term: i * spacing for i, term in enumerate(grouped_terms)}

            ax = axs[i][j]

            for k, subgroup in enumerate(subgroups):
                subgroup_data = combined_df[combined_df["Subgroup"] == subgroup]
                y_positions = np.array(
                    [term_to_y[term] for term in subgroup_data["Renamed_Term"]]
                )
                y_positions = y_positions + k * 0.2 - (len(subgroups) - 1) / 2 * 0.2

                ax.errorbar(
                    x=subgroup_data["Coefficient"],
                    y=y_positions,
                    xerr=(
                        subgroup_data["Coefficient"] - subgroup_data["CI_lower"],
                        subgroup_data["CI_upper"] - subgroup_data["Coefficient"],
                    ),
                    fmt="o",
                    label=subgroup,
                    capsize=4,
                    alpha=0.8,
                )

            # Only show y tick labels in first column
            if j == 0:
                ax.set_yticks(list(term_to_y.values()))
                ax.set_yticklabels(list(term_to_y.keys()), fontsize=8)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            if j == 0:
                if outcome_labels:
                    ax.set_ylabel(f"{outcome_labels[outcome]}")
                else:
                    ax.set_ylabel(f"{outcome}")
            else:
                ax.set_ylabel("")
            if i == 1:
                ax.set_xlabel("Coefficient")
            else:
                ax.set_xlabel("")
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                if j == 0:
                    ax.set_title("All")
                elif j == 1:
                    ax.set_title("Target = 1")
                else:
                    ax.set_title("Target = 0")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        frameon=False,
    )
    fig.tight_layout()


def plot_metric_mosaic_by_subgroup(
    df_level_all: pl.DataFrame,
    df_subgroup_all: pl.DataFrame,
    metric: str,
    pred_cols: list[str],
    calibrated_cols_binary: list[str],
    fig,
    ax_dict: dict,
    legend_mapper: Optional[dict],
    metric_mapper: Optional[dict],
) -> None:
    """
    Plot one large ax for overall metric and four smaller axes for subgroup metrics (one per subgroup).
    Each subgroup subplot includes all models with confidence intervals.
    Shared legend is shown below, and x-axis is log-scaled.
    """
    cols_to_use = pred_cols if metric == "auc" else calibrated_cols_binary

    df_level = (
        df_level_all.filter(pl.col("metric_type") == metric)
        .filter(pl.col("col").is_in(cols_to_use))
        .to_pandas()
    )

    df_subgroup = (
        df_subgroup_all.filter(pl.col("metric_type") == metric)
        .filter(pl.col("col").is_in(cols_to_use))
        .to_pandas()
    )

    # --- Overall plot ---
    ax_A = ax_dict["A"]
    for model in cols_to_use:
        df_model = df_level[df_level["col"] == model].sort_values("subset_size")

        ax_A.plot(
            df_model["subset_size"],
            df_model["metric"],
            label=legend_mapper[model],
            marker="o",
        )
        ax_A.fill_between(
            df_model["subset_size"],
            df_model["lb"],
            df_model["ub"],
            alpha=0.2,
        )

    ax_A.set_title("Overall performance")
    ax_A.set_xlabel("Train size (log scale)")
    ax_A.set_ylabel(metric_mapper[metric])

    # --- Subgroup plots ---
    subgroups = df_subgroup["subgroup"].unique()
    subplot_keys = ["B", "C", "D", "E"]
    for subgroup, key in zip(subgroups, subplot_keys):
        ax = ax_dict[key]
        df_sub = df_subgroup[df_subgroup["subgroup"] == subgroup]

        for model in cols_to_use:
            df_model = df_sub[df_sub["col"] == model].sort_values("subset_size")

            ax.plot(
                df_model["subset_size"],
                df_model["metric"],
                label=legend_mapper[model],
                marker="o",
            )
            if "lb" in df_model.columns and "ub" in df_model.columns:
                ax.fill_between(
                    df_model["subset_size"],
                    df_model["lb"],
                    df_model["ub"],
                    alpha=0.2,
                )

        ax.set_title(f"{subgroup}")
        ax.set_xlabel("Train size (log scale)")
        ax.set_ylabel(metric_mapper[metric])

    # --- Apply log scale and remove top/right spines
    for ax in ax_dict.values():
        ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Share x and y limits for subgroup plots
    sub_axes = [ax_dict[k] for k in ["B", "C", "D", "E"]]
    xlims = [ax.get_xlim() for ax in sub_axes]
    ylims = [ax.get_ylim() for ax in sub_axes]
    shared_xlim = (min(x[0] for x in xlims), max(x[1] for x in xlims))
    shared_ylim = (min(y[0] for y in ylims), max(y[1] for y in ylims))

    for key in ["B", "C", "D", "E"]:
        ax = ax_dict[key]
        ax.set_xlim(shared_xlim)
        ax.set_ylim(shared_ylim)
        if key in ["B", "C"]:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if key in ["C", "E"]:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    # --- Shared legend
    handles, labels = ax_A.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="",
        loc="lower center",
        ncol=len(cols_to_use),
        bbox_to_anchor=(0.5, -0.065),
        fontsize=10,
        frameon=False,
    )
