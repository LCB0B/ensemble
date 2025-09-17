# %%
import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT

from itertools import product
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


# %%
def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
) -> tuple[float, float]:
    """
    Compute a 95% bootstrap confidence interval for a metric.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted values.
        metric_func (Callable[[np.ndarray, np.ndarray], float]): Metric function to evaluate.
        n_resamples (int): Number of bootstrap resamples.
    """
    resampled_metrics = np.empty(n_resamples)
    n = len(y_true)

    for i in range(n_resamples):
        indices = np.random.choice(n, size=n, replace=True)
        resampled_metrics[i] = metric_func(y_true[indices], y_pred[indices])

    lower, upper = np.percentile(resampled_metrics, [2.5, 97.5])
    return lower, upper


def calculate_subgroup_gap_with_bootstrap(
    df: pl.DataFrame,
    prediction_cols: List[str],
    ground_truth_col: str,
    sensitive_cols: List[str],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
) -> pl.DataFrame:
    """
    Compute subgroup fairness gaps and 95% CIs using bootstrap resampling.

    Args:
        df (pl.DataFrame): Input data.
        prediction_cols (List[str]): Prediction columns to evaluate.
        ground_truth_col (str): Ground truth column.
        sensitive_cols (List[str]): Sensitive attribute columns.
        metric (Callable[[np.ndarray, np.ndarray], float]): Metric function.
        n_resamples (int): Number of bootstrap resamples.
    """
    results = []

    # Convert Polars DataFrame to Pandas for compatibility with metrics
    df_pd = df.to_pandas()

    # Get unique combinations of sensitive features to define subgroups
    unique_combinations = list(
        product(*[df_pd[col].unique() for col in sensitive_cols])
    )

    # Iterate over each prediction column
    for col in prediction_cols:
        # List to store the gap for each bootstrap sample
        bootstrap_gaps = []

        observed_subgroup_metrics = []
        for combination in unique_combinations:
            # Filter to the current subgroup
            observed_subgroup_filter = (df_pd[sensitive_cols] == combination).all(
                axis=1
            )
            observed_subgroup_df = df_pd[observed_subgroup_filter]

            observed_metric_value = metric(
                observed_subgroup_df[ground_truth_col], observed_subgroup_df[col]
            )
            observed_subgroup_metrics.append(observed_metric_value)

        observed_gap = np.max(observed_subgroup_metrics) - np.min(
            observed_subgroup_metrics
        )

        for _ in tqdm(range(n_resamples), desc=f"Bootstrapping for {col}"):
            # Bootstrap the entire DataFrame
            bootstrap_df = df_pd.sample(n=len(df_pd), replace=True)

            # Calculate metric for each subgroup in the bootstrap sample
            subgroup_metrics = []
            for combination in unique_combinations:
                # Filter to the current subgroup
                subgroup_filter = (bootstrap_df[sensitive_cols] == combination).all(
                    axis=1
                )
                subgroup_df = bootstrap_df[subgroup_filter]

                # Skip subgroup if insufficient data
                if (
                    len(subgroup_df) == 0
                    or len(np.unique(subgroup_df[ground_truth_col])) < 2
                ):
                    continue  # Skip if AUC or metric undefined due to only one class

                # Calculate metric for this subgroup
                metric_value = metric(subgroup_df[ground_truth_col], subgroup_df[col])
                subgroup_metrics.append(metric_value)

            # Calculate the gap within this bootstrap sample if there are enough subgroups
            if subgroup_metrics:
                gap = np.max(subgroup_metrics) - np.min(subgroup_metrics)
                bootstrap_gaps.append(gap)

        # Calculate observed gap and bootstrap confidence interval for the gap
        lb, ub = (
            np.percentile(bootstrap_gaps, [2.5, 97.5])
            if bootstrap_gaps
            else (np.nan, np.nan)
        )

        # Store results for this prediction column
        results.append({"col": col, "metric": observed_gap, "lb": lb, "ub": ub})

    # Convert results to a Polars DataFrame
    return pl.DataFrame(results)


def calculate_subgroup_metric_with_ci(
    df: pl.DataFrame,
    prediction_cols: List[str],
    ground_truth_col: str,
    sensitive_cols: List[str],
    metric: Callable[[Union[list, np.ndarray], Union[list, np.ndarray]], float],
    n_resamples: int = 1000,
) -> pl.DataFrame:
    """
    Compute subgroup-level metrics and 95% bootstrap CIs for intersectional subgroups.

    Args:
        df (pl.DataFrame): Input data.
        prediction_cols (List[str]): Prediction columns.
        ground_truth_col (str): Ground truth column.
        sensitive_cols (List[str]): Sensitive attribute columns.
        metric (Callable): Metric function.
        n_resamples (int): Number of bootstrap resamples.
    """

    results = []

    # Convert Polars DataFrame to Pandas for compatibility with sklearn metrics

    df_pd = df.to_pandas()

    # Get all possible combinations of the intersectional subgroups

    unique_combinations = list(
        product(*[df_pd[col].unique() for col in sensitive_cols])
    )

    for combination in tqdm(unique_combinations, desc="Calculating subgroup metrics"):

        # Filter the DataFrame for the current subgroup
        subgroup_filter = (df_pd[sensitive_cols] == combination).all(axis=1)
        subgroup_df = df_pd[subgroup_filter]

        # Check if there's enough data in the subgroup

        if len(subgroup_df) == 0:
            continue

        # Skip if AUC undefined
        if (len(np.unique(subgroup_df[ground_truth_col])) < 2) and (
            metric == roc_auc_score
        ):
            continue

        # Create a description string for the subgroup (e.g., "COL1=1 & COL2=0")
        subgroup_desc = " & ".join(
            [f"{col}={val}" for col, val in zip(sensitive_cols, combination)]
        )

        # Calculate the metric and bootstrap CI for each prediction column
        metrics = {"subgroup": subgroup_desc}

        for col in prediction_cols:

            # Calculate the metric value
            metric_value = metric(subgroup_df[ground_truth_col], subgroup_df[col])

            # Bootstrap the metric to calculate CIs
            lb, ub = bootstrap_ci(
                subgroup_df[ground_truth_col].to_numpy(),
                subgroup_df[col].to_numpy(),
                metric,
                n_resamples,
            )

            metrics[f"{col}_metric"] = metric_value

            metrics[f"{col}_lb"] = lb

            metrics[f"{col}_ub"] = ub

        results.append(metrics)

    # Convert the results into a Polars DataFrame

    result_df = pl.DataFrame(results)

    return result_df


def bootstrap_metrics_df(
    df: pl.DataFrame,
    target_col: str,
    pred_cols: List[str],
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
) -> pl.DataFrame:
    """
    Compute metrics and 95% bootstrap CIs for predictions vs. target.

    Args:
        df (pl.DataFrame): Input data.
        target_col (str): Ground truth column.
        pred_cols (List[str]): Prediction columns.
        metric_func (Callable[[np.ndarray, np.ndarray], float]): Metric function.
        n_resamples (int): Number of bootstrap resamples.
    """
    results = []

    y_true = df[target_col].to_numpy()

    for col in tqdm(pred_cols, desc="Evaluating metrics"):
        y_pred = df[col].to_numpy()
        metric = metric_func(y_true, y_pred)
        lb, ub = bootstrap_ci(y_true, y_pred, metric_func, n_resamples)
        results.append({"col": col, "metric": metric, "lb": lb, "ub": ub})

    return pl.DataFrame(results)


def melt_subgroup_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert wide-format subgroup metrics to long format with prediction name and value type.

    Args:
        df (pl.DataFrame): Wide-format subgroup metrics.
    """
    id_cols = ["subgroup"]
    value_vars = [c for c in df.columns if c not in id_cols]

    df_long = df.unpivot(index=id_cols, on=value_vars)

    # Extract prediction name and type (e.g. preds_lr_calibrated_binary, metric -> (preds_lr_calibrated_binary, metric))
    df_long = df_long.with_columns(
        [
            pl.col("variable")
            .str.extract(r"^(.*)_(metric|lb|ub)$", group_index=1)
            .alias("col"),
            pl.col("variable")
            .str.extract(r"^(.*)_(metric|lb|ub)$", group_index=2)
            .alias("value_type"),
        ]
    ).drop("variable")

    return df_long


# %%
def evaluate_all_metrics_by_outcome(
    df: pl.DataFrame,
    outcome_types: list[str],
    pred_cols: list[str],
    calibrated_cols_binary: list[str],
    ground_truth_col: str,
    sensitive_cols: list[str],
    n_bootstrap: int = 1000,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Evaluate overall, gap, and subgroup metrics for each outcome type.

    Args:
        df (pl.DataFrame): Input data.
        outcome_types (list[str]): Outcome types to evaluate.
        pred_cols (list[str]): Uncalibrated prediction columns.
        calibrated_cols_binary (list[str]): Calibrated binary prediction columns.
        ground_truth_col (str): Ground truth column.
        sensitive_cols (list[str]): Sensitive attribute columns.
        n_bootstrap (int): Number of bootstrap resamples.
    """
    level_results = []
    gap_results = []
    subgroup_results = []

    metric_specs = [
        ("recall", recall_score, calibrated_cols_binary),
        ("f1", f1_score, calibrated_cols_binary),
        ("precision", precision_score, calibrated_cols_binary),
        ("auc", roc_auc_score, pred_cols),
    ]

    for outcome in outcome_types:
        df_outcome = df.filter(pl.col("outcome_type") == outcome)

        for metric_name, metric_func, cols in metric_specs:
            # Level
            level_df = bootstrap_metrics_df(
                df_outcome, ground_truth_col, cols, metric_func, n_bootstrap
            ).with_columns(
                [
                    pl.lit(outcome).alias("outcome_type"),
                    pl.lit(metric_name).alias("metric_type"),
                    pl.lit("overall").alias("level"),
                ]
            )
            level_results.append(level_df)

            # Gap
            gap_df = calculate_subgroup_gap_with_bootstrap(
                df_outcome,
                prediction_cols=cols,
                ground_truth_col=ground_truth_col,
                sensitive_cols=sensitive_cols,
                metric=metric_func,
                n_resamples=n_bootstrap,
            ).with_columns(
                [
                    pl.lit(outcome).alias("outcome_type"),
                    pl.lit(metric_name).alias("metric_type"),
                    pl.lit("gap").alias("level"),
                ]
            )
            gap_results.append(gap_df)

            subgroup_df = calculate_subgroup_metric_with_ci(
                df_outcome,
                prediction_cols=cols,
                ground_truth_col=ground_truth_col,
                sensitive_cols=sensitive_cols,
                metric=metric_func,
                n_resamples=n_bootstrap,
            )

            subgroup_df_long = melt_subgroup_df(subgroup_df).with_columns(
                [
                    pl.lit(outcome).alias("outcome_type"),
                    pl.lit(metric_name).alias("metric_type"),
                    pl.lit("subgroup").alias("level"),
                ]
            )
            subgroup_results.append(subgroup_df_long)

    return (
        pl.concat(level_results),
        pl.concat(gap_results),
        pl.concat(subgroup_results),
    )


# def bootstrap_subgroup_metrics_and_gaps(
#     df: pl.DataFrame,
#     outcome_types: List[str],
#     ground_truth_col: str,
#     sensitive_cols: List[str],
#     metric_specs: List[Tuple[str, Callable, List[str]]],
#     sample_percents: List[int] = list(range(5, 105, 5)),
#     n_bootstrap: int = 25,
# ) -> pl.DataFrame:
#     """
#     Estimate subgroup metrics and fairness gaps with bootstrapped samples of increasing size.

#     Args:
#         df (pl.DataFrame): Input data.
#         outcome_types (List[str]): Outcome types to evaluate.
#         ground_truth_col (str): Ground truth column.
#         sensitive_cols (List[str]): Sensitive attribute columns.
#         metric_specs (List[Tuple[str, Callable, List[str]]]): Metric definitions and prediction columns.
#         sample_percents (List[int]): Sample size percentages.
#         n_bootstrap (int): Bootstrap iterations per sample size.
#     """
#     df_pd = df.to_pandas()
#     results = []

#     for outcome_no, outcome in enumerate(outcome_types):
#         df_outcome = df_pd[df_pd["outcome_type"] == outcome]

#         for sample_percent in tqdm(
#             sample_percents,
#             desc=f"Iterating over {outcome} sample percent, (outcome {outcome_no+1}/{len(outcome_types)}) (this does not scale linearly)",
#         ):
#             sample_size = int((sample_percent / 100) * len(df_outcome))

#             for i in range(n_bootstrap):
#                 sample = df_outcome.sample(n=sample_size, replace=True)

#                 for metric_type, metric_func, model_cols in metric_specs:
#                     for col in model_cols:
#                         subgroup_metrics = []
#                         groups = list(
#                             product(*[sample[c].unique() for c in sensitive_cols])
#                         )

#                         for g in groups:
#                             filt = (sample[sensitive_cols] == g).all(axis=1)
#                             subgroup = sample[filt]

#                             if (
#                                 len(subgroup) == 0
#                                 or len(np.unique(subgroup[ground_truth_col])) < 2
#                             ):
#                                 continue

#                             try:
#                                 score = metric_func(
#                                     subgroup[ground_truth_col], subgroup[col]
#                                 )
#                             except ValueError:
#                                 score = np.nan

#                             subgroup_metrics.append(
#                                 {
#                                     "subgroup": " & ".join(
#                                         f"{k}={v}" for k, v in zip(sensitive_cols, g)
#                                     ),
#                                     "score": score,
#                                     "col": col,
#                                     "metric_type": metric_type,
#                                     "outcome_type": outcome,
#                                     "bootstrap_iter": i,
#                                     "sample_percent": sample_percent,
#                                 }
#                             )

#                         # Compute gap
#                         scores = [
#                             s["score"]
#                             for s in subgroup_metrics
#                             if not np.isnan(s["score"])
#                         ]
#                         gap = np.max(scores) - np.min(scores) if scores else np.nan

#                         results.extend(subgroup_metrics)
#                         results.append(
#                             {
#                                 "subgroup": "GAP",
#                                 "score": gap,
#                                 "col": col,
#                                 "metric_type": metric_type,
#                                 "outcome_type": outcome,
#                                 "bootstrap_iter": i,
#                                 "sample_percent": sample_percent,
#                             }
#                         )

#     return pl.from_pandas(pd.DataFrame(results))


def progressive_sample_subgroup_metrics_and_gaps(
    df: pl.DataFrame,
    outcome_types: List[str],
    ground_truth_col: str,
    sensitive_cols: List[str],
    metric_specs: List[Tuple[str, Callable, List[str]]],
    sample_percents: List[int] = list(range(5, 105, 5)),
    n_repeats: int = 25,
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Compute subgroup metrics and fairness gaps using progressive (non-bootstrap) sampling.

    Args:
        df (pl.DataFrame): Input data.
        outcome_types (List[str]): Outcome types to evaluate.
        ground_truth_col (str): Ground truth column.
        sensitive_cols (List[str]): Sensitive attribute columns.
        metric_specs (List[Tuple[str, Callable, List[str]]]): Metric definitions and prediction columns.
        sample_percents (List[int]): Sample size percentages.
        n_repeats (int): Number of repetitions for each sample size.
        random_state (int): Seed for reproducibility.
    """
    df_pd = df.to_pandas()
    results = []
    rng = np.random.default_rng(random_state)

    for outcome in outcome_types:
        df_outcome = df_pd[df_pd["outcome_type"] == outcome].copy()

        for b in range(n_repeats):
            df_shuffled = df_outcome.sample(
                frac=1.0, random_state=rng.integers(0, 1e6), replace=False
            ).reset_index(drop=True)

            for sample_percent in tqdm(
                sample_percents,
                desc=f"{outcome}: progressive sample {b+1}/{n_repeats}",
            ):
                sample_size = int((sample_percent / 100) * len(df_shuffled))
                sample = df_shuffled.iloc[:sample_size]

                for metric_type, metric_func, model_cols in metric_specs:
                    for col in model_cols:
                        subgroup_metrics = []
                        groups = list(
                            product(*[sample[c].unique() for c in sensitive_cols])
                        )

                        for g in groups:
                            filt = (sample[sensitive_cols] == g).all(axis=1)
                            subgroup = sample[filt]

                            if (
                                len(subgroup) == 0
                                or len(np.unique(subgroup[ground_truth_col])) < 2
                            ):
                                continue

                            try:
                                score = metric_func(
                                    subgroup[ground_truth_col], subgroup[col]
                                )
                            except ValueError:
                                score = np.nan

                            subgroup_metrics.append(
                                {
                                    "subgroup": " & ".join(
                                        f"{k}={v}" for k, v in zip(sensitive_cols, g)
                                    ),
                                    "score": score,
                                    "col": col,
                                    "metric_type": metric_type,
                                    "outcome_type": outcome,
                                    "bootstrap_iter": b,
                                    "sample_percent": sample_percent,
                                }
                            )

                        scores = [
                            s["score"]
                            for s in subgroup_metrics
                            if not np.isnan(s["score"])
                        ]
                        gap = np.max(scores) - np.min(scores) if scores else np.nan

                        results.extend(subgroup_metrics)
                        results.append(
                            {
                                "subgroup": "GAP",
                                "score": gap,
                                "col": col,
                                "metric_type": metric_type,
                                "outcome_type": outcome,
                                "bootstrap_iter": b,
                                "sample_percent": sample_percent,
                            }
                        )

    return pl.from_pandas(pd.DataFrame(results))


def agreement_fraction(df: pd.DataFrame, threshold: float, min_columns: int) -> float:
    """
    Compute fraction of rows where at least `min_columns` predictions exceed `threshold`.

    Args:
        df (pd.DataFrame): DataFrame of predicted probabilities.
        threshold (float): Threshold to count agreement.
        min_columns (int): Minimum number of models required to agree.
    """
    count_above_threshold = (df > threshold).sum(axis=1)
    return (count_above_threshold >= min_columns).sum() / (
        count_above_threshold >= 1
    ).sum()


def normalized_agreement_fraction(
    df: pd.DataFrame,
    threshold: float,
    min_columns: int,
    n_permutations: int = 100,
    random_state: int = 42,
) -> float:
    """
    Compute normalized agreement fraction by subtracting expected agreement under random permutations.

    Args:
        df (pd.DataFrame): DataFrame of predicted probabilities.
        threshold (float): Threshold to count agreement.
        min_columns (int): Minimum number of models required to agree.
        n_permutations (int): Number of random permutations.
        random_state (int): Seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    observed = agreement_fraction(df, threshold, min_columns)

    expected_scores = []
    for _ in range(n_permutations):
        df_permuted = pd.DataFrame(
            {col: rng.permutation(df[col].values) for col in df.columns}
        )
        expected = agreement_fraction(df_permuted, threshold, min_columns)
        expected_scores.append(expected)

    expected_mean = np.mean(expected_scores)
    return observed - expected_mean
