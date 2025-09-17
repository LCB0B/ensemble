from typing import Dict, Literal, Tuple

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.paths import FPATH
from src.utils import calculate_datetime_from_abspos


def calculate_best_thresholds(
    df: pl.DataFrame, prob_cols: list[str], true_col: str, threshold_steps: int = 99
) -> Dict[str, Tuple[float, float]]:
    """
    Calculates the optimal threshold for each probability column based on the F1 score.

    Args:
        df (pl.DataFrame): DataFrame containing the probabilities and true labels.
        prob_cols (list[str]): List of column names with predicted probabilities.
        true_col (str): Column name with true labels (binary: 0 or 1).
        threshold_steps (int): Number of threshold values to evaluate between 0 and 1.

    Returns:
        dict: A dictionary where each key is a probability column name, and the value is a tuple
              containing the optimal threshold and the corresponding F1 score.
    """
    # Extract true labels as a numpy array
    true_labels = df[true_col].to_numpy()

    # Define threshold values to evaluate
    thresholds = np.linspace(0, 1, threshold_steps)

    best_thresholds = {}

    for col in prob_cols:
        probabilities = df[col].to_numpy()
        best_f1_score = 0
        best_threshold = 0

        # Iterate over thresholds to calculate F1 score
        for threshold in tqdm(thresholds, desc="Iterating over thresholds"):
            predictions = (probabilities >= threshold).astype(int)

            current_f1_score = f1_score(true_labels, predictions)

            # Update best threshold if this F1 score is higher
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                best_threshold = threshold

        best_thresholds[col] = (best_threshold, best_f1_score)

    return best_thresholds


def load_predictions(
    outcome: str,
    data_split: str,
    transformer_output_folder: str,
    transformer_output_file_name_1: str,
    transformer_output_file_name_2: str,
    cb_experiment_name: str,
    lr_experiment_name: str,
    data_folder: str,
) -> pl.DataFrame:
    """
    Loads predictions and ground truth for a given outcome and data split.

    Args:
        outcome (str): The prediction target name.
        data_split (str): The dataset split, e.g., 'train', 'val', 'test'.
        transformer_output_folder (str): Folder containing transformer output files.
        transformer_output_file_name_1 (str): Basename for pretrained transformer predictions.
        transformer_output_file_name_2 (str): Basename for finetuned transformer predictions.
        cb_experiment_name (str): Identifier for CatBoost experiment.
        lr_experiment_name (str): Identifier for Logistic Regression experiment.
        data_folder (str): Folder containing ground truth labels.
    """
    # Transformer predictions
    output_path_1 = (
        FPATH.DATA
        / transformer_output_folder
        / f"{transformer_output_file_name_1}_{outcome}_{data_split}.parquet"
    )
    output_path_2 = (
        FPATH.DATA
        / transformer_output_folder
        / f"{transformer_output_file_name_2}_{outcome}_{data_split}.parquet"
    )

    df_1 = pl.read_parquet(FPATH.swap_drives(output_path_1)).rename(
        {"predictions": f"preds_{transformer_output_file_name_1}"}
    )
    df_2 = pl.read_parquet(FPATH.swap_drives(output_path_2)).rename(
        {"predictions": f"preds_{transformer_output_file_name_2}"}
    )

    # Tabular predictions
    preds_cb = np.load(
        FPATH.swap_drives(
            FPATH.DATA
            / transformer_output_folder
            / f"cb_preds_{cb_experiment_name}_{outcome}_{data_split}.npy"
        )
    )
    preds_lr = np.load(
        FPATH.swap_drives(
            FPATH.DATA
            / transformer_output_folder
            / f"lr_preds_{lr_experiment_name}_{outcome}_{data_split}.npy"
        )
    )

    # Ground truth
    y_path = FPATH.DATA / data_folder / f"{outcome}_{data_split}_y.npy"
    y = np.load(FPATH.swap_drives(y_path))

    df = (
        df_1.with_columns(
            pl.Series(y).alias("target"),
            pl.Series(df_2.select(f"preds_{transformer_output_file_name_2}")),
            pl.Series(preds_cb).alias("preds_catboost"),
            pl.Series(preds_lr).alias("preds_lr"),
            pl.lit(outcome).alias("outcome_type"),
            pl.lit(data_split).alias("data_split"),
        )
        .cast({"person_id": int})
        .select(
            [
                "person_id",
                "outcome_type",
                "data_split",
                "target",
                f"preds_{transformer_output_file_name_1}",
                f"preds_{transformer_output_file_name_2}",
                "preds_catboost",
                "preds_lr",
                "censor",
            ]
        )
    )

    return df


def load_predictions_w_baseline(
    outcome: str,
    data_split: str,
    prediction_output_folder: str,
    transformer_output_file_name_1: str,
    transformer_output_file_name_2: str,
    cb_experiment_name: str,
    lr_experiment_name: str,
    baseline_experiment_name: str,
    data_folder: str,
) -> pl.DataFrame:
    """
    Loads predictions and ground truth for a given outcome and data split.

    Args:
        outcome (str): The prediction target name.
        data_split (str): The dataset split, e.g., 'train', 'val', 'test'.
        transformer_output_folder (str): Folder containing transformer output files.
        transformer_output_file_name_1 (str): Basename for pretrained transformer predictions.
        transformer_output_file_name_2 (str): Basename for finetuned transformer predictions.
        cb_experiment_name (str): Identifier for CatBoost experiment.
        lr_experiment_name (str): Identifier for Logistic Regression experiment.
        data_folder (str): Folder containing ground truth labels.
    """
    # Transformer predictions
    output_path_1 = (
        FPATH.NETWORK_DATA
        / prediction_output_folder
        / f"{transformer_output_file_name_1}_{outcome}_{data_split}.parquet"
    )
    output_path_2 = (
        FPATH.NETWORK_DATA
        / prediction_output_folder
        / f"{transformer_output_file_name_2}_{outcome}_{data_split}.parquet"
    )

    df_1 = pl.read_parquet(output_path_1).rename(
        {"predictions": f"preds_{transformer_output_file_name_1}"}
    )
    df_2 = pl.read_parquet(output_path_2).rename(
        {"predictions": f"preds_{transformer_output_file_name_2}"}
    )

    # Tabular predictions
    preds_cb = np.load(
        FPATH.NETWORK_DATA
        / prediction_output_folder
        / f"cb_preds_{cb_experiment_name}_{outcome}_{data_split}.npy"
    )
    preds_lr = np.load(
        FPATH.NETWORK_DATA
        / prediction_output_folder
        / f"lr_preds_{lr_experiment_name}_{outcome}_{data_split}.npy"
    )

    preds_baseline = np.load(
        FPATH.NETWORK_DATA
        / prediction_output_folder
        / f"{baseline_experiment_name}_{outcome}_{data_split}.npy"
    )

    # Ground truth
    y_path = FPATH.NETWORK_DATA / data_folder / f"{outcome}_{data_split}_y.npy"
    y = np.load(y_path)

    df = (
        df_1.with_columns(
            pl.Series(y).alias("target"),
            pl.Series(df_2.select(f"preds_{transformer_output_file_name_2}")),
            pl.Series(preds_cb).alias("preds_catboost"),
            pl.Series(preds_lr).alias("preds_lr"),
            pl.Series(preds_baseline).alias("preds_baseline"),
            pl.lit(outcome).alias("outcome_type"),
            pl.lit(data_split).alias("data_split"),
        )
        .cast({"person_id": int})
        .select(
            [
                "person_id",
                "outcome_type",
                "data_split",
                "target",
                f"preds_{transformer_output_file_name_1}",
                f"preds_{transformer_output_file_name_2}",
                "preds_catboost",
                "preds_lr",
                "preds_baseline",
                "censor",
            ]
        )
    )

    return df


def rank_to_score(
    df: pl.DataFrame, group_cols: list[str], rank_col: str, score_col: str
) -> pl.DataFrame:
    """
    Apply ranking and min-max normalization to a column grouped by other columns.

    Args:
        df (pl.DataFrame): Input DataFrame.
        group_cols (list[str]): Columns to group by.
        rank_col (str): Column to rank.
        score_col (str): Name of output score column.
    """
    return (
        df.with_columns(
            pl.col(rank_col).rank("random").over(group_cols).alias("rank_temp")
        )
        .with_columns(
            (
                (pl.col("rank_temp") - pl.col("rank_temp").min().over(group_cols))
                / (
                    pl.col("rank_temp").max().over(group_cols)
                    - pl.col("rank_temp").min().over(group_cols)
                )
            ).alias(score_col)
        )
        .drop("rank_temp")
    )


def calibrate_and_binarize_by_outcome(
    df: pl.DataFrame,
    pred_cols: list[str],
    true_col: str,
    outcome_col: str = "outcome_type",
    data_split_col: str = "data_split",
    val_tag: str = "val",
    test_tag: str = "test",
    threshold_steps: int = 99,
) -> pl.DataFrame:
    """
    Calibrates and binarizes predictions separately for each outcome_type using validation data.

    Args:
        df (pl.DataFrame): DataFrame with all outcome types and splits combined.
        pred_cols (list[str]): List of prediction column names.
        true_col (str): Ground truth column name (same for all outcomes).
        outcome_col (str): Column name for outcome type.
        data_split_col (str): Column name for val/test split.
        val_tag (str): Value in split column identifying validation rows.
        test_tag (str): Value in split column identifying test rows.
        threshold_steps (int): Number of thresholds for binarization.

    Returns:
        pl.DataFrame: Updated DataFrame with calibrated and binarized columns.
    """
    result_frames = []

    for outcome in df[outcome_col].unique().to_list():
        outcome_df = df.filter(pl.col(outcome_col) == outcome)
        df_val = outcome_df.filter(pl.col(data_split_col) == val_tag)
        df_test = outcome_df.filter(pl.col(data_split_col) == test_tag)

        # Skip if val or test data missing
        if df_val.is_empty() or df_test.is_empty():
            result_frames.append(outcome_df)
            continue

        # Calibrate predictions
        for col in pred_cols:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(df_val[col].to_numpy(), df_val[true_col].to_numpy())

            df_val = df_val.with_columns(
                pl.Series(ir.transform(df_val[col].to_numpy())).alias(
                    f"{col}_calibrated"
                )
            )
            df_test = df_test.with_columns(
                pl.Series(ir.transform(df_test[col].to_numpy())).alias(
                    f"{col}_calibrated"
                )
            )

        # Select thresholds and binarize
        calibrated_cols = [f"{col}_calibrated" for col in pred_cols]
        thresholds = calculate_best_thresholds(
            df_val, calibrated_cols, true_col, threshold_steps
        )

        print(thresholds)

        for col in calibrated_cols:
            threshold = thresholds[col][0]
            binary_col = (pl.col(col) >= threshold).cast(pl.Int8).alias(f"{col}_binary")
            df_val = df_val.with_columns(binary_col)
            df_test = df_test.with_columns(binary_col)

        result_frames.append(pl.concat([df_val, df_test]))

    return pl.concat(result_frames)


def load_all_subset_predictions(
    data_split: Literal["val", "test"],
    output_folder: str,
    subset_sizes: list[int],
    cb_experiment_template: str,
    lr_experiment_template: str,
    data_folder: str,
) -> pl.DataFrame:
    """
    Loads predictions for all subset sizes and both pretrained/scratch transformer models
    into a single dataframe with one row per person and separate prediction columns.

    Args:
        data_split (str): 'val' or 'test'.
        output_folder (str): Folder with saved prediction outputs.
        subset_sizes (list[int]): List of subset sizes to include.
        cb_experiment_template (str): Template like "cb_subset_size_{size}".
        lr_experiment_template (str): Template like "lr_subset_size_{size}".
        data_folder (str): Folder with ground truth arrays.
    """
    df_dict = {}

    y_path = FPATH.DATA / data_folder / f"at_risk_{data_split}_y.npy"
    y = np.load(FPATH.swap_drives(y_path))
    target_col = pl.Series(y).alias("target")

    background = pl.read_parquet(
        FPATH.swap_drives(FPATH.DUMP_DIR) / "all_background_integers.parquet"
    ).select(["person_id", "female", "native"])

    for size in subset_sizes:
        # Load all transformer predictions
        path_pretrained = (
            FPATH.DATA
            / output_folder
            / f"pretrained_True_size_{size}_at_risk_{data_split}.parquet"
        )
        path_scratch = (
            FPATH.DATA
            / output_folder
            / f"pretrained_False_size_{size}_at_risk_{data_split}.parquet"
        )

        df_pretrained = pl.read_parquet(FPATH.swap_drives(path_pretrained)).rename(
            {"predictions": "preds_pretrained"}
        )
        df_scratch = pl.read_parquet(FPATH.swap_drives(path_scratch)).rename(
            {"predictions": "preds_finetuned"}
        )

        # Load CB and LR predictions
        preds_cb = np.load(
            FPATH.swap_drives(
                FPATH.DATA
                / output_folder
                / f"cb_preds_{cb_experiment_template.format(size=size)}_{data_split}.npy"
            )
        )
        preds_lr = np.load(
            FPATH.swap_drives(
                FPATH.DATA
                / output_folder
                / f"lr_preds_{lr_experiment_template.format(size=size)}_{data_split}.npy"
            )
        )

        # Combine into one DataFrame
        df = (
            df_pretrained.with_columns(
                pl.Series(target_col).alias("target"),
                pl.Series(df_scratch.select("preds_finetuned")),
                pl.Series(preds_cb).alias("preds_catboost"),
                pl.Series(preds_lr).alias("preds_lr"),
                pl.lit("at_risk").alias("outcome_type"),
                pl.lit(data_split).alias("data_split"),
                pl.lit(size).alias("subset_size"),
                calculate_datetime_from_abspos(pl.col("censor"))
                .dt.year()
                .alias("year"),
            )
            .drop("censor")
            .cast({"person_id": int})
        )

        df = df.join(background, how="left", on="person_id")

        df_dict[size] = df

    return df_dict
