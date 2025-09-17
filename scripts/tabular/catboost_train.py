import json

import hydra
import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from omegaconf import DictConfig

from src.earlylife.src.paths import FPATH, check_and_copy_file_or_dir
from src.earlylife.src.tabular_models import objective_CatBoost

torch.set_float32_matmul_precision("medium")


@hydra.main(
    config_path=(FPATH.CONFIGS / "tabular").as_posix(),
    config_name="catboost_train.yaml",
    version_base=None,
)
def main(hparams: DictConfig) -> None:
    experiment_name = hparams.experiment_name
    n_trials = hparams.n_trials
    study_name = f"{experiment_name}_{hparams.outcome}"

    subset_size = hparams.subset_size
    if subset_size is not None:
        study_name += f"_{subset_size}"
        experiment_name += f"_{subset_size}"

    study_file = FPATH.OPTUNA / "cb" / f"cb_optuna_study_{study_name}.pkl"

    # Load data
    print("Loading data")
    train_data = pd.read_parquet(
        FPATH.swap_drives(
            FPATH.DATA / hparams.data_folder / f"{hparams.outcome}_train_dense.parquet"
        )
    )
    val_data = pd.read_parquet(
        FPATH.swap_drives(
            FPATH.DATA / hparams.data_folder / f"{hparams.outcome}_val_dense.parquet"
        )
    )
    train_labels = np.load(
        FPATH.swap_drives(
            FPATH.DATA / hparams.data_folder / f"{hparams.outcome}_train_y.npy"
        )
    )
    val_labels = np.load(
        FPATH.swap_drives(
            FPATH.DATA / hparams.data_folder / f"{hparams.outcome}_val_y.npy"
        )
    )

    # Optional subsetting
    if subset_size is not None:
        subset_file = (
            FPATH.DATA
            / hparams.subset_folder
            / "subsets"
            / f"train_idx_{subset_size}.npy"
        )
        check_and_copy_file_or_dir(subset_file)

        if not subset_file.exists():
            raise FileNotFoundError(f"Subset index file not found: {subset_file}")

        subset_idx = np.load(subset_file)
        # train_data = train_data[subset_idx]
        train_data = train_data.iloc[subset_idx]
        train_labels = train_labels[subset_idx]
        print(f"Using subset of size {subset_size}")

    # Load or create study
    if study_file.exists():
        study = joblib.load(study_file)
        print(f"Resuming existing study: {study_file}.")
    else:
        study = optuna.create_study(direction="maximize", study_name=study_name)
        print("Starting a new study.")

    # Load column types
    with open(
        FPATH.swap_drives(FPATH.DATA / hparams.data_folder / "column_types.json"),
        "r",
        encoding="utf-8",
    ) as f:
        column_types = json.load(f)

    # Faster for catboost to have as category than object (https://catboost.ai/docs/en/concepts/speed-up-training#pandas-instead-of-objects)
    for col in column_types["nominal"]:
        train_data[col] = train_data[col].astype("category")
        val_data[col] = val_data[col].astype("category")

    cols_to_drop = []
    for col in column_types["count"]:
        if (train_data[col] == "").all():
            cols_to_drop.append(col)
            print(f"Dropping col {col}")

    non_empty_count_cols = [x for x in column_types["count"] if (x not in cols_to_drop)]
    column_types["count"] = non_empty_count_cols

    train_data = train_data.drop(columns=cols_to_drop)
    val_data = val_data.drop(columns=cols_to_drop)

    top_tokens_count = hparams.top_tokens_count

    # Define objective function
    objective_with_params_set = lambda trial: objective_CatBoost(  # noqa: E731
        trial,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        experiment=experiment_name,
        column_types=column_types,
        top_tokens_count=top_tokens_count,
    )

    # Run optimization
    try:
        study.optimize(objective_with_params_set, n_trials=n_trials)
    except Exception as e:
        print(f"Optimization was interrupted due to {e}. Study saved.")
        joblib.dump(study, study_file)

    # Save the Optuna study to a file
    joblib.dump(study, study_file)

    # Print the best hyperparameters and best validation AUC
    print("Best hyperparameters:", study.best_params)
    print("Best validation AUC:", study.best_value)


if __name__ == "__main__":
    main()
