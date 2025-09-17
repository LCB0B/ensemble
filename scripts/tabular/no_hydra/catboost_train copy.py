# %%
import json

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import yaml

from src.paths import FPATH
from src.tabular_models import objective_CatBoost

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":

    with open(FPATH.CONFIGS / "tabular" / "catboost_train.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)

    experiment_name = hparams["experiment_name"]
    n_trials = hparams["n_trials"]
    study_file = FPATH.OPTUNA / "cb" / f"cb_optuna_study_{experiment_name}.pkl"
    min_df = hparams["min_df"]
    # Load data
    print("Loading data")
    train_data = pd.read_parquet(
        FPATH.swap_drives(
            FPATH.DATA
            / hparams["data_folder"]
            / f"{hparams['outcome']}_train_dense.parquet"
        )
    )
    val_data = pd.read_parquet(
        FPATH.swap_drives(
            FPATH.DATA
            / hparams["data_folder"]
            / f"{hparams['outcome']}_val_dense.parquet"
        )
    )
    train_labels = np.load(
        FPATH.swap_drives(
            FPATH.DATA / hparams["data_folder"] / f"{hparams['outcome']}_train_y.npy"
        )
    )
    val_labels = np.load(
        FPATH.swap_drives(
            FPATH.DATA / hparams["data_folder"] / f"{hparams['outcome']}_val_y.npy"
        )
    )

    # Load or create study
    if study_file.exists():
        study = joblib.load(study_file)
        print("Resuming existing study.")
    else:
        study = optuna.create_study(direction="maximize", study_name=experiment_name)
        print("Starting a new study.")

    with open(
        FPATH.swap_drives(FPATH.DATA / hparams["data_folder"] / "column_types.json"),
        "r",
        encoding="utf-8",
    ) as f:
        column_types = json.load(f)

    min_occurence = int(min_df * train_data.shape[0])

    objective_with_params_set = lambda trial: objective_CatBoost(  # noqa: E731
        trial,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        experiment=experiment_name,
        column_types=column_types,
        min_occurence=min_occurence,
    )

    study = optuna.create_study(direction="maximize")
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
