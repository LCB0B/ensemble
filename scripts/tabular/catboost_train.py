# %%
import numpy as np
import torch
import pandas as pd
from src.paths import FPATH, copy_file_or_dir
import optuna
import joblib
from src.tabular_models import objective_CatBoost

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    experiment = "CB2"
    N_TRIALS = 50
    # Load data
    print("Loading data")
    # copy_file_or_dir(FPATH.DATA / "train_dense.parquet")
    # copy_file_or_dir(FPATH.DATA / "val_dense.parquet")
    # copy_file_or_dir(FPATH.DATA / "train_y.npy")
    # copy_file_or_dir(FPATH.DATA / "val_y.npy")
    # Only read once -> Read straight from network
    train_data = pd.read_parquet(FPATH.swap_drives(FPATH.DATA / "train_dense.parquet"))
    val_data = pd.read_parquet(FPATH.swap_drives(FPATH.DATA / "val_dense.parquet"))
    train_labels = np.load(FPATH.swap_drives(FPATH.DATA / "train_y.npy"))
    val_labels = np.load(FPATH.swap_drives(FPATH.DATA / "val_y.npy"))

    objective_with_params_set = lambda trial: objective_CatBoost(  # noqa: E731
        trial,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        experiment=experiment,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_with_params_set, n_trials=N_TRIALS)  # Adjust n_trials as needed

    # Save the Optuna study to a file
    joblib.dump(study, FPATH.OPTUNA / f"{experiment}_optuna_study.pkl")

    # Print the best hyperparameters and best validation AUC
    print("Best hyperparameters:", study.best_params)
    print("Best validation AUC:", study.best_value)
