import warnings
from pathlib import Path

import hydra
import joblib
import numpy as np
import optuna
import torch
from omegaconf import DictConfig
from scipy.sparse import load_npz

from src.earlylife.src.paths import FPATH, check_and_copy_file_or_dir
from src.earlylife.src.tabular_models import objective_LR

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    module="lightning_fabric.utilities.cloud_io",
)


@hydra.main(
    config_path=(FPATH.CONFIGS / "tabular").as_posix(),
    config_name="logistic_regression_train.yaml",
    version_base=None,
)
def main(hparams: DictConfig) -> None:
    experiment_name = hparams.experiment_name
    max_epochs = hparams.max_epochs
    n_trials = hparams.n_trials
    study_name = f"{experiment_name}_{hparams.outcome}"

    subset_size = hparams.subset_size
    if subset_size is not None:
        study_name += f"_{subset_size}"
        experiment_name += f"_{subset_size}"

    study_file = FPATH.OPTUNA / "lr" / f"lr_optuna_study_{study_name}.pkl"

    # Load data
    print("Loading data")
    train_data = load_npz(
        FPATH.swap_drives(
            FPATH.DATA
            / hparams.data_folder
            / f"{hparams.outcome}_train_sparse_matrix.npz"
        )
    )
    val_data = load_npz(
        FPATH.swap_drives(
            FPATH.DATA
            / hparams.data_folder
            / f"{hparams.outcome}_val_sparse_matrix.npz"
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

    subset_size = hparams.get("subset_size", None)

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
        train_data = train_data[subset_idx]
        train_labels = train_labels[subset_idx]
        print(f"Using subset of size {subset_size}")

    # Load or create study
    if study_file.exists():
        study = joblib.load(study_file)
        print(f"Resuming existing study: {study_file}.")
    else:
        study = optuna.create_study(direction="maximize", study_name=study_name)
        print("Starting a new study.")

    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining_trials = n_trials - completed_trials
    if remaining_trials <= 0:
        print("Study already has the desired number of completed trials.")
        return
    else:
        print(f"Starting optimization with {remaining_trials} remaining trials.")

    # Optimization
    objective_with_params_set = lambda trial: objective_LR(  # noqa: E731
        trial,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        experiment_name=experiment_name,
        max_epochs=max_epochs,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
    )

    try:
        study.optimize(
            objective_with_params_set,
            n_trials=remaining_trials,
            callbacks=[lambda study, trial: joblib.dump(study, study_file)],
        )
    except Exception as e:
        print(f"Optimization was interrupted due to {e}. Study saved.")
        joblib.dump(study, study_file)

    joblib.dump(study, study_file)

    print("Best hyperparameters:", study.best_params)
    print("Best validation AUC:", study.best_value)
    print("Best model path:", study.best_trial.user_attrs["best_model_path"])


if __name__ == "__main__":
    main()
