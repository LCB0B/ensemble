# %%
import numpy as np
import torch

from src.paths import FPATH, check_and_copy_file_or_dir, copy_file_or_dir
from scipy.sparse import load_npz
import optuna
import joblib
from src.tabular_models import objective_LR

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":

    experiment_name = "LR5"
    MAX_EPOCHS = 10
    N_TRIALS = 50
    study_file = FPATH.OPTUNA / f"lr_optuna_study_{experiment_name}.pkl"

    # Load data
    
    print("Loading data")
    train_data = load_npz(FPATH.swap_drives(FPATH.DATA / "train_sparse_matrix.npz"))
    val_data = load_npz(FPATH.swap_drives(FPATH.DATA / "val_sparse_matrix.npz"))
    train_labels = np.load(FPATH.swap_drives(FPATH.DATA / "train_y.npy"))
    val_labels = np.load(FPATH.swap_drives(FPATH.DATA / "val_y.npy"))

    # Load existing study or create a new one
    if study_file.exists():
        study = joblib.load(study_file)
        print("Resuming existing study.")
    else:
        study = optuna.create_study(direction="maximize")
        print("Starting a new study.")

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining_trials = N_TRIALS - completed_trials
    if remaining_trials <= 0:
        print("Study already has the desired number of completed trials.")
        quit()
    else:
        print(f"Starting optimization with {remaining_trials} remaining trials.")
        

    # Define the optimization function
    objective_with_params_set = lambda trial: objective_LR(  # noqa: E731
        trial,
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        experiment_name=experiment_name,
        max_epochs=MAX_EPOCHS,
    )

    try:
        # Optimize and save the study after each trial
        study.optimize(objective_with_params_set, n_trials=remaining_trials, callbacks=[
            lambda study, trial: joblib.dump(study, study_file)
        ])
    except Exception as e:
        print(f"Optimization was interrupted due to {e}. Study saved.")
        joblib.dump(study, study_file)

    # Save the final study after completion
    joblib.dump(study, study_file)

    # Print the best hyperparameters and best validation AUC
    print("Best hyperparameters:", study.best_params)
    print("Best validation AUC:", study.best_value)
    print("Best model path:", study.best_trial.user_attrs["best_model_path"])

