import hydra
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from src.paths import FPATH


@hydra.main(
    config_path=(FPATH.CONFIGS / "socmob" / "tabular").as_posix(),
    config_name="catboost_inference.yaml",
    version_base=None,
)
def main(hparams: DictConfig) -> None:
    data_split = hparams.data_split  # test or val
    experiment_name = hparams.experiment_name
    study_name = f"{experiment_name}_{hparams.outcome}"

    subset_size = hparams.subset_size
    if subset_size is not None:
        study_name += f"_{subset_size}"
        experiment_name += f"_{subset_size}"

    study_file = FPATH.OPTUNA / "cb" / f"cb_optuna_study_{study_name}.pkl"

    print(f"Doing inference for experiment {study_name}, datasplit: {data_split}")
    print("Loading data")

    input_data_path = (
        FPATH.DATA
        / hparams.data_folder
        / f"{hparams.outcome}_{data_split}_dense.parquet"
    )

    # Load Optuna study and best model path
    study = joblib.load(study_file)
    best_model_path = study.best_trial.user_attrs["best_model_path"]

    # Load the best CatBoost model
    model = CatBoostClassifier()
    model.load_model(best_model_path)

    # Load test data for inference
    data = pd.read_parquet(FPATH.swap_drives(input_data_path))

    # Run inference
    predictions = model.predict_proba(data)[:, 1]

    # Save the predictions
    output_folder = FPATH.DATA / hparams.output_folder
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / f"cb_preds_{study_name}_{data_split}.npy"

    np.save(output_file, predictions)
    FPATH.alternative_copy_to_opposite_drive(output_file)

    # Evaluate the model

    outcome_data_path = (
        FPATH.DATA / hparams.data_folder / f"{hparams.outcome}_{data_split}_y.npy"
    )
    y_true = np.load(FPATH.swap_drives(outcome_data_path))
    try:
        auc_score = roc_auc_score(y_true, predictions)
        print(f"ROC AUC Score: {auc_score:.4f}")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
