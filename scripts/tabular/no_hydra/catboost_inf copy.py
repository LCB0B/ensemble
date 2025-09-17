import joblib
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.paths import FPATH

if __name__ == "__main__":

    # Load config
    with open(FPATH.CONFIGS / "tabular" / "catboost_inference.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)

    data_split = hparams["data_split"]  # test or val
    experiment_name = hparams["experiment_name"]
    print(f"Doing inference for experiment {experiment_name} datasplit {data_split}")
    print("Loading data")
    input_data_path = (
        FPATH.DATA
        / hparams["data_folder"]
        / f"{hparams['outcome']}_{data_split}_dense.parquet"
    )
    outcome_data_path = (
        FPATH.DATA / hparams["data_folder"] / f"{hparams['outcome']}_{data_split}_y.npy"
    )

    study_path = FPATH.OPTUNA / "cb" / f"cb_optuna_study_{experiment_name}.pkl"
    study = joblib.load(study_path)
    best_model_path = study.best_trial.user_attrs["best_model_path"]

    # Load the best CatBoost model
    model = CatBoostClassifier()
    model.load_model(best_model_path)

    # Load test data for inference
    data = pd.read_parquet(FPATH.swap_drives(input_data_path))

    # Run inference
    predictions = model.predict_proba(data)[:, 1]

    # Save the predictions

    output_folder = FPATH.DATA / hparams["output_folder"]
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / f"cb_preds_{experiment_name}_{data_split}.npy"

    np.save(output_file, predictions)
    FPATH.alternative_copy_to_opposite_drive(output_file)

    # Evaluate the model
    y_test = np.load(FPATH.swap_drives(outcome_data_path))
    auc_score = roc_auc_score(y_test, predictions)
    print(f"ROC AUC Score: {auc_score:.4f}")
