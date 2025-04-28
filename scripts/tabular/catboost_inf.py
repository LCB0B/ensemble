import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import joblib
from src.paths import FPATH, copy_file_or_dir

if __name__ == "__main__":
    
    data_split = "val" # test or val
    experiment = "CB2"
    print(f"Doing inference for experiment {experiment} datasplit {data_split}")
    print("Loading data")
    input_data_path = FPATH.DATA / f"{data_split}_dense.parquet"
    outcome_data_path = FPATH.DATA / f"{data_split}_y.npy"

    copy_file_or_dir(input_data_path)
    copy_file_or_dir(outcome_data_path)

    study = joblib.load(FPATH.OPTUNA / f"{experiment}_optuna_study.pkl")
    best_model_path = study.best_trial.user_attrs["best_model_path"]

    # Load the best CatBoost model
    model = CatBoostClassifier()
    model.load_model(best_model_path)

    # Load test data for inference
    data = pd.read_parquet(input_data_path)

    # Run inference
    predictions = model.predict_proba(data)[:, 1]

    # Save the predictions
    output_file_path = FPATH.DATA / f"cb_preds_{experiment}_{data_split}.npy"
    np.save(output_file_path, predictions)
    FPATH.alternative_copy_to_opposite_drive(output_file_path)

    # Evaluate the model
    y_test = np.load(outcome_data_path)
    auc_score = roc_auc_score(y_test, predictions)
    print(f"ROC AUC Score: {auc_score:.4f}")
