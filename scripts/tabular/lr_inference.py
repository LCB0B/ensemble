import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.paths import FPATH, copy_file_or_dir
from scipy.sparse import load_npz
import joblib
from src.tabular_models import LogisticRegression, SparseDataset
import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    # Load the saved Optuna study and retrieve the best hyperparameters and model path

    data_split = "val" # test or val
    experiment = "LR5"
    print(f"Doing inference for experiment {experiment} datasplit {data_split}")
    print("Loading data")
    input_data_path = FPATH.DATA / f"{data_split}_sparse_matrix.npz"
    outcome_data_path = FPATH.DATA / f"{data_split}_y.npy"
    
    copy_file_or_dir(input_data_path)
    copy_file_or_dir(outcome_data_path)

    study = joblib.load(FPATH.OPTUNA / f"lr_optuna_study_{experiment}.pkl")

    best_model_path = study.best_trial.user_attrs["best_model_path"]

    # Load the model from the checkpoint without specifying hyperparameters
    model = LogisticRegression.load_from_checkpoint(checkpoint_path=best_model_path)

    # Load test data for inference
    data = load_npz(input_data_path)
    dataset = SparseDataset(data)  # Dummy labels
    dataloader = DataLoader(
        dataset, batch_size=2048, shuffle=False, num_workers=32
    )

    # Run inference
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=dataloader)

    # Save the predictions
    predictions = torch.cat(predictions).detach().cpu().numpy()
    output_file_name = FPATH.DATA / f"lr_preds_{experiment}_{data_split}.npy"
    np.save(output_file_name, predictions)
    FPATH.alternative_copy_to_opposite_drive(output_file_name)

    # Evaluate the model
    y_true = np.load(outcome_data_path)
    auc_score = roc_auc_score(y_true, predictions)
    print(f"ROC AUC Score: {auc_score:.4f}")
