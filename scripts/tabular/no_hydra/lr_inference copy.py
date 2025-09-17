import joblib
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.paths import FPATH
from src.tabular_models import LogisticRegression, SparseDataset

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    # Load config
    with open(
        FPATH.CONFIGS / "tabular" / "logistic_regression_inference.yaml", "r"
    ) as stream:
        hparams = yaml.safe_load(stream)

    data_split = hparams["data_split"]  # test or val
    experiment_name = hparams["experiment_name"]

    print(f"Doing inference for experiment {experiment_name}, datasplit: {data_split}")
    print("Loading data")

    input_data_path = (
        FPATH.DATA
        / hparams["data_folder"]
        / f"{hparams['outcome']}_{data_split}_sparse_matrix.npz"
    )
    outcome_data_path = (
        FPATH.DATA / hparams["data_folder"] / f"{hparams['outcome']}_{data_split}_y.npy"
    )

    # Load best model from study
    study_path = FPATH.OPTUNA / "lr" / f"lr_optuna_study_{experiment_name}.pkl"
    study = joblib.load(study_path)
    best_model_path = study.best_trial.user_attrs["best_model_path"]
    model = LogisticRegression.load_from_checkpoint(checkpoint_path=best_model_path)

    # Inference
    data = load_npz(FPATH.swap_drives(input_data_path))
    dataset = SparseDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
    )

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    predictions = trainer.predict(model, dataloaders=dataloader)
    predictions = torch.cat(predictions).detach().cpu().numpy()

    # Save predictions

    output_folder = FPATH.DATA / hparams["output_folder"]
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / f"lr_preds_{experiment_name}_{data_split}.npy"

    np.save(output_file, predictions)
    FPATH.alternative_copy_to_opposite_drive(output_file)

    # Evaluate
    y_true = np.load(FPATH.swap_drives(outcome_data_path))
    auc_score = roc_auc_score(y_true, predictions)
    print(f"ROC AUC Score: {auc_score:.4f}")
