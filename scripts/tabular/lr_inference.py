import warnings
from pathlib import Path

import hydra
import joblib
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.earlylife.src.paths import FPATH
from src.earlylife.src.tabular_models import LogisticRegression, SparseDataset

torch.set_float32_matmul_precision("medium")

warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    module="lightning_fabric.utilities.cloud_io",
)


@hydra.main(
    config_path=(FPATH.CONFIGS / "tabular").as_posix(),
    config_name="logistic_regression_inference.yaml",
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

    study_file = FPATH.OPTUNA / "lr" / f"lr_optuna_study_{study_name}.pkl"

    print(
        f"Doing inference for experiment {experiment_name}_{hparams.outcome}, datasplit: {data_split}"
    )
    print("Loading data")

    input_data_path = (
        FPATH.DATA
        / hparams.data_folder
        / f"{hparams.outcome}_{data_split}_sparse_matrix.npz"
    )
    outcome_data_path = (
        FPATH.DATA / hparams.data_folder / f"{hparams.outcome}_{data_split}_y.npy"
    )

    # Load best model from study

    study = joblib.load(study_file)
    best_model_path = study.best_trial.user_attrs["best_model_path"]
    model = LogisticRegression.load_from_checkpoint(checkpoint_path=best_model_path)

    # Inference
    data = load_npz(FPATH.swap_drives(input_data_path))
    dataset = SparseDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
    )

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    predictions = trainer.predict(model, dataloaders=dataloader)
    predictions = torch.cat(predictions).detach().cpu().numpy()

    # Save predictions
    output_folder = FPATH.DATA / hparams.output_folder
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / f"lr_preds_{study_name}_{data_split}.npy"

    np.save(output_file, predictions)
    FPATH.alternative_copy_to_opposite_drive(output_file)

    # Evaluate
    y_true = np.load(FPATH.swap_drives(outcome_data_path))
    try:
        auc_score = roc_auc_score(y_true, predictions)
        print(f"ROC AUC Score: {auc_score:.4f}")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
