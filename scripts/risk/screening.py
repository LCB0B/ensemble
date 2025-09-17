import os
import torch._dynamo.compiled_autograd
import yaml
import torch
import pyarrow.dataset as ds
import polars as pl
from pathlib import PosixPath
import hydra
from omegaconf import DictConfig, open_dict  # noqa: E402

from src.datamodule2 import PredictFinetuneLifeLDM
from src.screening_models import (
    AgeScreeningModel,
    RiskScreeningModel,
    RandomScreeningModel,
)
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid
from lightning.pytorch import Trainer
from src.loggers import RetryOrSkipTensorBoardLogger
from lightning.pytorch import seed_everything


@hydra.main(
    config_path=f"{FPATH.CONFIGS.as_posix()}/risk",
    config_name="hparams_screen.yaml",
    version_base=None,
)
def main(hparams: DictConfig):
    torch.serialization.add_safe_globals([PosixPath])
    torch._dynamo.config.cache_size_limit = 16

    number = get_wandb_runid(FPATH.TB_LOGS / hparams.experiment_name).split("_")[0]
    run_id = f"{number}_{hparams.dir_path}-{hparams.outcome['train'].split('_')[-2]}"
    seed_everything(73)

    # Set training variables
    torch.set_float32_matmul_precision(hparams.float32_matmul_precision)
    N_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    background_path = (
        FPATH.DATA / hparams.source_dir / hparams.background
    ).with_suffix(".parquet")
    outcomes_path = {
        key: (FPATH.DATA / hparams.source_dir / cohort).with_suffix(".parquet")
        for key, cohort in hparams.outcome.items()
    }

    for s in [background_path] + list(outcomes_path.values()):
        check_and_copy_file_or_dir(s, verbosity=2)

    background = pl.read_parquet(background_path)
    outcomes = {key: pl.read_parquet(path) for key, path in outcomes_path.items()}

    dm = PredictFinetuneLifeLDM(
        dir_path=FPATH.DATA / hparams.dir_path,
        sources=None,
        outcomes=outcomes,
        background=background,
        subset_background=None,
        n_tokens=hparams.n_tokens,
        lengths=hparams.lengths,
        num_workers=hparams.num_workers,
        max_seq_len=hparams.max_seq_len,
        prediction_windows=hparams.prediction_windows,
        source_dir=None,
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    with open_dict(hparams):
        hparams.vocab_size = len(dm.pipeline.vocab)

    if hparams.model_type == "age":
        model = AgeScreeningModel(**hparams)
        run_id += (
            f"-age{hparams['screening_age_lower']}-{hparams['screening_age_upper']}"
        )
    elif hparams.model_type == "risk":
        model = RiskScreeningModel(**hparams)
        run_id += f"-risk-{hparams['risk_threshold']}"
        model.risk_model.load_state_dict(
            torch.load(
                FPATH.CHECKPOINTS_TRANSFORMER / hparams.checkpoint,
                weights_only=False,
            )["state_dict"]
        )
        run_id += f"-{hparams['checkpoint'].split('/')[1].split('-')[0]}"

    print(f"Experiment: {hparams['experiment_name']} / {run_id}")

    # Trainer setup
    logger = RetryOrSkipTensorBoardLogger(
        save_dir=FPATH.TB_LOGS / hparams.experiment_name,
        name="",
        version=run_id,
        default_hp_metric=False,
    )

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="auto",
        devices=N_DEVICES,
        callbacks=None,
        logger=logger,
        strategy="auto",
        deterministic=False,
        precision=hparams.precision,
        log_every_n_steps=1000,
        # fast_dev_run=10,
    )

    # Train
    # trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
