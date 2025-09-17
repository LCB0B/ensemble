import os
import warnings
from pathlib import PosixPath

import polars as pl
import pyarrow.dataset as ds
import torch
import torch._dynamo.compiled_autograd
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.earlylife.src.datamodule2 import ScreenRiskFinetuneLifeLightningDataModule
from src.earlylife.src.loggers import RetryOrSkipTensorBoardLogger  # noqa: E402
from src.earlylife.src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid
from src.earlylife.src.q_learning import EnvelopeQScreeningModel, QScreeningModel

if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch._dynamo.config.cache_size_limit = 16
    # Load hparams
    with open(
        FPATH.CONFIGS / "hparams_qlearning.yaml", "r", encoding="utf-8"
    ) as stream:
        hparams = yaml.safe_load(stream)
    # run_id = f"{get_wandb_runid(FPATH.TB_LOGS / hparams['experiment_name'])}-simple-{hparams['outcome'].split('/')[1]}-grid"
    num = get_wandb_runid(FPATH.TB_LOGS / hparams["experiment_name"]).split("_")[0]
    run_id = f"{num}_{hparams['dir_path']}-{hparams['outcome'].split('/')[-1].split('_')[1]}-qlearning"

    seed_everything(73)
    # Set training variables
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    N_DEVICES = (
        len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if torch.cuda.is_available()
        else 1
    )

    #### Data ####
    source_paths = [
        (FPATH.DATA / path).with_suffix(".parquet") for path in hparams["sources"]
    ]
    background_path = (FPATH.DATA / hparams["background"]).with_suffix(".parquet")
    outcomes_path = (FPATH.DATA / hparams["outcome"]).with_suffix(".parquet")

    for s in source_paths + [background_path, outcomes_path]:
        check_and_copy_file_or_dir(s, verbosity=2)

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    outcomes = pl.read_parquet(outcomes_path)

    dm = ScreenRiskFinetuneLifeLightningDataModule(
        dir_path=FPATH.DATA / hparams["dir_path"],
        sources=sources,
        outcomes=outcomes,
        background=background,
        cls_token=hparams["cls_token"],
        sep_token=hparams["sep_token"],
        segment=hparams["segment"],
        subset_background=hparams["subset_background"],
        n_tokens=hparams["n_tokens"],
        lengths=hparams["lengths"],
        num_workers=hparams["num_workers"],
        max_seq_len=hparams["max_seq_len"],
        prediction_windows=hparams["prediction_windows"],
        negative_censor=hparams["negative_censor"],
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    hparams["vocab_size"] = len(dm.pipeline.vocab)

    model = QScreeningModel(**hparams)
    # model = EnvelopeQScreeningModel(**hparams)
    if isinstance(model, EnvelopeQScreeningModel):
        run_id += "-envelope"
    else:
        run_id += f"-alpha{hparams['alpha']}"
    torch.serialization.add_safe_globals([PosixPath])
    model.risk_model.load_state_dict(
        torch.load(
            FPATH.CHECKPOINTS_TRANSFORMER / hparams["checkpoint"] / "last.ckpt",
            weights_only=False,
        )["state_dict"]
    )

    print(f"Experiment: {hparams['experiment_name']} / {run_id}")

    # Trainer setup
    logger = RetryOrSkipTensorBoardLogger(
        save_dir=FPATH.TB_LOGS / hparams["experiment_name"],
        name="",
        version=run_id,
        default_hp_metric=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=FPATH.CHECKPOINTS_TRANSFORMER / hparams["experiment_name"] / run_id,
        filename="best",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        lr_monitor,
        # checkpoint_callback,
    ]

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",
        devices=N_DEVICES,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=False,
        precision=hparams["precision"],
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        val_check_interval=500,
        # limit_train_batches=1000,
        # limit_val_batches=1000,
        fast_dev_run=1000,
    )

    # Train
    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm, ckpt_path="best")
