import os
import torch._dynamo.compiled_autograd
import yaml
import torch
import pyarrow.dataset as ds
import polars as pl

from src.datamodule2 import (
    RiskFinetuneLifeLightningDataModule,
    ParentsRiskFinetuneLifeLightningDataModule,
)
from src.encoder_nano_risk import RiskNanoEncoder, ParentRiskNanoEncoder
from src.prediction_writer import SaveSimpleInfo, SaveSelectiveInfo
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 16
    # Load hparams
    with open(
        FPATH.CONFIGS / "hparams_finetune2.yaml", "r", encoding="utf-8"
    ) as stream:
        hparams = yaml.safe_load(stream)
    # assert hparams["outcome"].split("_")[-1] in hparams["checkpoint"]
    run_id = hparams["checkpoint"].split("/")[1]
    print(f"Checkpoint: {hparams['experiment_name']} / {run_id}")

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

    if "parent" in hparams["checkpoint"].split("/")[1]:
        dm_cls = ParentsRiskFinetuneLifeLightningDataModule
    else:
        dm_cls = RiskFinetuneLifeLightningDataModule
    dm = dm_cls(
        dir_path=FPATH.DATA / hparams["dir_path"],
        sources=sources,
        outcomes=outcomes,
        background=background,
        cls_token=hparams["cls_token"],
        sep_token=hparams["sep_token"],
        segment=hparams["segment"],
        subset_background=hparams["subset_background"],
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        max_seq_len=hparams["max_seq_len"],
        prediction_windows=hparams["prediction_windows"],
        negative_censor=hparams["negative_censor"],
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    hparams["vocab_size"] = len(dm.pipeline.vocab)

    # Load checkpoint if fine-tuning
    if "parent" in hparams["checkpoint"].split("/")[1]:
        model_cls = ParentRiskNanoEncoder
    else:
        model_cls = RiskNanoEncoder
    model = model_cls(**hparams)

    # Trainer setup
    logger = TensorBoardLogger(
        save_dir=FPATH.TB_LOGS / "risk_inf",
        name="",
        version=run_id,
        default_hp_metric=False,
    )
    pred_writer = SaveSimpleInfo(
        fname=(
            FPATH.FPATH_PROJECT / "data" / "preds" / hparams["experiment_name"] / run_id
        ).with_suffix(".pt")
    )

    checkpoint_path = FPATH.CHECKPOINTS_TRANSFORMER / hparams["checkpoint"]

    model = model.load_from_checkpoint(checkpoint_path)
    model.eval()

    callbacks = [pred_writer]

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",
        devices=N_DEVICES,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=False,
        precision=hparams["precision"],
        log_every_n_steps=1000,
    )

    # Train
    trainer.predict(model, datamodule=dm, return_predictions=False)
