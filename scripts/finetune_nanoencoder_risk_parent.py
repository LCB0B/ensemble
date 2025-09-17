import os
import warnings
import yaml
import torch
import pyarrow.dataset as ds
import polars as pl
from src.datamodule2 import FamilyRiskFinetuneLifeLightningDataModule
from src.encoder_nano_risk import FamilyRiskNanoEncoder
from src.prediction_writer import SaveSimpleInfo
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch._dynamo.config.cache_size_limit = 20
    # Load hparams
    with open(
        FPATH.CONFIGS / "hparams_finetune2.yaml", "r", encoding="utf-8"
    ) as stream:
        hparams = yaml.safe_load(stream)
    num = get_wandb_runid(FPATH.TB_LOGS / hparams["experiment_name"]).split("_")[0]
    run_id = f"{num}_{hparams['dir_path']}-{hparams['outcome'].split('/')[-1].split('_')[1]}-parent-grid"

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

    # TODO: Move all to config
    dm = ParentsRiskFinetuneLifeLightningDataModule(
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

    # The iterations are spread out over the devices, hence the division by devices
    hparams["steps_per_epoch"] = dm.get_steps_per_train_epoch() / N_DEVICES
    hparams["optimizer_max_iters"] = hparams["max_epochs"] * hparams["steps_per_epoch"]

    model = ParentRiskNanoEncoder(**hparams)
    # Load pretrained model
    if ckpt_path := hparams.get("load_pretrained_model"):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt_path = FPATH.CHECKPOINTS_TRANSFORMER / ckpt_path

        model = model.load_from_checkpoint(
            ckpt_path,
            strict=False,
            **hparams,
        )

        # Need to set the optimizer information, else it will use those from pretrained model
        set_hparams_list = [
            "learning_rate",
            "steps_per_epoch",
            "optimizer_max_iters",
            "optimizer_warmup_epochs",
        ]
        for hparam in set_hparams_list:
            model.hparams[hparam] = hparams[hparam]
        optimizers_and_schedulers = model.configure_optimizers()

        run_id += f"-ckpt_{ckpt_path.parts[-2]}"
    print(f"Experiment: {hparams['experiment_name']} / {run_id}")

    if hparams.get("retry_checkpoint"):
        print("Loading from checkpoint:", hparams["retry_checkpoint"])
        model = ParentRiskNanoEncoder.load_from_checkpoint(
            FPATH.CHECKPOINTS_TRANSFORMER / hparams["retry_checkpoint"], strict=True
        )

    # Trainer setup
    logger = TensorBoardLogger(
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
        checkpoint_callback,
    ]
    # Prediction writing
    if hparams.get("save_preds"):
        pred_writer = SaveSimpleInfo(
            fname=(
                FPATH.FPATH_PROJECT
                / "data"
                / "preds"
                / hparams["experiment_name"]
                / run_id
            ).with_suffix(".pt")
        )
        callbacks.append(pred_writer)

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
        # fast_dev_run=50,
    )

    # Train
    trainer.fit(model, datamodule=dm)
    trainer.predict(model, datamodule=dm, ckpt_path="best")
