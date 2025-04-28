import os
import yaml
import torch
import warnings
import pyarrow.dataset as ds
import polars as pl

from src.datamodule2 import (
    RiskFinetuneLifeLightningDataModule,
    ParentsRiskFinetuneLifeLightningDataModule,
)
from src.encoder_nano_risk import RiskNanoEncoder, ParentRiskNanoEncoder
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
)

# This is an erroneous warning, the mask is indeed already bool
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.",
    category=UserWarning,
    module="torch.nn.modules.activation",
)

if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 16
    # Load hparams
    with open(
        FPATH.CONFIGS / "hparams_finetune2.yaml", "r", encoding="utf-8"
    ) as stream:
        hparams = yaml.safe_load(stream)
    run_id = f"{get_wandb_runid(FPATH.TB_LOGS / hparams['experiment_name'])}-{hparams['outcome'].split('_')[1]}-parents-newrope_long"
    print(f"Experiment: {hparams['experiment_name']} / {run_id}")

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
    parents_path = FPATH.DATA / "health" / "parents.parquet"

    for s in source_paths + [background_path, outcomes_path, parents_path]:
        check_and_copy_file_or_dir(s)

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    outcomes = pl.read_parquet(outcomes_path)
    parents = pl.read_parquet(parents_path)

    # TODO: Move all to config
    dm = ParentsRiskFinetuneLifeLightningDataModule(
        dir_path=FPATH.DATA / hparams["dir_path"],
        sources=sources,
        outcomes=outcomes,
        parents=parents,
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

    # Load checkpoint if fine-tuning
    model = ParentRiskNanoEncoder(**hparams)

    if hparams["compile"]:
        model = torch.compile(model, dynamic=False)
        print("Model has been compiled")

    # Trainer setup
    logger = TensorBoardLogger(
        save_dir=FPATH.TB_LOGS / hparams["experiment_name"],
        name="",
        version=run_id,
        default_hp_metric=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=FPATH.CHECKPOINTS / hparams["experiment_name"] / run_id,
        filename="best",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        lr_monitor,
        checkpoint_callback,
    ]

    # profiler = SimpleProfiler(filename="simple_profiler")

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",
        devices=N_DEVICES,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=False,
        precision=hparams["precision"],
        log_every_n_steps=50,
        # profiler=profiler,
    )

    # Train
    trainer.fit(model, datamodule=dm)
