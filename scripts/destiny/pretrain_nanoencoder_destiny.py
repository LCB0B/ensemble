import os
import yaml  # noqa: E402
import torch  # noqa: E402
import polars as pl  # noqa: E402
from pathlib import PosixPath
from src.datamodule2 import PretrainLDM  # noqa: E402
from src.encoder_nano_risk import PretrainNanoEncoder  # noqa: E402
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid  # noqa: E402
from lightning.pytorch import Trainer  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from src.loggers import RetryOrSkipTensorBoardLogger  # noqa: E402
from lightning.pytorch import seed_everything  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402


if __name__ == "__main__":
    torch.serialization.add_safe_globals([PosixPath])
    # Load hparams
    with open(
        FPATH.CONFIGS / "destiny" / "hparams_destiny_pretrain.yaml",
        "r",
        encoding="utf-8",
    ) as stream:
        hparams = yaml.safe_load(stream)
    run_id = (
        get_wandb_runid(FPATH.TB_LOGS / hparams["experiment_name"])
        + f"-pretrain-{hparams['dir_path']}"
    )

    seed_everything(73)
    print(f"Experiment: {hparams['experiment_name']} / {run_id}")

    # Set training variables
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    #### Data ####
    source_paths = [
        (FPATH.DATA / hparams["source_dir"] / path).with_suffix(".parquet")
        for path in hparams["sources"]
    ]
    background_path = (
        FPATH.DATA / hparams["source_dir"] / hparams["background"]
    ).with_suffix(".parquet")
    cohort_paths = {
        key: (FPATH.DATA / hparams["source_dir"] / cohort).with_suffix(".parquet")
        for key, cohort in hparams["cohorts"].items()
    }

    for s in source_paths + [background_path] + list(cohort_paths.values()):
        check_and_copy_file_or_dir(s, verbosity=2)

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    cohorts = {
        key: pl.read_parquet(path, columns=["person_id"])
        for key, path in cohort_paths.items()
    }

    dm = PretrainLDM(
        dir_path=FPATH.DATA / hparams["dir_path"],
        sources=sources,
        background=background,
        cohorts=cohorts,
        subset_background=hparams["subset_background"],
        n_tokens=hparams["n_tokens"],
        lengths=hparams["lengths"],
        num_workers=hparams["num_workers"],
        max_seq_len=hparams["max_seq_len"],
        source_dir=hparams["source_dir"],
    )

    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    if "warmup_epochs" in hparams:
        dm.setup()
        sampler = dm.get_sampler(dm.train_dataset)
        avg_batch_size = hparams["n_tokens"] / (
            sum(sampler.lengths) / len(sampler.lengths)
        )
        n_steps = len(dm.train_dataset) / avg_batch_size
        print(f"Info: Steps ~{int(n_steps)} per epoch")

    # get vocab size
    hparams["vocab_size"] = len(dm.pipeline.vocab)
    if "warmup_epochs" in hparams:
        hparams["warmup_steps"] = int(n_steps * hparams["warmup_epochs"])
        print(
            f"Info: Warmup is {hparams['warmup_steps']} steps ({hparams['warmup_epochs']} epochs)"
        )

    # Model
    model = PretrainNanoEncoder(**hparams)

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
        monitor="val/loss",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=2,
    )
    from lightning.pytorch.callbacks import ModelSummary

    callbacks = [lr_monitor, checkpoint_callback, early_stop, ModelSummary()]

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",
        devices=n_devices,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=False,
        precision=hparams["precision"],
        log_every_n_steps=100,
        fast_dev_run=500,
        # profiler="simple",
    )

    # Train
    trainer.fit(model, datamodule=dm)
    # print(torch.cuda.memory_summary())
    # trainer.validate(model, datamodule=dm)
