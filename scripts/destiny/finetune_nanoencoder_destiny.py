import os
import torch._dynamo.compiled_autograd
import torch
import pyarrow.dataset as ds
import polars as pl
from pathlib import PosixPath
import hydra
from omegaconf import DictConfig, open_dict  # noqa: E402

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch import seed_everything
from src.datamodule2 import PredictFinetuneLifeLDM
from src.encoder_nano_risk import PredictionFinetuneNanoEncoder
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid
from src.loggers import RetryOrSkipTensorBoardLogger
from src.prediction_writer import SaveSimpleInfo
from src.utils import (
    LegacyTorchCheckpointIO,
    auto_ntokens,
    get_n_steps_per_epoch,
    set_warmup_steps,
)


@hydra.main(
    config_path=f"{FPATH.CONFIGS.as_posix()}/destiny",
    config_name="hparams_destiny_finetune.yaml",
    version_base=None,
)
def main(hparams: DictConfig):
    torch.serialization.add_safe_globals([PosixPath])
    torch._dynamo.config.cache_size_limit = 16

    run_id = f"{get_wandb_runid(FPATH.TB_LOGS / hparams.experiment_name)}-{hparams.dir_path}-{hparams.outcome.split('/')[1]}"
    seed_everything(73)
    # Set training variables
    torch.set_float32_matmul_precision(hparams.float32_matmul_precision)
    N_DEVICES = (
        len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if torch.cuda.is_available()
        else 1
    )

    #### Data ####
    source_paths = [
        (FPATH.DATA / hparams.source_dir / path).with_suffix(".parquet")
        for path in hparams.sources
    ]
    background_path = (
        FPATH.DATA / hparams.source_dir / hparams.background
    ).with_suffix(".parquet")
    outcomes_path = (FPATH.DATA / hparams.source_dir / hparams.outcome).with_suffix(
        ".parquet"
    )
    cohort_paths = {
        key: (FPATH.DATA / hparams.source_dir / cohort).with_suffix(".parquet")
        for key, cohort in hparams["cohorts"].items()
    }
    for s in (
        source_paths + [background_path, outcomes_path] + list(cohort_paths.values())
    ):
        check_and_copy_file_or_dir(s, verbosity=2)

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    outcomes = pl.read_parquet(outcomes_path)
    cohorts = {
        key: pl.read_parquet(path, columns=["person_id"])
        for key, path in cohort_paths.items()
    }

    dm = PredictFinetuneLifeLDM(
        dir_path=FPATH.DATA / hparams.dir_path,
        sources=sources,
        cohorts=cohorts,
        outcomes=outcomes,
        background=background,
        subset_background=hparams.subset_background,
        n_tokens=hparams.n_tokens,
        lengths=hparams.lengths,
        num_workers=hparams.num_workers,
        max_seq_len=hparams.max_seq_len,
        prediction_windows=hparams.prediction_windows,
        source_dir=hparams.source_dir,
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    with open_dict(hparams):
        hparams.vocab_size = len(dm.pipeline.vocab)

    # Load checkpoint if fine-tuning
    model = PredictionFinetuneNanoEncoder(**hparams)
    if ckpt_path := hparams.get("load_pretrained_model"):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt_path = FPATH.CHECKPOINTS_TRANSFORMER / ckpt_path

        model = model.load_from_checkpoint(
            ckpt_path,
            strict=False,
            **hparams,
        )

        # Need to set the optimizer information, else it will use those from pretrained model
        set_hparams_list = ["learning_rate", "warmup_steps"]
        for hparam in set_hparams_list:
            model.hparams[hparam] = hparams[hparam]
        optimizers_and_schedulers = model.configure_optimizers()
        run_id += f"-ckpt_{ckpt_path.parts[-2]}"

    # Trainer setup
    logger = RetryOrSkipTensorBoardLogger(
        save_dir=FPATH.TB_LOGS / hparams.experiment_name,
        name="",
        version=run_id,
        default_hp_metric=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=FPATH.CHECKPOINTS_TRANSFORMER / hparams["experiment_name"] / run_id,
        filename="{epoch}-{AUROC_mean:.3f}-{PRAUC_mean:.3f}",
        monitor="AUROC_mean",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="AUROC_mean",
        mode="max",
        patience=2,
    )
    pred_writer = SaveSimpleInfo(
        fname=(
            FPATH.NETWORK_DATA / "preds" / hparams.experiment_name / run_id
        ).with_suffix(".pt")
    )

    callbacks = [lr_monitor, checkpoint_callback, pred_writer, early_stop]
    print(f"Info: {hparams.outcome}")
    print(f"Info: experiment {hparams['experiment_name']} / {run_id}")
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="auto",
        devices=N_DEVICES,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=False,
        precision=hparams.precision,
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        plugins=[LegacyTorchCheckpointIO()],
        # profiler="simple",
        # fast_dev_run=200,
        # val_check_interval=2000,
    )
    if hparams.auto_ntokens:
        auto_ntokens(
            trainer, model, dm, init_val=200_000, steps_per_trial=50, max_trials=5
        )

    if dm.train_dataset is None:
        dm.setup()

    n_steps = get_n_steps_per_epoch(dm)

    if "warmup_epochs" in hparams:
        set_warmup_steps(model, n_steps, hparams.warmup_epochs)

    # Train
    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
