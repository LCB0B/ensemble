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
from src.encoder_nano_risk import PredictionFinetuneNanoEncoder
from src.paths import FPATH, check_and_copy_file_or_dir, get_wandb_runid
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from src.loggers import RetryOrSkipTensorBoardLogger
from lightning.pytorch import seed_everything
from src.prediction_writer import SaveSimpleInfo


@hydra.main(
    config_path=f"{FPATH.CONFIGS.as_posix()}/destiny",
    config_name="hparams_destiny_finetune.yaml",
    version_base=None,
)
def main(hparams: DictConfig):
    torch.serialization.add_safe_globals([PosixPath])
    torch._dynamo.config.cache_size_limit = 16

    run_id = f"{get_wandb_runid(FPATH.TB_LOGS / hparams.experiment_name)}-{hparams.dir_path}-{hparams.outcome['train'].split('/')[1]}"
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
    outcomes_path = {
        key: (FPATH.DATA / hparams.source_dir / cohort).with_suffix(".parquet")
        for key, cohort in hparams.outcome.items()
    }

    for s in source_paths + [background_path] + list(outcomes_path.values()):
        check_and_copy_file_or_dir(s, verbosity=2)

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    outcomes = {key: pl.read_parquet(path) for key, path in outcomes_path.items()}

    dm = PredictFinetuneLifeLDM(
        dir_path=FPATH.DATA / hparams.dir_path,
        sources=sources,
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
        dirpath=FPATH.CHECKPOINTS_TRANSFORMER / hparams.experiment_name / run_id,
        filename="best",
        save_top_k=1,
        save_last=True,
    )
    pred_writer = SaveSimpleInfo(
        fname=(
            FPATH.NETWORK_DATA / "preds" / hparams.experiment_name / run_id
        ).with_suffix(".pt")
    )

    callbacks = [lr_monitor, checkpoint_callback, pred_writer]

    print(f"Info Experiment: {hparams['experiment_name']} / {run_id}")
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
        # profiler=profiler,
        # fast_dev_run=200,
    )

    # Train

    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, datamodule=dm)
    # trainer.predict(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
