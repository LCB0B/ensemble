# %%
import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT


from datetime import datetime  # noqa: E402
from pathlib import PosixPath

import hydra  # noqa: E402
import polars as pl
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import torch
import torch._dynamo.compiled_autograd
import yaml
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, open_dict  # noqa: E402

from src.calculate_lengths import calculate_finetune_sequence_lengths
from src.datamodule2_Magnus import FamilyAutoRegressiveDataModule
from src.encoder_nano_socmob import FamilyPretrainNanoEncoder
from src.loggers import RetryOrSkipTensorBoardLogger
from src.paths import (
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
    get_newest_path,
    get_wandb_runid,
    get_wandb_runid_filter_away_nonconforming,
)
from src.prediction_writer import SaveSimpleInfo

# REGEX FROM HYDRA TO CONFIG
# Search
# hparams\.([a-zA-Z_][a-zA-Z0-9_]*)
# Replace
# hparams["$1"]
# REGEX FROM CONFIG TO HYDRA
# Search
# hparams\["([a-zA-Z_][a-zA-Z0-9_]*)"\]
# Replace
# hparams.$1


@hydra.main(
    config_path=(FPATH.CONFIGS / "socmob").as_posix(),
    config_name="hparams_family_pretrain.yaml",
    version_base=None,
)
def main(hparams: DictConfig):
    # %%
    torch.serialization.add_safe_globals([PosixPath])
    torch._dynamo.config.cache_size_limit = 16
    # # Load hparams
    # with open(
    #     FPATH.CONFIGS / "socmob" / "hparams_family_pretrain.yaml", "r", encoding="utf-8"
    # ) as stream:
    #     hparams = yaml.safe_load(stream)
    if hparams.deterministic_run_id:
        run_id = hparams.deterministic_run_id
    else:
        run_id = f"{get_wandb_runid_filter_away_nonconforming(FPATH.TB_LOGS / hparams.experiment_name)}-pretrain-{'-'.join(hparams.feature_set)}"
    seed_everything(73)
    # Set training variables
    torch.set_float32_matmul_precision(hparams.float32_matmul_precision)
    N_DEVICES = (
        len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if torch.cuda.is_available()
        else 1
    )
    # Where to save
    dir_path = FPATH.DATA / hparams.dir_path

    # ----------- Sample info -----------

    sample_folder = FPATH.NETWORK_DATA / hparams.sample_folder
    sample_folder.mkdir(exist_ok=True)

    # Sample splits
    train_pids_file = sample_folder / "pretrain_train_pids.parquet"
    # copy_file_or_dir(train_pids_file)

    val_pids_file = sample_folder / "pretrain_val_pids.parquet"
    # copy_file_or_dir(val_pids_file)

    train_person_ids = (
        pl.read_parquet(train_pids_file)
        .with_columns(pl.col("person_id").cast(str))["person_id"]
        .to_list()
    )
    val_person_ids = (
        pl.read_parquet(val_pids_file)
        .with_columns(pl.col("person_id").cast(str))["person_id"]
        .to_list()
    )

    # ----------- Input info -----------
    data_path = FPATH.DATA / hparams.source_dir
    data_path.mkdir(exist_ok=True)

    source_paths = [
        (data_path / path).with_suffix(".parquet") for path in hparams.sources
    ]
    background_path = data_path / "background.parquet"

    for s in source_paths + [background_path]:
        if hparams.force_copy_of_sources:
            copy_file_or_dir(s)
        elif hparams.use_network_io_for_sources:
            background_path = FPATH.swap_drives(background_path)
            source_paths = [FPATH.swap_drives(path) for path in source_paths]
        else:
            check_and_copy_file_or_dir(s)
    background_path = (
        FPATH.NETWORK_DATA / hparams.source_dir / hparams.background
    ).with_suffix(".parquet")
    outcomes_path = (
        FPATH.NETWORK_DATA / hparams.sample_folder / "pretrain_outcomes_transformer"
    ).with_suffix(".parquet")

    # sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    outcomes = pl.read_parquet(outcomes_path)

    # CENSORING
    print(f"Filtering sources to be before {hparams.cutoff_year}")
    cutoff_date = datetime(hparams.cutoff_year, 1, 1)
    pretrain_cutoff = pa.scalar(cutoff_date, type=pa.timestamp("ns"))
    filtered_sources = [
        ds.dataset(filepath, format="parquet").filter(
            pc.less(ds.field("date_col"), pretrain_cutoff)
        )
        for filepath in source_paths
    ]

    dm = FamilyAutoRegressiveDataModule(
        dir_path=dir_path,
        sources=filtered_sources,
        outcomes=outcomes,
        background=background,
        subset_background=hparams.subset_background,
        n_tokens=hparams.n_tokens,
        lengths=hparams.lengths,
        train_person_ids=train_person_ids,
        val_person_ids=val_person_ids,
        num_workers=hparams.num_workers,
        max_seq_len=hparams.max_seq_len,
        # prediction_windows=hparams.prediction_windows,
        source_dir=hparams.source_dir,
        feature_set=hparams.feature_set,
        pretrain_style="AR",  # MLM collate not implemented for Family variant, but pretrain_style is expected as input
        cutoff=hparams.token_freq_cutoff,
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    # Hydra version
    with open_dict(hparams):
        hparams.vocab_size = len(dm.pipeline.vocab)
    # ## Non-hydra version
    # hparams.vocab_size = len(dm.pipeline.vocab)

    model = FamilyPretrainNanoEncoder(**hparams)

    # ---------- Callback setup ----------
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
        monitor="val/loss",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [lr_monitor, checkpoint_callback]

    ckpt_path = None

    # ---------- Load failed model ----------
    if hparams.load_failed_model:
        ckpt_path = (
            FPATH.CHECKPOINTS_TRANSFORMER
            / hparams.failed_experiment_name
            / hparams.failed_run_name
            / "last.ckpt"
        )
        print("Loading failed model")

    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="auto",
        devices=N_DEVICES,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=hparams.deterministic,
        precision=hparams.precision,
        log_every_n_steps=500,
        fast_dev_run=hparams.fast_dev_run,
        val_check_interval=hparams.val_check_interval,
    )
    # Train
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
