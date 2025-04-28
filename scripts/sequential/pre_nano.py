import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT

import yaml  # noqa: E402
import torch  # noqa: E402
import warnings  # noqa: E402
import polars as pl  # noqa: E402
import numpy as np # noqa: E402
# from datetime import datetime

from src.datamodule import PretrainLifeLightningDataModule  # noqa: E402
from src.encoder_nano_bigru_rope_lr_automatic import PretrainNanoEncoder  # noqa: E402
from src.paths import FPATH, check_and_copy_file_or_dir, copy_file_or_dir, get_wandb_runid  # noqa: E402
from lightning.pytorch import Trainer  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    LearningRateMonitor,
    ModelCheckpoint,
)
from src.loggers import RetryTensorBoardLogger  # noqa: E402
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: E402
from lightning.pytorch import seed_everything  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
from datetime import datetime  # noqa: E402


# This is an erroneous warning, the mask is indeed already bool
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.",
    category=UserWarning,
    module="torch.nn.modules.activation",
)

if __name__ == "__main__":
    # Load hparams
    with open(FPATH.CONFIGS / "hparams_pretrain.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)
    run_id = get_wandb_runid(FPATH.TB_LOGS / hparams["experiment_name"])

    seed_everything(73)
    print(f"Experiment: {hparams['experiment_name']}")

    # Set training variables
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    #### Data ####

    # TODO: Give data through config?

    # Where to save
    dir_path = FPATH.DATA / hparams["dir_path"]

    copy_file_or_dir(FPATH.DATA / "sample_pretrain_train.npy")
    copy_file_or_dir(FPATH.DATA / "sample_pretrain_val.npy")
    train_person_ids = np.load(FPATH.DATA / "sample_pretrain_train.npy").astype(str)
    val_person_ids = np.load(FPATH.DATA / "sample_pretrain_val.npy").astype(str)

    # Input data
    source_paths = [
        FPATH.DATA / f"{hparams['fname_prefix']}_amrun.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_lpr.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_lmdb.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_high_school_grades.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_primary_school_grades.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_higher_ed_grades.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_graduation.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_enrollment.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_diploma_grades.parquet",
        # FPATH.DATA / f"{hparams['fname_prefix']}_ras.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_ras_neet_only.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_decisions.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_jail_parole_start.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_jail_end.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_parole_end.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_conferred_charges.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_minor_charges.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_victims.parquet",
        FPATH.DATA / f"{hparams['fname_prefix']}_charges.parquet",
    ]

    background_path = FPATH.DATA / f"{hparams['fname_prefix']}_background.parquet"

    for s in source_paths + [background_path]:
        check_and_copy_file_or_dir(s)

    background = pl.read_parquet(background_path)

    # CENSORING
    print(f'Filtering sources to be before {hparams["cutoff_year"]}')
    cutoff_date = datetime(hparams["cutoff_year"], 1, 1)
    pretrain_cutoff = pa.scalar(cutoff_date, type=pa.timestamp("ns"))
    filtered_sources = [
        ds.dataset(filepath, format="parquet").filter(
            pc.less(ds.field("date_col"), pretrain_cutoff)
        )
        for filepath in source_paths
    ]

    # TODO: Move all to config
    dm = PretrainLifeLightningDataModule(
        dir_path=dir_path,
        sources=filtered_sources,
        background=background,
        cls_token=hparams["include_cls"],
        sep_token=hparams["include_sep"],
        train_person_ids=train_person_ids,
        val_person_ids=val_person_ids,
        segment=hparams["include_segment"],
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        collate_method=hparams["collate_method"],
        max_seq_len=hparams["max_seq_len"],
        cutoff=hparams['token_freq_cutoff'],
    )

    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    hparams["vocab_size"] = len(dm.pipeline.vocab)

    train_dataloader_length = dm.get_steps_per_train_epoch()

    # The iterations are spread out over the devices, hence the division by devices
    hparams["steps_per_epoch"] = train_dataloader_length / n_devices
    
    hparams["optimizer_max_iters"] = hparams["max_epochs"] * hparams["steps_per_epoch"]

    # Model

    model = PretrainNanoEncoder(**hparams)
    if hparams["compile"]:
        model = torch.compile(model)

    # Trainer setup

    logger = RetryTensorBoardLogger(
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
    # early_stopping_callback = EarlyStopping("val/acc", patience=3)

    callbacks = [lr_monitor, checkpoint_callback]  # , early_stopping_callback]

    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./performance_profiling'),
    #     sort_by_key="self_cuda_memory_usage",
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, skip_first=10),
    #     profile_memory=True,
    # )

    #strategy = DDPStrategy(find_unused_parameters=False)

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",
        devices=n_devices,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",  # strategy,
        deterministic=True,
        precision=hparams["precision"],
        log_every_n_steps=1000,
        # profiler=profiler,
        # fast_dev_run=50,
        # limit_val_batches=0
    )

    # Train
    if hparams['load_failed_model']:
        ckpt_path = FPATH.CHECKPOINTS_TRANSFORMER / "pretrain_split" / "014_thriving_bison" / "last.ckpt"
        print('LOADING OLD MODEL')
    else:
        ckpt_path = None
    
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
