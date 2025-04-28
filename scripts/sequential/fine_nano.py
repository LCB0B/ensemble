import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT

import yaml  # noqa: E402
import torch  # noqa: E402
import warnings  # noqa: E402
import polars as pl  # noqa: E402
import numpy as np  # noqa: E402
from datetime import datetime  # noqa: E402

from src.datamodule import FinetuneLifeLightningDataModule  # noqa: E402
from src.encoder_nano_bigru_rope_lr import FinetuneNanoEncoder  # noqa: E402
from src.paths import (  # noqa: E402
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
    get_wandb_runid,
    get_newest_path,
)
from lightning.pytorch import Trainer  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: E402
from lightning.pytorch import seed_everything  # noqa: E402
from lightning.pytorch.profilers import PyTorchProfiler  # noqa: E402
from torchmetrics.utilities.exceptions import TorchMetricsUserWarning  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

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

# It's slow, but we need it ¯\_(ツ)_/¯
warnings.filterwarnings(
    "ignore",
    category=TorchMetricsUserWarning,
    message="You are trying to use a metric in deterministic mode on GPU that uses `torch.cumsum`, which is currently not supported. The tensor will be copied to the CPU memory to compute it and then copied back to GPU. Expect some slowdowns.*",
)

# # https://github.com/Lightning-AI/pytorch-lightning/issues/18123#issuecomment-1668830955
# from functools import wraps
# from typing import Union

# def overwrite_torch_functions():
#     # wrap torch.nn.Module.__setattr__
#     module_set_attr_orig = torch.nn.Module.__setattr__

#     @wraps(torch.nn.Module.__setattr__)
#     def wrap_set_attr(self, name: str, value: Union[torch.Tensor, 'torch.nn.Module']) -> None:
#         if isinstance(value, torch.nn.Module):# and not isinstance(value, Metric):
#             try:
#                 print("------value :: ", value)
#             except (RuntimeError):
#                 pass
#         module_set_attr_orig(self, name, value)

#     torch.nn.Module.__setattr__ = wrap_set_attr



if __name__ == "__main__":
    #overwrite_torch_functions()
    
    # Load hparams
    with open(FPATH.CONFIGS / "hparams_finetune.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)
    run_id = get_wandb_runid(FPATH.TB_LOGS / hparams["experiment_name"])

    seed_everything(73)
    # Set training variables
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    # assert (
    #     n_devices == 4
    # ), """Not training on all four GPUs. If this is intended, comment out this line.
    # To enable all GPUs, run 'export CUDA_VISIBLE_DEVICES=0,1,2,3' from terminal"""

    #### Data ####

    # TODO: Give data through config?

    # Where to save
    dir_path = FPATH.DATA / hparams["dir_path"]

    copy_file_or_dir(FPATH.DATA / "sample_neet_train.npy")
    copy_file_or_dir(FPATH.DATA / "sample_neet_val.npy")
    neet_train_person_ids = np.load(FPATH.DATA / "sample_neet_train.npy")
    neet_val_person_ids = np.load(FPATH.DATA / "sample_neet_val.npy")

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
        # print("ALWAYS COPYING DATA RIGHT NOW")
        check_and_copy_file_or_dir(s)

    background = pl.read_parquet(background_path)

    # CENSORING

    cutoff_date = datetime(hparams["cutoff_year"], 1, 1)
    pretrain_cutoff = pa.scalar(cutoff_date, type=pa.timestamp("ns"))
    filtered_sources = [
        ds.dataset(filepath, format="parquet").filter(
            pc.less(ds.field("date_col"), pretrain_cutoff)
        )
        for filepath in source_paths
    ]

    # CENSORING
    outcome_path = FPATH.DATA / "outcomes_processed_NEET.parquet"
    copy_file_or_dir(outcome_path)
    df_neet = pl.read_parquet(outcome_path)
    print(df_neet.head())
    outcomes_df = df_neet.select(["person_id", "target", "censor"])

    # TODO: Move all to config
    dm = FinetuneLifeLightningDataModule(
        dir_path=dir_path,
        sources=filtered_sources,
        background=background,
        outcomes=outcomes_df,
        cls_token=hparams["include_cls"],
        sep_token=hparams["include_sep"],
        train_person_ids=neet_train_person_ids,
        val_person_ids=neet_val_person_ids,
        segment=hparams["include_segment"],
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        collate_method=hparams["collate_method"],
        max_seq_len=hparams["max_seq_len"],
        cutoff=hparams["token_freq_cutoff"],
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)

    # get vocab size
    hparams["vocab_size"] = len(dm.pipeline.vocab)

    train_dataloader_length = dm.get_steps_per_train_epoch()

    # The iterations are spread out over the devices, hence the division by devices
    hparams["steps_per_epoch"] = train_dataloader_length / n_devices
    hparams["optimizer_max_iters"] = hparams["max_epochs"] * hparams["steps_per_epoch"]

    # Load checkpoint if fine-tuning
    checkpoint_path_for_trainer = None

    if hparams["load_failed_model"]:
        checkpoint_path = (
            FPATH.CHECKPOINTS_TRANSFORMER
            / "finetune_split"
            / "030_unsightly_wolf"
            / "last.ckpt"
        )
        print("Loading failed model")

        model = FinetuneNanoEncoder.load_from_checkpoint(checkpoint_path, strict=False)
        checkpoint_path_for_trainer = checkpoint_path

    elif hparams["load_checkpoint"]:
        checkpoint_folder = (
            FPATH.CHECKPOINTS_TRANSFORMER / hparams["checkpoint_experiment_name"]
        )
        if str(hparams["checkpoint_run_name"]).lower() == "latest":
            checkpoint_name = get_newest_path(checkpoint_folder)
        else:
            checkpoint_name = str(hparams["checkpoint_run_name"])

        print(
            f"Loading checkpoint: {checkpoint_name} from experiment {hparams['checkpoint_experiment_name']}"
        )

        checkpoint_path = checkpoint_folder / checkpoint_name / "best.ckpt"

        model = FinetuneNanoEncoder.load_from_checkpoint(checkpoint_path, strict=False)

        set_hparams_list = [
            "layer_lr_decay",
            "learning_rate",
            "steps_per_epoch",
            "optimizer_max_iters",
            "optimizer_warmup_epochs"
        ]
        for hparam in set_hparams_list:
            model.hparams[hparam] = hparams[hparam]
        optimizers_and_schedulers = model.configure_optimizers()

    else:
        print("Not loading checkpoint.")
        print(
            f"Setting layer_lr_decay to 1, it was previously {hparams['layer_lr_decay']}"
        )
        hparams["layer_lr_decay"] = 1
        model = FinetuneNanoEncoder(**hparams)

    if hparams["compile"]:
        model = torch.compile(model)

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
        monitor="AUROC",
        mode="max",
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

    strategy = DDPStrategy(find_unused_parameters=False)

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",  # "auto",
        devices=n_devices,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=True,
        precision=hparams["precision"],
        log_every_n_steps=1000,
        # profiler=profiler,
        # fast_dev_run=50,
        # limit_val_batches=0,
        # profiler=SimpleProfiler(filename="simple_profile"),
    )

    # Train
    trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path_for_trainer)
