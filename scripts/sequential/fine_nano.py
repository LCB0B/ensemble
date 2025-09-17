import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT

import warnings  # noqa: E402
from datetime import datetime  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch import Trainer, seed_everything  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig, open_dict  # noqa: E402
from torchmetrics.utilities.exceptions import TorchMetricsUserWarning  # noqa: E402

from src.earlylife.src.calculate_lengths import (
    calculate_finetune_sequence_lengths,  # noqa: E402
)
from src.earlylife.src.datamodule import (
    FinetuneLifeLightningDataModule,  # noqa: E402
)
from src.earlylife.src.encoder_nano import FinetuneNanoEncoder  # noqa: E402
from src.earlylife.src.loggers import RetryTensorBoardLogger  # noqa: E402
from src.earlylife.src.paths import (  # noqa: E402
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
    get_newest_path,
    get_wandb_runid,
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

warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    module="lightning_fabric.utilities.cloud_io",
)


@hydra.main(
    config_path=FPATH.CONFIGS.as_posix(),
    config_name="hparams_finetune.yaml",
    version_base=None,
)
def main(hparams: DictConfig):

    if hparams.run_id is not None:
        run_id = hparams.run_id
    else:
        run_id = get_wandb_runid(FPATH.TB_LOGS / hparams.experiment_name)

    seed_everything(73)

    # Set training variables
    torch.set_float32_matmul_precision(hparams.float32_matmul_precision)
    n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    # Where to save
    dir_path = FPATH.DATA / hparams.dir_path

    # ----------- Outcome info -----------

    sample_folder = FPATH.DATA / hparams.sample_folder
    sample_folder.mkdir(exist_ok=True)

    outcome_fname = hparams.outcome_fname
    print(f"Prediction target: {outcome_fname}")

    # CENSORING
    outcome_path = sample_folder / f"{outcome_fname}_targets.parquet"
    copy_file_or_dir(outcome_path)
    df_targets = pl.read_parquet(outcome_path)
    df_targets = df_targets.select(["person_id", "target", "censor"])

    # Sample splits
    train_pids_file = sample_folder / f"{outcome_fname}_train_pids.parquet"
    copy_file_or_dir(train_pids_file)

    val_pids_file = sample_folder / f"{outcome_fname}_val_pids.parquet"
    copy_file_or_dir(val_pids_file)

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

    subset_size = hparams.get("subset_size", None)

    # Optional subsetting
    if subset_size is not None:
        subset_file = (
            FPATH.DATA
            / hparams.subset_folder
            / "subsets"
            / f"train_idx_{subset_size}.npy"
        )
        check_and_copy_file_or_dir(subset_file)

        run_id = f"pretrained_{hparams.load_pretrained_model}_size_{subset_size}"

        if not subset_file.exists():
            raise FileNotFoundError(f"Subset index file not found: {subset_file}")

        subset_idx = np.load(subset_file)

        df_train = df_targets.cast({"person_id": str}).filter(
            (pl.col("person_id").is_in(train_person_ids))
            & (pl.col("censor").dt.year() < hparams.cutoff_year)
        )

        df_train = df_train[pl.Series("row_idx", subset_idx)]

        df_val = df_targets.cast({"person_id": str}).filter(
            (pl.col("person_id").is_in(val_person_ids))
            & (pl.col("censor").dt.year() < hparams.cutoff_year)
        )

        df_targets = pl.concat([df_train, df_val]).cast({"person_id": int})

        print(f"Using subset of size {subset_size}")

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

    background = pl.read_parquet(background_path)

    # CENSORING
    cutoff_date = datetime(hparams.cutoff_year, 1, 1)
    pretrain_cutoff = pa.scalar(cutoff_date, type=pa.timestamp("ns"))
    filtered_sources = [
        ds.dataset(filepath, format="parquet").filter(
            pc.less(ds.field("date_col"), pretrain_cutoff)
        )
        for filepath in source_paths
    ]

    # ---------- Training setup ----------
    dm = FinetuneLifeLightningDataModule(
        dir_path=dir_path,
        sources=filtered_sources,
        background=background,
        outcomes=df_targets,
        cls_token=hparams.include_cls,
        sep_token=hparams.include_sep,
        train_person_ids=train_person_ids,
        val_person_ids=val_person_ids,
        segment=hparams.include_segment,
        n_tokens=hparams.n_tokens,
        num_workers=hparams.num_workers,
        max_seq_len=hparams.max_seq_len,
        cutoff=hparams.token_freq_cutoff,
        cutoff_year=hparams.cutoff_year,
        source_dir=hparams.source_dir,
        lengths=hparams.lengths,
    )
    dm.prepare_data()  # TODO: Ideally we should not call this and let Lightning call it (and get the dm info somewhere else)
    calculate_finetune_sequence_lengths(
        outcome_fname=hparams.outcome_fname,
        dir_path=hparams.dir_path,
        sample_folder=hparams.sample_folder,
    )
    with open_dict(hparams):
        hparams.vocab_size = len(dm.pipeline.vocab)

    # ---------- Load failed or pretrained model ----------
    checkpoint_path_for_trainer = None

    if hparams.load_failed_model:
        checkpoint_path = (
            FPATH.CHECKPOINTS_TRANSFORMER
            / hparams.failed_experiment_name
            / hparams.failed_run_name
            / "last.ckpt"
        )
        print("Loading failed model")
        model = FinetuneNanoEncoder.load_from_checkpoint(checkpoint_path, strict=False)
        checkpoint_path_for_trainer = checkpoint_path

    elif hparams.load_pretrained_model:
        checkpoint_folder = (
            FPATH.CHECKPOINTS_TRANSFORMER / hparams.checkpoint_experiment_name
        )
        checkpoint_name = (
            get_newest_path(checkpoint_folder)
            if str(hparams.checkpoint_run_name).lower() == "latest"
            else str(hparams.checkpoint_run_name)
        )
        print(
            f"Loading checkpoint: {checkpoint_name} from experiment {hparams.checkpoint_experiment_name}"
        )
        checkpoint_path = checkpoint_folder / checkpoint_name / "best.ckpt"
        model = FinetuneNanoEncoder.load_from_checkpoint(checkpoint_path, strict=False)

        set_hparams_list = [
            # "layer_lr_decay", # If implemented, also need to be reset
            "learning_rate",
            "warmup_steps",
        ]
        for hparam in set_hparams_list:
            model.hparams[hparam] = hparams[hparam]

        _ = model.configure_optimizers()

    else:
        print("Not loading checkpoint.")
        model = FinetuneNanoEncoder(**hparams)

    # ---------- Callback setup ----------
    logger = RetryTensorBoardLogger(
        save_dir=FPATH.TB_LOGS / hparams.experiment_name,
        name="",
        version=run_id,
        default_hp_metric=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=FPATH.CHECKPOINTS_TRANSFORMER / hparams.experiment_name / run_id,
        filename="best",
        monitor="AUROC",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [lr_monitor, checkpoint_callback]

    # ---------- Train ----------
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="auto",
        devices=n_devices,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=hparams.deterministic,
        precision=hparams.precision,
        log_every_n_steps=500,
        fast_dev_run=hparams.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path_for_trainer)


if __name__ == "__main__":
    main()
