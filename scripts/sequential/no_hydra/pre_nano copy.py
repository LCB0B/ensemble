import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT

import warnings  # noqa: E402
from datetime import datetime  # noqa: E402

import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from lightning.pytorch import (
    Trainer,  # noqa: E402
    seed_everything,  # noqa: E402
)
from lightning.pytorch.callbacks import (  # noqa: E402
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig

from src.datamodule import PretrainLifeLightningDataModule  # noqa: E402
from src.encoder_nano import PretrainNanoEncoder  # noqa: E402
from src.loggers import RetryTensorBoardLogger  # noqa: E402
from src.paths import (  # noqa: E402
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
    get_wandb_runid,
)

# from datetime import datetime


torch._dynamo.config.optimize_ddp = False

# This is an erroneous warning, the mask is indeed already bool
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.",
    category=UserWarning,
    module="torch.nn.modules.activation",
)


@hydra.main(
    config_path=FPATH.CON, config_name="hparams_pretrain.yaml", version_base=None
)
def main(hparams: DictConfig):
    # Load hparams
    with open(FPATH.CONFIGS / "hparams_pretrain.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)
    run_id = get_wandb_runid(FPATH.TB_LOGS / hparams["experiment_name"])

    seed_everything(73)
    print(f"Experiment: {hparams['experiment_name']}")

    # Set training variables
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    # Where to save
    dir_path = FPATH.DATA / hparams["dir_path"]

    # ----------- Sample info -----------

    # Sample splits
    sample_folder = FPATH.DATA / hparams["sample_folder"]
    sample_folder.mkdir(exist_ok=True)

    train_pids_file = sample_folder / "pretrain_train_pids.parquet"
    copy_file_or_dir(train_pids_file)

    val_pids_file = sample_folder / "pretrain_val_pids.parquet"
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

    # ----------- Input info -----------
    data_path = FPATH.DATA / hparams["source_dir"]
    data_path.mkdir(exist_ok=True)

    source_paths = [
        (data_path / path).with_suffix(".parquet") for path in hparams["sources"]
    ]

    background_path = data_path / "background.parquet"

    for s in source_paths + [background_path]:
        if hparams["force_copy_of_sources"]:
            copy_file_or_dir(s)
        else:
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

    # ---------- Training setup ----------
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

    # Model

    model = PretrainNanoEncoder(**hparams)

    # ---------- Callback setup ----------
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

    callbacks = [lr_monitor, checkpoint_callback]

    # ---------- Load failed model ----------
    if hparams["load_failed_model"]:
        ckpt_path = (
            FPATH.CHECKPOINTS_TRANSFORMER
            / hparams["failed_experiment_name"]
            / hparams["failed_run_name"]
            / "last.ckpt"
        )
        print("Loading failed model")
    else:
        ckpt_path = None

    # ---------- Train ----------

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="auto",
        devices=n_devices,
        callbacks=callbacks,
        logger=logger,
        strategy="auto",
        deterministic=True,
        precision=hparams["precision"],
        log_every_n_steps=500,
        fast_dev_run=hparams["fast_dev_run"],
    )

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
