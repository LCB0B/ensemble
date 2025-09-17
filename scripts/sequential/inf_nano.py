import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT

import warnings  # noqa: E402
from datetime import datetime  # noqa: E402

import hydra  # noqa: E402
import polars as pl
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch import Trainer, seed_everything  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.earlylife.src.datamodule import FinetuneLifeLightningDataModule  # noqa: E402
from src.earlylife.src.encoder_nano import FinetuneNanoEncoder  # noqa: E402
from src.earlylife.src.paths import (  # noqa: E402
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
    get_newest_path,
)
from src.earlylife.src.prediction_writer import SaveSelectiveInfo  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message=".*You are using `torch.load` with `weights_only=False`.*",
    module="lightning_fabric.utilities.cloud_io",
)


@hydra.main(
    config_path=FPATH.CONFIGS.as_posix(),
    config_name="hparams_inference.yaml",
    version_base=None,
)
def main(hparams: DictConfig) -> None:
    seed_everything(73)

    # Set testing variables
    torch.set_float32_matmul_precision(hparams.float32_matmul_precision)

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
    df_targets = df_targets.select(["person_id", "censor"]).with_columns(
        pl.lit(10_000).alias("target")
    )  # No target, but expects a target due to some conversion into tensors.

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

    # ---------- Setup DataModule ----------
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
        inference_type=hparams.inference_type,
        cutoff_year=hparams.cutoff_year,
        source_dir=hparams.source_dir,
        lengths=hparams.lengths,
    )
    dm.prepare_data()

    # ---------- Load finetuned model ----------
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

    checkpoint_path = (
        FPATH.CHECKPOINTS_TRANSFORMER
        / hparams.checkpoint_experiment_name
        / checkpoint_name
        / "best.ckpt"
    )

    model = FinetuneNanoEncoder.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()

    # ---------- Prediction writer ----------
    pred_writer = SaveSelectiveInfo(
        fname=f"{hparams.output_file_name}_{hparams.outcome_fname}_{hparams.inference_type}",
        folder=hparams.inference_data_folder,
    )

    # ---------- Trainer setup for inference ----------
    trainer = Trainer(
        accelerator="auto",
        devices=1,  # NO DISTRIBUTION DURING INFERENCE
        precision=hparams.precision,
        callbacks=[pred_writer],
        logger=False,
    )

    # ---------- Perform inference ----------
    trainer.predict(model, datamodule=dm, return_predictions=False)


if __name__ == "__main__":
    main()
