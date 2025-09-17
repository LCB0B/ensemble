import os
import pickle
import warnings
from datetime import datetime  # noqa: E402

import numpy as np
import polars as pl
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import torch
import yaml
from lightning.pytorch import Trainer, seed_everything

from src.datamodule import FinetuneLifeLightningDataModule
from src.encoder_nano import FinetuneNanoEncoder
from src.paths import (
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
    get_newest_path,
)
from src.prediction_writer import SaveAllInfoWriter, SaveSelectiveInfo, SaveSimpleInfo
from src.utils import filter_sources_by_date

if __name__ == "__main__":
    seed_everything(73)

    # Load hparams
    with open(FPATH.CONFIGS / "hparams_inference.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)

    # Set testing variables
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    # Where to save
    dir_path = FPATH.DATA / hparams["dir_path"]

    # ----------- Outcome info -----------

    sample_folder = FPATH.DATA / hparams["sample_folder"]
    sample_folder.mkdir(exist_ok=True)

    outcome_fname = hparams["outcome_fname"]
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

    cutoff_date = datetime(hparams["cutoff_year"], 1, 1)
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
        cls_token=hparams["include_cls"],
        sep_token=hparams["include_sep"],
        train_person_ids=train_person_ids,
        val_person_ids=val_person_ids,
        segment=hparams["include_segment"],
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        max_seq_len=hparams["max_seq_len"],
        cutoff=hparams["token_freq_cutoff"],
        inference_type=hparams["inference_type"],
    )
    dm.prepare_data()

    # Load finetuned model

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

    checkpoint_path = (
        FPATH.CHECKPOINTS_TRANSFORMER
        / hparams["checkpoint_experiment_name"]
        / checkpoint_name
        / "best.ckpt"
    )

    model = FinetuneNanoEncoder.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()

    ## Define the prediction writer (Write on batch to accumulate predictions and write on epoch to save)
    pred_writer = SaveSelectiveInfo(
        fname=f'{hparams["output_file_name"]}_{hparams["inference_type"]}',
        folder=hparams["inference_data_folder"],
    )

    # Trainer setup for testing
    trainer = Trainer(
        accelerator="auto",
        devices=1,  # NO DISTRIBUTION DURING INFERENCE
        precision=hparams["precision"],
        callbacks=[pred_writer],
        logger=False,
    )

    # Perform inference
    trainer.predict(model, datamodule=dm, return_predictions=False)
