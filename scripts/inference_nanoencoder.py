import yaml
import torch
import warnings
import polars as pl
import pickle
import os
import numpy as np
from src.datamodule import FinetuneLifeLightningDataModule
from src.encoder_nano_bigru_rope import FinetuneNanoEncoder
from src.prediction_writer import SaveAllInfoWriter, SaveSelectiveInfo
from src.paths import FPATH, check_and_copy_file_or_dir, get_newest_path, copy_file_or_dir
from src.utils import filter_sources_by_date
from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything


from datetime import datetime  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

if __name__ == "__main__":
    seed_everything(73)

    # Load hparams
    with open(FPATH.CONFIGS / "hparams_inference.yaml", "r") as stream:
        hparams = yaml.safe_load(stream)

    # Set testing variables
    torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
    n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    #### Data ####
    
    dir_path = FPATH.DATA / hparams["dir_path"]

    # Input data    
            
    source_paths = [
        (FPATH.DATA / path).with_suffix(".parquet") for path in hparams["sources"]
    ]

    background_path = FPATH.DATA / f"{hparams['fname_prefix']}_background.parquet"

    for s in source_paths + [background_path]:
        check_and_copy_file_or_dir(s)

    background = pl.read_parquet(background_path)

    # CENSORING

    cutoff_date = datetime(hparams["cutoff_year"], 1, 1)
    pretrain_cutoff = pa.scalar(cutoff_date, type=pa.timestamp("ns"))
    filtered_sources = [ds.dataset(filepath, format='parquet').filter(pc.less(
                ds.field("date_col"), pretrain_cutoff
            )) for filepath in source_paths]
    
    # CENSORING
    outcome_path = FPATH.DATA / "outcomes_processed_NEET.parquet"
    #FPATH.alternative_copy_to_opposite_drive(FPATH.swap_drives(outcome_path))
    df_neet = pl.read_parquet(FPATH.swap_drives(outcome_path))
    #print(df_neet.head())
    outcomes_df = df_neet.select(['person_id', 'target', 'censor'])

    # Needed for validation data inference only:

    copy_file_or_dir(FPATH.DATA / "sample_neet_train.npy")
    copy_file_or_dir(FPATH.DATA / "sample_neet_val.npy")
    neet_train_person_ids = np.load(FPATH.DATA / "sample_neet_train.npy")
    neet_val_person_ids = np.load(FPATH.DATA / "sample_neet_val.npy")

    # Data Module setup
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
        cutoff=hparams['token_freq_cutoff'],
        inference_type=hparams['inference_type'],
    )
    dm.prepare_data()

    # Load finetuned model

    checkpoint_folder = FPATH.CHECKPOINTS / hparams["checkpoint_experiment_name"]
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
    #pred_writer = SaveAllInfoWriter(fname=hparams["output_file_name"])
    pred_writer = SaveSelectiveInfo(fname=f'{hparams["output_file_name"]}_{hparams["inference_type"]}')

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
