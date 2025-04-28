import os

os.environ["POLARS_MAX_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["RAYON_NUM_THREADS"] = "8"  # MUST BE BEFORE POLARS IMPORT
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import yaml  # noqa: E402
import torch  # noqa: E402
import warnings  # noqa: E402
import polars as pl  # noqa: E402
import numpy as np 

from src.datamodule2 import LifeLightningDataModule  # noqa: E402
from src.encoder_nano_risk import CausalEncoder  # noqa: E402
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

import pickle
from pathlib import Path

from tqdm import tqdm  


# This is an erroneous warning, the mask is indeed already bool
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.",
    category=UserWarning,
    module="torch.nn.modules.activation",
)


param_path = FPATH.TB_LOGS / "stable_pretrain" / "018_lovely_cheetah" / "hparams.yaml"
# Load hparams "kdrev"/ "22SSI" / "ensemble"/"project2vec"/"logs"/"transformer_logs"/"stable_pretrain"/"018_lovely_cheetah"
with open(param_path, "r") as stream:
    hparams = yaml.safe_load(stream)

seed_everything(73)

# Set training variables
torch.set_float32_matmul_precision(hparams["float32_matmul_precision"])
n_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
# assert (
#     n_devices == 4
# ), "Not training on all four GPUs. If this is intended, comment out this line. To enable all GPUs, run 'export CUDA_VISIBLE_DEVICES=0,1,2,3' from terminal"

#### Data ####

# TODO: Give data through config?

# Where to save
dir_path = FPATH.DATA / hparams["dir_path"]

# copy_file_or_dir(FPATH.DATA / "sample_pretrain_train.npy")
# copy_file_or_dir(FPATH.DATA / "sample_pretrain_val.npy")
# train_person_ids = np.load(FPATH.DATA / "sample_pretrain_train.npy").astype(str)
# val_person_ids = np.load(FPATH.DATA / "sample_pretrain_val.npy").astype(str)

# Input data
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
dm = LifeLightningDataModule( dir_path=dir_path,
    sources=filtered_sources,
    background=background,
    cls_token=hparams["include_cls"],
    sep_token=hparams["include_sep"],
    segment=hparams["include_segment"],
    batch_size=hparams["batch_size"],
    num_workers=hparams["num_workers"],
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

#hparams["swiglu"] = True

model = CausalEncoder(**hparams)
# if hparams["compile"]:
#     model = torch.compile(model)
#     print("Model has been compiled")

#load checkpoint (load model)
ckpt_path = FPATH.CHECKPOINTS_TRANSFORMER / "stable_pretrain"  / "018_lovely_cheetah" / "best.ckpt"
print('Loading model')

checkpoint = torch.load(ckpt_path, map_location="cuda")
model.load_state_dict(checkpoint["state_dict"])

model.eval().to("cuda")

dm.setup(stage="fit")
train_dataloader = dm.train_dataloader()


dm.setup(stage="predict")

# Get a single batch
predict_dataloader = dm.predict_dataloader()


save_dir = FPATH.DATA / "probability"
os.makedirs(save_dir, exist_ok=True)

# Define file paths
save_dir = Path(save_dir)  # Ensure save_dir is a Path object


# Put the model in evaluation mode and move it to GPU
model.eval().to("cuda")
num_batches_to_process = 10

# Initialize a list to store all sequence log probabilities
all_log_probs = []

# Iterate over the dataloader
for batch_idx, batch in enumerate(tqdm(predict_dataloader, desc="Processing Batches")):
    if batch_idx >= num_batches_to_process:
        break
    # Move batch to GPU
    batch = {key: value.to("cuda") for key, value in batch.items()}
    batch = dm.on_after_batch_transfer(batch, dataloader_idx=0)

    # Compute sequence log probabilities
    seq_log_prob = model.compute_sequence_log_prob(batch)
    
    # Append log probabilities to the list (move to CPU)
    all_log_probs.extend(seq_log_prob.cpu().tolist())

# After processing all batches, aggregate into a Polars DataFrame
df = pl.DataFrame({
    "sequence_log_prob": all_log_probs
})

# Define the output Parquet file path and save
parquet_path = save_dir / "sequence_log_probs.parquet"
df.write_parquet(parquet_path)

print(f"Aggregated sequence log probabilities saved to {parquet_path}")