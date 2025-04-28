import os
import yaml
import torch
import warnings
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
from datetime import datetime
from src.datamodule2 import LifeLightningDataModule
from src.encoder_nano_risk import CausalEncoder
from src.paths import FPATH, check_and_copy_file_or_dir
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.",
    category=UserWarning,
    module="torch.nn.modules.activation",
)


# Environment variables
os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["RAYON_NUM_THREADS"] = "8"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Load hyperparameters
with open(FPATH.CONFIGS / "hparams_pretrain2.yaml", "r") as stream:
    hparams = yaml.safe_load(stream)
print(f"Experiment: {hparams['experiment_name']}")

data_path = FPATH.DATA /'life_pretrain'


# Paths for input data
source_paths =  list(data_path.glob("*.parquet"))
background_path = data_path / f"background.parquet"

for s in source_paths + [background_path]:
    check_and_copy_file_or_dir(s)

# Filter data based on cutoff date
cutoff_date = datetime(hparams["cutoff_year"], 1, 1)
pretrain_cutoff = pa.scalar(cutoff_date, type=pa.timestamp("ns"))
filtered_sources = [
    ds.dataset(filepath, format="parquet").filter(
        pc.less(ds.field("date_col"), pretrain_cutoff)
    )
    for filepath in source_paths
]
background = pl.read_parquet(background_path)

# Prepare the datamodule
dm = LifeLightningDataModule(
    dir_path=FPATH.DATA / hparams["dir_path"],
    sources=filtered_sources,
    background=background,
    cls_token=hparams["include_cls"],
    sep_token=hparams["include_sep"],
    segment=hparams["include_segment"],
    batch_size=hparams["batch_size"],
    num_workers=hparams["num_workers"],
    max_seq_len=hparams["max_seq_len"],
    cutoff=hparams["token_freq_cutoff"],
)
dm.prepare_data()
dm.setup(stage="fit")
dataloader = dm.train_dataloader()

# Load model checkpoint
checkpoint_path = FPATH.CHECKPOINTS_TRANSFORMER / hparams["experiment_name"] / "best.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cuda")
model = CausalEncoder(**checkpoint["hyper_parameters"])
model.load_state_dict(checkpoint["state_dict"])
model.eval().to("cuda")

# Generate tokens
print("Generating predictions...")
predictions = []
with torch.no_grad():
    for batch in dataloader:
        # Move batch to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to("cuda")

        # Forward pass
        output = model(batch)
        logits = model.decoder(output[:, -1, :])  # Last token's logits
        predicted_indices = torch.argmax(logits, dim=-1)
        predicted_tokens = [
            dm.pipeline.vocab.idx_to_token[idx.item()] for idx in predicted_indices
        ]
        predictions.extend(predicted_tokens)

# Print predictions
print(f"Generated Tokens: {predictions}")
