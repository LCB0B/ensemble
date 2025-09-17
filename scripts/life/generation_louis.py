import sys


import torch
from src.generation_utils import (
    load_pretrained_model,
    setup_datamodule, 
    generate_from_dataloader,
    scenario_analysis_example,
    interactive_generation,
    force_model_dtype
)
from src.paths import FPATH

# Configuration
name_model = "014_shrewd_fox-pretrain-autoregressive-lr0.0003"

print("Setting up datamodule...")
dm, hparams = setup_datamodule(name_model)

print("Loading model...")
checkpoint_path = FPATH.CHECKPOINTS_TRANSFORMER / "destiny" / name_model / "best.ckpt"
model = load_pretrained_model(checkpoint_path, hparams)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model loaded on {device}")
print(f"Model parameters dtype: {next(model.parameters()).dtype}")
print(f"Vocab size: {hparams['vocab_size']}")

# Get dataloader
val_dataloader = dm.val_dataloader()

model.eval()
device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype

val_dataloader = dm.val_dataloader()
batch = next(iter(val_dataloader))
batch = dm.on_after_batch_transfer(batch,1)

batch = dm.transfer_batch_to_device(batch,device,1)
device = next(model.parameters()).device

from torch.amp import autocast

# Test forward pass
with torch.no_grad():
    with autocast('cuda'):  # or autocast(device_type='cuda')
        output = model.forward(batch)

        