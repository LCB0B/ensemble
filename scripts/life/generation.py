"""
Simple generation script using src/generation_utils.py
Save this as scripts/generation.py and run from project root.
"""

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
hparams["num_workers"] = 0
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

print("\n" + "="*50)
print("STARTING GENERATION EXAMPLES")
print("="*50)

# 1. Generate from validation dataloader (WITH flash attention)
print("\n1. Basic Generation Examples...")
results = generate_from_dataloader(model, val_dataloader, num_samples=2)
print(f"Generated {len(results)} results successfully!")

# 2. Scenario analysis (WITH flash attention)
print("\n2. Scenario Analysis...")
scenarios = scenario_analysis_example(model, val_dataloader, num_batches=2, num_scenarios=3)
print(f"Generated {len(scenarios)} scenario batches successfully!")

# 3. Interactive session (optional)
use_interactive = input("\n3. Run interactive session? (y/n): ").lower().strip()
if use_interactive == 'y':
    interactive_generation(model, dm.pipeline.vocab, device)

print("\nGeneration complete!")
