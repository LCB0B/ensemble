#!/usr/bin/env python
# filepath: /home/louibo/ensemble/scripts/perplexity_life.py

import os
os.environ["POLARS_MAX_THREADS"] = "4"
os.environ["RAYON_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import yaml
import torch
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# Import your relevant classes
from src.paths import FPATH
from src.encoder_nano_risk import CausalEncoder, PretrainNanoEncoder
from src.datamodule4 import LifeLightningDataModule

def load_model_from_checkpoint(
    model_class,
    checkpoint_path: Path,
    hparams_path: Path,
    device: str = "cuda"
):
    """Load model from checkpoint file"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load hyperparameters
    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)
    
    # Load model
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(device),
        strict=False,
        **hparams
    )
    model.to(device)
    model.eval()
    print(f"Model loaded successfully: {model.__class__.__name__}")
    
    return model, hparams



def get_dataloader(
    datamodule: LifeLightningDataModule,
    stage: str = 'predict'
) -> torch.utils.data.DataLoader:
    """Gets the dataloader for the specified stage."""
    print(f"Setting up datamodule for stage: '{stage}'")
    
    if stage == 'validate':
        datamodule.setup(stage='validate')
        dataloader = datamodule.val_dataloader()
    elif stage == 'test':
        datamodule.setup(stage='test')
        dataloader = datamodule.test_dataloader()
    elif stage == 'predict':
        datamodule.setup(stage='predict')
        dataloader = datamodule.predict_dataloader()
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    if dataloader is None:
        raise RuntimeError(f"Failed to create dataloader for stage '{stage}'.")

    print(f"Dataloader obtained. Collate fn: {dataloader.collate_fn.__class__.__name__}")
    return dataloader

def calculate_perplexity(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    datamodule: LifeLightningDataModule,
    device: str = "cuda",
    num_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate perplexity using the model and dataloader
    """
    print("-" * 50)
    print(f"Starting perplexity calculation on {device}")
    
    # Track metrics
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    
    # Track sequence-level perplexities
    sequence_perplexities = []
    
    # Check that evaluate_batch_perplexity_accuracy is available in model
    if not hasattr(model, 'evaluate_batch_perplexity_accuracy'):
        raise AttributeError(f"Model {model.__class__.__name__} does not have evaluate_batch_perplexity_accuracy method")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calculating perplexity", total=num_batches)):
            if num_batches is not None and batch_idx >= num_batches:
                break
                
            # Process batch
            batch = datamodule.transfer_batch_to_device(batch, device, 0)
            if not batch or 'event' not in batch:
                continue
                
            try:
                batch = datamodule.on_after_batch_transfer(batch, 0)
                if 'attn_mask' not in batch:
                    continue
            except Exception as e:
                print(f"Error preparing batch: {e}")
                continue
            
            # Calculate perplexity metrics using the model's method
            metrics = model.evaluate_batch_perplexity_accuracy(batch, pad_token_id=0,
                                                               return_token_loss=True)
            
            # Accumulate metrics
            total_loss += metrics['batch_loss']
            total_tokens += metrics['total_tokens']
            total_batches += 1
            
            # Calculate per-sequence perplexity if token_losses is available
            if 'per_token_loss' in metrics:
                # Reshape losses to get per-sequence values
                per_token_loss = metrics['per_token_loss']
                valid_mask = metrics['valid_mask_flat']
                
                # Get the original batch shape
                batch_size = batch['event'].size(0)
                seq_len = batch['event'].size(1) - 1  # -1 because targets are shifted
                
                # Reshape to (batch_size, seq_len)
                per_token_loss = per_token_loss.view(batch_size, seq_len)
                valid_mask = valid_mask.view(batch_size, seq_len)
                
                # Calculate per-sequence metrics
                for seq_idx in range(batch_size):
                    seq_mask = valid_mask[seq_idx]
                    if seq_mask.sum() > 0:  # Only process sequences with valid tokens
                        seq_loss = per_token_loss[seq_idx][seq_mask].sum().item()
                        seq_tokens = seq_mask.sum().item()
                        seq_ppl = math.exp(min(seq_loss / seq_tokens, 100))
                        sequence_perplexities.append(seq_ppl)
            
            if batch_idx % 10 == 0 and metrics['total_tokens'] > 0:
                current_ppl = math.exp(min(metrics['batch_loss'] / max(metrics['total_tokens'], 1), 100))
                print(f"Batch {batch_idx}: tokens={metrics['total_tokens']}, perplexity={current_ppl:.4f}")
    
    # Calculate final metrics
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    
    results = {
        'batches_evaluated': total_batches,
        'tokens_evaluated': total_tokens,
        'average_loss': float(avg_loss),
        'perplexity': float(perplexity),
        'sequence_perplexities': sequence_perplexities  # Add sequence perplexities to results
    }
    
    print("\nPerplexity evaluation complete:")
    for key, value in results.items():
        if key != 'sequence_perplexities':  # Don't print the list of perplexities
            print(f"  {key}: {value}")
    
    return results

def calculate_perplexity_with_disk_offloading(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    datamodule: LifeLightningDataModule,
    device: str = "cuda",
    num_batches: Optional[int] = None,
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Calculate perplexity using the model and dataloader, saving sequence perplexities to disk
    to reduce memory usage.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with evaluation data
        datamodule: DataModule for batch processing
        device: Device to run evaluation on
        num_batches: Number of batches to evaluate (None for all)
        output_dir: Directory to save intermediate perplexity files
        
    Returns:
        Dict with perplexity metrics and path to perplexity files
    """
    import os
    import pickle
    
    print("-" * 50)
    print(f"Starting perplexity calculation on {device}")
    
    # Create output directory for perplexity files
    if output_dir is None:
        output_dir = Path("perplexity_data")
    perp_dir = output_dir / "sequence_perplexities"
    perp_dir.mkdir(parents=True, exist_ok=True)
    
    # Track metrics
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    
    # Track batch perplexity files
    perplexity_files = []
    
    # Check that evaluate_batch_perplexity_accuracy is available in model
    if not hasattr(model, 'evaluate_batch_perplexity_accuracy'):
        raise AttributeError(f"Model {model.__class__.__name__} does not have evaluate_batch_perplexity_accuracy method")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calculating perplexity", total=num_batches)):
            if num_batches is not None and batch_idx >= num_batches:
                break
                
            # Process batch
            batch = datamodule.transfer_batch_to_device(batch, device, 0)
            if not batch or 'event' not in batch:
                continue
                
            try:
                batch = datamodule.on_after_batch_transfer(batch, 0)
                if 'attn_mask' not in batch:
                    continue
            except Exception as e:
                print(f"Error preparing batch: {e}")
                continue
            
            # Calculate perplexity metrics using the model's method
            metrics = model.evaluate_batch_perplexity_accuracy(batch, pad_token_id=0,
                                                               return_token_loss=True)
            
            # Accumulate metrics
            total_loss += metrics['batch_loss']
            total_tokens += metrics['total_tokens']
            total_batches += 1
            
            # Calculate per-sequence perplexity if token_losses is available
            sequence_perplexities = []
            if 'per_token_loss' in metrics:
                # Reshape losses to get per-sequence values
                per_token_loss = metrics['per_token_loss']
                valid_mask = metrics['valid_mask_flat']
                
                # Get the original batch shape
                batch_size = batch['event'].size(0)
                seq_len = batch['event'].size(1) - 1  # -1 because targets are shifted
                
                # Reshape to (batch_size, seq_len)
                per_token_loss = per_token_loss.view(batch_size, seq_len)
                valid_mask = valid_mask.view(batch_size, seq_len)
                
                # Calculate per-sequence metrics
                for seq_idx in range(batch_size):
                    seq_mask = valid_mask[seq_idx]
                    if seq_mask.sum() > 0:  # Only process sequences with valid tokens
                        seq_loss = per_token_loss[seq_idx][seq_mask].sum().item()
                        seq_tokens = seq_mask.sum().item()
                        seq_ppl = math.exp(min(seq_loss / seq_tokens, 100))
                        sequence_perplexities.append(seq_ppl)
            
            # Save perplexities to disk
            if sequence_perplexities:
                batch_perp_file = perp_dir / f"perplexities_batch_{batch_idx}.pkl"
                with open(batch_perp_file, 'wb') as f:
                    pickle.dump(sequence_perplexities, f)
                perplexity_files.append(batch_perp_file)
            
            if batch_idx % 10 == 0 and metrics['total_tokens'] > 0:
                current_ppl = math.exp(min(metrics['batch_loss'] / max(metrics['total_tokens'], 1), 100))
                print(f"Batch {batch_idx}: tokens={metrics['total_tokens']}, perplexity={current_ppl:.4f}")
    
    # Calculate final metrics
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    
    results = {
        'batches_evaluated': total_batches,
        'tokens_evaluated': total_tokens,
        'average_loss': float(avg_loss),
        'perplexity': float(perplexity),
        'perplexity_files': perplexity_files,  # Store file paths instead of actual values
        'perplexity_dir': perp_dir
    }
    
    print("\nPerplexity evaluation complete:")
    for key, value in results.items():
        if key not in ['perplexity_files', 'perplexity_dir']: 
            print(f"  {key}: {value}")
    print(f"  Sequence perplexities saved to: {perp_dir}")
    print(f"  Number of perplexity files: {len(perplexity_files)}")
    
    return results


def plot_perplexity_distribution(perplexities, output_path=None):
    """
    Plot the distribution of sequence-level perplexities using only matplotlib.
    
    Args:
        perplexities: List of perplexity values for each sequence
        output_path: Path to save the plot (if None, plot will be displayed)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy array and handle any inf/nan values
    perplexities = np.array(perplexities)
    perplexities = perplexities[np.isfinite(perplexities)]
    
    # Set the style and create figure
    plt.figure(figsize=(12, 8))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate statistics
    mean_ppl = np.mean(perplexities)
    median_ppl = np.median(perplexities)
    min_ppl = np.min(perplexities)
    max_ppl = np.min([np.max(perplexities), 1000])  # Cap at 1000 for visualization
    
    # Create histogram
    n, bins, patches = plt.hist(perplexities, bins=50, alpha=0.7, color='steelblue', density=True)
    
    # Add a kernel density estimate (similar to KDE from seaborn)
    from scipy.stats import gaussian_kde
    density = gaussian_kde(perplexities)
    x_range = np.linspace(min_ppl, max_ppl, 1000)
    plt.plot(x_range, density(x_range), 'r-', linewidth=2)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_ppl, color='red', linestyle='--', 
                label=f'Mean: {mean_ppl:.2f}')
    plt.axvline(median_ppl, color='green', linestyle='-', 
                label=f'Median: {median_ppl:.2f}')
    
    # Set labels and title
    plt.xlabel('Perplexity', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distribution of Sequence-Level Perplexities', fontsize=16)
    
    # Add statistics annotation
    stats_text = (f"Min: {min_ppl:.2f}\nMax: {max_ppl:.2f}\n"
                 f"Mean: {mean_ppl:.2f}\nMedian: {median_ppl:.2f}\n"
                 f"Total Sequences: {len(perplexities)}")
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                fontsize=12, verticalalignment='top')
    
    plt.legend()
    plt.tight_layout()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def plot_saved_perplexities(perplexity_dir: Path, output_path: Optional[Path] = None):
    """
    Load perplexities from saved files and create a distribution plot.
    
    Args:
        perplexity_dir: Directory containing perplexity files
        output_path: Path to save the plot (if None, plot will be displayed)
    """
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    
    # Load all perplexity values from files
    perplexities = []
    for file_path in tqdm(list(perplexity_dir.glob("perplexities_batch_*.pkl")), 
                        desc="Loading perplexity files"):
        try:
            with open(file_path, 'rb') as f:
                batch_perplexities = pickle.load(f)
                perplexities.extend(batch_perplexities)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(perplexities)} perplexity values")
    
    # Create plot using the same plotting function
    plot_perplexity_distribution(perplexities, output_path)
    
    return perplexities

if __name__ == "__main__":
     # ----- Configuration -----
    HPARAMS_PATH = FPATH.CONFIGS / "hparams_pretrain2.yaml"
    HPARAMS_PATH = Path('logs/transformer_logs/stable_pretrain/8192/hparams.yaml')

    if not HPARAMS_PATH.exists(): 
        raise FileNotFoundError(f"Hparams file not found: {HPARAMS_PATH}")
    
    with open(HPARAMS_PATH, "r") as stream: 
        hparams = yaml.safe_load(stream)

    # Model checkpoint paths
    EXPERIMENT_NAME = hparams.get("experiment_name", "default_experiment")
    RUN_ID = ""  # Or load dynamically
    CHECKPOINT_NAME = "best.ckpt"
    CHECKPOINT_DIR = FPATH.CHECKPOINTS / EXPERIMENT_NAME / RUN_ID
    CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME
    CHECKPOINT_PATH = FPATH.CHECKPOINTS / 'stable_pretrain'/ '8192' / 'last.ckpt'
    
    # Data paths
    DATA_DIR_PATH = Path('/home/louibo/ensemble/data/life_test_compiled')
    LMDB_PATH = DATA_DIR_PATH / "dataset.lmdb"
    VOCAB_PATH = DATA_DIR_PATH / "vocab.json"
    PNR_MAP_PATH = DATA_DIR_PATH / "pnr_to_database_idx.json"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    STAGE = 'predict'  # or 'validate', 'test'
    # Generation parameters
    MODEL_CLASS = CausalEncoder

    
    # ----- Setup -----
    print("Instantiating DataModule...")
    datamodule = LifeLightningDataModule(
        dir_path=DATA_DIR_PATH,
        lmdb_path=LMDB_PATH,
        vocab_path=VOCAB_PATH,
        pnr_to_idx_path=PNR_MAP_PATH,
        background_length=0,
        cls_token=True,
        sep_token=False,
        segment=False,
        max_seq_len=hparams["max_seq_len"],
        batch_size=hparams["batch_size"],
        num_workers=0
    )
    
    # Load model
    model, _= load_model_from_checkpoint(
        model_class=MODEL_CLASS,
        checkpoint_path=CHECKPOINT_PATH,
        hparams_path=HPARAMS_PATH,
        device=DEVICE
    )
    
    # Get dataloader
    dataloader = get_dataloader(
        datamodule=datamodule,
        stage=STAGE
    )
    
    # Prepare model information for model card
    model_info = {
        "model_name": MODEL_CLASS.__name__,
        "experiment_name": EXPERIMENT_NAME,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "hparams": {
            key: value for key, value in hparams.items() 
            if isinstance(value, (str, int, float, bool, list, dict))
        }
    }
    
   

    # # Calculate perplexity on validation data
    # validation_results = calculate_perplexity(
    #     model=model,
    #     dataloader=dataloader,
    #     datamodule=datamodule,
    #     device=DEVICE,
    #     num_batches=1
    # )


     # Calculate perplexity on validation data
    validation_results = calculate_perplexity(
        model=model,
        dataloader=dataloader,
        datamodule=datamodule,
        device=DEVICE,
        num_batches=1  # Adjust as needed
    )
    
    output_dir = Path(f"perplexity_data/{EXPERIMENT_NAME}")
    validation_results = calculate_perplexity_with_disk_offloading(
        model=model,
        dataloader=dataloader,
        datamodule=datamodule,
        device=DEVICE,
        num_batches=None,  # Process all batches
        output_dir=output_dir
    )
    
    # Create output directory for plots if it doesn't exist
    plot_dir = Path(f"perplexity_plots/{EXPERIMENT_NAME}")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plot filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = plot_dir / f"perplexity_distribution_{timestamp}.png"
    
    # Load saved perplexities and create plot
    plot_saved_perplexities(validation_results['perplexity_dir'], plot_path)