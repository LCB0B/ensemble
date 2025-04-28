import os
os.environ["POLARS_MAX_THREADS"] = "4"
os.environ["RAYON_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import yaml
import torch
import warnings
import pickle
from pathlib import Path
import json
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Type
from datetime import datetime

# Import your relevant classes
from src.datamodule4 import LifeLightningDataModule
from src.encoder_nano_risk import PretrainNanoEncoder, CausalEncoder
from src.paths import FPATH

warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool")

def load_model_from_checkpoint(
    model_class: Type[torch.nn.Module],
    checkpoint_path: Path,
    hparams: Dict[str, Any],
    device: str = "cpu"
) -> torch.nn.Module:
    """Loads a model from a Lightning checkpoint."""
    print(f"Loading model class '{model_class.__name__}' from checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(device),
        strict=False,
        **hparams
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    return model

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


def generate_sequences(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    datamodule: LifeLightningDataModule,
    device: str,
    prompt_length: int = 500,
    max_new_tokens: int = 100,
    num_simulations: int = 10,
    num_batches: Optional[int] = None,
    generation_strategy: str = "top_p",
    output_dir: Path = None,
    model_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generates sequences given prompts from the dataloader.
    
    Args:
        model: The model to use for generation
        dataloader: DataLoader containing sequences to use as prompts
        datamodule: DataModule for batch processing
        device: Device to run on
        prompt_length: Length of prompt to use from each sequence
        max_new_tokens: Maximum number of new tokens to generate
        num_simulations: Number of simulations to run per prompt
        num_batches: Maximum number of batches to process (None for all)
        generation_strategy: Strategy for generation (most_likely, top_k, top_p)
        output_dir: Directory to save generation results
        model_info: Dictionary containing model information for the model card
        
    Returns:
        Dictionary with generation metadata
    """
    print("-" * 50)
    print(f"Starting sequence generation with {generation_strategy} strategy...")
    
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths for saving results
        sequences_path = output_dir / "generated_sequences.pkl"
        prompts_path = output_dir / "prompts.pkl"
        original_batches_path = output_dir / "original_batches.pkl"
        metadata_path = output_dir / "generation_metadata.json"
        model_card_path = output_dir / "model_card.json"
        
        # Initialize files
        with open(sequences_path, "wb") as f:
            pickle.dump([], f)
        with open(prompts_path, "wb") as f:
            pickle.dump([], f)
        with open(original_batches_path, "wb") as f:
            pickle.dump([], f)
    
    # Track generation metadata
    metadata = {
        "total_sequences": 0,
        "batches_processed": 0,
        "max_new_tokens": max_new_tokens,
        "prompt_length": prompt_length,
        "generation_strategy": generation_strategy,
        "num_simulations": num_simulations,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating sequences",total=num_batches)):
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
                
            # Store batch size and update counters
            batch_size = batch['event'].size(0)
            metadata["batches_processed"] += 1
            metadata["total_sequences"] += batch_size * num_simulations
            
            # Store the prompts (beginning of each sequence)
            prompts = batch['event'][:, :prompt_length].clone()
            
            # Create a copy of the original batch for saving
            original_batch = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            
            # Run multiple simulations
            all_generations = []
            
            for sim in range(num_simulations):
                try:
                    # Generate sequences
                    generated = model.generate_sequence(
                        batch=batch,
                        prompt_length=prompt_length,
                        max_new_tokens=max_new_tokens,
                        strategy=generation_strategy
                    )
                    
                    # Add to collection
                    all_generations.append(generated)
                    
                except Exception as e:
                    print(f"Error during generation (simulation {sim+1}): {e}")
            
            # Save results for this batch
            if output_dir and all_generations:
                # Stack all simulations into a tensor of shape [batch_size, num_simulations, seq_len]
                stacked_generations = torch.stack(all_generations, dim=1)
                
                # Save generated sequences
                with open(sequences_path, "rb") as f:
                    existing = pickle.load(f)
                existing.append(stacked_generations.cpu())
                with open(sequences_path, "wb") as f:
                    pickle.dump(existing, f)
                
                # Save prompts
                with open(prompts_path, "rb") as f:
                    existing = pickle.load(f)
                existing.append(prompts.cpu())
                with open(prompts_path, "wb") as f:
                    pickle.dump(existing, f)
                
                # Save original batches
                #remove the 'attn_mask' key from the original batch (cannot be pickled)
                original_batch.pop('attn_mask', None)
                with open(original_batches_path, "rb") as f:
                    existing = pickle.load(f)
                existing.append(original_batch)
                with open(original_batches_path, "wb") as f:
                    pickle.dump(existing, f)
                
                print(f"Saved batch {batch_idx+1} (size: {batch_size})")
    
    # Finalize results
    if output_dir:
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create and save model card
        model_card = {
            "model_name": model.__class__.__name__,
            "model_parameters": {
                param_name: getattr(model, param_name) 
                for param_name in ["vocab_size", "n_layer", "n_head", "n_embd", "max_seq_len"]
                if hasattr(model, param_name)
            },
            "generation_parameters": {
                "prompt_length": prompt_length,
                "max_new_tokens": max_new_tokens,
                "generation_strategy": generation_strategy,
                "num_simulations": num_simulations
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add any additional model info provided
        if model_info:
            model_card.update(model_info)
        
        # Save model card
        try:
            with open(model_card_path, "w") as f:
                json.dump(model_card, f, indent=2)
        except TypeError:
            # Handle non-serializable objects
            print("Warning: Some model parameters couldn't be serialized to JSON.")
            # Create a simplified model card
            simple_model_card = {
                "model_name": model.__class__.__name__,
                "generation_parameters": {
                    "prompt_length": prompt_length,
                    "max_new_tokens": max_new_tokens,
                    "generation_strategy": generation_strategy,
                    "num_simulations": num_simulations
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(model_card_path, "w") as f:
                json.dump(simple_model_card, f, indent=2)
        
        print(f"Generation complete. Results saved to {output_dir}")
    
    return metadata


if __name__ == "__main__":
    # ----- Configuration (Updated to match load_model.py) -----
    # Load hparams from logs like load_model.py does
    HPARAMS_PATH = Path('logs/transformer_logs/stable_pretrain/8192/hparams.yaml')
    if not HPARAMS_PATH.exists():
        raise FileNotFoundError(f"Hparams file not found: {HPARAMS_PATH}")
    
    with open(HPARAMS_PATH, "r") as stream: 
        hparams = yaml.safe_load(stream)

    # Model checkpoint paths (match load_model.py)
    EXPERIMENT_NAME = hparams["experiment_name"]
    RUN_ID = ""  # Or load dynamically
    CHECKPOINT_NAME = "last.ckpt"
    CHECKPOINT_PATH = Path("checkpoints/stable_pretrain/8192") / CHECKPOINT_NAME
    
    # Data paths
    #DATA_DIR_PATH = FPATH.DATA / hparams["dir_path"]
    DATA_DIR_PATH = Path('/home/louibo/ensemble/data/life_test_compiled')
    LMDB_PATH = DATA_DIR_PATH / "dataset.lmdb"
    VOCAB_PATH = DATA_DIR_PATH / "vocab.json"
    PNR_MAP_PATH = DATA_DIR_PATH / "pnr_to_database_idx.json"
    
    # Generation parameters
    MODEL_CLASS = CausalEncoder
    STAGE = 'predict'
    BATCH_SIZE = 32
    NUM_BATCHES = 3
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    PROMPT_LENGTH = 2000
    MAX_NEW_TOKENS = 200
    NUM_SIMULATIONS = 20
    GENERATION_STRATEGY = "top_p"  # 'most_likely', 'top_k', 'top_p'
    
    # Output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = FPATH.GENERATED / EXPERIMENT_NAME / f"generated_sequences_{timestamp}_{PROMPT_LENGTH}_{MAX_NEW_TOKENS}_{NUM_SIMULATIONS}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ----- Load background data (similar to load_model.py) -----
    background = None
    BACKGROUND_PATH = FPATH.DATA / hparams['sources'] / "background.parquet"
    if BACKGROUND_PATH.exists():
        import polars as pl
        print(f"Loading background data from {BACKGROUND_PATH}")
        background = pl.read_parquet(BACKGROUND_PATH)
    
    # ----- Setup -----
    print("Instantiating DataModule...")
    datamodule = LifeLightningDataModule(
        dir_path=DATA_DIR_PATH,
        lmdb_path=LMDB_PATH,
        vocab_path=VOCAB_PATH,
        pnr_to_idx_path=PNR_MAP_PATH,
        # Updated to include background like in load_model.py
        background=background,
        cls_token=True,
        sep_token=False,
        segment=False,
        max_seq_len=hparams["max_seq_len"],
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    # Load model
    model = load_model_from_checkpoint(
        model_class=MODEL_CLASS,
        checkpoint_path=CHECKPOINT_PATH,
        hparams=hparams,
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
    
    # ----- Generate sequences -----
    metadata = generate_sequences(
        model=model,
        dataloader=dataloader,
        datamodule=datamodule,
        device=DEVICE,
        prompt_length=PROMPT_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        num_simulations=NUM_SIMULATIONS,
        num_batches=NUM_BATCHES,
        generation_strategy=GENERATION_STRATEGY,
        output_dir=OUTPUT_DIR,
        model_info=model_info
    )
    
    # ----- Print summary -----
    print("\n=== Generation Summary ===")
    print(f"Total sequences: {metadata['total_sequences']}")
    print(f"Batches processed: {metadata['batches_processed']}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=========================")