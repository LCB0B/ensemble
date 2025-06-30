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


import os
import torch
import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime
import json

def launch_parallel_generation(
    num_processes: int = 4,
    total_batches: int = 40,
    gpu_id: int = 0,
    base_output_dir: str = None,
    prompt_length: int = 1000,
    max_new_tokens: int = 200,
    simulations_per_process: int = 25,  # Split simulations among processes
    combine_results: bool = True  # New parameter to control combining
):
    """
    Launch multiple generation processes in parallel on the same GPU.
    
    Args:
        num_processes: Number of parallel processes to run
        total_batches: Total number of batches to process across all processes
        gpu_id: GPU ID to use
        base_output_dir: Base directory for outputs
        prompt_length: Length of prompt for generation
        max_new_tokens: Maximum new tokens to generate
        simulations_per_process: Number of simulations per process
    """
    ## Calculate batches per process
    batches_per_process = total_batches // num_processes
    
    # Create base output directory with the desired format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_simulations = simulations_per_process * num_processes
    
    if base_output_dir is None:
        from src.paths import FPATH
        # Create directory with the requested format
        base_output_dir = Path(FPATH.GENERATED) / "stable_pretrain" / f"generated_sequences_parallel_{timestamp}_{prompt_length}_{max_new_tokens}_{total_simulations}"
    else:
        base_output_dir = Path(base_output_dir)
    
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base output directory: {base_output_dir}")
    
    
    # Save configuration
    config = {
        "num_processes": num_processes,
        "total_batches": total_batches,
        "batches_per_process": batches_per_process,
        "gpu_id": gpu_id,
        "prompt_length": prompt_length,
        "max_new_tokens": max_new_tokens,
        "simulations_per_process": simulations_per_process,
        "timestamp": timestamp
    }
    
    with open(base_output_dir / "parallel_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Launch processes
    processes = []
    process_dirs = []
    
    print(f"Launching {num_processes} parallel generation processes...")
    
    for i in range(num_processes):
        # Determine batch range for this process
        start_batch = i * batches_per_process
        num_batches = batches_per_process
        
        # Create output directory for this process
        process_dir = base_output_dir / f"process_{i}"
        process_dir.mkdir(exist_ok=True)
        process_dirs.append(process_dir)
        
        # Modify environment with GPU settings
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Build command
        cmd = [
            "python", "scripts/generate_life.py",
            "--num_batches", str(num_batches),
            "--start_batch", str(start_batch),
            "--prompt_length", str(prompt_length),
            "--max_new_tokens", str(max_new_tokens),
            "--num_simulations", str(simulations_per_process),
            "--output_dir", str(process_dir)
        ]
        
        # Launch process
        print(f"Launching process {i}: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(process)
    
    # Monitor processes
    while any(p.poll() is None for p in processes):
        time.sleep(300) 
        statuses = []
        for i, p in enumerate(processes):
            if p.poll() is None:
                statuses.append(f"Process {i}: Running")
            else:
                statuses.append(f"Process {i}: Exited ({p.returncode})")
        print(" | ".join(statuses))
    
    # Check results
    all_successful = True
    for i, p in enumerate(processes):
        if p.returncode != 0:
            all_successful = False
            print(f"Process {i} failed with code {p.returncode}")
            print("STDOUT:")
            print(p.stdout.read())
            print("STDERR:")
            print(p.stderr.read())
    
    if all_successful:
        print("All processes completed successfully!")
        
        # Optionally combine results
        print(f"Generation results saved in: {base_output_dir}")
        print("Each process directory contains:")
        for i, dir_path in enumerate(process_dirs):
            print(f"  Process {i}: {dir_path}")
        
        # Combine results if requested
        if combine_results:
            combined_dir = combine_parallel_outputs(base_output_dir)
            print(f"Combined output directory: {combined_dir}")
            return combined_dir
    
    return base_output_dir

def combine_parallel_outputs(base_output_dir: Path) -> Path:
    """
    Combine outputs from parallel generation processes into a single directory.
    
    Args:
        base_output_dir: Path to the base output directory containing process_X subdirectories
        
    Returns:
        Path to the combined output directory
    """
    print(f"Combining outputs from {base_output_dir}...")
    
    # Create combined directory
    combined_dir = base_output_dir
    combined_dir.mkdir(exist_ok=True)
    
    # Find all process directories
    process_dirs = sorted([d for d in base_output_dir.glob("process_*") if d.is_dir()])
    
    if not process_dirs:
        print("No process directories found!")
        return base_output_dir
    
    print(f"Found {len(process_dirs)} process directories")
    
    # Initialize empty lists for combined data
    all_sequences = []
    all_prompts = []
    all_original_batches = []
    total_sequences = 0
    total_batches = 0
    generation_params = None
    model_info = None
    
    # Process each directory
    for process_dir in process_dirs:
        print(f"Processing {process_dir}")
        
        # Load sequences
        sequences_path = process_dir / "generated_sequences.pkl"
        if sequences_path.exists():
            with open(sequences_path, "rb") as f:
                sequences = pickle.load(f)
                all_sequences.extend(sequences)
        
        # Load prompts
        prompts_path = process_dir / "prompts.pkl"
        if prompts_path.exists():
            with open(prompts_path, "rb") as f:
                prompts = pickle.load(f)
                all_prompts.extend(prompts)
        
        # Load original batches
        original_batches_path = process_dir / "original_batches.pkl"
        if original_batches_path.exists():
            with open(original_batches_path, "rb") as f:
                batches = pickle.load(f)
                all_original_batches.extend(batches)
        
        # Load metadata to aggregate statistics
        metadata_path = process_dir / "generation_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                total_sequences += metadata.get("total_sequences", 0)
                total_batches += metadata.get("batches_processed", 0)
                generation_params = generation_params or {
                    k: v for k, v in metadata.items() 
                    if k in ["max_new_tokens", "prompt_length", "generation_strategy"]
                }
        
        # Get model info from any process (should be the same in all)
        if not model_info:
            model_card_path = process_dir / "model_card.json"
            if model_card_path.exists():
                with open(model_card_path, "r") as f:
                    model_info = json.load(f)
    
    # Save combined sequences
    print("Saving combined sequences...")
    with open(combined_dir / "generated_sequences.pkl", "wb") as f:
        pickle.dump(all_sequences, f)
    
    # Save combined prompts
    print("Saving combined prompts...")
    with open(combined_dir / "prompts.pkl", "wb") as f:
        pickle.dump(all_prompts, f)
    
    # Save combined original batches
    print("Saving combined original batches...")
    with open(combined_dir / "original_batches.pkl", "wb") as f:
        pickle.dump(all_original_batches, f)
    
    # Save combined metadata
    combined_metadata = {
        "total_sequences": total_sequences,
        "batches_processed": total_batches,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "combined_from_processes": len(process_dirs)
    }
    
    # Add generation parameters if available
    if generation_params:
        combined_metadata.update(generation_params)
    
    with open(combined_dir / "generation_metadata.json", "w") as f:
        json.dump(combined_metadata, f, indent=2)
    
    # Save model card
    if model_info:
        with open(combined_dir / "model_card.json", "w") as f:
            json.dump(model_info, f, indent=2)
    
    # Copy parallel config file if exists
    config_path = base_output_dir / "parallel_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        with open(combined_dir / "parallel_config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    print(f"Combined data saved to {combined_dir}")
    
    # Delete original process directories to save space
    print("Deleting temporary process directories...")
    for process_dir in process_dirs:
        import shutil
        shutil.rmtree(process_dir)
    
    print("Cleanup complete!")
    return combined_dir

# Modify the main function to accept command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel sequence generation")
    parser.add_argument("--processes", type=int, default=6, help="Number of parallel processes")
    parser.add_argument("--total_batches", type=int, default=6, help="Total batches to process")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--prompt_length", type=int, default=1000, help="Prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens")
    parser.add_argument("--simulations_per_process", type=int, default=200, help="Simulations per process")
    parser.add_argument("--no_combine", action="store_true", help="Don't combine results (keep separate process folders)")
    
    args = parser.parse_args()
    
    # Launch parallel generation
    output_dir = launch_parallel_generation(
        num_processes=args.processes,
        total_batches=args.total_batches,
        gpu_id=args.gpu_id,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        simulations_per_process=args.simulations_per_process,
        combine_results=not args.no_combine
    )
    
    print(f"Output directory: {output_dir}")