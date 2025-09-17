#!/usr/bin/env python3
"""
A wrapper script to run the single-GPU generation script in parallel.

This launcher splits the total number of simulations across all available GPUs,
runs an independent process for each, and then aggregates the results into a
single final output.

This script does NOT require any modification to the original script.

Usage:
    python run_parallel_ensemble.py --model_path /path/to/checkpoint --num_people 100 --num_simulations 500
"""
import argparse
import os
import sys
import subprocess
import torch
import yaml
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import json

# We need the exact save function from the original script to maintain the format.
# It's self-contained, so we can copy it here for convenience.
def save_ensemble_data(ensemble_data, save_folder):
    """Save ensemble data to structured folder. (Copied from original script)"""
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_dir = save_path / f"ensemble_{timestamp}"
    ensemble_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving aggregated ensemble data to: {ensemble_dir}")
    
    metadata_path = ensemble_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(ensemble_data['metadata'], f, default_flow_style=False)
    
    person_data_path = ensemble_dir / "person_data.pt"
    person_data = {
        'person_ids': ensemble_data['person_ids'],
        'prompt_lengths': ensemble_data['prompt_lengths']
    }
    torch.save(person_data, person_data_path)
    
    sequences_path = ensemble_dir / "sequences.pt"
    torch.save({
        'full_sequences': ensemble_data['generated_sequences'],
        'generated_only': ensemble_data['generated_events'],
    }, sequences_path)
    
    summary_path = ensemble_dir / "summary.json"
    num_people = ensemble_data['metadata']['num_people']
    num_sims = ensemble_data['metadata']['num_simulations']
    prompt_len = ensemble_data['metadata']['prompt_length']
    max_new = ensemble_data['metadata']['max_new_tokens']
    
    summary_data = {
        "ensemble_summary": {
            "description": "Aggregated Ensemble Life Generation Summary",
            "generated_on": ensemble_data['metadata']['generation_timestamp'],
            "dataset": {
                "num_people": num_people,
                "simulations_per_person": num_sims,
                "prompt_length": prompt_len,
                "max_new_tokens": max_new
            },
            "generation_parameters": {
                "temperature": ensemble_data['metadata']['temperature'],
                "top_k": ensemble_data['metadata']['top_k'],
                "top_p": ensemble_data['metadata']['top_p']
            },
            "model_info": {
                "model_path": ensemble_data['metadata']['model_path'],
                "experiment_name": ensemble_data['metadata']['experiment_name']
            },
            "data_structure": {
                "sequences_pt": {
                    "full_sequences": {"shape": list(ensemble_data['generated_sequences'].shape)},
                    "generated_only": {"shape": list(ensemble_data['generated_events'].shape)}
                },
                "person_data_pt": "person_ids and prompt_lengths",
                "metadata_yaml": "Complete generation configuration"
            }
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"✓ Saved aggregated ensemble data successfully.")
    print(f"  - Output directory: {ensemble_dir}")
    return ensemble_dir

def main():
    # --- Step 1: Parse all arguments, same as the original script ---
    # This ensures we can pass them through to the child processes.
    # The name of the original script file.
    # IMPORTANT: Change this if your file is named differently.
    original_script_name = "scripts/generate_sequences.py"
    
    parser = argparse.ArgumentParser(description=f"Parallel launcher for {original_script_name}")
    # All arguments from the original script are copied here
    parser.add_argument("--model_path", type=str, 
                       default="checkpoints/transformer/destiny/021_muddy_cobra-pretrain-lr0.0003/best.ckpt",
                       help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--config_path", type=str, 
                       help="Path to config YAML file")
    parser.add_argument("--experiment_name", type=str,
                       default="destiny/021_muddy_cobra-pretrain-lr0.0003",
                       help="Experiment name (loads config from TB logs)")

    parser.add_argument("--prompt_length", type=int, default=1000, help="Truncate prompts to this length (None = use full)")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_compiled_dataset", action="store_true", default=True, help="Use pre-compiled dataset")
    parser.add_argument("--use_source_files", action="store_true", help="Use source files instead of compiled dataset")
    parser.add_argument("--num_people", type=int, default=2, help="Number of people to generate life sequences for")
    parser.add_argument("--num_simulations", type=int, default=10, help="*Total* number of simulations per person across all GPUs")
    parser.add_argument("--save_folder", type=str, default="generated", help="Final folder name to save aggregated results")
    
    args = parser.parse_args()

    # --- Step 2: Set up the parallel execution ---
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("Error: No GPUs found. This script requires at least one GPU.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {world_size} GPUs. Splitting work...")

    # Split simulations among GPUs
    sims_per_gpu = args.num_simulations // world_size
    remainder = args.num_simulations % world_size
    sims_distribution = [sims_per_gpu] * world_size
    for i in range(remainder):
        sims_distribution[i] += 1

    print(f"Total simulations: {args.num_simulations}. Distribution per GPU: {sims_distribution}")

    # Create a main temporary directory to hold results from each process
    temp_dir = tempfile.mkdtemp(prefix="ensemble_parallel_")
    print(f"Using temporary directory for partial results: {temp_dir}")

    processes = []
    # --- Step 3: Launch a process for each GPU ---
    for rank in range(world_size):
        if sims_distribution[rank] == 0:
            continue # Skip GPUs that have no work

        # Each process gets a unique folder inside the main temp dir
        gpu_temp_save_path = Path(temp_dir) / f"gpu_{rank}"
        gpu_temp_save_path.mkdir()

        # Build the command for the subprocess
        cmd = [
            sys.executable,  # Use the same python interpreter
            original_script_name,
            "--ensemble_mode" # This must be enabled
        ]

        # Pass through all arguments, overriding where necessary
        for arg, value in vars(args).items():
            if value is None:
                continue
            if arg == 'num_simulations':
                cmd.extend([f"--{arg.replace('_', '_')}", str(sims_distribution[rank])])
            elif arg == 'save_folder':
                cmd.extend([f"--{arg.replace('_', '_')}", str(gpu_temp_save_path)])
            elif arg == 'seed':
                 cmd.extend([f"--{arg.replace('_', '_')}", str(args.seed + rank)]) # Different seed per proc
            elif isinstance(value, bool):
                if value:
                    cmd.append(f"--{arg.replace('_', '_')}")
            else:
                cmd.extend([f"--{arg.replace('_', '_')}", str(value)])
        
        # Set the environment variable to assign this process to a specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)

        print(f"\n[Launcher] Spawning process for GPU {rank} with {sims_distribution[rank]} simulations.")
        print(f"  > Command: {' '.join(cmd)}")
        
        # Launch the process
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)

    # --- Step 4: Wait for all processes to complete ---
    for proc in processes:
        proc.wait()

    print("\nAll generation processes have finished.")

    # --- Step 5: Aggregate the results ---
    print("Aggregating results from all GPUs...")
    
    all_sequences_tensors = []
    all_generated_only_tensors = []
    final_person_data = None
    final_metadata = None

    for rank in range(world_size):
        if sims_distribution[rank] == 0:
            continue

        gpu_temp_save_path = Path(temp_dir) / f"gpu_{rank}"
        try:
            # The original script creates a timestamped folder, find it
            ensemble_output_dir = next(gpu_temp_save_path.glob("ensemble_*"))
            print(f"  > Loading results from: {ensemble_output_dir}")

            # Load the sequence tensors
            seq_data = torch.load(ensemble_output_dir / "sequences.pt", map_location='cpu')
            all_sequences_tensors.append(seq_data['full_sequences'])
            all_generated_only_tensors.append(seq_data['generated_only'])

            # Load metadata and person_ids only from the first GPU (they are all the same)
            if rank == 0:
                final_person_data = torch.load(ensemble_output_dir / "person_data.pt")
                with open(ensemble_output_dir / "metadata.yaml", "r") as f:
                    final_metadata = yaml.safe_load(f)

        except (StopIteration, FileNotFoundError) as e:
            print(f"Warning: Could not find or load results from GPU {rank}'s directory. Skipping. Error: {e}")
            continue

    if not all_sequences_tensors:
        print("Error: No results were found to aggregate. Aborting.", file=sys.stderr)
        shutil.rmtree(temp_dir)
        sys.exit(1)
        
    # Concatenate along the 'simulations' dimension (dim=1)
    final_sequences = torch.cat(all_sequences_tensors, dim=1)
    final_generated_only = torch.cat(all_generated_only_tensors, dim=1)
    
    # Update the metadata to reflect the total number of simulations
    final_metadata['num_simulations'] = args.num_simulations
    
    final_ensemble_data = {
        'person_ids': final_person_data['person_ids'],
        'prompt_lengths': final_person_data['prompt_lengths'],
        'generated_sequences': final_sequences,
        'generated_events': final_generated_only,
        'metadata': final_metadata
    }

    # --- Step 6: Save the final aggregated file and clean up ---
    save_ensemble_data(final_ensemble_data, args.save_folder)

    try:
        shutil.rmtree(temp_dir)
        print(f"Successfully cleaned up temporary directory: {temp_dir}")
    except OSError as e:
        print(f"Error cleaning up temporary directory {temp_dir}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()