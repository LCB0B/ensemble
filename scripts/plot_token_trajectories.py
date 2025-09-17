#!/usr/bin/env python3
"""
Enhanced script to visualize token trajectories from ensemble generation data.
Works with actual tokenizer to decode tokens and extract values.
"""

import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
import sys
import os

# Add src to path to import tokenizer utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_ensemble_data(ensemble_dir):
    """Load ensemble data from a generated folder"""
    ensemble_path = Path(ensemble_dir)
    
    print(f"Loading ensemble data from: {ensemble_path}")
    
    # Load metadata
    with open(ensemble_path / "metadata.yaml", 'r') as f:
        metadata = yaml.safe_load(f)
    
    # Load JSON summary if available
    summary_json_path = ensemble_path / "summary.json"
    summary = None
    if summary_json_path.exists():
        with open(summary_json_path, 'r') as f:
            summary = json.load(f)
    
    # Load person data
    person_data = torch.load(ensemble_path / "person_data.pt")
    
    # Load sequence tensors
    sequences = torch.load(ensemble_path / "sequences.pt")
    
    data = {
        'metadata': metadata,
        'summary': summary,
        'person_ids': person_data['person_ids'],
        'prompt_lengths': person_data['prompt_lengths'],
        'full_sequences': sequences['full_sequences'],      # [num_people, num_simulations, max_seq_length]
        'generated_only': sequences['generated_only'],      # [num_people, num_simulations, max_new_tokens]
    }
    
    return data


def load_tokenizer_from_data_path(data_path):
    """
    Load tokenizer from the data path used in the model.
    This is a placeholder - you'll need to implement based on your tokenizer setup.
    """
    # Try to load vocabulary or tokenizer from the data directory
    vocab_path = Path(data_path) / "vocab.json"
    
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        # Create reverse mapping from token_id to token_string
        id_to_token = {v: k for k, v in vocab.items()}
        return id_to_token
    else:
        print(f"Warning: Could not find tokenizer at {vocab_path}")
        return None


def extract_token_values_with_tokenizer(sequence, token_prefix, id_to_token):
    """
    Extract values from tokens matching a prefix pattern using actual tokenizer.
    
    Args:
        sequence: List of token IDs
        token_prefix: Prefix pattern (e.g., "LAB_perindkialt_13_Q")
        id_to_token: Mapping from token ID to token string
    
    Returns:
        List of (position, value) tuples
    """
    values = []
    
    for pos, token_id in enumerate(sequence):
        if token_id == 0:  # Skip padding tokens
            continue
            
        # Convert token ID to string
        if token_id in id_to_token:
            token_str = id_to_token[token_id]
            
            # Check if token matches the prefix pattern
            if token_str.startswith(token_prefix):
                # Extract the quantile value
                # For income tokens: LAB_perindkialt_13_Q{XX} where XX is 1-100
                match = re.search(r'Q(\d+)', token_str)
                if match:
                    value = int(match.group(1))
                    values.append((pos, value))
    
    return values


def extract_token_values_heuristic(sequence, token_prefix="LAB_perindkialt_13_Q"):
    """
    Extract token values using heuristics when tokenizer is not available.
    This assumes income tokens are in a specific ID range.
    """
    values = []
    
    # Remove padding tokens
    clean_sequence = [token for token in sequence if token != 0]
    
    # Heuristic: assume income tokens are in a range based on their frequency
    # We'll look for tokens that appear in patterns suggesting they're income tokens
    
    # Get unique tokens and their frequencies
    unique_tokens = {}
    for token in clean_sequence:
        unique_tokens[token] = unique_tokens.get(token, 0) + 1
    
    # Sort by frequency (income tokens might appear less frequently)
    sorted_tokens = sorted(unique_tokens.items(), key=lambda x: x[1])
    
    # Heuristic: assume income tokens are in a specific range
    # This is dataset-specific and would need adjustment
    potential_income_tokens = []
    
    # Example heuristic: tokens in certain ranges might be income tokens
    for token_id, freq in sorted_tokens:
        # Adjust these ranges based on your actual tokenizer
        if 1000 <= token_id <= 2000:  # Example range
            potential_income_tokens.append(token_id)
    
    # Map positions to estimated quantiles
    for pos, token_id in enumerate(clean_sequence):
        if token_id in potential_income_tokens:
            # Estimate quantile based on token position in the range
            if len(potential_income_tokens) > 1:
                idx = potential_income_tokens.index(token_id)
                # Map to quantile 1-100
                quantile = int((idx / (len(potential_income_tokens) - 1)) * 99) + 1
                values.append((pos, quantile))
    
    return values


def plot_token_trajectories(data, person_idx, token_prefix="LAB_perindkialt_13_Q", 
                          save_dir="figures/income", id_to_token=None, 
                          value_name="Income Quantile"):
    """
    Plot token trajectories for a specific person.
    
    Args:
        data: Ensemble data dictionary
        person_idx: Index of the person to plot
        token_prefix: Token prefix to look for
        save_dir: Directory to save plots
        id_to_token: Mapping from token IDs to token strings
        value_name: Name of the value being plotted (for labels)
    """
    person_id = data['person_ids'][person_idx]
    prompt_length = data['prompt_lengths'][person_idx]
    
    # Get sequences for this person
    full_sequences = data['full_sequences'][person_idx]  # [num_simulations, max_seq_length]
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Extract values from each simulation
    all_trajectories = []
    prompt_values = None
    sequence_lengths = []
    
    for sim_idx in range(full_sequences.shape[0]):
        sequence = full_sequences[sim_idx].tolist()
        
        # Calculate actual sequence length (without padding)
        actual_length = len([token for token in sequence if token != 0])
        sequence_lengths.append(actual_length)
        
        # Extract values using tokenizer if available
        if id_to_token:
            values = extract_token_values_with_tokenizer(sequence, token_prefix, id_to_token)
        else:
            values = extract_token_values_heuristic(sequence, token_prefix)
        
        if values:
            positions, vals = zip(*values)
            all_trajectories.append((positions, vals))
            
            # Extract prompt values (first trajectory as reference)
            if prompt_values is None:
                prompt_positions = [p for p in positions if p < prompt_length]
                prompt_vals = [v for p, v in zip(positions, vals) if p < prompt_length]
                if prompt_positions:
                    prompt_values = (prompt_positions, prompt_vals)
    
    # Print debugging info
    if sequence_lengths:
        min_length = min(sequence_lengths)
        max_length = max(sequence_lengths)
        avg_length = sum(sequence_lengths) / len(sequence_lengths)
        short_sequences = sum(1 for length in sequence_lengths if length < prompt_length)
        
        print(f"    Sequence lengths: min={min_length}, max={max_length}, avg={avg_length:.1f}")
        print(f"    Prompt length: {prompt_length}")
        print(f"    Sequences shorter than prompt: {short_sequences}/{len(sequence_lengths)}")
        print(f"    Found {len(all_trajectories)} trajectories with {token_prefix} tokens")
    
    if not all_trajectories:
        print(f"No {value_name.lower()} trajectories found for {person_id}")
        return 0
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Separate trajectories into prompt and generated parts
    prompt_trajectories = []
    generated_trajectories = []
    
    for positions, vals in all_trajectories:
        # Split into prompt and generated parts
        prompt_pos = [p for p in positions if p < prompt_length]
        prompt_vals_part = [v for p, v in zip(positions, vals) if p < prompt_length]
        
        generated_pos = [p for p in positions if p >= prompt_length]
        generated_vals_part = [v for p, v in zip(positions, vals) if p >= prompt_length]
        
        if prompt_pos:
            prompt_trajectories.append((prompt_pos, prompt_vals_part))
        if generated_pos:
            generated_trajectories.append((generated_pos, generated_vals_part))
    
    # Plot prompt trajectories with low transparency (should be similar)
    for positions, vals in prompt_trajectories:
        plt.plot(positions, vals, 'r-', alpha=0.05, linewidth=1)
    
    # Plot generated trajectories with higher transparency
    for positions, vals in generated_trajectories:
        plt.plot(positions, vals, 'black', alpha=0.05, linewidth=1)
    
    # Plot prompt as solid line (average or first trajectory)
    if prompt_values:
        prompt_positions, prompt_vals = prompt_values
        plt.plot(prompt_positions, prompt_vals, 'r-', linewidth=3, 
                label=f'Prompt ({value_name})', zorder=10)
    
    # Add vertical line at generation start
    plt.axvline(x=prompt_length, color='black', linestyle='--', linewidth=2, 
                label=f'Generation Start (position {prompt_length})', zorder=5)
    
    # Customize plot
    plt.xlabel('Event Position in Sequence')
    plt.ylabel(f'{value_name} (1-100)')
    
    # Add more detailed title
    short_sequences = sum(1 for length in sequence_lengths if length < prompt_length)
    title = f'{value_name} Trajectories for {person_id}\n'
    title += f'{len(all_trajectories)} trajectories with {token_prefix} tokens, '
    title += f'prompt length: {prompt_length}, '
    title += f'short sequences: {short_sequences}/{len(sequence_lengths)}'
    
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(0, 101)
    
    # Add shaded region for prompt
    plt.axvspan(0, prompt_length, alpha=0.1, color='red', label='Prompt Region')
    
    # Save plot
    safe_prefix = token_prefix.replace('/', '_').replace('\\', '_')
    filename = f"{person_id}_{safe_prefix}_trajectories.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {person_id}: {save_path / filename}")
    
    return len(all_trajectories)


def plot_all_trajectories(data, token_prefix="LAB_perindkialt_13_Q", 
                         save_dir="figures/income", max_people=None,
                         id_to_token=None, value_name="Income Quantile"):
    """
    Plot trajectories for all people in the dataset.
    """
    num_people = len(data['person_ids'])
    if max_people:
        num_people = min(num_people, max_people)
    
    print(f"Creating {value_name.lower()} trajectory plots for {num_people} people...")
    
    trajectories_found = 0
    
    for person_idx in range(num_people):
        try:
            num_trajectories = plot_token_trajectories(
                data, person_idx, token_prefix, save_dir, id_to_token, value_name
            )
            if num_trajectories > 0:
                trajectories_found += 1
        except Exception as e:
            print(f"Error plotting person {data['person_ids'][person_idx]}: {e}")
            continue
    
    print(f"Successfully created plots for {trajectories_found} people with {value_name.lower()} data")


def main():
    parser = argparse.ArgumentParser(description="Plot token trajectories from ensemble generation data")
    parser.add_argument("ensemble_dir", type=str, help="Path to ensemble directory")
    parser.add_argument("--token_prefix", type=str, default="LAB_perindkialt_13_Q",
                       help="Token prefix to look for (default: LAB_perindkialt_13_Q)")
    parser.add_argument("--save_dir", type=str, default="figures/income",
                       help="Directory to save plots")
    parser.add_argument("--max_people", type=int, default=None,
                       help="Maximum number of people to plot (default: all)")
    parser.add_argument("--person_id", type=str, default=None,
                       help="Plot specific person ID only")
    parser.add_argument("--value_name", type=str, default="Income Quantile",
                       help="Name of the value being plotted (for labels)")
    parser.add_argument("--data_path", type=str, default="data/life_all_compiled",
                       help="Path to data directory for tokenizer")
    
    args = parser.parse_args()
    
    # Load ensemble data
    data = load_ensemble_data(args.ensemble_dir)
    
    print(f"Loaded ensemble data:")
    print(f"  - {len(data['person_ids'])} people")
    print(f"  - {data['metadata']['num_simulations']} simulations per person")
    print(f"  - Sequence shape: {data['full_sequences'].shape}")
    
    # Try to load tokenizer
    id_to_token = load_tokenizer_from_data_path(args.data_path)
    if id_to_token:
        print(f"Loaded tokenizer with {len(id_to_token)} tokens")
    else:
        print("Using heuristic token extraction (tokenizer not available)")
    
    # Plot trajectories
    if args.person_id:
        # Plot specific person
        try:
            person_idx = data['person_ids'].index(args.person_id)
            plot_token_trajectories(data, person_idx, args.token_prefix, 
                                  args.save_dir, id_to_token, args.value_name)
        except ValueError:
            print(f"Person ID {args.person_id} not found in data")
    else:
        # Plot all people
        plot_all_trajectories(data, args.token_prefix, args.save_dir, 
                            args.max_people, id_to_token, args.value_name)


if __name__ == "__main__":
    main()