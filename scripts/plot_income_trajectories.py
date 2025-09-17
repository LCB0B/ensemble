#!/usr/bin/env python3
"""
Script to visualize income trajectories from ensemble generation data.
Plots the evolution of income tokens through time with prompt (solid) and generated trajectories (transparent).
"""

import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
from collections import defaultdict


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


def extract_token_values(sequence, token_prefix):
    """
    Extract values from tokens matching a prefix pattern.
    
    Args:
        sequence: List of token IDs
        token_prefix: Prefix pattern (e.g., "LAB_perindkialt_13_Q")
    
    Returns:
        List of (position, value) tuples
    """
    # For now, we'll work with token IDs directly
    # In a real implementation, you'd need a tokenizer to convert IDs to strings
    # This is a placeholder that assumes token IDs follow a pattern
    
    values = []
    for pos, token_id in enumerate(sequence):
        if token_id == 0:  # Skip padding tokens
            continue
            
        # Placeholder: convert token_id to string representation
        # This would need actual tokenizer in real implementation
        token_str = f"token_{token_id}"
        
        # Extract value from token if it matches the pattern
        # For income tokens: LAB_perindkialt_13_Q{XX} where XX is 1-100
        if token_prefix in token_str:
            # Extract the quantile value (this is a placeholder)
            # In real implementation, you'd parse the actual token string
            match = re.search(r'Q(\d+)', token_str)
            if match:
                value = int(match.group(1))
                values.append((pos, value))
    
    return values


def extract_income_values_from_ids(sequence,vocab, token_prefix="LAB_perindkialt_13_Q" ):
    """
    Extract income values from token IDs based on pattern.
    
    This is a simplified version that assumes income tokens follow a specific ID pattern.
    In reality, you'd need the actual tokenizer to decode tokens to strings.
    """
    values = []
    # find the token prefix in the vocab
    token_ids = [token_id for token, token_id in vocab.items() if token.startswith(token_prefix)]
    #map token IDs to quantiles (they are not ordered, so we need to extract the quantile value, after the prefix)
    if not token_ids:
        print(f"No tokens found with prefix '{token_prefix}' in vocabulary.")
        return values
    reverse_vocab = {v: k for k, v in vocab.items()}
    token_id_to_quantile = {token_id: int(reverse_vocab[token_id].split('_')[-1][1:]) for token_id in token_ids if token_id in reverse_vocab}  
    for pos, token_id in enumerate(sequence):
        if int(token_id) in token_ids:
            quantile = token_id_to_quantile[int(token_id)]
            values.append((pos, quantile)) 
    return values


def create_income_mapping_from_pattern(vocab_size=10000):
    """
    Create a mapping from token IDs to income quantiles.
    This is a placeholder - in real usage, you'd load this from your tokenizer.
    """
    income_mapping = {}
    
    # Example: assume income tokens are consecutive IDs starting from 5000
    for quantile in range(1, 101):  # Quantiles 1-100
        token_id = 5000 + quantile - 1  # Example mapping
        income_mapping[token_id] = quantile
    
    return income_mapping


def plot_income_trajectories(data, person_idx, token_prefix="LAB_perindkialt_13_Q", 
                           save_dir="figures/income", income_mapping=None):
    """
    Plot income trajectories for a specific person.
    
    Args:
        data: Ensemble data dictionary
        person_idx: Index of the person to plot
        token_prefix: Token prefix to look for
        save_dir: Directory to save plots
        income_mapping: Mapping from token IDs to income quantiles
    """
    person_id = data['person_ids'][person_idx]
    prompt_length = data['prompt_lengths'][person_idx]
    
    # Get sequences for this person
    full_sequences = data['full_sequences'][person_idx]  # [num_simulations, max_seq_length]
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Extract income values from each simulation
    all_trajectories = []
    prompt_income = None
    
    for sim_idx in range(full_sequences.shape[0]):
        sequence = full_sequences[sim_idx].tolist()
        
        # Extract income values
        if income_mapping:
            income_values = []
            for pos, token_id in enumerate(sequence):
                if token_id in income_mapping:
                    income_values.append((pos, income_mapping[token_id]))
        else:
            income_values = extract_income_values_from_ids(sequence, token_prefix)
        
        if income_values:
            positions, values = zip(*income_values)
            all_trajectories.append((positions, values))
            
            # Extract prompt income (first trajectory as reference)
            if prompt_income is None:
                prompt_positions = [p for p in positions if p < prompt_length]
                prompt_values = [v for p, v in zip(positions, values) if p < prompt_length]
                if prompt_positions:
                    prompt_income = (prompt_positions, prompt_values)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot all trajectories with transparency
    for positions, values in all_trajectories:
        plt.plot(positions, values, 'b-', alpha=0.1, linewidth=1)
    
    # Plot prompt as solid line
    if prompt_income:
        prompt_positions, prompt_values = prompt_income
        plt.plot(prompt_positions, prompt_values, 'r-', linewidth=2, label='Prompt (Real Life)')
    
    # Add vertical line at generation start
    plt.axvline(x=prompt_length, color='red', linestyle='--', linewidth=2, 
                label=f'Generation Start (position {prompt_length})')
    
    # Customize plot
    plt.xlabel('Event Position')
    plt.ylabel('Income Quantile (1-100)')
    plt.title(f'Income Trajectories for {person_id}\n'
              f'{len(all_trajectories)} simulations, prompt length: {prompt_length}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(0, 101)
    
    # Save plot
    filename = f"{person_id}_income_trajectories.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {person_id}: {save_path / filename}")
    
    return len(all_trajectories)


def plot_all_income_trajectories(data, token_prefix="LAB_perindkialt_13_Q", 
                                save_dir="figures/income", max_people=None):
    """
    Plot income trajectories for all people in the dataset.
    
    Args:
        data: Ensemble data dictionary
        token_prefix: Token prefix to look for
        save_dir: Directory to save plots
        max_people: Maximum number of people to plot (None for all)
    """
    num_people = len(data['person_ids'])
    if max_people:
        num_people = min(num_people, max_people)
    
    print(f"Creating income trajectory plots for {num_people} people...")
    
    # Create income mapping (placeholder)
    income_mapping = create_income_mapping_from_pattern()
    
    trajectories_found = 0
    
    for person_idx in range(num_people):
        try:
            num_trajectories = plot_income_trajectories(
                data, person_idx, token_prefix, save_dir, income_mapping
            )
            if num_trajectories > 0:
                trajectories_found += 1
        except Exception as e:
            print(f"Error plotting person {data['person_ids'][person_idx]}: {e}")
            continue
    
    print(f"Successfully created plots for {trajectories_found} people with income data")


# parser = argparse.ArgumentParser(description="Plot income trajectories from ensemble generation data")
# parser.add_argument("ensemble_dir", type=str, default = 'generated/ensemble_20250708_082217', help="Path to ensemble directory")
# parser.add_argument("--token_prefix", type=str, default="LAB_perindkialt_13_Q",
#                     help="Token prefix to look for (default: LAB_perindkialt_13_Q)")
# parser.add_argument("--save_dir", type=str, default="figures/income",
#                     help="Directory to save plots")
# parser.add_argument("--max_people", type=int, default=5,
#                     help="Maximum number of people to plot (default: all)")
# parser.add_argument("--person_id", type=str, default=None,
#                     help="Plot specific person ID only")

# args = parser.parse_args()

ensemble_dir = 'generated/ensemble_20250716_140755'
token_prefix = "LAB_perindkialt_13_Q"
#token_prefix = 'LAB_bredt_loen_beloeb_Q'
save_dir = f"figures/income/{token_prefix[:10]}"
max_people = 250
person_id = None  # Set to specific person ID or None for all

#load json vocab in data/life_all_compiled/vocab.json
vocab_path = Path('data/destiny_dataset/vocab.json')
with open(vocab_path, 'r') as f:
    vocab = json.load(f)


# Load ensemble data
data = load_ensemble_data(ensemble_dir)

full_sequences = data['full_sequences']
person_ids_str = data['person_ids']
person_idx = [int(pid.split('_')[1]) for pid in person_ids_str]

prompt_lengths = data['prompt_lengths']
generation_length = full_sequences.shape[2] - prompt_lengths[0]

print(f"Loaded ensemble data:")
print(f"  - {len(data['person_ids'])} people")
print(f"  - {data['metadata']['num_simulations']} simulations per person")
print(f"  - Sequence shape: {data['full_sequences'].shape}")


#load the data for the list of person in person_idx

#full_sequences.shape  torch.Size([n_people, n_simulations, max_seq_length])


#just plot one person

person_id = person_idx[4]  # For example, plot the first person

for person_id in person_idx:
    fig, ax = plt.subplots(figsize=(12, 8))
    #plot the prompt trajectory
    prompt_sequence = full_sequences[person_id, 0, :prompt_lengths[person_id]]
    print(f"Prompt sequence for person {person_id}: {prompt_sequence[:10]}")
    prompt_income = extract_income_values_from_ids(prompt_sequence,vocab, token_prefix)
    print(f"Prompt income for person {person_id}: {prompt_income}")
    if prompt_income:
        prompt_positions, prompt_values = zip(*prompt_income)
    _ = ax.plot(prompt_positions, prompt_values, 'firebrick', linewidth=2, label='Prompt (Real Life)')
    # Plot all generated trajectories
    prompt_length = prompt_lengths[person_id]
    for sim_idx in range(full_sequences.shape[1]):
        sequence = full_sequences[person_id, sim_idx, prompt_length:]
        income_values = extract_income_values_from_ids(sequence, vocab, token_prefix)
        if income_values:
            positions, values = zip(*income_values)
            #add prompt length to positions
            positions = [pos + prompt_length for pos in positions]
            _ = ax.plot(positions, values, 'k-', alpha=0.1, linewidth=0.5)
    plt.axvline(x=prompt_lengths[person_id], color='k', linestyle='--', linewidth=2, 
                label=f'Generation Start (position {prompt_lengths[person_id]})')
    plt.savefig(f"{save_dir}/{person_id}_income_trajectory.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot for person {person_id} to {save_dir}/{person_id}_income_trajectory.png")


#print sequence through reverse_vocab
reverse_vocab = {v: k for k, v in vocab.items()}
print(f"Sequence for person {person_id}:")
for i,token_id in enumerate(full_sequences[person_id, 0, :prompt_lengths[person_id] ]):
    if int(token_id) in reverse_vocab:
        print(f"{i}:{token_id}: {reverse_vocab[int(token_id)]}")
    else:
        print(f"{token_id}: <unknown>")  # Handle unknown tokens


for person_id in person_idx:
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Plot the prompt trajectory (unchanged) ---
    prompt_sequence = full_sequences[person_id, 0, :prompt_lengths[person_id]]
    prompt_income = extract_income_values_from_ids(prompt_sequence, vocab, token_prefix)
    
    if prompt_income:
        prompt_positions, prompt_values = zip(*prompt_income)
        ax.plot(prompt_positions, prompt_values, 'firebrick', linewidth=2, label='Prompt (Real Life)', zorder=3)

    # --- Collect all generated points for the heatmap ---
    all_gen_positions = []
    all_gen_values = []
    prompt_length = prompt_lengths[person_id]
    
    for sim_idx in range(full_sequences.shape[1]):
        # We look at the full sequence to get correct positions relative to the start
        sequence = full_sequences[person_id, sim_idx]
        income_values = extract_income_values_from_ids(sequence, vocab, token_prefix)
        
        if income_values:
            positions, values = zip(*income_values)
            # Filter for generated part only
            gen_positions = [p for p in positions if p >= prompt_length]
            gen_values = [v for p, v in zip(positions, values) if p >= prompt_length]
            
            all_gen_positions.extend(gen_positions)
            all_gen_values.extend(gen_values)

    # --- Plot the generated trajectories as a heatmap ---
    if all_gen_positions:
        # Define bins for the heatmap
        # X-axis: from prompt end to sequence end. Bin for each position.
        # Y-axis: from 0 to 101. Bin for each quantile.
        x_bins = np.arange(prompt_length, full_sequences.shape[2] + 1,10)
        y_bins = np.arange(0, 100 + 2)  # 1-100 quantiles, so 101 bins
        
        # Create the 2D histogram
        h, xedges, yedges = np.histogram2d(all_gen_positions, all_gen_values, bins=(x_bins, y_bins))
        
        # Use imshow for better control over the heatmap
        im = ax.imshow(h.T, origin='lower', aspect='auto', 
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       cmap='Blues', zorder=1)
        fig.colorbar(im, ax=ax, label='Trajectory Density')

    # --- Plot formatting (unchanged) ---
    ax.axvline(x=prompt_lengths[person_id], color='k', linestyle='--', linewidth=2, 
                label=f'Generation Start (position {prompt_lengths[person_id]})', zorder=2)

    # Customize plot
    ax.set_xlabel('Event Position')
    ax.set_ylabel('Income Quantile (1-100)')
    ax.set_title(f'Income Trajectory Heatmap for Person {person_id}')
    ax.legend()
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_ylim(0, 101)
    ax.set_xlim(0, full_sequences.shape[2]) # Ensure x-axis covers the whole sequence

    plt.savefig(f"{save_dir}/{person_id}_income_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved heatmap for person {person_id} to {save_dir}/{person_id}_income_heatmap.png")
    plt.close(fig)
