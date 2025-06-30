import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm
import re
import torch


def load_data(output_dir: Path) -> Tuple[List, List, List]:
    """Load generated sequences, prompts, and original batches from the output directory."""
    sequences_path = output_dir / "generated_sequences.pkl"
    prompts_path = output_dir / "prompts.pkl"
    original_batches_path = output_dir / "original_batches.pkl"
    
    print(f"Loading data from {output_dir}")
    with open(sequences_path, "rb") as f:
        all_sequences = pickle.load(f)
    with open(prompts_path, "rb") as f:
        all_prompts = pickle.load(f)
    with open(original_batches_path, "rb") as f:
        all_original_batches = pickle.load(f)
        
    return all_sequences, all_prompts, all_original_batches

def build_token_map(vocab: Dict[str, int], prefix: str, reverse: bool = False) -> Dict[int, float]:
    """
    Build a mapping from token IDs to numerical values based on the token prefix.
    """
    token_map = {}
    token_pattern = {}
    
    # Extract mapping from tokens to values
    for token, tid in vocab.items():
        if token.startswith(prefix):
            suffix = token[len(prefix):]
            # Try to extract a number value
            try:
                # Look for numbers in the suffix
                match = re.search(r'(\d+)', suffix)
                if match:
                    value = float(match.group(1))
                    token_map[tid] = value
                    token_pattern[tid] = token
            except ValueError:
                continue
    
    # Sort by value
    sorted_tokens = sorted(token_map.items(), key=lambda x: x[1], reverse=reverse)
    
    # Normalize values between 0 and 1 for easier plotting
    if sorted_tokens:
        min_val = sorted_tokens[-1][1] if reverse else sorted_tokens[0][1]
        max_val = sorted_tokens[0][1] if reverse else sorted_tokens[-1][1]
        range_val = max_val - min_val
        
        if range_val > 0:
            normalized_map = {
                tid: (val - min_val) / range_val for tid, val in token_map.items()
            }
        else:
            normalized_map = {tid: 0.5 for tid in token_map.keys()}
            
        return normalized_map
    
    return {}

def extract_attribute_trajectory(
    sequence: np.ndarray, 
    token_map: Dict[int, float],
    default_value: Optional[float] = None
) -> List[Tuple[int, float]]:
    """
    Extract the trajectory of a specific attribute from a sequence.
    """
    trajectory = []
    
    for pos, token in enumerate(sequence):
        if token in token_map:
            trajectory.append((pos, token_map[token]))
    
    return trajectory

def extract_attribute_data(
    sequences: torch.Tensor,      # Shape: [batch_size, num_simulations, seq_len]
    prompts: torch.Tensor,        # Shape: [batch_size, prompt_length] 
    original_sequences: torch.Tensor,  # Shape: [batch_size, seq_len]
    vocab: Dict[str, int],
    token_prefix: str,
    reverse_mapping: bool = False,
    max_sequences: int = 100,
    max_simulations: int = 100,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Extract attribute data from generated sequences without plotting.
    
    Args:
        sequences: Tensor of shape [batch_size, num_simulations, seq_len]
        prompts: Tensor of shape [batch_size, prompt_length]
        original_sequences: Tensor of shape [batch_size, seq_len]
        vocab: Dictionary mapping token strings to IDs
        token_prefix: Prefix for tokens to analyze
        reverse_mapping: If True, higher numbers in token map to lower values
        max_sequences: Maximum number of sequences to process
        max_simulations: Maximum number of simulations to include per sequence
        debug: Whether to print debug information
        
    Returns:
        Dictionary with extracted trajectory data
    """
    if debug:
        print(f"Processing tensor with shape: {sequences.shape}")
        print(f"Prompts shape: {prompts.shape}")
        print(f"Original sequences shape: {original_sequences.shape}")
    
    # Build token mapping
    token_map = build_token_map(vocab, token_prefix, reverse=reverse_mapping)
    
    if debug:
        print(f"Found {len(token_map)} tokens matching prefix '{token_prefix}'")
        matching_tokens = [token for token, tid in vocab.items() 
                          if token.startswith(token_prefix) and tid in token_map]
        print(f"Example matching tokens: {matching_tokens[:5] if matching_tokens else 'None'}")
    
    if not token_map:
        raise ValueError(f"No mappable tokens found with prefix '{token_prefix}'")
    
    # Prepare data structure for all trajectories
    trajectory_data = []
    
    # Process sequences
    sequences_with_tokens = 0
    batch_size = sequences.shape[0]  # Number of sequences
    
    # Process each sequence
    for i in range(min(batch_size, max_sequences)):
        # Extract original sequence and prompt length
        original_seq = original_sequences[i].cpu().numpy()
        prompt_length = prompts.shape[1]  # All prompts have the same length
        
        # Extract original trajectory
        orig_trajectory = extract_attribute_trajectory(original_seq, token_map)
        
        if orig_trajectory and debug:
            sequences_with_tokens += 1
            if sequences_with_tokens <= 3:  # Just show the first few
                print(f"Person {i}: Found {len(orig_trajectory)} matching tokens")
                print(f"  First few positions and values: {orig_trajectory[:3]}")
        
        # Split into before and after prompt
        orig_before_prompt = [(pos, val) for pos, val in orig_trajectory if pos < prompt_length]
        orig_after_prompt = [(pos, val) for pos, val in orig_trajectory if pos >= prompt_length]
        
        # Extract generated trajectories
        generated_trajectories = []
        sim_predictions = sequences[i]  # Shape: [num_simulations, seq_length]
        num_sims = min(sim_predictions.shape[0], max_simulations)
        
        found_in_simulations = 0
        
        for sim in range(num_sims):
            gen_seq = sim_predictions[sim].cpu().numpy()
            gen_trajectory = extract_attribute_trajectory(gen_seq, token_map)
            gen_after_prompt = [(pos, val) for pos, val in gen_trajectory]
            
            if gen_after_prompt:
                found_in_simulations += 1
            
            generated_trajectories.append({
                'sim_index': sim,
                'trajectory': gen_after_prompt
            })
        
        if debug and i < 3:
            print(f"  Found tokens in {found_in_simulations}/{num_sims} simulations")
        
        # Store all data for this person
        person_data = {
            'person_idx': i,
            'prompt_length': prompt_length,
            'original_before_prompt': orig_before_prompt,
            'original_after_prompt': orig_after_prompt,
            'generated_trajectories': generated_trajectories
        }
        
        trajectory_data.append(person_data)
    
    if debug:
        print(f"Found attribute tokens in {sequences_with_tokens}/{min(max_sequences, batch_size)} sequences")
        print(f"Created {len(trajectory_data)} person trajectories")
    
    return {
        'token_prefix': token_prefix,
        'trajectories': trajectory_data
    }


def filter_sequences_by_length(
    all_sequences: List, 
    all_prompts: List, 
    all_original_batches: List,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    debug: bool = False
) -> Tuple[List, List, List]:
    """
    Filter sequences by their length.
    
    Args:
        all_sequences: List of sequence batches
        all_prompts: List of prompts
        all_original_batches: List of original batches
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        debug: Whether to print debug information
        
    Returns:
        Filtered sequences, prompts, and original batches
    """
    import torch
    
    filtered_sequences = []
    filtered_prompts = []
    filtered_original_batches = []
    
    total_sequences = 0
    filtered_count = 0
    
    # Iterate through each batch
    for seq_batch, prompts, original_batch in zip(all_sequences, all_prompts, all_original_batches):
        # Get sequence lengths from the batch
        sequence_lens = original_batch['sequence_lens']
        total_sequences += len(sequence_lens)
        
        # Create mask for sequences that meet the criteria
        mask = torch.ones(sequence_lens.shape[0], dtype=torch.bool)
        
        if min_length is not None:
            mask = mask & (sequence_lens >= min_length)
            
        if max_length is not None:
            mask = mask & (sequence_lens <= max_length)
        
        filtered_count += mask.sum().item()
        
        # Check if any sequences passed the filter
        if mask.any():
            # Filter the batch
            filtered_sequences.append(seq_batch[mask])
            filtered_prompts.append(prompts[mask])
            
            # Create a filtered original batch with all fields filtered by the mask
            filtered_original = {k: v[mask] if torch.is_tensor(v) and v.shape[0] == len(mask) else v 
                               for k, v in original_batch.items()}
            filtered_original_batches.append(filtered_original)
    
    if debug:
        print(f"Filtered {filtered_count}/{total_sequences} sequences " +
              f"(min_length={min_length}, max_length={max_length})")
        
        if filtered_sequences:
            print(f"First filtered batch shape: {filtered_sequences[0].shape}")
    
    return filtered_sequences, filtered_prompts, filtered_original_batches


def filter_sequences_by_length(
    all_sequences: List, 
    all_prompts: List, 
    all_original_batches: List,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    debug: bool = False
) -> Tuple[List, List, List]:
    """
    Filter sequences by their length.
    
    Args:
        all_sequences: List of sequence batches
        all_prompts: List of prompts
        all_original_batches: List of original batches
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        debug: Whether to print debug information
        
    Returns:
        Filtered sequences, prompts, and original batches
    """
    import torch
    
    filtered_sequences = []
    filtered_prompts = []
    filtered_original_batches = []
    
    total_sequences = 0
    filtered_count = 0
    
    # Iterate through each batch
    for seq_batch, prompts, original_batch in zip(all_sequences, all_prompts, all_original_batches):
        # Get sequence lengths from the batch
        sequence_lens = original_batch['sequence_lens']
        total_sequences += len(sequence_lens)
        
        # Create mask for sequences that meet the criteria
        mask = torch.ones(sequence_lens.shape[0], dtype=torch.bool)
        
        if min_length is not None:
            mask = mask & (sequence_lens >= min_length)
            
        if max_length is not None:
            mask = mask & (sequence_lens <= max_length)
        
        filtered_count += mask.sum().item()
        
        # Check if any sequences passed the filter
        if mask.any():
            # Filter the batch
            filtered_sequences.append(seq_batch[mask])
            filtered_prompts.append(prompts[mask])
            
            # Create a filtered original batch with all fields filtered by the mask
            filtered_original = {k: v[mask] if torch.is_tensor(v) and v.shape[0] == len(mask) else v 
                               for k, v in original_batch.items()}
            filtered_original_batches.append(filtered_original)
    
    if debug:
        print(f"Filtered {filtered_count}/{total_sequences} sequences " +
              f"(min_length={min_length}, max_length={max_length})")
        
        if filtered_sequences:
            print(f"First filtered batch shape: {filtered_sequences[0].shape}")
    
    return filtered_sequences, filtered_prompts, filtered_original_batches

def filter_sequences_by_token_frequency(
    sequences,
    prompts, 
    original_sequences,
    vocab: Dict[str, int],
    token_prefix: str,
    position_window: Tuple[int, int] = (0, 1000),
    min_occurrences: int = 10,
    debug: bool = False
) -> Union[Tuple[List, List, List], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Filter sequences based on the frequency of specific tokens within a position window.
    Works with either:
    - Lists of batch tensors (standard format)
    - Already concatenated tensors (filtered format)
    
    Args:
        sequences: Either list of batched tensors or single tensor
        prompts: Either list of prompts or single tensor
        original_sequences: Either list of original batches or single tensor
        vocab: Dictionary mapping token strings to IDs
        token_prefix: Prefix to identify the group of tokens
        position_window: Tuple of (start_position, end_position) to look for tokens
        min_occurrences: Minimum number of times tokens must appear
        debug: Whether to print debug information
        
    Returns:
        Filtered data in same format as input (list or tensor)
    """
    import torch
    
    # Determine if inputs are already concatenated tensors
    is_tensor_input = isinstance(sequences, torch.Tensor) and not isinstance(sequences, list)
    
    # Convert to list format if tensor input
    if is_tensor_input:
        sequences = [sequences]
        prompts = [prompts]
        
        # Handle original_sequences which might be tensor or dict
        if isinstance(original_sequences, torch.Tensor):
            # Create a dummy batch dict with 'event' key
            original_sequences = [{'event': original_sequences}]
    
    # Build a set of token IDs that match the prefix
    matching_token_ids = set()
    for token_str, token_id in vocab.items():
        if token_str.startswith(token_prefix):
            matching_token_ids.add(token_id)
    
    if debug:
        print(f"Found {len(matching_token_ids)} tokens matching prefix '{token_prefix}'")
        if len(matching_token_ids) > 0:
            sample_tokens = [token for token, tid in vocab.items() 
                            if token.startswith(token_prefix)][:5]
            print(f"Sample tokens: {sample_tokens}")
        else:
            print("No matching tokens found! Filter will return empty results.")
    
    filtered_sequences = []
    filtered_prompts = []
    filtered_original_batches = []
    
    total_sequences = 0
    filtered_count = 0
    window_start, window_end = position_window
    
    # Iterate through each batch
    for seq_idx, (seq_batch, prompts_batch, original_batch) in enumerate(zip(sequences, prompts, original_sequences)):
        batch_size = seq_batch.shape[0]
        total_sequences += batch_size
        
        # Create a mask for this batch
        mask = torch.zeros(batch_size, dtype=torch.bool)
        
        # Check each sequence in the batch
        for i in range(batch_size):
            # Get original sequence based on input type
            if is_tensor_input:
                original_seq = original_batch['event'][i].cpu().numpy() if 'event' in original_batch else original_batch[i].cpu().numpy()
            else:
                original_seq = original_batch['event'][i].cpu().numpy()
            
            # Count matching tokens in the window
            token_count = 0
            token_positions = []
            
            for pos, token_id in enumerate(original_seq):
                if (pos >= window_start and pos <= window_end and 
                    token_id in matching_token_ids):
                    token_count += 1
                    token_positions.append(pos)
            
            # If enough tokens, mark this sequence to keep
            if token_count >= min_occurrences:
                mask[i] = True
                filtered_count += 1
                
                if debug and filtered_count <= 3:
                    print(f"Sequence {seq_idx * batch_size + i} has {token_count} matching tokens")
                    print(f"  Token positions: {token_positions[:10]}...")
        
        # Check if any sequences passed the filter
        if mask.any():
            # Filter the batch
            filtered_sequences.append(seq_batch[mask])
            filtered_prompts.append(prompts_batch[mask])
            
            # Create a filtered original batch
            if is_tensor_input:
                if 'event' in original_batch:
                    filtered_original = {'event': original_batch['event'][mask]}
                else:
                    filtered_original = original_batch[mask]
            else:
                filtered_original = {k: v[mask] if torch.is_tensor(v) and v.shape[0] == len(mask) else v 
                                for k, v in original_batch.items()}
                
            filtered_original_batches.append(filtered_original)
    
    if debug:
        print(f"Filtered {filtered_count}/{total_sequences} sequences " +
              f"with at least {min_occurrences} occurrences of '{token_prefix}' tokens " +
              f"in positions {position_window}")
        
        if filtered_sequences:
            print(f"First filtered batch shape: {filtered_sequences[0].shape}")
    
    # Return in the same format as input
    if is_tensor_input and filtered_sequences:
        # Concatenate results back to tensor format
        filtered_sequences = torch.cat(filtered_sequences, dim=0) if filtered_sequences else torch.tensor([])
        filtered_prompts = torch.cat(filtered_prompts, dim=0) if filtered_prompts else torch.tensor([])
        
        # Handle original sequences based on their type
        if isinstance(original_sequences[0], dict) and 'event' in original_sequences[0]:
            filtered_original_batches = torch.cat([batch['event'] for batch in filtered_original_batches], dim=0) if filtered_original_batches else torch.tensor([])
        else:
            filtered_original_batches = torch.cat(filtered_original_batches, dim=0) if filtered_original_batches else torch.tensor([])
            
    return filtered_sequences, filtered_prompts, filtered_original_batches

# STEP 2: PLOTTING FUNCTION
def plot_attribute_trajectories(
    trajectory_data: Dict[str, Any],
    plot_title: Optional[str] = None,
    save_path: Optional[Path] = None,
    alpha: float = 0.05
) -> None:
    """
    Plot attribute trajectories from pre-extracted data.
    
    Args:
        trajectory_data: Data structure from extract_attribute_data
        plot_title: Title for the plot
        save_path: Path to save the plot (if None, plot will be displayed)
    """
    token_prefix = trajectory_data['token_prefix']
    trajectories = trajectory_data['trajectories']
    
    # Setup plot
    num_people = len(trajectories)
    fig, axes = plt.subplots(num_people, 1, figsize=(12, 4*num_people))
    
    if num_people == 1:
        axes = [axes]  # Make it iterable
    
    plot_title = plot_title or f"Trajectory of {token_prefix.replace('_', ' ').strip()}"
    fig.suptitle(plot_title, fontsize=16)
    
    # Plot each person's data
    for person_idx, person_data in enumerate(trajectories):
        ax = axes[person_idx]
        
        # Extract data
        prompt_length = person_data['prompt_length']
        orig_before = person_data['original_before_prompt']
        orig_after = person_data['original_after_prompt']
        generated = person_data['generated_trajectories']
        
        # Plot original data before prompt
        if orig_before:
            x_before, y_before = zip(*orig_before)
            ax.plot(x_before, y_before, '-', color='blue', linewidth=2, 
                    label='Original (prompt)', alpha=0.7)
        
        # Plot original data after prompt
        if orig_after:
            x_after, y_after = zip(*orig_after)
            ax.plot(x_after, y_after, '-', color='green', linewidth=2, 
                    label='Original (after prompt)', alpha=0.7)
        
        # Plot generated trajectories
        for i, gen_data in enumerate(generated):
            gen_trajectory = gen_data['trajectory']
            if gen_trajectory:
                x_gen, y_gen = zip(*gen_trajectory)
                #make x_gen relative to the prompt length
                x_gen = [x + prompt_length for x in x_gen]
                ax.plot(x_gen, y_gen, '-', color='k', alpha=alpha, linewidth=1, 
                       label=f'Generated {i+1}' if i == 0 else "_nolegend_")
        
        # Add vertical line at prompt boundary
        ax.axvline(x=prompt_length, color='grey', linestyle='--', alpha=0.7,
                  label='Prompt Boundary')
        
        # Label for this subplot
        ax.set_title(f"Person {person_idx+1}")
        ax.set_xlabel("Position in Sequence")
        ax.set_ylabel(f"{token_prefix.replace('_', ' ').strip()} Level")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Add legend to first plot only
        if person_idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        #set x lim max to be the min between all sequences and the real in original length
        #ax.set_xlim(0, min(len(generated)+prompt_length, max(x_after))+100)
        ax.set_xlim(0, len(generated)+prompt_length+100)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save or display
    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

# Update the combined function for convenience
def analyze_and_plot_trajectories(
    all_sequences: List,
    all_prompts: List,
    all_original_batches: List,
    vocab: Dict[str, int],
    token_prefix: str,
    output_dir: Optional[Path] = None,
    reverse_mapping: bool = False,
    max_sequences: int = 10,
    max_simulations: int = 5,
    plot_title: Optional[str] = None,
    save_plot: bool = True
) -> Dict[str, Any]:
    """
    Combined function that extracts data and plots it.
    """
    # Extract data
    trajectory_data = extract_attribute_data(
        all_sequences,
        all_prompts,
        all_original_batches,
        vocab,
        token_prefix=token_prefix,
        reverse_mapping=reverse_mapping,
        max_sequences=max_sequences,
        max_simulations=max_simulations
    )
    
    # Define save path
    save_path = None
    if save_plot and output_dir:
        plot_dir = output_dir / "trajectory_plots"
        save_path = plot_dir / f"{token_prefix.replace('_', '')}_trajectories.png"
    
    # Plot data
    plot_attribute_trajectories(
        trajectory_data=trajectory_data,
        plot_title=plot_title,
        save_path=save_path
    )
    
    return trajectory_data
    
def compute_jaccard_similarities_over_time(
    original_sequences: torch.Tensor,  # Shape: [batch_size, seq_len]
    generated_sequences: torch.Tensor,  # Shape: [batch_size, num_simulations, seq_len]
    prompts: torch.Tensor,             # Shape: [batch_size, prompt_length]
    time_window: int = 100,            # Window size for time steps
    max_time_steps: int = None,        # Maximum number of time steps to compute
    padding_idx: int = 0,              # Index for padding tokens
    ignore_indices: List[int] = None,  # Indices to ignore (e.g., special tokens)
    debug: bool = False
) -> Dict[str, Any]:
    """
    Compute Jaccard similarity index between original sequences and generated simulations over time.
    
    For each time step t, calculates Jaccard index for tokens from prompt_length to prompt_length+t.
    
    Args:
        original_sequences: Original sequences tensor [batch_size, seq_len]
        generated_sequences: Generated sequences tensor [batch_size, num_simulations, seq_len]
        prompts: Prompt tensor to determine where to start comparison [batch_size, prompt_length]
        time_window: Number of tokens to include in each time step window
        max_time_steps: Maximum number of time steps to compute (None = compute all possible)
        padding_idx: Token ID for padding (to be ignored in comparison)
        ignore_indices: List of token IDs to ignore in comparison (e.g., special tokens)
        debug: Whether to print debug information
        
    Returns:
        Dictionary with Jaccard similarity statistics over time
    """
    import torch
    import numpy as np
    from collections import defaultdict
    
    if ignore_indices is None:
        ignore_indices = []
    ignore_indices.append(padding_idx)
    
    prompt_length = prompts.shape[1]
    batch_size = original_sequences.shape[0]
    num_simulations = generated_sequences.shape[1]
    
    # Calculate maximum possible length after prompt
    max_orig_len = max([len(original_sequences[i][prompt_length:]) for i in range(batch_size)])
    max_gen_len = generated_sequences.shape[2]  # Maximum length of generated sequences
    
    # Determine number of time steps
    if max_time_steps is None:
        # Calculate maximum number of steps possible
        max_length = min(max_orig_len, max_gen_len)
        num_time_steps = (max_length + time_window - 1) // time_window  # Ceiling division
    else:
        num_time_steps = max_time_steps
    
    if debug:
        print(f"Computing Jaccard similarities over time for {batch_size} sequences with {num_simulations} simulations each")
        print(f"Prompt length: {prompt_length}")
        print(f"Time window: {time_window} tokens")
        print(f"Number of time steps: {num_time_steps}")
        print(f"Ignoring token indices: {ignore_indices}")
    
    # Store results with time dimension
    # Shape: [batch_size, num_simulations, num_time_steps]
    all_jaccard_indices = np.zeros((batch_size, num_simulations, num_time_steps))
    all_intersection_sizes = np.zeros((batch_size, num_simulations, num_time_steps))
    all_union_sizes = np.zeros((batch_size, num_simulations, num_time_steps))
    
    # Process each sequence
    for i in range(batch_size):
        # Get original tokens after prompt
        orig_tokens = original_sequences[i].cpu().numpy()
        orig_tokens_after_prompt = orig_tokens[prompt_length:]
        
        # Process each simulation
        for sim_idx in range(num_simulations):
            # Get generated tokens for this simulation
            gen_tokens = generated_sequences[i, sim_idx].cpu().numpy()
            
            # Process each time step
            for t in range(num_time_steps):
                # Calculate window end for this time step
                window_end = min((t+1) * time_window, len(orig_tokens_after_prompt), max_gen_len)
                
                # If we've reached the end of either sequence, stop processing time steps
                if window_end <= t * time_window:
                    break
                
                # Get tokens for this time window (from prompt end to current time step)
                orig_window = orig_tokens_after_prompt[:window_end]
                gen_window = gen_tokens[:window_end]
                
                # Create sets of tokens (filtering out padding and ignored tokens)
                orig_token_set = {t.item() for t in torch.tensor(orig_window) 
                                 if t.item() not in ignore_indices}
                gen_token_set = {t.item() for t in torch.tensor(gen_window)
                                if t.item() not in ignore_indices}
                
                # Calculate intersection and union sizes
                intersection_size = len(orig_token_set.intersection(gen_token_set))
                union_size = len(orig_token_set.union(gen_token_set))
                
                # Calculate Jaccard index
                jaccard_index = intersection_size / union_size if union_size > 0 else 0
                
                # Store results
                all_jaccard_indices[i, sim_idx, t] = jaccard_index
                all_intersection_sizes[i, sim_idx, t] = intersection_size
                all_union_sizes[i, sim_idx, t] = union_size
    
    # Calculate statistics over time
    # Average across simulations for each sequence and time step
    avg_jaccard_per_sequence_time = all_jaccard_indices.mean(axis=1)  # [batch_size, num_time_steps]
    
    # Average across all sequences for each time step
    avg_jaccard_per_time = all_jaccard_indices.mean(axis=(0, 1))  # [num_time_steps]
    
    # Overall average across all dimensions
    avg_jaccard_overall = all_jaccard_indices.mean()
    
    if debug:
        print(f"Average Jaccard similarity (overall): {avg_jaccard_overall:.4f}")
        print(f"Jaccard similarity at first time step: {avg_jaccard_per_time[0]:.4f}")
        print(f"Jaccard similarity at last time step: {avg_jaccard_per_time[-1]:.4f}")
    
    return {
        "jaccard_indices": all_jaccard_indices,  # [batch_size, num_simulations, num_time_steps]
        "avg_jaccard_per_sequence_time": avg_jaccard_per_sequence_time,  # [batch_size, num_time_steps]
        "avg_jaccard_per_time": avg_jaccard_per_time,  # [num_time_steps]
        "avg_jaccard_overall": avg_jaccard_overall,  # scalar
        "intersection_sizes": all_intersection_sizes,  # [batch_size, num_simulations, num_time_steps]
        "union_sizes": all_union_sizes,  # [batch_size, num_simulations, num_time_steps]
        "time_window": time_window,  # Window size used
        "num_time_steps": num_time_steps  # Number of time steps computed
    }

# Example usage
if __name__ == "__main__":
    from src.paths import FPATH
    
    # Define paths
    #generated/stable_pretrain/generated_sequences_20250429_001600_1000_200_100
    output_directory = FPATH.GENERATED / "stable_pretrain" / "generated_sequences_20250429_001456_1000_200_100"
    vocab_file = FPATH.DATA / "life_all_compiled" / "vocab.json"

    data_dir = Path("generated") / "analysis_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = Path('figures') / "trajectory_plots"

    # Load data outside the function
    print("Loading data...")
    all_sequences, all_prompts, all_original_batches = load_data(output_directory)
     
    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    
    # Optional: Filter sequences by length
    min_seq_length = 1200  # Example: only sequences at least 1200 tokens long
    filtered_sequences, filtered_prompts, filtered_original_batches = filter_sequences_by_length(
        all_sequences, all_prompts, all_original_batches, 
        min_length=min_seq_length, 
        debug=True
    )
    #concatenate the first dimension of the filtered sequences, prompts and original batches (remove the batching dimension)
    filtered_sequences = torch.cat(filtered_sequences, dim=0)
    filtered_prompts = torch.cat(filtered_prompts, dim=0)
    filtered_original_batches = torch.cat([original_batch['event'] for original_batch in filtered_original_batches], dim=0)


    #SET OF TOKENS COMPARAISON
    # Compute Jaccard similarities over time
    jaccard_results = compute_jaccard_similarities_over_time(
        original_sequences=filtered_original_batches,
        generated_sequences=filtered_sequences,
        prompts=filtered_prompts,
        time_window=1,  # Use 100 tokens per time step
        padding_idx=0,
        ignore_indices=[1, 2, 3, 4],
        debug=True
    )

    # Plot average Jaccard similarity over time
    plt.figure(figsize=(10, 6))
    time_steps = list(range(1, jaccard_results['num_time_steps']+1))
    plt.plot(time_steps, jaccard_results['avg_jaccard_per_time'], '-', linewidth=2)
    plt.title('Average Jaccard Similarity Over Time')
    plt.xlabel('Time Step ')
    plt.ylabel('Jaccard Similarity')
    plt.grid(True, alpha=0.3)
    plt.savefig(figure_dir / 'jaccard_similarity_over_time.png', dpi=300)
    plt.show()

    # Plot heatmap of Jaccard similarity over time for each sequence
    plt.figure(figsize=(12, 8))
    plt.imshow(
        jaccard_results['avg_jaccard_per_sequence_time'], 
        aspect='auto', 
        cmap='viridis',
        vmin=0, 
        vmax=1
    )
    plt.colorbar(label='Jaccard Similarity')
    plt.title('Jaccard Similarity Over Time by Sequence')
    plt.xlabel('Time')
    plt.ylabel('Sequence (life)')
    #integer y ticks
    plt.yticks(ticks=range(jaccard_results['avg_jaccard_per_sequence_time'].shape[0]), 
               labels=[f"Seq {i+1}" for i in range(jaccard_results['avg_jaccard_per_sequence_time'].shape[0])])
    plt.savefig(figure_dir / 'jaccard_similarity_heatmap.png', dpi=300)
    plt.show()

    #for one sequence, plot the jaccard similarity over time of every simulation
    for seq_idx in range(10):
        plt.figure(figsize=(10, 6))
        for sim_idx in range(jaccard_results['jaccard_indices'].shape[1]):
            plt.plot(
                time_steps, 
                jaccard_results['jaccard_indices'][seq_idx, sim_idx], 
                'k-', 
                linewidth=1, 
                alpha=0.1
            )
        plt.title(f'Jaccard Similarity Over Time for Sequence {seq_idx+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Jaccard Similarity')
        plt.grid(True, alpha=0.3)
        plt.savefig(figure_dir / f'jaccard/jaccard_similarity_sequence_{seq_idx+1}.png', dpi=300)

    # COMPARE INCOME

    # Filter by income token frequencyß
    token_prefix = "LAB_bredt_loen_beloeb_"
    #token_prefix = "LAB_perindkialt_13_"
    window = (0, 1000)  # Look for tokens in the first 1000 positions
    min_occurrences = 5  # Require at least 10 occurrences
    
    filtered_sequences, filtered_prompts, filtered_original_batches = filter_sequences_by_token_frequency(
        filtered_sequences, filtered_prompts, filtered_original_batches,
        vocab, token_prefix, window, min_occurrences,
        debug=True
    )



    #print the shape of the filtered sequences, prompts and original batches
    print(f"Filtered sequences shape: {filtered_sequences.shape}")

    
    # Use filtered data for analysis
    total_income_data = extract_attribute_data(
        filtered_sequences,  
        filtered_prompts,
        filtered_original_batches,
        vocab,
        token_prefix=token_prefix,
        max_sequences=16,
        max_simulations=100,
        debug=True,
    )
    
    # Save extracted data for later use if needed
    with open(data_dir / "total_income_trajectories.pkl", "wb") as f:
        pickle.dump(total_income_data, f)
    
    # Plot the data
    plot_attribute_trajectories(
        trajectory_data=total_income_data,
        plot_title="Total Income Trajectories (Filtered by Length)",
        save_path=figure_dir / f"trajectories_{token_prefix}.png",
        alpha=0.1
    )