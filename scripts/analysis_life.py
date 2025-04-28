import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm
import re

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

# STEP 1: DATA EXTRACTION AND ANALYSIS FUNCTION
def extract_attribute_data(
    output_dir: Path,
    vocab_path: Path,
    token_prefix: str,
    reverse_mapping: bool = False,
    max_sequences: int = 100,
    max_simulations: int = 100
) -> Dict[str, Any]:
    """
    Extract attribute data from generated sequences without plotting.
    
    Returns:
        Dictionary with structured trajectory data that can be used for plotting
    """
    # Load data
    all_sequences, all_prompts, all_original_batches = load_data(output_dir)
    
    # Load vocabulary
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    
    # Build token mapping
    token_map = build_token_map(vocab, token_prefix, reverse=reverse_mapping)
    
    if not token_map:
        raise ValueError(f"No mappable tokens found with prefix '{token_prefix}'")
    
    # Extract token names for legend
    token_names = {tid: token for token, tid in vocab.items() if tid in token_map}
    
    # Prepare data structure for all trajectories
    trajectory_data = []
    
    # Process sequences
    for seq_idx, (seq_batch, prompts, original) in enumerate(zip(all_sequences, all_prompts, all_original_batches)):
        if seq_idx >= max_sequences:
            break
            
        # Get batch size
        batch_size = prompts.shape[0]
        
        # Process each sequence in the batch
        for i in range(min(batch_size, max_sequences - seq_idx)):
            person_idx = seq_idx * batch_size + i
            if person_idx >= max_sequences:
                break
                
            # Extract original sequence
            original_seq = original['event'][i].cpu().numpy()
            prompt_length = prompts.shape[1]
            
            # Extract original trajectory
            orig_trajectory = extract_attribute_trajectory(original_seq, token_map)
            
            # Split into before and after prompt
            orig_before_prompt = [(pos, val) for pos, val in orig_trajectory if pos < prompt_length]
            orig_after_prompt = [(pos, val) for pos, val in orig_trajectory if pos >= prompt_length]
            
            # Extract generated trajectories
            generated_trajectories = []
            sim_predictions = seq_batch[i]  # [num_simulations, seq_len]
            num_sims = min(sim_predictions.shape[0], max_simulations)
            
            for sim in range(num_sims):
                gen_seq = sim_predictions[sim].cpu().numpy()
                gen_trajectory = extract_attribute_trajectory(gen_seq, token_map)
                gen_after_prompt = [(pos, val) for pos, val in gen_trajectory if pos >= prompt_length]
                
                generated_trajectories.append({
                    'sim_index': sim,
                    'trajectory': gen_after_prompt
                })
            
            # Store all data for this person
            person_data = {
                'person_idx': person_idx,
                'prompt_length': prompt_length,
                'original_before_prompt': orig_before_prompt,
                'original_after_prompt': orig_after_prompt,
                'generated_trajectories': generated_trajectories
            }
            
            trajectory_data.append(person_data)
    
    return {
        'token_prefix': token_prefix,
        'trajectories': trajectory_data
    }

# STEP 2: PLOTTING FUNCTION
def plot_attribute_trajectories(
    trajectory_data: Dict[str, Any],
    plot_title: Optional[str] = None,
    save_path: Optional[Path] = None
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
            ax.plot(x_before, y_before, 'o-', color='blue', linewidth=2, 
                    label='Original (prompt)', alpha=0.7)
        
        # Plot original data after prompt
        if orig_after:
            x_after, y_after = zip(*orig_after)
            ax.plot(x_after, y_after, 'o-', color='green', linewidth=2, 
                    label='Original (after prompt)', alpha=0.7)
        
        # Plot generated trajectories
        for i, gen_data in enumerate(generated):
            gen_trajectory = gen_data['trajectory']
            if gen_trajectory:
                x_gen, y_gen = zip(*gen_trajectory)
                ax.plot(x_gen, y_gen, 'o--', alpha=0.4, linewidth=1, 
                       label=f'Generated {i+1}' if i == 0 else "_nolegend_")
        
        # Add vertical line at prompt boundary
        ax.axvline(x=prompt_length, color='red', linestyle='--', alpha=0.7, 
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
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save or display
    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

# STEP 3: COMBINED FUNCTION (FOR CONVENIENCE)
def analyze_and_plot_trajectories(
    output_dir: Path,
    vocab_path: Path,
    token_prefix: str,
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
        output_dir=output_dir,
        vocab_path=vocab_path,
        token_prefix=token_prefix,
        reverse_mapping=reverse_mapping,
        max_sequences=max_sequences,
        max_simulations=max_simulations
    )
    
    # Define save path
    save_path = None
    if save_plot:
        plot_dir = output_dir / "trajectory_plots"
        save_path = plot_dir / f"{token_prefix.replace('_', '')}_trajectories.png"
    
    # Plot data
    plot_attribute_trajectories(
        trajectory_data=trajectory_data,
        plot_title=plot_title,
        save_path=save_path
    )
    
    return trajectory_data

# Example usage
if __name__ == "__main__":
    from src.paths import FPATH
    
    # Define paths
    output_directory = FPATH.GENERATED / "stable_pretrain" / "generated_sequences_20250428_151150_1000_50_20"
    vocab_file = FPATH.DATA / "life_test_compiled" / "vocab.json"

    data_dir = Path("generated") / "analysis_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = Path('figures') / "trajectory_plots"

    token_prefix = "LAB_bredt_loen_beloeb_"

    # Option 2: Separate data extraction and plotting
    # Extract data
    total_income_data = extract_attribute_data(
        output_dir=output_directory,
        vocab_path=vocab_file,
        token_prefix=token_prefix,
        max_sequences=10,
        max_simulations=100
    )
    
    # Save extracted data for later use if needed

    with open(data_dir / "total_income_trajectories.pkl", "wb") as f:
        pickle.dump(total_income_data, f)
    
    # Plot the data
    plot_attribute_trajectories(
        trajectory_data=total_income_data,
        plot_title="Total Income Trajectories",
        save_path=figure_dir / f"trajectories_{token_prefix}.png"
    )