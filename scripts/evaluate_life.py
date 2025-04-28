import os
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List
import re
from tqdm import tqdm

def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with open(vocab_path, "r") as f:
        import json
        vocab = json.load(f)
    return vocab

def build_ordered_group(vocab: Dict[str, int], prefix: str) -> List[int]:
    """
    Return an ordered list of token IDs for tokens whose string starts with `prefix`.
    Ordering is determined by the integer part that follows the prefix.
    Tokens without an integer part are ordered lexicographically.
    """
    group = []
    for token, tid in vocab.items():
        if token.startswith(prefix):
            suffix = token[len(prefix):]
            # Try to extract an integer; if not, use 0
            try:
                num = int(suffix)
            except ValueError:
                num = 0
            group.append((num, tid))
    # Sort by the extracted integer value
    group.sort(key=lambda x: x[0])
    # Return only the token IDs in order
    return [tid for num, tid in group]

def evaluate_generated_sequences(
    output_dir: Path,
    token_group_prefix: str,
    vocab_path: Path
) -> Dict[str, Any]:
    """
    Loads the generated sequences, prompts, and original batches; then evaluates the
    predictions by finding the next occurrence of a target token (from the ordered group)
    after the prompt. Metrics include exact match rate and average normalized rank distance.
    
    Args:
        output_dir: Directory where generated files live.
        token_group_prefix: Prefix to identify target tokens (e.g., "LAB_perindkialt_13_").
        vocab_path: Path to the vocab JSON file.
        
    Returns:
        A dictionary with evaluation metrics.
    """
    output_dir = Path(output_dir)
    sequences_path = output_dir / "generated_sequences.pkl"
    prompts_path = output_dir / "prompts.pkl"
    original_batches_path = output_dir / "original_batches.pkl"
    
    # Load saved data
    with open(sequences_path, "rb") as f:
        all_sequences = pickle.load(f)  # list of tensors, each shape [batch_size, num_simulations, seq_len]
    with open(prompts_path, "rb") as f:
        all_prompts = pickle.load(f)    # list of tensors, each shape [batch_size, prompt_length]
    with open(original_batches_path, "rb") as f:
        all_original_batches = pickle.load(f)  # list of dicts; we expect key 'event'
    
    # Load vocabulary and build ordered target token group
    vocab = load_vocab(vocab_path)
    target_group: List[int] = build_ordered_group(vocab, token_group_prefix)
    if not target_group:
        raise ValueError(f"No tokens found in vocab with prefix '{token_group_prefix}'.")
    # print with order their token names
    # target_group_names = [k for k, v in vocab.items() if v in target_group]
    # target_group_names.sort(key=lambda x: int(re.search(r'\d+', x[len(token_group_prefix):]).group()))
    # print(f"Target token group names (ordered by prefix '{token_group_prefix}'): {target_group_names}")
    # print(f"Target token group (ordered by prefix '{token_group_prefix}'): {target_group}")
    # Check if all sequences and prompts are aligned

    # Evaluation counters
    total_sequences = 0
    exact_match_count = 0
    total_rank_distance = 0.0
    total_steps = 0
    count_evaluated = 0
    recall_at_5_count = 0  # count for recall@5
    
    # For each batch (assumed aligned by order with prompts and original_batches)
    for seq_tensor, prompt_tensor, original in tqdm(zip(all_sequences, all_prompts, all_original_batches),total=len(all_sequences), desc="Evaluating sequences"):
        # seq_tensor: [batch_size, num_simulations, seq_len]
        # prompt_tensor: [batch_size, prompt_length]
        # original['event']: [batch_size, seq_len_full]
        batch_size = prompt_tensor.shape[0]
        prompt_length = prompt_tensor.shape[1]
        # For each sequence in the batch
        for i in range(batch_size):
            orig_seq = original['event'][i]  # original sequence (assume already a tensor or list)
            # Determine the original target token:
            # Find the first token in orig_seq (after prompt_length) that is in target_group.
            original_target = None
            for token in orig_seq[prompt_length:]:
                if token in target_group:
                    original_target = token
                    break
            if original_target is None:
                continue  # skip if no target token in the original sequence
            
            # For each simulation for this sequence
            sim_predictions = seq_tensor[i]  # shape: [num_simulations, seq_len]
            for sim in range(sim_predictions.shape[0]):
                pred_seq = sim_predictions[sim]
                # Find first token after prompt_length in pred_seq that is in target_group.
                predicted_target = None
                steps = 0
                for token in pred_seq[:]:
                    steps += 1
                    if token in target_group:
                        predicted_target = token
                        break
                if predicted_target is None:
                    continue
                count_evaluated += 1
                total_steps += steps
                # Exact match
                if predicted_target == original_target:
                    exact_match_count += 1
                # For group metrics, compute rank distance based on ordering in target_group
                true_rank = target_group.index(original_target)
                pred_rank = target_group.index(predicted_target)
                rank_distance = abs(true_rank - pred_rank)
                normalized_distance = rank_distance / (len(target_group) - 1) if len(target_group) > 1 else 0
                total_rank_distance += rank_distance
                if rank_distance <= 5:
                    recall_at_5_count += 1


                total_sequences += 1

    avg_steps = total_steps / count_evaluated if count_evaluated > 0 else None
    exact_match_rate = exact_match_count / count_evaluated if count_evaluated > 0 else None
    avg_norm_rank_distance = total_rank_distance / count_evaluated if count_evaluated > 0 else None
    recall_at_5 = recall_at_5_count / count_evaluated if count_evaluated > 0 else None
    #add the number of sequences to the results
    results = {
        "n_simulations": len(all_sequences)*len(all_sequences[0])*len(all_sequences[0][0]),
        "total_evaluated": count_evaluated,
        "exact_match_rate": exact_match_rate,
        "avg_steps_to_target": avg_steps,
        "avg_normalized_rank_distance": avg_norm_rank_distance,
        "recall_at_5": recall_at_5,
        "target_group_size": len(target_group)
    }
    print("Evaluation results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
        
    return results

if __name__ == "__main__":
    # Example usage:
    from src.paths import FPATH
    # Replace with your actual output directory path generated/stable_pretrain/generated_sequences_20250418_180103_200_200_10
    output_directory = FPATH.GENERATED / "stable_pretrain" / "generated_sequences_20250421_001832_1000_300_30"
    # Replace with your vocab file path
    vocab_file = FPATH.DATA / "life_all_compiled" / "vocab.json"
    
    # Define the prefix for target tokens (e.g., income tokens)
    target_prefix = "LAB_bredt_loen_beloeb_"
    
    evaluate_generated_sequences(
        output_dir=output_directory,
        token_group_prefix=target_prefix,
        vocab_path=vocab_file
    )

    target_prefix = "LAB_perindkialt_13_"
    
    evaluate_generated_sequences(
        output_dir=output_directory,
        token_group_prefix=target_prefix,
        vocab_path=vocab_file
    )

# Example to decode tokens
# decode_tokens(all_sequences[0][0][0], vocab)