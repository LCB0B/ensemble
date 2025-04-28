import os
os.environ["POLARS_MAX_THREADS"] = "4"
os.environ["RAYON_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set GPU for evaluation

import yaml
import torch
import warnings
from pathlib import Path
import json
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Type
import torch.nn.functional as F # For cross_entropy

# --- Import your relevant classes ---
from src.datamodule4 import LifeLightningDataModule # Or other specific DM
from src.encoder_nano_risk import PretrainNanoEncoder, CausalEncoder # Import relevant model classes
from src.paths import FPATH

# Filter warnings if needed
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool; this will negatively affect performance. Prefer to use a boolean mask directly.",
    category=UserWarning,
    module="torch.nn.modules.activation",
)

# === Core Functions ===

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

    # Use strict=False if checkpoint has extra/missing keys not needed for inference
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(device),
        strict=False, # Often needed
        **hparams # Pass hparams needed for model __init__
    )
    model.to(device)
    model.eval() # Set to evaluation mode
    print("Model loaded successfully and set to evaluation mode.")
    return model

def get_evaluation_dataloader(
    datamodule: LifeLightningDataModule,
    stage: str = 'validate' # Or 'test', 'predict'
) -> torch.utils.data.DataLoader:
    """Gets the dataloader for the specified evaluation stage."""
    print(f"Setting up datamodule for stage: '{stage}'")
    # Ensure the dataset for the stage exists
    if stage == 'validate' and datamodule.val_dataset is None:
        datamodule.setup(stage='validate')
    elif stage == 'test' and (not hasattr(datamodule, 'test_dataset') or datamodule.test_dataset is None):
         datamodule.setup(stage='test')
    elif stage == 'predict' and datamodule.predict_dataset is None:
        datamodule.setup(stage='predict')
    # Add more stages if needed

    print(f"Getting dataloader for stage: '{stage}'")
    if stage == 'validate':
        dataloader = datamodule.val_dataloader()
    elif stage == 'test':
         dataloader = datamodule.test_dataloader()
    elif stage == 'predict':
         dataloader = datamodule.predict_dataloader()
    else:
        raise ValueError(f"Unsupported evaluation stage: {stage}")

    if dataloader is None:
        raise RuntimeError(f"Failed to create dataloader for stage '{stage}'.")

    print(f"Dataloader for stage '{stage}' obtained. Collate fn: {dataloader.collate_fn.__class__.__name__}")
    return dataloader

def evaluate_targeted_generation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    datamodule: LifeLightningDataModule,
    device: str,
    target_token_ids: Optional[List[int]] = None,
    token_groups: Optional[Dict[str, List[int]]] = None,
    max_generation_steps: int = 50,
    num_batches: Optional[int] = None,
    pad_token_id: int = 0
) -> Dict[str, Any]:
    """
    Evaluates model's ability to predict specific tokens using its built-in generation methods.
    """
    print("-" * 50)
    print("Starting targeted token generation evaluation...")
    
    # Initialize metrics
    results = {
        'exact_match_count': 0,
        'total_sequences': 0,
        'avg_steps_to_target': 0,
        'groups': {}
    }
    
    # Set up metrics for each token group
    if token_groups:
        for group_name, tokens in token_groups.items():
            results['groups'][group_name] = {
                'exact_matches': 0, 
                'total_sequences': 0,
                'avg_rank_distance': 0,
                'hit_rate': 0
            }
    
    # Combine all target tokens for efficient lookup
    all_target_tokens = set(target_token_ids or [])
    if token_groups:
        for tokens in token_groups.values():
            all_target_tokens.update(tokens)
    all_target_tokens = list(all_target_tokens)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating generation")):
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
                
            # Process each sequence in the batch
            for seq_idx in range(batch['event'].shape[0]):
                original_seq = batch['event'][seq_idx].clone()
                
                # Find target tokens in original sequence
                target_positions = []
                for pos in range(1, len(original_seq)):
                    if original_seq[pos].item() in all_target_tokens:
                        target_positions.append(pos)
                
                if not target_positions:
                    continue
                    
                # Use prefix before first target as prompt
                first_target_pos = min(target_positions)
                if first_target_pos < 3:  # Need reasonable context
                    continue
                    
                prompt_length = first_target_pos - 1
                original_target = original_seq[first_target_pos].item()
                
                # Create a single-sequence batch for generation
                gen_batch = {k: v[seq_idx:seq_idx+1].clone() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Use model's built-in generation method
                try:
                    generated_sequence = model.generate_sequence(
                        batch=gen_batch,
                        prompt_length=prompt_length,
                        max_new_tokens=max_generation_steps,
                        strategy="most_likely"
                    )
                    
                    # Find target token in generated sequence
                    found_target = False
                    predicted_target = None
                    steps_taken = 0
                    
                    for i, token_id in enumerate(generated_sequence[0]):
                        steps_taken = i + 1
                        if token_id.item() in all_target_tokens:
                            found_target = True
                            predicted_target = token_id.item()
                            break
                    
                    # Only count sequences where we found a target
                    if found_target:
                        results['total_sequences'] += 1
                        results['avg_steps_to_target'] += steps_taken
                        
                        # Exact match evaluation
                        if predicted_target == original_target:
                            results['exact_match_count'] += 1
                        
                        # Group-specific evaluation
                        if token_groups:
                            for group_name, tokens in token_groups.items():
                                if original_target in tokens and predicted_target in tokens:
                                    group_results = results['groups'][group_name]
                                    group_results['total_sequences'] += 1
                                    
                                    # Calculate rank-based distance
                                    true_rank = tokens.index(original_target)
                                    pred_rank = tokens.index(predicted_target)
                                    rank_distance = abs(true_rank - pred_rank)
                                    normalized_distance = rank_distance / (len(tokens) - 1) if len(tokens) > 1 else 0
                                    
                                    group_results['avg_rank_distance'] += normalized_distance
                                    
                                    if predicted_target == original_target:
                                        group_results['exact_matches'] += 1
                                        
                except Exception as e:
                    print(f"Error during generation: {e}")
                    continue
    
    # Finalize results
    if results['total_sequences'] > 0:
        results['exact_match_rate'] = results['exact_match_count'] / results['total_sequences']
        results['avg_steps_to_target'] /= results['total_sequences']
        
        # Finalize group metrics
        for group_name, metrics in results['groups'].items():
            if metrics['total_sequences'] > 0:
                metrics['hit_rate'] = metrics['exact_matches'] / metrics['total_sequences']
                metrics['avg_rank_distance'] /= metrics['total_sequences']
    
    # Print results
    print(f"\nEvaluated {results['total_sequences']} sequences")
    print(f"Exact match rate: {results.get('exact_match_rate', 0):.4f}")
    
    for group_name, metrics in results.get('groups', {}).items():
        if metrics['total_sequences'] > 0:
            print(f"\n{group_name} metrics:")
            print(f"  Exact match rate: {metrics['hit_rate']:.4f}")
            print(f"  Avg rank distance: {metrics['avg_rank_distance']:.4f}")
    
    print("-" * 50)
    return results

if __name__ == "__main__":
    # 1. --- Configuration ---
    # !! Load hparams (ensure path is correct) !!
    HPARAMS_PATH = 'logs/transformer_logs/stable_pretrain/8192/hparams.yaml"
    if not HPARAMS_PATH.exists(): raise FileNotFoundError(f"Hparams file not found: {HPARAMS_PATH}")
    with open(HPARAMS_PATH, "r") as stream: 
        hparams = yaml.safe_load(stream)

    # !! Define Checkpoint and Data Paths !!
    EXPERIMENT_NAME = hparams["experiment_name"]
    RUN_ID = "" # Or load dynamically if possible e.g., using get_wandb_runid
    CHECKPOINT_NAME = "last.ckpt" # Or "best.ckpt"
    #CHECKPOINT_PATH = FPATH.CHECKPOINTS / EXPERIMENT_NAME / RUN_ID / CHECKPOINT_NAME
    CHECKPOINT_PATH = Path("checkpoints/stable_pretrain/8192")
    DATA_DIR_PATH = FPATH.DATA / hparams["dir_path"]
    LMDB_PATH = DATA_DIR_PATH / "dataset.lmdb"
    VOCAB_PATH = DATA_DIR_PATH / "vocab.json"
    PNR_MAP_PATH = DATA_DIR_PATH / "pnr_to_database_idx.json"

    # !! Choose Model Class !!
    MODEL_CLASS = CausalEncoder 

    # !! Evaluation Parameters !!
    EVALUATION_STAGE = 'validate' # Stage to evaluate on
    EVAL_BATCH_SIZE = hparams["batch_size"] * 2 # Larger batch size for eval is fine
    NUM_BATCHES_TO_EVAL = 100 # Set to None for full dataset
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PAD_TOKEN_ID = 0 # <<< Check your vocab for the actual PAD ID! Assumed 0.

    # !! (Optional) Define Specific Tokens to Track !!
    INCOME_TOKEN_PREFIX = "LAB_perindkialt_13_"
    # Load vocab once here to find IDs
    temp_vocab = {}
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, 'r') as f: temp_vocab = json.load(f)
    SPECIFIC_TOKEN_IDS_TO_TRACK = [
        token_id for token, token_id in temp_vocab.items()
        if token.startswith(INCOME_TOKEN_PREFIX)
    ]
    if not SPECIFIC_TOKEN_IDS_TO_TRACK:
         print(f"Warning: No tokens found with prefix '{INCOME_TOKEN_PREFIX}' in {VOCAB_PATH}")

    # Load background data if needed
    background = None
    BACKGROUND_PATH = FPATH.DATA  / hparams['sources']/ "background.parquet"  # Adjust path as needed
    if BACKGROUND_PATH.exists():
        import polars as pl
        print(f"Loading background data from {BACKGROUND_PATH}")
        background = pl.read_parquet(BACKGROUND_PATH)

    # 2. --- Setup ---
    # Instantiate DataModule
    # Use parameters consistent with the model being evaluated
    print("Instantiating DataModule...")
    datamodule = LifeLightningDataModule(
        dir_path=DATA_DIR_PATH,
        lmdb_path=LMDB_PATH,
        vocab_path=VOCAB_PATH,
        pnr_to_idx_path=PNR_MAP_PATH,
        # !! Ensure these match the loaded model's training config !!
        background=background,  # Add the background data
        cls_token=True,
        sep_token=False,
        segment=False,
        max_seq_len=hparams["max_seq_len"],
        # Eval specific config
        batch_size=EVAL_BATCH_SIZE,
        num_workers=0,
    )

    # Load Model
    model = load_model_from_checkpoint(
        model_class=MODEL_CLASS,
        checkpoint_path=CHECKPOINT_PATH / CHECKPOINT_NAME,
        hparams=hparams,
        device=DEVICE
    )

    # Get Dataloader
    dataloader = get_evaluation_dataloader(
        datamodule=datamodule,
        stage=EVALUATION_STAGE
    )

    income_tokens = [token_id for token, token_id in datamodule.vocab.items() 
                if token.startswith("LAB_perindkialt_13_")]


    # Define token groups for evaluation
    token_groups = {
        "income": income_tokens,
    }

    # Run targeted generation evaluation
    generation_results = evaluate_targeted_generation(
        model=model,
        dataloader=dataloader,
        datamodule=datamodule,
        device=DEVICE,
        target_token_ids=SPECIFIC_TOKEN_IDS_TO_TRACK,  # For backward compatibility
        token_groups=token_groups,
        max_generation_steps=100,
        num_batches=10
    )

    # 4. --- Print Results ---
    print("\n=== Final Evaluation Results ===")
    for metric, value in generation_results.items():
        if isinstance(value, float):
            print(f"{metric:<25}: {value:.4f}")
        else:
            print(f"{metric:<25}: {value}")
    print("==============================")