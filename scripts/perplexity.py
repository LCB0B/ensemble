
def run_perplexity_accuracy_evaluation(
    model: CausalEncoder, # Type hint specific model
    dataloader: torch.utils.data.DataLoader,
    datamodule: LifeLightningDataModule,
    device: str,
    num_batches: Optional[int] = None,
    target_token_ids: Optional[List[int]] = None,
    pad_token_id: int = 0
) -> Dict[str, float]:
    """
    Runs perplexity and accuracy evaluation using model's internal method.
    """
    print("-" * 30)
    print("Starting Perplexity/Accuracy Evaluation...")
    if target_token_ids:
        print(f"Calculating specific accuracy for {len(target_token_ids)} token IDs.")
    print("-" * 30)

    total_loss_agg = 0.0
    total_tokens_agg = 0
    total_correct_agg = 0
    total_specific_targets_agg = 0
    correct_specific_agg = 0

    target_token_ids_tensor = torch.tensor(target_token_ids if target_token_ids else [],
                                          device=device, dtype=torch.long)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Batches")):
        if num_batches is not None and batch_idx >= num_batches:
            print(f"\nReached evaluation limit of {num_batches} batches.")
            break

        batch = datamodule.transfer_batch_to_device(batch, device, 0)
        if not batch or 'event' not in batch:
            print(f"Warning: Skipping invalid or empty batch {batch_idx}.")
            continue

        # Call the model's evaluation method
        batch_results = model.evaluate_batch_perplexity_accuracy(batch, pad_token_id)

        # Aggregate results
        total_loss_agg += batch_results['batch_loss']
        total_tokens_agg += batch_results['total_tokens']
        total_correct_agg += batch_results['correct_preds']

        # Aggregate specific accuracy if needed
        if target_token_ids and batch_results['total_tokens'] > 0:
            targets_flat = batch_results['targets_flat']
            predictions_flat = batch_results['predictions_flat']
            valid_mask_flat = batch_results['valid_mask_flat']

            specific_target_mask_flat = torch.isin(targets_flat, target_token_ids_tensor) & valid_mask_flat
            num_specific_targets_in_batch = specific_target_mask_flat.sum().item()

            if num_specific_targets_in_batch > 0:
                total_specific_targets_agg += num_specific_targets_in_batch
                correct_specific_preds_mask = (predictions_flat == targets_flat) & specific_target_mask_flat
                correct_specific_agg += correct_specific_preds_mask.sum().item()

    # --- Calculate Final Metrics ---
    print("\nEvaluation loop finished.")
    results = {}
    if total_tokens_agg > 0:
        avg_loss = total_loss_agg / total_tokens_agg
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        overall_accuracy = (total_correct_agg / total_tokens_agg) * 100
        results['average_loss'] = avg_loss
        results['perplexity'] = perplexity
        results['overall_accuracy'] = overall_accuracy
    else:
        print("No valid (non-padding) tokens were evaluated.")
        results.update({'average_loss': float('nan'), 'perplexity': float('nan'), 'overall_accuracy': float('nan')})


    if target_token_ids:
        if total_specific_targets_agg > 0:
            specific_accuracy = (correct_specific_agg / total_specific_targets_agg) * 100
            results['specific_token_accuracy'] = specific_accuracy
        else:
            print("No target tokens matching the specified list were found.")
            results['specific_token_accuracy'] = 0.0

    print("-" * 30)
    print("--- Perplexity/Accuracy Results ---")
    for metric, value in results.items(): print(f"{metric:<25}: {value:.4f}" if isinstance(value, float) else f"{metric:<25}: {value}")
    print("-" * 30)
    return results


def run_generative_evaluation(
    model: CausalEncoder,
    dataloader: torch.utils.data.DataLoader,
    datamodule: LifeLightningDataModule, # Pass the DM
    device: str,
    prompt_length: int,
    max_new_tokens: int,
    num_batches: Optional[int] = None,
    strategy: str = "most_likely",
    k: int = 5,
    p: float = 0.9,
    temperature: float = 1.0,
    pad_token_id: int = 0
) -> Dict[str, float]:
    """
    Runs generative evaluation, calculating perplexity of generated sequences.
    MODIFIED: Applies on_after_batch_transfer to create mask for evaluation.
    """
    # ... (initial setup and prints remain the same) ...

    total_generated_loss_agg = 0.0
    total_generated_tokens_agg = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating/Evaluating")):
        # ... (batch limit check, initial batch transfer, empty batch check remain the same) ...

        # Apply the mask creation hook to the *original* batch first
        # This ensures the model's generation step uses the correct mask internally
        try:
            batch = datamodule.on_after_batch_transfer(batch, 0)
            if 'attn_mask' not in batch:
                 print(f"Warning: 'attn_mask' not created for original batch {batch_idx} for generation. Skipping.")
                 continue
        except Exception as e:
            print(f"\nERROR during on_after_batch_transfer for generation batch {batch_idx}: {e}. Skipping batch.")
            continue

        # ... (prompt length check remains the same) ...

        # --- Generate ---
        generated_ids = model.generate_sequence(
            batch=batch, # Pass the batch *with* the initial attn_mask
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            strategy=strategy, k=k, p=p, temperature=temperature
        )

        # --- Evaluate Generated Part ---
        prompt_ids = batch['event'][:, :prompt_length]
        full_sequence_ids = torch.cat((prompt_ids, generated_ids), dim=1)

        # --- Create a new batch dictionary for evaluation ---
        # Start with the full sequence
        eval_batch = {'event': full_sequence_ids}

        # !! Crucially, add other necessary components for mask creation !!
        # We need sequence_lens for the *full* sequence. Assuming no padding
        # was introduced during generation, the lengths are constant.
        # If padding *could* occur (e.g., EOS token), this needs adjustment.
        bs, full_seq_len = full_sequence_ids.shape
        eval_batch['sequence_lens'] = torch.full((bs,), full_seq_len, dtype=torch.long, device=device)

        # Add any other keys from the original batch that might be needed
        # by on_after_batch_transfer (unlikely based on code, but good practice)
        # for key in ['acc_event_lens', 'first_abspos', 'abspos', ...]: # Add keys if needed
        #     if key in batch:
        #         eval_batch[key] = batch[key] # Potentially slice/adjust these too

        # --- Apply the mask creation hook to the *new* evaluation batch ---
        try:
            # eval_batch is already on the device implicitly (components came from device tensors)
            eval_batch = datamodule.on_after_batch_transfer(eval_batch, 0) # Create mask for the full seq
            if 'attn_mask' not in eval_batch:
                print(f"Warning: 'attn_mask' not created for eval_batch {batch_idx}. Skipping eval for this batch.")
                continue
        except Exception as e:
            print(f"\nERROR during on_after_batch_transfer for eval_batch {batch_idx}: {e}. Skipping eval for this batch.")
            continue

        # --- Evaluate using the model's method ---
        try:
             gen_results = model.evaluate_generated_sequence_perplexity(
                 full_sequence_batch=eval_batch, # Use the batch with the correct mask
                 prompt_length=prompt_length,
                 pad_token_id=pad_token_id
             )
             # Aggregate results
             total_generated_loss_agg += gen_results['generated_loss_sum']
             total_generated_tokens_agg += gen_results['generated_token_count']
        except Exception as e:
             print(f"\nERROR evaluating generated sequence for batch {batch_idx}: {e}. Skipping.")
             # import pdb; pdb.set_trace()
             continue

    # --- Calculate Final Metrics ---
    # (Remains the same)
    print("\nGenerative evaluation loop finished.")
    results = {}
    if total_generated_tokens_agg > 0:
        avg_generated_loss = total_generated_loss_agg / total_generated_tokens_agg
        perplexity_generated = torch.exp(torch.tensor(avg_generated_loss)).item()
        results['generated_sequence_perplexity'] = perplexity_generated
    else:
        print("No valid generated tokens were evaluated.")
        results['generated_sequence_perplexity'] = float('nan')

    print("-" * 30)
    print("--- Generative Evaluation Results ---")
    for metric, value in results.items(): print(f"{metric:<30}: {value:.4f}" if isinstance(value, float) else f"{metric:<30}: {value}")
    print("-" * 30)
    return results
# === Example Usage (Modified) ===

if __name__ == "__main__":
    # 1. --- Configuration ---
    HPARAMS_PATH = FPATH.CONFIGS / "hparams_pretrain2.yaml"
    if not HPARAMS_PATH.exists(): raise FileNotFoundError(f"Hparams file not found: {HPARAMS_PATH}")
    with open(HPARAMS_PATH, "r") as stream: hparams = yaml.safe_load(stream)

    EXPERIMENT_NAME = hparams["experiment_name"]
    RUN_ID = "" # TODO: Set your Run ID
    CHECKPOINT_NAME = "last.ckpt"
    CHECKPOINT_PATH = Path("checkpoints/sample") # TODO: Set correct base path FPATH.CHECKPOINTS / EXPERIMENT_NAME / RUN_ID
    DATA_DIR_PATH = FPATH.DATA / hparams["dir_path"]
    LMDB_PATH = DATA_DIR_PATH / "dataset.lmdb"
    VOCAB_PATH = DATA_DIR_PATH / "vocab.json"
    PNR_MAP_PATH = DATA_DIR_PATH / "pnr_to_database_idx.json" # Assuming JSON

    MODEL_CLASS = CausalEncoder # <<< Load as CausalEncoder for evaluation methods

    EVALUATION_STAGE = 'validate'
    EVAL_BATCH_SIZE = hparams["batch_size"] # Use training batch size or adjust
    NUM_BATCHES_TO_EVAL = 50 # Limit for testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PAD_TOKEN_ID = 0 # <<< Check your vocab!

    # Generation parameters
    PROMPT_LENGTH = 50 # How many tokens to use as prompt
    MAX_NEW_TOKENS = 100 # How many tokens to generate
    GENERATION_STRATEGY = "most_likely" # or "top_k", "top_p"

    # Specific token tracking
    INCOME_TOKEN_PREFIX = "LAB_perindkialt_13_"
    temp_vocab = {}
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, 'r') as f: temp_vocab = json.load(f)
    SPECIFIC_TOKEN_IDS_TO_TRACK = [
        token_id for token, token_id in temp_vocab.items()
        if token.startswith(INCOME_TOKEN_PREFIX)
    ] or None # Set to None if prefix not found or tracking not desired

    # 2. --- Setup ---
    print("Instantiating DataModule...")
    datamodule = LifeLightningDataModule(
        dir_path=DATA_DIR_PATH, lmdb_path=LMDB_PATH, vocab_path=VOCAB_PATH,
        pnr_to_idx_path=PNR_MAP_PATH, background_length=0, cls_token=True,
        sep_token=False, segment=False, max_seq_len=hparams["max_seq_len"],
        batch_size=EVAL_BATCH_SIZE, num_workers=0,
    )

    model = load_model_from_checkpoint(
        model_class=MODEL_CLASS, checkpoint_path=CHECKPOINT_PATH / CHECKPOINT_NAME,
        hparams=hparams, device=DEVICE
    )

    dataloader = get_evaluation_dataloader(datamodule=datamodule, stage=EVALUATION_STAGE)

    # 3. --- Run Evaluations ---

    # --- A: Perplexity / Accuracy Evaluation ---
    print("\n" + "="*10 + " Running Perplexity/Accuracy Evaluation " + "="*10)
    perplexity_results = run_perplexity_accuracy_evaluation(
        model=model, dataloader=dataloader, datamodule=datamodule,
        device=DEVICE, num_batches=NUM_BATCHES_TO_EVAL,
        target_token_ids=SPECIFIC_TOKEN_IDS_TO_TRACK,
        pad_token_id=PAD_TOKEN_ID
    )

    # --- B: Generative Evaluation ---
    print("\n" + "="*10 + " Running Generative Evaluation " + "="*10)
    # Need to re-get dataloader if previous loop consumed it
    dataloader_gen = get_evaluation_dataloader(datamodule=datamodule, stage=EVALUATION_STAGE)
    generative_results = run_generative_evaluation(
        model=model, dataloader=dataloader_gen, datamodule=datamodule,
        device=DEVICE, prompt_length=PROMPT_LENGTH, max_new_tokens=MAX_NEW_TOKENS,
        num_batches=NUM_BATCHES_TO_EVAL, # Limit number of prompts
        strategy=GENERATION_STRATEGY,
        pad_token_id=PAD_TOKEN_ID
    )

    print("\nCombined Evaluation Complete.")