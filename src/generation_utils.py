"""
Generation utilities for the NanoEncoder model.
Contains all generation-related functions with proper dtype handling.
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Any, List
from src.encoder_nano_risk import GenerativeNanoEncoder
from src.datamodule2 import PretrainDataModule
import polars as pl
import pyarrow.dataset as ds
from src.paths import FPATH
import json


_VOCAB_CACHE = {}

def load_vocab(path: str):
    """
    Load a vocab file into {id: token_str}.
    Supports:
      - JSON list: ["<pad>", "A", "B", ...]
      - JSON dict: {"token": id, ...} or {"id": "token", ...}
      - Plain text file: one token per line (line index = id)
    Cached by absolute path.
    """
    if not path:
        return {}
    import os
    abs_path = os.path.abspath(path)
    if abs_path in _VOCAB_CACHE:
        return _VOCAB_CACHE[abs_path]

    mapping = {}
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            head = f.read(2048)
            f.seek(0)
            if head.lstrip().startswith("{") or head.lstrip().startswith("["):
                data = json.load(f)
                if isinstance(data, list):
                    for i, tok in enumerate(data):
                        mapping[i] = str(tok)
                elif isinstance(data, dict):
                    # Detect direction heuristic
                    sample_val = next(iter(data.values()))
                    if isinstance(sample_val, int):
                        # token -> id
                        for tok, idx in data.items():
                            mapping[int(idx)] = str(tok)
                    else:
                        # id -> token
                        for k, v in data.items():
                            try:
                                mapping[int(k)] = str(v)
                            except:
                                continue
                else:
                    print("[WARN] Unsupported JSON vocab structure.")
            else:
                # Plain text file
                f.seek(0)
                for i, line in enumerate(f):
                    mapping[i] = line.rstrip("\n")
    except Exception as e:
        print(f"[WARN] Failed to load vocab '{path}': {e}")
        mapping = {}

    _VOCAB_CACHE[abs_path] = mapping
    return mapping

def decode_ids(ids, vocab: dict, unknown_token="?"):
    """
    Decode a list of integer ids using vocab mapping.
    Falls back to raw id if not found.
    """
    if not vocab:
        return [str(i) for i in ids]
    return [vocab.get(int(i), f"{i}:{unknown_token}") for i in ids]


def load_pretrained_model(checkpoint_path: str, hparams: dict) -> GenerativeNanoEncoder:
    """Load a pretrained model for generation"""
    model = GenerativeNanoEncoder(**hparams)
    
    # Load weights from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    return model


def setup_datamodule(name_model: str) -> tuple[PretrainDataModule, dict]:
    """Setup datamodule and load hparams from model directory"""
    # Load configuration
    with open(FPATH.TB_LOGS / "destiny" / name_model / 'hparams.yaml', "r") as f:
        hparams = yaml.safe_load(f)

    # Set up data
    source_paths = [
        (FPATH.DATA / hparams["source_dir"] / path).with_suffix(".parquet")
        for path in hparams["sources"]
    ]
    background_path = (
        FPATH.DATA / hparams["source_dir"] / hparams["background"]
    ).with_suffix(".parquet")

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)

    dm = PretrainDataModule(
        dir_path=FPATH.DATA / hparams["dir_path"],
        sources=sources,
        background=background,
        subset_background=hparams["subset_background"],
        n_tokens=hparams["n_tokens"],
        lengths=hparams["lengths"],
        num_workers=0,  # Use 0 for generation
        max_seq_len=hparams["max_seq_len"],
        source_dir=hparams["source_dir"],
    )

    dm.prepare_data()
    dm.setup()

    # Get vocab size
    hparams["vocab_size"] = len(dm.pipeline.vocab)
    
    return dm, hparams


def truncate_batch_sequences(batch, max_length):
    """Truncate sequences in batch to max_length"""
    if 'cu_seqlens' in batch:
        # Batch is in unpacked format - need to repack first, truncate, then unpack again
        from flash_attn.bert_padding import pad_input, unpad_input
        
        # Repack to standard format
        B = len(batch['cu_seqlens']) - 1
        T = batch['max_seqlen_in_batch']
        
        repacked = {}
        for key in ['event', 'abspos', 'age', 'segment']:
            if key in batch:
                padded = pad_input(batch[key].unsqueeze(-1), batch['indices'], B, T)
                repacked[key] = padded.squeeze(-1)
        
        repacked['attn_mask'] = (repacked['event'] != 0)
        
        # Truncate in standard format
        truncated = {}
        for key, value in repacked.items():
            if torch.is_tensor(value) and value.dim() == 2:
                truncated[key] = value[:, :max_length]
            else:
                truncated[key] = value
        
        # Re-unpack to Flash Attention format
        _, indices, cu_seqlens, max_seqlen_in_batch, total = unpad_input(
            truncated["event"].unsqueeze(-1), truncated["attn_mask"]
        )
        
        truncated.update({
            "indices": indices,
            "max_seqlen_in_batch": max_seqlen_in_batch,
            "cu_seqlens": cu_seqlens,
            "total": total.sum().item()
        })
        
        # Flatten features
        for key in ["event", "abspos", "age", "segment"]:
            if key in truncated:
                truncated[key] = truncated[key].flatten()[indices]
        
        return truncated
    else:
        # Batch is in standard format - simple truncation
        truncated = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() == 2:
                truncated[key] = value[:, :max_length]
            else:
                truncated[key] = value
        
        # Update attention mask
        if 'attn_mask' in truncated:
            truncated['attn_mask'] = (truncated['event'] != 0)
        
        return truncated

def _merge_batches(batches, max_prompt_length=None):
    """
    Merge a list of standard-format batches (each (B_i, T)) into one big batch.
    Truncates to max_prompt_length if provided (same across keys).
    Only concatenates 2D tensors (event, abspos, age, segment, attn_mask, target if present).
    Non-2D / scalar keys: keeps from first batch if identical shape else skips.
    """
    if not batches:
        return {}
    merged = {}
    two_d_keys = set()
    for b in batches:
        for k, v in b.items():
            if torch.is_tensor(v) and v.ndim == 2:
                two_d_keys.add(k)

    for k in two_d_keys:
        tensors = []
        for b in batches:
            if k not in b:
                continue
            t = b[k]
            if max_prompt_length is not None and t.shape[1] > max_prompt_length:
                t = t[:, :max_prompt_length]
            tensors.append(t)
        if tensors:
            merged[k] = torch.cat(tensors, dim=0)

    # Derive attn_mask if missing
    if "attn_mask" not in merged and "event" in merged:
        merged["attn_mask"] = (merged["event"] != 0).to(merged["event"].dtype)

    # Copy over consistent 1D/scalar meta keys (optional)
    first = batches[0]
    for k, v in first.items():
        if k in merged:
            continue
        if torch.is_tensor(v) and v.ndim == 2:
            continue
        merged[k] = v
    return merged


def generate_from_dataloader(
    model,
    dataloader,
    num_samples=1,
    generation_configs=None,
    max_prompt_length=None,
    accumulate_batches: int = 1,
    target_batch_size: int = None,
):
    """
    Generate from dataloader.
    accumulate_batches: number of consecutive dataloader batches to merge per generation call.
    target_batch_size: override to collect multiple dataloader batches until reaching this batch size.
                       (Takes precedence over accumulate_batches if provided.)
    """
    results = []
    if generation_configs is None:
        generation_configs = [{}]

    total_prompt_tokens = 0
    total_gen_tokens = 0
    total_ms = 0.0

    data_iter = iter(dataloader)
    batch_call_idx = 0

    while batch_call_idx < num_samples:
        raw_batches = []
        collected = 0
        # Collect batches
        while True:
            try:
                b = next(data_iter)
            except StopIteration:
                break
            raw_batches.append(b)
            if target_batch_size is not None:
                # Estimate B from 'event' tensor if present
                if "event" in b and torch.is_tensor(b["event"]):
                    collected += b["event"].shape[0]
                else:
                    # Fallback: count 1
                    collected += 1
                if collected >= target_batch_size:
                    break
            else:
                if len(raw_batches) >= accumulate_batches:
                    break
        if not raw_batches:
            break  # No more data

        # Move each small batch to device & truncate BEFORE merge
        device = next(model.parameters()).device
        prepared_batches = []
        for sb in raw_batches:
            for k, v in sb.items():
                if torch.is_tensor(v):
                    sb[k] = v.to(device)
            if max_prompt_length is not None and "event" in sb:
                if sb["event"].shape[1] > max_prompt_length:
                    sb["event"] = sb["event"][:, :max_prompt_length]
                    for aux_key in ("attn_mask", "abspos", "age", "segment", "target"):
                        if aux_key in sb and torch.is_tensor(sb[aux_key]) and sb[aux_key].shape[1] > max_prompt_length:
                            sb[aux_key] = sb[aux_key][:, :max_prompt_length]
                if "attn_mask" in sb and sb["attn_mask"].dtype != torch.float32:
                    sb["attn_mask"] = sb["attn_mask"].float()
            prepared_batches.append(sb)

        merged_batch = _merge_batches(
            prepared_batches, max_prompt_length=max_prompt_length
        )
        if not merged_batch:
            break

        B_merged = merged_batch["event"].shape[0]
        print(f"\nMerged generation batch {batch_call_idx}: size={B_merged} "
              f"(from {len(prepared_batches)} dataloader batches)")

        for ci, gen_config in enumerate(generation_configs):
            print(f"Config {ci}: {gen_config}")
            out = model.generate(merged_batch, **gen_config)
            generated_batch = out
            metrics = compute_generation_metrics(
                generated_batch, vocab_size=model.hparams.vocab_size
            )
            prof = generated_batch.get("profile")
            if prof:
                metrics["tokens_per_sec_total"] = f"{prof['tokens_per_sec_total']:.2f}"
                metrics["gen_tokens_per_sec"] = f"{prof['gen_tokens_per_sec']:.2f}"
                metrics["decode_gen_tokens_per_sec"] = f"{prof['decode_gen_tokens_per_sec']:.2f}"
                metrics["first_step_ms"] = f"{(prof['first_step_ms'] or 0):.2f}"
                total_prompt_tokens += prof["prompt_tokens"]
                total_gen_tokens += prof["gen_tokens"]
                total_ms += prof["total_ms"]

            print(f"Metrics: {metrics}")
            generated_batch["metrics"] = metrics
            results.append(generated_batch)

        batch_call_idx += 1
        if batch_call_idx >= num_samples:
            break

    if total_ms > 0 and total_gen_tokens > 0:
        agg_total_tps = (total_prompt_tokens + total_gen_tokens) / (total_ms / 1000.0)
        agg_gen_tps = total_gen_tokens / (total_ms / 1000.0)
        print(f"[AGG SPEED] total_prompt_tokens={total_prompt_tokens} "
              f"total_gen_tokens={total_gen_tokens} wall_ms={total_ms:.1f} "
              f"total_toks/s={agg_total_tps:.1f} gen_toks/s={agg_gen_tps:.1f}")
    return results

def print_generation_sample(
    result,
    sample_idx=0,
    vocab: dict | None = None,
    max_tokens: int = 128,
    decode: bool = False,
):
    """Print a sample generation with optional vocab decoding."""
    generated = result['generated_events'][sample_idx]
    generation_length = result['generation_lengths'][sample_idx].item()
    total_length = generated.shape[0]
    prompt_length = result.get('prompt_length', total_length - generation_length)

    print(f"Total sequence length: {total_length}")
    print(f"Prompt length: {prompt_length}")
    print(f"Generation length: {generation_length}")

    prompt_tokens = generated[:prompt_length].tolist()
    gen_tokens = generated[prompt_length:prompt_length + generation_length].tolist()

    def _show(label, tokens):
        shown = tokens[:max_tokens]
        print(f"\n--- {label} ({len(tokens)} tokens; showing {len(shown)}) ---")
        print("IDs:", shown)
        if decode:
            print("TOK:", decode_ids(shown, vocab or {}))

    _show("PROMPT", prompt_tokens)
    _show("GENERATED", gen_tokens)
    _show("FULL SEQUENCE", generated.tolist())

    # Metrics formatting
    if 'metrics' in result:
        formatted = {}
        for k, v in result['metrics'].items():
            if isinstance(v, float):
                if 'time' in k.lower():
                    formatted[k] = f"{v:.1f}ms"
                else:
                    formatted[k] = f"{v:.3f}"
            else:
                formatted[k] = v
        print(f"\nMetrics: {formatted}")

def compute_generation_metrics(generated_batch: Dict[str, torch.Tensor], vocab_size: int = None) -> Dict[str, float]:
    """
    Compute various metrics for the generated sequences.
    
    Args:
        generated_batch: Output from generate() method
        vocab_size: Size of vocabulary for diversity calculations
        
    Returns:
        Dictionary of computed metrics
    """
    generated_events = generated_batch['generated_events']
    prompt_length = generated_batch['prompt_length']
    
    # Extract only the newly generated tokens
    new_tokens = generated_events[:, prompt_length:]
    
    metrics = {}
    
    # Basic statistics
    metrics['avg_generation_length'] = generated_batch['generation_lengths'].float().mean().item()
    metrics['max_generation_length'] = generated_batch['generation_lengths'].max().item()
    metrics['min_generation_length'] = generated_batch['generation_lengths'].min().item()
    metrics['completion_rate'] = generated_batch['finished'].float().mean().item()
    
    # Token diversity
    if vocab_size is not None:
        unique_tokens = torch.unique(new_tokens[new_tokens != 0])  # Exclude padding
        metrics['vocab_usage'] = len(unique_tokens) / vocab_size
        metrics['unique_tokens_generated'] = len(unique_tokens)
    
    # Repetition analysis (simple n-gram repetition)
    for n in [2, 3, 4]:
        if new_tokens.shape[1] >= n:
            ngrams = []
            for i in range(new_tokens.shape[1] - n + 1):
                ngram = new_tokens[:, i:i+n]
                ngrams.append(ngram)
            
            if ngrams:
                all_ngrams = torch.cat(ngrams, dim=0)
                unique_ngrams = torch.unique(all_ngrams, dim=0)
                repetition_rate = 1.0 - (len(unique_ngrams) / len(all_ngrams))
                metrics[f'{n}gram_repetition_rate'] = repetition_rate
    
    return metrics


def interactive_generation(model: GenerativeNanoEncoder, vocab, device='cuda'):
    """Interactive generation session"""
    print("Interactive Generation Session")
    print("Enter medical event codes separated by spaces (or 'quit' to exit)")
    
    model_dtype = next(model.parameters()).dtype
    
    while True:
        user_input = input("\nEnter events: ").strip()
        if user_input.lower() == 'quit':
            break
        
        try:
            # Parse user input
            event_codes = [int(x) for x in user_input.split()]
            if not event_codes:
                continue
            
            # Create a simple batch
            batch_size = 1
            seq_len = len(event_codes)
            
            batch = {
                'event': torch.tensor(event_codes, dtype=torch.long).unsqueeze(0).to(device),
                'abspos': torch.randn(batch_size, seq_len, dtype=model_dtype).to(device),
                'age': torch.randn(batch_size, seq_len, dtype=model_dtype).to(device),
                'segment': torch.zeros(batch_size, seq_len, dtype=torch.long).to(device),
            }
            
            # Get generation parameters from user
            temp = float(input("Temperature (0.1-2.0, default 1.0): ") or "1.0")
            max_tokens = int(input("Max new tokens (default 20): ") or "20")
            top_k = input("Top-k (default None): ")
            top_k = int(top_k) if top_k else None
            top_p = input("Top-p (default None): ")
            top_p = float(top_p) if top_p else None
            
            # Generate
            generated = model.generate(
                batch,
                temperature=temp,
                max_new_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p
            )
            
            # Show results
            result = {
                'generated_events': generated['generated_events'].cpu(),
                'generation_lengths': generated['generation_lengths'].cpu(),
                'metrics': compute_generation_metrics(generated, vocab_size=len(vocab)),
                'prompt_length': generated['prompt_length']
            }
            
            print_generation_sample(result, sample_idx=0)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()


def scenario_analysis(
    model: GenerativeNanoEncoder, 
    dataloader, 
    num_batches=3, 
    num_scenarios=5
):
    """
    Generate multiple scenarios for each input in the batch.
    
    Args:
        model: GenerativeNanoEncoder model
        dataloader: PyTorch DataLoader
        num_batches: Number of batches to process
        num_scenarios: Number of scenarios to generate per input
        
    Returns:
        List of scenario results
    """
    print("Scenario Analysis: Generating continuations for patient trajectories")
    
    scenarios = []
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Generate multiple continuations for the same prompt
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"Analyzing batch {batch_idx}...")
        
        # Move to device and fix dtypes
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.dtype.is_floating_point:
                    batch[key] = value.to(device=device, dtype=model_dtype)
                else:
                    batch[key] = value.to(device=device)
        
        try:
            # Generate multiple scenarios using the generate method
            scenario_results = []
            for scenario_idx in range(num_scenarios):
                generated = model.generate(
                    batch,
                    temperature=1.0,
                    top_p=0.9,
                    max_new_tokens=25
                )
                scenario_results.append(generated)
            
            scenarios.append({
                'batch_idx': batch_idx,
                'scenarios': scenario_results
            })
            
            # Print example scenarios for first patient in batch
            print(f"\nBatch {batch_idx} scenarios:")
            if 'event' in batch:
                original = batch['event'][0].cpu()
                print(f"Original: {original.tolist()[:10]}...")  # Show first 10 events
            
            for scenario_idx, scenario in enumerate(scenario_results):
                scenario_seq = scenario['generated_events'][0].cpu()
                prompt_length = scenario['prompt_length']
                new_events = scenario_seq[prompt_length:]
                print(f"Scenario {scenario_idx}: {new_events.tolist()}")
                
        except Exception as e:
            print(f"Scenario analysis failed for batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return scenarios