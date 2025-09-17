#!/usr/bin/env python3
"""
Simplified ensemble generation script (ensemble mode only).
Defaults:
  - top_p = 0.99
  - torch.compile enabled by default (disable with --no_compile)
  - ensemble mode only (one code path)
"""

import argparse
import yaml
import torch
import polars as pl
import pyarrow.dataset as ds
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from lightning.pytorch import seed_everything

from src.paths import FPATH, check_and_copy_file_or_dir
from src.datamodule2 import PretrainDataModule
from src.encoder_nano_risk import GenerativeNanoEncoder
from src.generation_utils import load_vocab, decode_ids


def load_model_and_data(model_path: Path, hparams_path: Path, use_compiled_dataset: bool = True):
    if not hparams_path.is_file():
        raise FileNotFoundError(f"Hparams file not found: {hparams_path}")
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)

    print(f"Loaded hparams: {hparams_path}")
    print(f"Model config: d_model={hparams.get('d_model')} layers={hparams.get('num_layers')}")

    data_dir = hparams.get("source_dir", "destiny_dataset")
    background_name = hparams.get("background", "destiny/background")
    background_path = (FPATH.DATA / data_dir / background_name).with_suffix(".parquet")
    check_and_copy_file_or_dir(background_path, verbosity=1)
    background = pl.read_parquet(background_path)

    if use_compiled_dataset:
        sources = []
        print("Using pre-compiled dataset directory")
    else:
        sources_list = hparams.get("sources", [])
        sources = []
        for s in sources_list:
            p = (FPATH.DATA / data_dir / s).with_suffix(".parquet")
            check_and_copy_file_or_dir(p, verbosity=1)
            sources.append(ds.dataset(p, format="parquet"))
        print(f"Loaded {len(sources)} raw source parquet files")

    dm = PretrainDataModule(
        dir_path=FPATH.DATA / hparams.get("dir_path", data_dir),
        sources=sources,
        background=background,
        subset_background=hparams.get("subset_background", False),
        n_tokens=hparams.get("n_tokens", 8e5),
        lengths=hparams.get("lengths", "lengths"),
        num_workers=0,
        max_seq_len=hparams.get("max_seq_len", 2048),
        source_dir=data_dir,
        pretrain_style=hparams.get("pretrain_style", "AR"),
        masking_ratio=hparams.get("masking_ratio"),
    )
    dm.prepare_data()
    dm.setup()
    hparams["vocab_size"] = len(dm.pipeline.vocab)
    print(f"Vocab size: {hparams['vocab_size']}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    print(f"Loading model: {model_path}")
    model = GenerativeNanoEncoder(**hparams)
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    precision = hparams.get("precision", "32-true")
    if "16" in str(precision):
        model._mixed_precision = True
        model.half()
    else:
        model._mixed_precision = False
    print(f"Device: {device} dtype: {next(model.parameters()).dtype} mixed_precision={model._mixed_precision}")
    return model, dm, hparams


def check_sequence_length(single_person_batch, prompt_length, person_id, verbose=False):
    if "event" not in single_person_batch:
        return False
    events = single_person_batch["event"][0]
    actual = (events != 0).sum().item()
    if actual < prompt_length:
        if verbose:
            print(f"  {person_id} too short ({actual} < {prompt_length})")
        return False
    return True


# ...existing code above...

def extract_source_pid(sample):
    """
    Extract original person id from a single-person batch dict if present.
    Supports:
      - sample['person_id'] tensor (shape [1] or scalar)
      - int / str
    Returns str or None.
    """
    v = sample.get("person_id")
    if v is None:
        return None
    if torch.is_tensor(v):
        if v.numel() == 1:
            return str(int(v.flatten()[0].item()))
        return ",".join(str(int(x)) for x in v.flatten().tolist())
    return str(v)


def generate_ensemble(model, dm, args):
    print("\n=== Ensemble Generation ===")
    print(f"People: {args.num_people}  Simulations/person: {args.num_simulations}")
    print(f"Prompt length: {args.prompt_length}  Max new: {args.max_new_tokens}")
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"GPU memory free {free_mem/1024**2:.0f}MiB / {total_mem/1024**2:.0f}MiB")

    dataloader = dm.val_dataloader()
    vocab_size = len(dm.pipeline.vocab)

    meta = {
        "num_people_requested": args.num_people,
        "num_simulations": args.num_simulations,
        "prompt_length": args.prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "generation_timestamp": datetime.now().isoformat(),
        "model_path": str(args.model_path),
        "hparams_path": str(args.hparams_path),
        "vocab_size": vocab_size,
        "flat_sim": getattr(args, "flat_sim", False),
    }

    generation_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "use_cache": args.use_cache,
        "profile": args.profile,
    }

    device = next(model.parameters()).device

    total_prompt_tokens = 0
    total_gen_tokens = 0
    total_ms = 0.0
    gen_calls = 0

    # -------------------- FLAT MODE --------------------
    if getattr(args, "flat_sim", False):
        person_ids = []          # synthetic ids
        source_person_ids = []   # original ids (if available)
        prompt_lengths = []
        skipped = []
        valid_people = []        # list of (synthetic_pid, source_pid, single_batch)
        data_iter = iter(dataloader)

        while len(valid_people) < args.num_people:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            if "attn_mask" not in batch:
                batch["attn_mask"] = (batch["event"] != 0)
            bsz = batch["event"].shape[0]
            for i in range(bsz):
                pid = f"person_{len(valid_people):06d}"
                single = {k: (v[i:i+1] if torch.is_tensor(v) and v.dim() >= 2 else v)
                          for k, v in batch.items()}
                if not check_sequence_length(single, args.prompt_length, pid, args.verbose):
                    skipped.append({"person_id": pid, "reason": "too_short"})
                    continue
                for k, v in single.items():
                    if torch.is_tensor(v) and v.dim() == 2:
                        single[k] = v[:, :args.prompt_length]
                single["attn_mask"] = (single["event"] != 0)
                src_pid = extract_source_pid(single)
                valid_people.append((pid, src_pid, single))
                if len(valid_people) >= args.num_people:
                    break

        collected_people = len(valid_people)
        if collected_people == 0:
            raise RuntimeError("No valid people gathered for flat_sim mode.")
        if collected_people < args.num_people:
            print(f"[WARN] Only collected {collected_people} valid people (requested {args.num_people}).")
        meta["num_people_collected"] = collected_people

        flat_chunks = []
        flat_pids = []
        flat_src_pids = []
        for pid, src_pid, single in valid_people:
            rep = {}
            for k, v in single.items():
                if torch.is_tensor(v) and v.dim() == 2:  # (1,T)
                    rep[k] = v.repeat(args.num_simulations, 1)
                else:
                    rep[k] = v
            if src_pid is not None:
                # ensure a person_id tensor is propagated (keep original if already there)
                if "person_id" not in rep:
                    if src_pid.isdigit():
                        rep["person_id"] = torch.tensor([int(src_pid)], device=device)
                    else:
                        # keep as string (will be ignored by model)
                        rep["person_id"] = src_pid
            flat_chunks.append(rep)
            flat_pids.append(pid)
            flat_src_pids.append(src_pid)

        merged = {}
        for k in flat_chunks[0]:
            if torch.is_tensor(flat_chunks[0][k]) and flat_chunks[0][k].dim() == 2:
                merged[k] = torch.cat([c[k] for c in flat_chunks], 0)
            else:
                merged[k] = flat_chunks[0][k]

        with torch.no_grad():
            out = model.generate(merged, **generation_cfg)

        if args.profile:
            prof = out.get("profile")
            if prof:
                total_prompt_tokens += prof.get("prompt_tokens", 0)
                total_gen_tokens += prof.get("gen_tokens", 0)
                total_ms += prof.get("total_ms", 0.0)
                gen_calls += 1

        full = out["full"]
        gen = out["generated"]
        gen_lens = out["generation_lengths"]
        Bflat = full.size(0)
        expected = collected_people * args.num_simulations
        if Bflat != expected:
            raise RuntimeError(f"Flat batch size mismatch: got {Bflat}, expected {expected}.")

        max_seq_len = args.prompt_length + args.max_new_tokens
        all_full = torch.zeros(collected_people, args.num_simulations, max_seq_len, dtype=torch.long)
        all_gen = torch.zeros(collected_people, args.num_simulations, args.max_new_tokens, dtype=torch.long)

        gen_len_actual = gen.size(1)
        for p_idx in range(collected_people):
            person_ids.append(flat_pids[p_idx])
            source_person_ids.append(flat_src_pids[p_idx])
            prompt_lengths.append(args.prompt_length)
            start = p_idx * args.num_simulations
            end = start + args.num_simulations
            full_slice = full[start:end, :args.prompt_length + gen_len_actual]
            gen_slice = gen[start:end]
            all_full[p_idx, :, :full_slice.size(1)] = full_slice.cpu()
            all_gen[p_idx, :, :gen_slice.size(1)] = gen_slice.cpu()

        expose_ids = source_person_ids if any(sp is not None for sp in source_person_ids) else person_ids
        ensemble = {
            "person_ids": expose_ids,
            "generated_person_ids": person_ids,
            "source_person_ids": source_person_ids,
            "prompt_lengths": prompt_lengths,
            "generated_sequences": all_full,
            "generated_events": all_gen,
            "skipped_people": skipped,
            "metadata": meta,
        }
        if args.profile and total_ms > 0 and total_gen_tokens > 0:
            total_tps = (total_prompt_tokens + total_gen_tokens) / (total_ms / 1000.0)
            gen_tps = total_gen_tokens / (total_ms / 1000.0)
            ensemble["profile"] = {
                "gen_calls": gen_calls,
                "prompt_tokens": total_prompt_tokens,
                "gen_tokens": total_gen_tokens,
                "wall_ms": total_ms,
                "total_toks_per_sec": total_tps,
                "gen_toks_per_sec": gen_tps,
            }
            print(f"[AGG SPEED] total_toks/s={total_tps:.1f} gen_toks/s={gen_tps:.1f}")
        print(f"\nSummary(flat): collected={collected_people} skipped={len(skipped)}")
        return ensemble

    # -------------------- BATCHED MODE --------------------
    collected = 0
    processed = 0
    skipped = []
    person_ids = []
    source_person_ids = []
    prompt_lengths = []

    max_seq_len = args.prompt_length + args.max_new_tokens
    all_full = torch.zeros(args.num_people, args.num_simulations, max_seq_len, dtype=torch.long)
    all_gen = torch.zeros(args.num_people, args.num_simulations, args.max_new_tokens, dtype=torch.long)

    pbar_people = tqdm(total=args.num_people, desc="People", unit="p")
    people_buffers = []
    data_iter = iter(dataloader)

    def fetch_next_people():
        nonlocal processed
        while len(people_buffers) < args.people_per_batch and len(person_ids) + len(people_buffers) < args.num_people:
            try:
                batch = next(data_iter)
            except StopIteration:
                return False
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            if "attn_mask" not in batch:
                batch["attn_mask"] = (batch["event"] != 0)
            bsz = batch["event"].shape[0]
            for i in range(bsz):
                if len(people_buffers) + len(person_ids) >= args.num_people:
                    break
                pid = f"person_{processed:06d}"
                processed += 1
                single = {k: (v[i:i+1] if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in batch.items()}
                if not check_sequence_length(single, args.prompt_length, pid, args.verbose):
                    skipped.append({"person_id": pid, "reason": "too_short"})
                    continue
                for k, v in single.items():
                    if torch.is_tensor(v) and v.dim() == 2:
                        single[k] = v[:, :args.prompt_length]
                single["attn_mask"] = (single["event"] != 0)
                src_pid = extract_source_pid(single)
                people_buffers.append((pid, src_pid, single))
                if len(people_buffers) >= args.people_per_batch:
                    break
        return len(people_buffers) > 0

    while collected < args.num_people:
        if not fetch_next_people():
            break

        if args.auto_sim_batch:
            if not torch.cuda.is_available():
                sims_per_person = min(args.sim_batch_size, args.num_simulations)
            else:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                target_bytes = int(total_mem * args.memory_target_pct)

                def build(sp):
                    chunks = []
                    for _, _, single in people_buffers:
                        multi = {k: (v.repeat(sp, 1) if torch.is_tensor(v) and v.dim() == 2 else v)
                                 for k, v in single.items()}
                        chunks.append(multi)
                    merged_local = {}
                    for k in chunks[0]:
                        if torch.is_tensor(chunks[0][k]) and chunks[0][k].dim() == 2:
                            merged_local[k] = torch.cat([c[k] for c in chunks], 0)
                        else:
                            merged_local[k] = chunks[0][k]
                    return merged_local

                def mem_delta(sp):
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        mb = build(sp)
                        start = torch.cuda.memory_allocated()
                        with torch.no_grad():
                            model.generate(mb, **generation_cfg)
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated()
                        return max(0, peak - start)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            return -1
                        raise

                base = mem_delta(1)
                if base <= 0:
                    sims_per_person = min(args.sim_batch_size, args.num_simulations)
                else:
                    per_sim = base
                    people_batch = len(people_buffers)
                    est = int(target_bytes / (per_sim * max(1, people_batch)))
                    sims_per_person = max(1, min(args.num_simulations, est))
                    for cand in (512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2):
                        if sims_per_person >= cand:
                            sims_per_person = cand
                            break
                    while sims_per_person > 1 and mem_delta(sims_per_person) < 0:
                        sims_per_person //= 2
                print(f"[AUTO] sims_per_person={sims_per_person} people_batch={len(people_buffers)} "
                      f"global_batch={sims_per_person * len(people_buffers)}")
        else:
            sims_per_person = min(args.sim_batch_size, args.num_simulations)

        progress = {pid: 0 for pid, _, _ in people_buffers}
        while True:
            remaining = {pid: args.num_simulations - done for pid, done in progress.items()}
            if all(r == 0 for r in remaining.values()):
                break
            take_pp = min(sims_per_person, max(remaining.values()))
            chunks = []
            offsets = []
            for pid, src_pid, single in people_buffers:
                need = remaining[pid]
                if need <= 0:
                    continue
                take = min(take_pp, need)
                multi = {k: (v.repeat(take, 1) if torch.is_tensor(v) and v.dim() == 2 else v)
                         for k, v in single.items()}
                # ensure original person_id propagated
                if src_pid is not None and "person_id" not in multi:
                    if src_pid.isdigit():
                        multi["person_id"] = torch.tensor([int(src_pid)], device=device)
                    else:
                        multi["person_id"] = src_pid
                chunks.append(multi)
                offsets.append((pid, take))
            merged = {}
            for k in chunks[0]:
                if torch.is_tensor(chunks[0][k]) and chunks[0][k].dim() == 2:
                    merged[k] = torch.cat([c[k] for c in chunks], 0)
                else:
                    merged[k] = chunks[0][k]
            with torch.no_grad():
                out = model.generate(merged, **generation_cfg)

            if args.profile:
                prof = out.get("profile")
                if prof:
                    total_prompt_tokens += prof.get("prompt_tokens", 0)
                    total_gen_tokens += prof.get("gen_tokens", 0)
                    total_ms += prof.get("total_ms", 0.0)
                    gen_calls += 1

            full = out.get("full")
            gen = out.get("generated")
            gen_lens = out.get("generation_lengths")
            if full is None or gen is None or gen_lens is None:
                raise RuntimeError("model.generate missing keys")

            cursor = 0
            for pid, take in offsets:
                if pid not in person_ids:
                    # record synthetic id; source id appended later
                    person_ids.append(pid)
                    prompt_lengths.append(args.prompt_length)
                pidx = person_ids.index(pid)
                for j in range(take):
                    if progress[pid] + j >= args.num_simulations:
                        break
                    full_seq = full[cursor + j]
                    gen_seq = gen[cursor + j]
                    gen_len = gen_lens[cursor + j].item()
                    dest = progress[pid] + j
                    padded_full = torch.zeros(max_seq_len, dtype=torch.long, device=full_seq.device)
                    store_len = min(full_seq.shape[0], max_seq_len)
                    padded_full[:store_len] = full_seq[:store_len]
                    padded_gen = torch.zeros(args.max_new_tokens, dtype=torch.long, device=gen_seq.device)
                    g_store = min(gen_len, args.max_new_tokens, gen_seq.shape[0])
                    padded_gen[:g_store] = gen_seq[:g_store]
                    all_full[pidx, dest] = padded_full.cpu()
                    all_gen[pidx, dest] = padded_gen.cpu()
                progress[pid] += take
                cursor += take

        newly = 0
        # append source ids in same order
        for pid, src_pid, _ in people_buffers:
            idx = person_ids.index(pid)
            if idx >= collected:
                newly += 1
                source_person_ids.append(src_pid)
        collected = len(person_ids)
        pbar_people.update(newly)
        pbar_people.set_postfix(collected=collected, skipped=len(skipped))
        people_buffers = []
        if collected >= args.num_people:
            break

    pbar_people.close()
    expose_ids = source_person_ids if any(sp is not None for sp in source_person_ids) else person_ids
    ensemble = {
        "person_ids": expose_ids[:args.num_people],
        "generated_person_ids": person_ids[:args.num_people],
        "source_person_ids": source_person_ids[:args.num_people],
        "prompt_lengths": prompt_lengths[:args.num_people],
        "generated_sequences": all_full[:args.num_people],
        "generated_events": all_gen[:args.num_people],
        "skipped_people": skipped,
        "metadata": meta,
    }
    if args.profile and total_ms > 0 and total_gen_tokens > 0:
        total_tps = (total_prompt_tokens + total_gen_tokens) / (total_ms / 1000.0)
        gen_tps = total_gen_tokens / (total_ms / 1000.0)
        ensemble["profile"] = {
            "gen_calls": gen_calls,
            "prompt_tokens": total_prompt_tokens,
            "gen_tokens": total_gen_tokens,
            "wall_ms": total_ms,
            "total_toks_per_sec": total_tps,
            "gen_toks_per_sec": gen_tps,
        }
        print(f"[AGG SPEED] total_toks/s={total_tps:.1f} gen_toks/s={gen_tps:.1f}")
    print(f"\nSummary: collected={collected} processed={processed} skipped={len(skipped)}")
    return ensemble


def save_ensemble(ensemble, folder: str):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = out_dir / f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ts_dir.mkdir(exist_ok=True)
    with open(ts_dir / "metadata.yaml", "w") as f:
        yaml.dump(ensemble["metadata"], f)
    torch.save(
        {"full_sequences": ensemble["generated_sequences"],
         "generated_only": ensemble["generated_events"]},
        ts_dir / "sequences.pt",
    )
    torch.save(
        {
            "person_ids": ensemble["person_ids"],
            "generated_person_ids": ensemble.get("generated_person_ids"),
            "source_person_ids": ensemble.get("source_person_ids"),
            "prompt_lengths": ensemble["prompt_lengths"],
        },
        ts_dir / "person_data.pt",
    )
    print(f"Saved ensemble to {ts_dir}")
    return ts_dir


def main():
    parser = argparse.ArgumentParser(description="Ensemble sequence generation (only ensemble mode).")
    parser.add_argument("--model_path", type=Path, default=FPATH.DEFAULT_MODEL)
    parser.add_argument("--hparams_path", type=Path, default=FPATH.DEFAULT_HPARAMS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_length", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--num_people", type=int, default=8)
    parser.add_argument("--num_simulations", type=int, default=32)
    parser.add_argument("--people_per_batch", type=int, default=4)
    parser.add_argument("--sim_batch_size", type=int, default=128)
    parser.add_argument("--auto_sim_batch", action="store_true")
    parser.add_argument("--memory_target_pct", type=float, default=0.10,
                        help="Target fraction of total GPU memory for auto batch sizing.")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (enabled by default).")
    parser.add_argument("--flat_sim", action="store_true",
                            help="Generate all simulations in a single flat batch (people*num_simulations).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--decode_tokens", action="store_true")
    parser.add_argument("--vocab_path", type=Path, default=FPATH.DATA / 'destiny_dataset' / 'vocab.json')
    parser.add_argument("--print_max_tokens", type=int, default=48)
    parser.add_argument("--save_folder", type=str, default="generated_ensemble")
    args = parser.parse_args()

    seed_everything(args.seed)

    vocab_map = {}
    if args.decode_tokens and args.vocab_path.is_file():
        vocab_map = load_vocab(str(args.vocab_path))
        print(f"[INFO] Loaded vocab size={len(vocab_map)}")

    model, dm, hparams = load_model_and_data(
        args.model_path, args.hparams_path, use_compiled_dataset=True
    )

    if not args.no_compile:
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="default", fullgraph=False)
            print("Model compiled.")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    ensemble = generate_ensemble(model, dm, args)
    save_ensemble(ensemble, args.save_folder)

    if args.decode_tokens and len(ensemble["generated_events"]) > 0:
        ids = ensemble["generated_events"][0, 0].tolist()
        trimmed = [i for i in ids if i != 0][:args.print_max_tokens]
        print("\nSample decoded (first person / first sim):")
        print("IDs:", trimmed)
        print("TOK:", decode_ids(trimmed, vocab_map))


if __name__ == "__main__":
    main()