#!/usr/bin/env python3
"""
Ensemble generation analysis.

Sources for real (reference) sequences (choose one):
  1) --dataset_dir : LMDB dataset; pulls only needed person_ids lazily
  2) --gold        : pre-built .pt file with {"person_ids","full_sequences"}

Metrics per person:
  - prompt_len, real_future_len
  - unique token counts (prompt, real future, sim union, novel)
  - coverage & Jaccard (sim union vs real future)
  - JS divergence (freq) sims vs real future
  - entropy of sim tokens
  - positional exact match (mean, std) over first K positions
  - majority vote positional match
  - unigram / bigram overlap
  - Levenshtein distance (mean/std, truncated)

Aggregate: mean / median for numeric fields.

Usage:
  python scripts/analyze_generation_stats.py ENSEMBLE_DIR --dataset_dir data/destiny_dataset
"""

import argparse, json, math, yaml
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from collections import Counter

# Lazy import of utils_data helpers
try:
    from src.utils_data import load_dataset, PersonIndex, fetch_by_person_ids
except ImportError:
    load_dataset = PersonIndex = fetch_by_person_ids = None  # dataset_dir mode disabled

PAD_TOKEN = 0


def load_ensemble(ensemble_dir: Path):
    """
    Load ensemble outputs.
    Chooses source_person_ids (original dataset ids) when present; else generated ids.
    """
    meta = yaml.safe_load(open(ensemble_dir / "metadata.yaml"))
    person_data = torch.load(ensemble_dir / "person_data.pt")
    seqs = torch.load(ensemble_dir / "sequences.pt")

    source = person_data.get("source_person_ids")
    generated = person_data.get("generated_person_ids")
    legacy = person_data.get("person_ids")

    if source and any(s is not None for s in source):
        base_generated = generated if generated is not None else legacy
        person_ids = [
            (s if s is not None else (g if g is not None else l))
            for s, g, l in zip(
                source,
                base_generated if base_generated is not None else [None]*len(source),
                legacy if legacy is not None else [None]*len(source),
            )
        ]
    else:
        person_ids = generated if generated is not None else legacy

    return {
        "meta": meta,
        "person_ids": person_ids,
        "prompt_lengths": person_data["prompt_lengths"],
        "full_sequences": seqs["full_sequences"],
    }

def load_real_from_dataset(dataset_dir: Path, person_ids, field="event"):
    if load_dataset is None:
        raise ImportError("utils_data not available for dataset load")
    ds = load_dataset(dataset_dir)
    pindex = PersonIndex.build(ds)
    kept, padded, lengths = fetch_by_person_ids(ds, person_ids, person_index=pindex, field=field, pad=True)
    # Map back; some might be missing
    pid_to_row = {pid: i for i, pid in enumerate(kept)}
    return pid_to_row, padded  # padded: [N,L]


def load_real_from_gold(gold_path: Path):
    obj = torch.load(gold_path)
    pid_to_row = {pid: i for i, pid in enumerate(obj["person_ids"])}
    return pid_to_row, obj["full_sequences"]  # [N,L]


def js_divergence(p_counts, q_counts):
    vocab = set(p_counts) | set(q_counts)
    if not vocab:
        return float("nan")
    p = np.array([p_counts.get(t, 0) for t in vocab], dtype=float)
    q = np.array([q_counts.get(t, 0) for t in vocab], dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return (a[mask] * np.log(a[mask] / b[mask])).sum()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def entropy(counts):
    total = sum(counts.values())
    if total == 0: return float("nan")
    probs = np.array(list(counts.values())) / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def lev_distance(a, b, max_len=None):
    if max_len is not None:
        a = a[:max_len]; b = b[:max_len]
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = np.arange(lb + 1)
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def ngram_set(seq, n):
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


def analyze_person(pidx, ens, real_map, real_tensor,
                   max_positions_eval=128, max_lev_len=128):
    person_id = ens["person_ids"][pidx]
    if person_id not in real_map:
        return None
    row = real_map[person_id]
    real_seq = real_tensor[row].tolist()
    # trim padding
    if PAD_TOKEN in real_seq:
        real_seq = real_seq[:real_seq.index(PAD_TOKEN)]
    prompt_len = ens["prompt_lengths"][pidx]
    real_future = real_seq[prompt_len:] if prompt_len < len(real_seq) else []
    sims = ens["full_sequences"][pidx]  # [S,L]
    S, L = sims.shape

    # Prompt tokens (same across sims)
    prompt_tokens = sims[0, :prompt_len].tolist()
    prompt_tokens = [t for t in prompt_tokens if t != PAD_TOKEN]
    prompt_set = set(prompt_tokens)

    real_future_set = {t for t in real_future if t != PAD_TOKEN}

    sim_union = set()
    sim_counts = Counter()
    novel_union = set()
    uniq_per_sim = []
    gen_lengths = []
    lev_dists = []
    pos_matches = []

    max_pos_eval = min(max_positions_eval, len(real_future))
    from collections import Counter as Cntr
    vote = [Cntr() for _ in range(max_pos_eval)]

    for s in range(S):
        seq = sims[s].tolist()
        if PAD_TOKEN in seq:
            seq = seq[:seq.index(PAD_TOKEN)]
        gen_part = seq[prompt_len:]
        gen_lengths.append(len(gen_part))
        tokens_clean = [t for t in gen_part if t != PAD_TOKEN]
        uset = set(tokens_clean)
        uniq_per_sim.append(len(uset))
        sim_union |= uset
        sim_counts.update(tokens_clean)
        novel_union |= (uset - prompt_set)
        if real_future:
            lev_dists.append(lev_distance(real_future, gen_part, max_len=max_lev_len))
            if max_pos_eval > 0:
                m = sum(1 for i in range(max_pos_eval)
                        if i < len(gen_part) and i < len(real_future) and gen_part[i] == real_future[i])
                pos_matches.append(m / max_pos_eval)
                for i in range(max_pos_eval):
                    if i < len(gen_part):
                        vote[i][gen_part[i]] += 1

    # Metrics
    coverage = (len(real_future_set & sim_union) / len(real_future_set)
                if real_future_set else float("nan"))
    jaccard = (len(sim_union & real_future_set) / len(sim_union | real_future_set)
               if (sim_union or real_future_set) else float("nan"))

    jsd = js_divergence(sim_counts, Counter([t for t in real_future if t != PAD_TOKEN]))
    ent = entropy(sim_counts)

    majority = float("nan")
    if real_future and max_pos_eval > 0:
        correct = 0; denom = 0
        for i in range(max_pos_eval):
            if i >= len(real_future): break
            if not vote[i]: continue
            if vote[i].most_common(1)[0][0] == real_future[i]:
                correct += 1
            denom += 1
        if denom > 0:
            majority = correct / denom

    unigram_overlap = (len(real_future_set & sim_union) / len(real_future_set)
                       if real_future_set else float("nan"))
    bigram_overlap = float("nan")
    if len(real_future) >= 2:
        rf_bi = ngram_set(real_future, 2)
        sim_bi = set()
        for s in range(S):
            seq = sims[s].tolist()
            if PAD_TOKEN in seq:
                seq = seq[:seq.index(PAD_TOKEN)]
            gen_part = seq[prompt_len:]
            if len(gen_part) >= 2:
                sim_bi |= ngram_set(gen_part, 2)
        if rf_bi:
            bigram_overlap = len(rf_bi & sim_bi) / len(rf_bi)

    return {
        "person_id": person_id,
        "prompt_len": prompt_len,
        "real_future_len": len(real_future),
        "num_sims": S,
        "gen_len_mean": float(np.mean(gen_lengths)) if gen_lengths else 0.0,
        "gen_len_min": int(min(gen_lengths)) if gen_lengths else 0,
        "gen_len_max": int(max(gen_lengths)) if gen_lengths else 0,
        "prompt_unique": len(prompt_set),
        "real_future_unique": len(real_future_set),
        "sim_union_unique": len(sim_union),
        "avg_sim_unique": float(np.mean(uniq_per_sim)) if uniq_per_sim else 0.0,
        "novel_unique": len(novel_union),
        "novel_ratio_vs_prompt": (len(novel_union) / max(1, len(prompt_set))) if prompt_set else float("nan"),
        "coverage_real_future": coverage,
        "jaccard_future": jaccard,
        "js_divergence": jsd,
        "sim_entropy": ent,
        "pos_match_mean": float(np.mean(pos_matches)) if pos_matches else float("nan"),
        "pos_match_std": float(np.std(pos_matches)) if pos_matches else float("nan"),
        "majority_vote_pos_match": majority,
        "unigram_overlap": unigram_overlap,
        "bigram_overlap": bigram_overlap,
        "levenshtein_mean": float(np.mean(lev_dists)) if lev_dists else float("nan"),
        "levenshtein_std": float(np.std(lev_dists)) if lev_dists else float("nan"),
    }


def aggregate(stats):
    if not stats: return {}
    out = {}
    numeric_keys = [k for k in stats[0] if k != "person_id"]
    for k in numeric_keys:
        vals = [s[k] for s in stats if isinstance(s[k], (int, float)) and not (isinstance(s[k], float) and math.isnan(s[k]))]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_median"] = float(np.median(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ensemble_dir")
    ap.add_argument("--dataset_dir", help="LMDB dataset directory (preferred)")
    ap.add_argument("--gold", help=".pt file with real sequences (fallback)")
    ap.add_argument("--field", default="event")
    ap.add_argument("--max_positions_eval", type=int, default=128)
    ap.add_argument("--max_lev_len", type=int, default=128)
    ap.add_argument("--out", default="ensemble_stats")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    if not args.dataset_dir and not args.gold:
        raise SystemExit("Provide --dataset_dir or --gold")

    ens = load_ensemble(Path(args.ensemble_dir))
    person_ids = ens["person_ids"]

    if args.dataset_dir:
        real_map, real_tensor = load_real_from_dataset(Path(args.dataset_dir), person_ids, field=args.field)
    else:
        real_map, real_tensor = load_real_from_gold(Path(args.gold))

    stats = []
    for pidx in range(len(person_ids)):
        s = analyze_person(
            pidx, ens, real_map, real_tensor,
            max_positions_eval=args.max_positions_eval,
            max_lev_len=args.max_lev_len
        )
        if s:
            stats.append(s)

    if not stats:
        print("No matching persons found in reference source.")
        return

    agg = aggregate(stats)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "per_person_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(out_dir / "aggregate_stats.json", "w") as f:
        json.dump(agg, f, indent=2)
    with open(out_dir / "aggregate_stats.yaml", "w") as f:
        yaml.safe_dump(agg, f)

    if args.csv:
        pd.DataFrame(stats).to_csv(out_dir / "per_person_stats.csv", index=False)

    print(f"Analyzed persons: {len(stats)}")
    for k, v in list(agg.items())[:12]:
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

# python scripts/analyze_generation_stats.py  generated_ensemble/ensemble_20250814_155842 --dataset_dir data/destiny_dataset --out ensemble_stats --csv
