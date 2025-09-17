#!/usr/bin/env python3
"""
Monte Carlo income trajectory plot:
- Many semi‑transparent lines (one per simulation) for a chosen person
- Prompt segment highlighted
- Optional mean + percentile band
"""

import json, re, argparse
from pathlib import Path
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

INCOME_PREFIX_DEFAULT = "LAB_perindkialt_13_Q"   # Adjust if you want another variable

def load_ensemble(ensemble_dir: Path):
    meta = yaml.safe_load(open(ensemble_dir / "metadata.yaml"))
    persons = torch.load(ensemble_dir / "person_data.pt")
    seqs = torch.load(ensemble_dir / "sequences.pt")
    return {
        "meta": meta,
        "person_ids": persons["person_ids"],
        "prompt_lengths": persons["prompt_lengths"],
        "full_sequences": seqs["full_sequences"],   # [P,S,L]
        "generated_only": seqs["generated_only"],
    }

def build_income_id_map(vocab_path: Path, prefix: str):
    vocab = json.load(open(vocab_path))
    id2q = {}
    pat = re.compile(re.escape(prefix) + r"(\d+)$")
    for token, idx in vocab.items():
        m = pat.search(token)
        if m:
            q = int(m.group(1))
            id2q[idx] = q
    if not id2q:
        print(f"[WARN] No tokens matched prefix '{prefix}'")
    return id2q

def extract_quantiles_per_sim(seqs_person, id2q):
    """
    seqs_person: (S,L) LongTensor
    Returns list length S; each is (positions, quantiles)
    """
    sims = []
    S, L = seqs_person.shape
    for s in range(S):
        seq = seqs_person[s].tolist()
        pos = []
        val = []
        for i,tok in enumerate(seq):
            if tok == 0:  # padding
                continue
            q = id2q.get(tok)
            if q is not None:
                pos.append(i)
                val.append(q)
        if pos:
            sims.append((np.array(pos), np.array(val)))
        else:
            sims.append((np.array([]), np.array([])))
    return sims

def densify_step(positions, values, L):
    """
    Convert sparse (pos,q) to length L step array (forward fill).
    Missing -> NaN until first observed.
    """
    arr = np.full(L, np.nan)
    if len(positions)==0: return arr
    last = np.nan
    j = 0
    for i in range(L):
        if j < len(positions) and i == positions[j]:
            last = values[j]
            j += 1
        arr[i] = last
    return arr

def plot_mc(person_idx, data, id2q, prefix, outdir, band=True, show_mean=True, alpha=0.12):
    person_id = data["person_ids"][person_idx]
    prompt_len = data["prompt_lengths"][person_idx]
    seqs_person = data["full_sequences"][person_idx]  # (S,L)
    S, L = seqs_person.shape

    sims_sparse = extract_quantiles_per_sim(seqs_person, id2q)
    # Densify
    dense = np.vstack([densify_step(p,v,L) for p,v in sims_sparse])  # (S,L)
    # Keep only sims that have at least one observed value
    mask_valid = ~np.isnan(dense).all(1)
    dense = dense[mask_valid]
    if dense.size == 0:
        print(f"[INFO] No {prefix} tokens for {person_id}")
        return
    # Stats over generated portion
    mean = np.nanmean(dense, axis=0)
    p05  = np.nanpercentile(dense, 5, axis=0)
    p95  = np.nanpercentile(dense, 95, axis=0)

    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12,6))

    # Plot each simulation
    for arr in dense:
        plt.plot(arr, color="black", alpha=alpha, linewidth=0.7)

    # Prompt highlight (use first valid sim forward-filled)
    prompt_ref = dense[0].copy()
    plt.plot(range(prompt_len), prompt_ref[:prompt_len], color="firebrick", linewidth=2, label="Prompt sims")

    # Mean & band
    if band:
        plt.fill_between(range(prompt_len, L), p05[prompt_len:], p95[prompt_len:],
                         color="cornflowerblue", alpha=0.25, label="Gen 5–95%")
    if show_mean:
        plt.plot(mean, color="blue", linewidth=2, label="Mean")

    plt.axvline(prompt_len, color="k", linestyle="--", linewidth=1.5, label=f"Gen start ({prompt_len})")
    plt.ylim(0, 101)
    plt.xlim(0, L)
    plt.xlabel("Event position")
    plt.ylabel("Income quantile (1–100)")
    plt.title(f"Income Monte Carlo Trajectories\nPerson {person_id}  Sims={dense.shape[0]}")
    plt.legend()
    plt.grid(alpha=0.25)

    fname = outdir / f"{person_id}_income_mc.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ensemble_dir")
    ap.add_argument("--vocab", default="data/destiny_dataset/vocab.json")
    ap.add_argument("--prefix", default=INCOME_PREFIX_DEFAULT)
    ap.add_argument("--outdir", default="figures/income_mc")
    ap.add_argument("--person_id", default=None, help="Specific person_id (e.g. person_000012)")
    ap.add_argument("--max_people", type=int, default=10)
    ap.add_argument("--no_band", action="store_true")
    ap.add_argument("--no_mean", action="store_true")
    args = ap.parse_args()

    ensemble_dir = Path(args.ensemble_dir)
    data = load_ensemble(ensemble_dir)
    id2q = build_income_id_map(Path(args.vocab), args.prefix)

    if not id2q:
        return

    if args.person_id:
        try:
            idx = data["person_ids"].index(args.person_id)
        except ValueError:
            print("Person not found.")
            return
        plot_mc(idx, data, id2q, args.prefix, Path(args.outdir),
                band=not args.no_band, show_mean=not args.no_mean)
    else:
        n = len(data["person_ids"])
        limit = min(n, args.max_people) if args.max_people else n
        for i in range(limit):
            plot_mc(i, data, id2q, args.prefix, Path(args.outdir),
                    band=not args.no_band, show_mean=not args.no_mean)

if __name__ == "__main__":
    main()


# python scripts/plot_mc.py generated_ensemble/ensemble_20250814_151546