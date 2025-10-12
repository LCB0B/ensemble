#!/usr/bin/env python3
"""
Manual time-token generation (mirrors claude_function/test_generation.py)
"""

import re
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn.functional import softmax

from src.generation_utils import (
    load_time_token_model,
    setup_time_token_datamodule,
    get_sequences_from_dataloader,
    load_vocab,
    find_start_position,
)
from src.paths import FPATH

RUN = "037_hurt_pelican-pretrain-lr0.05"
NUM_PEOPLE = 5
NUM_GENERATIONS = 5
START_CONDITION = {"type": "year", "value": 2000}
STOP_TOKENS = 500
TEMPERATURE = 1.0
TOP_P = 0.9
DEVICE = "cuda"

OUTPUT_DIR = Path("figures") / RUN
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ckpt_dir = FPATH.CHECKPOINTS_TRANSFORMER / "destiny" / RUN
logs_dir = FPATH.TB_LOGS / "destiny" / RUN
hparams_path = logs_dir / "hparams.yaml"
if not hparams_path.exists():
    raise FileNotFoundError(f"Missing hparams: {hparams_path}")

ckpt_path = next((ckpt_dir / name for name in ("best.ckpt", "last.ckpt") if (ckpt_dir / name).exists()), None)
if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")

with open(hparams_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
vocab_path = FPATH.DATA / cfg["dir_path"] / "vocab.json"
id2tok = load_vocab(str(vocab_path))

model, hparams = load_time_token_model(str(ckpt_path), str(hparams_path), vocab_path=str(vocab_path))
model = model.to(DEVICE).eval()
datamodule, _ = setup_time_token_datamodule(hparams)

sequences = get_sequences_from_dataloader(datamodule, NUM_PEOPLE, DEVICE)

def apply_top_p(logits: torch.Tensor) -> torch.Tensor:
    if TOP_P >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    mask = cum_probs > TOP_P
    mask[..., 0] = False
    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(-1, sorted_idx, mask)
    logits = logits.masked_fill(remove_mask, float("-inf"))
    return logits

results = []

for seq_data in sequences:
    person_id = seq_data["synthetic_person_id"]
    sequence = seq_data["event"][0].tolist()
    valid_len = (seq_data["event"][0] != 0).sum().item()
    sequence = sequence[:valid_len]

    start_pos = find_start_position(sequence, id2tok, START_CONDITION)
    if start_pos is None:
        print(f"Start condition not found for person {person_id}")
        continue

    prompt_tokens = sequence[: start_pos + 1]
    prompt_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    prompt_segments = seq_data["segment"][:, : prompt_ids.shape[1]].to(DEVICE)
    prompt_mask = torch.ones_like(prompt_ids, dtype=torch.float32)

    generations = []

    for gen_idx in range(NUM_GENERATIONS):
        torch.manual_seed(42 + gen_idx + person_id)

        event = prompt_ids.clone()
        segment = prompt_segments.clone()
        attn_mask = prompt_mask.clone()

        generated = []
        seg_tail = segment[:, -1].item()

        for step in range(STOP_TOKENS):
            cur_batch = {
                "event": event,
                "segment": segment,
                "attn_mask": attn_mask,
            }

            with torch.no_grad():
                hidden = model.forward_generation(cur_batch, use_autocast=False)
                logits = model.decoder(hidden[:, -1]).float()
            logits = apply_top_p(logits / TEMPERATURE)
            probs = softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(int(next_token.item()))
            seg_tail += 1

            new_event = next_token.long()
            new_segment = torch.tensor([[seg_tail]], device=DEVICE, dtype=torch.long)
            new_mask = torch.ones_like(new_event, dtype=torch.float32)

            event = torch.cat([event, new_event], dim=1)
            segment = torch.cat([segment, new_segment], dim=1)
            attn_mask = torch.cat([attn_mask, new_mask], dim=1)

        generations.append(generated)

    results.append(
        {
            "person_id": person_id,
            "full_sequence": sequence,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generations,
        }
    )

def plot_time_tokens(person_result):
    person_id = person_result["person_id"]
    prompt = person_result["prompt_tokens"]
    real_seq = person_result["full_sequence"]
    generated_list = person_result["generated_tokens"]

    pattern_age = re.compile(r"^AGE_(\d+)$")
    pattern_year = re.compile(r"^YEAR_(\d+)$")

    def extract(xs):
        idx_age, val_age, idx_year, val_year = [], [], [], []
        for idx, tid in enumerate(xs):
            token = id2tok.get(int(tid), "")
            if m := pattern_age.match(token):
                idx_age.append(idx)
                val_age.append(int(m.group(1)))
            if m := pattern_year.match(token):
                idx_year.append(idx)
                val_year.append(int(m.group(1)))
        return idx_age, val_age, idx_year, val_year

    real_age_idx, real_age_val, real_year_idx, real_year_val = extract(real_seq)

    fig_age, ax_age = plt.subplots(figsize=(12, 6))
    fig_year, ax_year = plt.subplots(figsize=(12, 6))

    if len(real_age_idx) > 1:
        ax_age.plot(real_age_idx, real_age_val, label="Real", color="blue")
    if len(real_year_idx) > 1:
        ax_year.plot(real_year_idx, real_year_val, label="Real", color="blue")

    colors = ["red", "orange", "green", "purple", "brown"]

    for idx, gen_tokens in enumerate(generated_list):
        full_seq = prompt + gen_tokens
        g_age_idx, g_age_val, g_year_idx, g_year_val = extract(full_seq)
        color = colors[idx % len(colors)]
        if len(g_age_idx) > 1:
            ax_age.plot(g_age_idx, g_age_val, color=color, alpha=0.35, label=f"Gen {idx+1}" if idx == 0 else None)
        if len(g_year_idx) > 1:
            ax_year.plot(g_year_idx, g_year_val, color=color, alpha=0.35, label=f"Gen {idx+1}" if idx == 0 else None)

    prompt_len = len(prompt)
    for ax in (ax_age, ax_year):
        ax.axvline(prompt_len, linestyle=":", color="gray")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xlabel("Token index")
    ax_age.set_ylabel("Years since birth")
    ax_age.set_title(f"AGE Token Positions - Person {person_id}")
    ax_year.set_ylabel("Calendar year")
    ax_year.set_title(f"YEAR Token Positions - Person {person_id}")

    fig_age.tight_layout()
    fig_year.tight_layout()

    fig_age.savefig(OUTPUT_DIR / f"time_tokens_person_{person_id}_age.png", dpi=300)
    fig_year.savefig(OUTPUT_DIR / f"time_tokens_person_{person_id}_year.png", dpi=300)
    plt.close(fig_age)
    plt.close(fig_year)

for res in results:
    plot_time_tokens(res)

print("Processing complete!")