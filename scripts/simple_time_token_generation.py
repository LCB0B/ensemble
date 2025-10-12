#!/usr/bin/env python3
"""
Simple Time Token Generation Script

Direct generation using model.generate() for clarity and flexibility.
No hidden wrapper functions - all generation logic visible in script.

Usage: python scripts/simple_time_token_generation.py
"""

import re
import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from src.generation_utils import (
    load_time_token_model,
    setup_time_token_datamodule,
    get_sequences_from_dataloader,
    load_vocab,
    find_start_position,
)
from src.paths import FPATH

#checkpoints/transformer/destiny/032_victorious_rhino-pretrain-lr0.1
#checkpoints/transformer/destiny/033_delightful_ocelot-pretrain-lr0.01
#checkpoints/transformer/destiny/035_obedient_leopard-pretrain-lr0.01
# ——— Configuration ———
RUN             = "037_hurt_pelican-pretrain-lr0.05"
NUM_PEOPLE      = 5
NUM_GENERATIONS = 10
START_CONDITION = {"type":"year", "value":2000}
STOP_CONDITION  = {"type":"tokens", "value":100}
TEMPERATURE     = 1.0
TOP_P           = 0.9
DEVICE          = "cuda"
# ————————————————

print(f"Loading experiment: {RUN}")

# Create output directory
OUTPUT_DIR = Path("figures") / RUN
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Load model checkpoint & hparams
ckpt_dir = FPATH.CHECKPOINTS_TRANSFORMER / "destiny" / RUN
logs_dir = FPATH.TB_LOGS / "destiny" / RUN
hparams_path = logs_dir / "hparams.yaml"

if not logs_dir.exists() or not hparams_path.exists():
    raise FileNotFoundError(f"Missing logs or hparams in {logs_dir}")

ckpt_path = None
for name in ("best.ckpt", "last.ckpt"):
    candidate = ckpt_dir / name
    if candidate.exists():
        ckpt_path = candidate
        break

if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

print(f"Checkpoint: {ckpt_path}")
print(f"Config: {hparams_path}")

# Load vocab (id→token) - do this BEFORE loading model
with open(hparams_path) as f:
    hparams_config = yaml.safe_load(f)
vocab_path = FPATH.DATA / hparams_config["dir_path"] / "vocab.json"
id2tok = load_vocab(str(vocab_path))
print(f"Vocabulary size: {len(id2tok)}")

# Load model + datamodule (pass vocab_path to enable masking)
model, hparams = load_time_token_model(str(ckpt_path), str(hparams_path), vocab_path=str(vocab_path))
dm, _ = setup_time_token_datamodule(hparams)

# Move model to device and set to eval mode
model = model.to(DEVICE).eval()
print(f"Model loaded on {DEVICE}")

# Extract sequences from dataloader
print(f"\nExtracting {NUM_PEOPLE} sequences from dataset...")
sequences = get_sequences_from_dataloader(dm, NUM_PEOPLE, DEVICE)
print(f"Found {len(sequences)} sequences")

# Generate sequences for each person
print(f"\nGenerating sequences:")
print(f"  People: {NUM_PEOPLE}")
print(f"  Generations per person: {NUM_GENERATIONS}")
print(f"  Start condition: {START_CONDITION}")
print(f"  Stop condition: {STOP_CONDITION}")

results = []

for seq_data in sequences:
    person_id = seq_data["synthetic_person_id"]

    try:
        # Extract sequence and remove padding
        sequence = seq_data['event'][0].cpu().tolist()
        actual_length = (seq_data['event'][0] != 0).sum().item()
        sequence = sequence[:actual_length]

        # Find start position using helper
        start_pos = find_start_position(sequence, id2tok, START_CONDITION)

        if start_pos is None:
            print(f"Start condition not found for person {person_id}")
            continue

        # Create prompt from start position
        prompt_sequence = sequence[:start_pos + 1]

        # Create base prompt batch (template to clone from)
        prompt_batch_base = {
            'event': torch.tensor([prompt_sequence], dtype=torch.long).to(DEVICE),
            'segment': seq_data['segment'][:, :len(prompt_sequence)].to(DEVICE),
            'attn_mask': torch.ones(1, len(prompt_sequence), dtype=torch.float32).to(DEVICE)  # IMPORTANT: float32 for attention
        }

        # Generate multiple sequences directly with model.generate()
        generated_sequences = []
        generation_lengths = []

        for gen_idx in range(NUM_GENERATIONS):
            torch.manual_seed(42 + gen_idx + person_id)

            prompt_ids = prompt_batch_base["event"].clone()
            prompt_segments = prompt_batch_base["segment"].clone()
            attn_mask = prompt_batch_base["attn_mask"].clone()

            generated = []
            segments = []

            for step in range(STOP_CONDITION["value"]):
                new_tokens = torch.tensor(generated, dtype=torch.long, device=DEVICE).unsqueeze(0)
                new_segments = torch.tensor(segments, dtype=torch.long, device=DEVICE).unsqueeze(0)

                cur_event = torch.cat([prompt_ids, new_tokens], dim=1)
                cur_segment = torch.cat([prompt_segments, new_segments], dim=1)
                cur_mask = torch.ones_like(cur_event, dtype=torch.float32, device=DEVICE)

                batch = {
                    "event": cur_event,
                    "segment": cur_segment,
                    "attn_mask": cur_mask,
                }

                with torch.no_grad():
                    logits = model.get_next_token_logits(batch, use_autocast=False)

                # Apply temperature
                logits = logits / TEMPERATURE

                # Apply top-p (nucleus) filtering
                if TOP_P < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > TOP_P
                    sorted_indices_to_remove[..., 0] = False  # Keep at least the top token

                    # Create removal mask and scatter back to original order
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample from filtered distribution
                probs = logits.softmax(-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1).item()

                generated.append(next_token)
                segments.append(
                    segments[-1] + 1 if segments else prompt_segments[0, -1].item() + 1
                )

            generated_sequences.append(generated)
            generation_lengths.append(len(generated))

        # Package results
        result = {
            "person_id": person_id,
            "full_sequence": sequence,
            "prompt_tokens": prompt_sequence,
            "generated_tokens": generated_sequences,
            "generation_lengths": generation_lengths,
        }

        results.append(result)
        print(f"Generated {NUM_GENERATIONS} sequences for person {person_id}")

    except Exception as e:
        print(f"Error generating for person {person_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\nGenerated {len(results)} total results")

# Plot comparison per person
for gen_result in results:
    person_id = gen_result['person_id']

    print(f"\n{'='*50}")
    print(f"PERSON {person_id}")
    print(f"{'='*50}")

    # Get real sequence and generated results
    real_tokens = gen_result['full_sequence']
    prompt_tokens = gen_result['prompt_tokens']
    generated_tokens_list = gen_result['generated_tokens']

    # Verify prompt matches beginning of real sequence
    prompt_len = len(prompt_tokens)
    real_prompt_portion = real_tokens[:prompt_len]

    if prompt_tokens == real_prompt_portion:
        print(f"✓ Prompt matches real sequence beginning")
    else:
        print(f"⚠ WARNING: Prompt does not match real sequence!")
        continue

    print(f"Real sequence length: {len(real_tokens)} tokens")
    print(f"Prompt length: {len(prompt_tokens)} tokens")
    print(f"Generated sequences: {len(generated_tokens_list)}")

    # Debug: Check if generated sequences contain time tokens
    for idx, gen_tokens in enumerate(generated_tokens_list):
        age_count = sum(1 for tid in gen_tokens if 'AGE_' in id2tok.get(int(tid), ''))
        year_count = sum(1 for tid in gen_tokens if 'YEAR_' in id2tok.get(int(tid), ''))
        print(f"  Generated sequence {idx}: {len(gen_tokens)} tokens")
        print(f"    - Contains {age_count} AGE tokens, {year_count} YEAR tokens")
        # Show first few tokens
        first_tokens = [id2tok.get(int(tid), f"UNK_{tid}") for tid in gen_tokens[:10]]
        print(f"    - First tokens: {first_tokens}")

    # Extract time token positions from real sequence
    real_age_indices, real_age_values = [], []
    real_year_indices, real_year_values = [], []

    pattern_age = re.compile(r'^AGE_(\d+)$')
    pattern_year = re.compile(r'^YEAR_(\d+)$')

    for i, tid in enumerate(real_tokens):
        name = id2tok.get(int(tid), "")
        m_age = pattern_age.match(name)
        if m_age:
            real_age_indices.append(i)
            real_age_values.append(int(m_age.group(1)))
        m_year = pattern_year.match(name)
        if m_year:
            real_year_indices.append(i)
            real_year_values.append(int(m_year.group(1)))

    # Create separate figures for AGE and YEAR
    fig_age, ax_age = plt.subplots(1, 1, figsize=(12, 6))
    fig_year, ax_year = plt.subplots(1, 1, figsize=(12, 6))

    # Plot real data as lines
    if len(real_age_indices) > 1:
        ax_age.plot(real_age_indices, real_age_values, c="blue", alpha=0.7, label="Real")
    if len(real_year_indices) > 1:
        ax_year.plot(real_year_indices, real_year_values, c="blue", alpha=0.7, label="Real")

    # Plot generated sequences
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for idx, gen_tokens in enumerate(generated_tokens_list):
        color = colors[idx % len(colors)]

        # Create full sequence (prompt + generated)
        full_sequence = prompt_tokens + gen_tokens

        # Extract time token positions from full sequence
        gen_age_indices, gen_age_values = [], []
        gen_year_indices, gen_year_values = [], []

        for i, tid in enumerate(full_sequence):
            name = id2tok.get(int(tid), "")
            m_age = pattern_age.match(name)
            if m_age:
                gen_age_indices.append(i)
                gen_age_values.append(int(m_age.group(1)))
            m_year = pattern_year.match(name)
            if m_year:
                gen_year_indices.append(i)
                gen_year_values.append(int(m_year.group(1)))

        # Plot generated tokens with low alpha
        if len(gen_age_indices) > 1:
            ax_age.plot(gen_age_indices, gen_age_values, c=color, alpha=0.1,
                       label=f"Gen {idx+1}" if idx == 0 else None)
        if len(gen_year_indices) > 1:
            ax_year.plot(gen_year_indices, gen_year_values, c=color, alpha=0.1,
                        label=f"Gen {idx+1}" if idx == 0 else None)

    # Draw prompt/generation boundary
    ax_age.axvline(prompt_len, color="gray", linestyle=":", linewidth=1)
    ax_year.axvline(prompt_len, color="gray", linestyle=":", linewidth=1)

    # Add boundary labels
    _, ymax_age = ax_age.get_ylim()
    _, ymax_year = ax_year.get_ylim()
    ax_age.text(prompt_len, ymax_age, "Prompt|Gen", rotation=90,
                va="bottom", ha="right", color="gray")
    ax_year.text(prompt_len, ymax_year, "Prompt|Gen", rotation=90,
                 va="bottom", ha="right", color="gray")

    # Finalize plots
    ax_age.set_title(f"AGE Token Positions - Person {person_id}")
    ax_age.set_ylabel("Years since birth")
    ax_age.set_xlabel("Token index")
    ax_age.grid(True, alpha=0.3)
    #ax_age.legend()

    ax_year.set_title(f"YEAR Token Positions - Person {person_id}")
    ax_year.set_ylabel("Calendar year")
    ax_year.set_xlabel("Token index")
    ax_year.grid(True, alpha=0.3)
    #ax_year.legend()

    # Save plots
    age_path = OUTPUT_DIR / f"time_tokens_person_{person_id}_age.png"
    year_path = OUTPUT_DIR / f"time_tokens_person_{person_id}_year.png"

    fig_age.tight_layout()
    fig_year.tight_layout()

    fig_age.savefig(age_path, dpi=300, bbox_inches="tight")
    fig_year.savefig(year_path, dpi=300, bbox_inches="tight")

    print(f"Saved AGE plot: {age_path}")
    print(f"Saved YEAR plot: {year_path}")

    # Clean up memory
    plt.close(fig_age)
    plt.close(fig_year)

print(f"\nProcessing complete!")
