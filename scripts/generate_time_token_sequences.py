#!/usr/bin/env python3
"""
Time Token Sequence Generation Script

Generate sequences using trained time token models with sophisticated time-aware controls.
Supports starting at specific time points (age/year tokens or indices) and stopping
at specified conditions (token count, age, or year).

Usage:
    python scripts/generate_time_token_sequences.py --model_path MODEL.ckpt --config CONFIG.yaml
    python scripts/generate_time_token_sequences.py --experiment 097_modern_falcon-pretrain-lr0.2
"""

import argparse
import json
import torch
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

from src.generation_utils import (
    load_time_token_model,
    setup_time_token_datamodule,
    generate_time_token_sequences,
    extract_time_tokens,
    load_vocab
)
from src.paths import FPATH


def load_experiment_model(experiment_name: str):
    """Load model and config from experiment directory"""
    # Look for checkpoint in checkpoints directory
    ckpt_dir = FPATH.CHECKPOINTS_TRANSFORMER / "destiny" / experiment_name

    # Look for hparams in logs directory (correct current location)
    logs_dir = FPATH.TB_LOGS / "destiny" / experiment_name
    hparams_path = logs_dir / "hparams.yaml"

    if not logs_dir.exists():
        raise FileNotFoundError(f"Experiment logs directory not found: {logs_dir}")

    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.yaml not found in {logs_dir}")

    # Check for standard checkpoint names in order of preference
    checkpoint_candidates = [
        ckpt_dir / "best.ckpt",
        ckpt_dir / "last.ckpt",
    ]

    # Try standard checkpoint names first
    ckpt_path = None
    for candidate in checkpoint_candidates:
        if candidate.exists():
            ckpt_path = candidate
            break

    # If no standard checkpoints found, look for any .ckpt files
    if ckpt_path is None:
        if ckpt_dir.exists():
            ckpt_files = list(ckpt_dir.glob("*.ckpt"))
            if ckpt_files:
                # Use the latest checkpoint
                ckpt_path = max(ckpt_files, key=lambda x: x.stat().st_mtime)

    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}. Expected locations: {[str(c) for c in checkpoint_candidates]}")

    print(f"Loading model from: {ckpt_path}")
    print(f"Loading config from: {hparams_path}")

    return str(ckpt_path), str(hparams_path)


# Removed - using dataloader approach instead


def parse_condition(condition_str: str) -> Dict[str, Any]:
    """
    Parse condition string into structured format.

    Examples:
        "index:100" -> {"type": "index", "value": 100}
        "age:25" -> {"type": "age", "value": 25}
        "year:2010" -> {"type": "year", "value": 2010}
        "tokens:50" -> {"type": "tokens", "value": 50}
    """
    try:
        condition_type, value_str = condition_str.split(":", 1)
        value = int(value_str)
        return {"type": condition_type, "value": value}
    except ValueError:
        raise ValueError(f"Invalid condition format: {condition_str}. Expected 'type:value'")


def save_results(results: List[Dict], output_path: Path, vocab: Dict[int, str] = None):
    """Save generation results to file"""
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp and metadata
    output_data = {
        "generation_timestamp": datetime.now().isoformat(),
        "num_people": len(results),
        "results": results
    }

    # Save as JSON
    if output_path.suffix == '.json':
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    # Also create a human-readable summary
    summary_path = output_path.with_suffix('.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Time Token Generation Results\n")
        f.write(f"Generated: {output_data['generation_timestamp']}\n")
        f.write(f"Number of people: {output_data['num_people']}\n\n")

        for i, result in enumerate(results):
            f.write(f"Person {result['person_id']} (Result {i+1}):\n")
            f.write(f"  Start: {result['start_condition']}\n")
            f.write(f"  Stop: {result['stop_condition']}\n")
            f.write(f"  Generations: {result['num_generations']}\n")
            f.write(f"  Prompt length: {len(result['prompt_tokens'])}\n")

            for j, gen_tokens in enumerate(result['generated_tokens']):
                f.write(f"  Generation {j+1}: {len(gen_tokens)} tokens\n")
                if vocab:
                    # Show first few tokens with names
                    sample_tokens = gen_tokens[:10]
                    token_names = [vocab.get(token_id, f"UNK_{token_id}") for token_id in sample_tokens]
                    f.write(f"    Sample: {token_names}\n")
            f.write("\n")

    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate time token sequences")

    # Model loading options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", type=str, help="Experiment name (e.g., 097_modern_falcon-pretrain-lr0.2)")
    group.add_argument("--model_path", type=str, help="Path to model checkpoint")

    parser.add_argument("--config_path", type=str, help="Path to config YAML (required if using --model_path)")

    # Generation parameters
    parser.add_argument("--num_people", type=int, default=5, help="Number of people to generate for")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations per person")

    # Time token controls
    parser.add_argument("--start_condition", type=str, default="age:18",
                       help="Start condition (format: type:value, e.g., 'age:25', 'year:2010', 'index:100')")
    parser.add_argument("--stop_condition", type=str, default="tokens:50",
                       help="Stop condition (format: type:value, e.g., 'age:30', 'year:2015', 'tokens:100')")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")

    # Output options
    parser.add_argument("--output_dir", type=str, default="generated_time_tokens",
                       help="Output directory for results")
    parser.add_argument("--output_name", type=str, help="Output filename (auto-generated if not specified)")

    # Device and performance
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model and config
    if args.experiment:
        model_path, config_path = load_experiment_model(args.experiment)
    else:
        model_path = args.model_path
        config_path = args.config_path
        if not config_path:
            raise ValueError("--config_path required when using --model_path")

    print(f"Loading time token model...")
    model, hparams = load_time_token_model(model_path, config_path)

    print(f"Setting up datamodule...")
    datamodule, hparams = setup_time_token_datamodule(hparams)

    # Get vocabulary for decoding
    vocab = {i: token for token, i in datamodule.pipeline.vocab.items()}
    time_tokens = extract_time_tokens(vocab)

    print(f"Found {len(time_tokens['year_tokens'])} year tokens and {len(time_tokens['age_tokens'])} age tokens")

    # Parse conditions
    start_condition = parse_condition(args.start_condition)
    stop_condition = parse_condition(args.stop_condition)

    print(f"Start condition: {start_condition}")
    print(f"Stop condition: {stop_condition}")

    print(f"Generating for {args.num_people} people")

    # Generate sequences
    print(f"Generating {args.num_generations} sequences per person...")
    results = generate_time_token_sequences(
        model=model,
        datamodule=datamodule,
        num_people=args.num_people,
        start_condition=start_condition,
        stop_condition=stop_condition,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device
    )

    # Create output filename
    if args.output_name:
        output_name = args.output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_str = f"{start_condition['type']}{start_condition['value']}"
        stop_str = f"{stop_condition['type']}{stop_condition['value']}"
        output_name = f"timetoken_gen_{start_str}_to_{stop_str}_{timestamp}.json"

    output_path = Path(args.output_dir) / output_name

    # Save results
    save_results(results, output_path, vocab)

    # Print summary
    total_generations = sum(len(r['generated_tokens']) for r in results)
    avg_length = np.mean([np.mean(r['generation_lengths']) for r in results])

    print(f"\nGeneration Summary:")
    print(f"  People processed: {len(results)}")
    print(f"  Total generations: {total_generations}")
    print(f"  Average generation length: {avg_length:.1f} tokens")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()