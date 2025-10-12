#!/usr/bin/env python3
"""
Time Token Sequence Generation Script

Generate sequences using trained time token models with sophisticated time-aware controls.
Supports starting at specific time points (age/year tokens or indices) and stopping
at specified conditions (token count, age, or year).

Usage:
    python scripts/generate_time_token_sequences.py \
        --model_name 029 \
        --experiment destiny \
        --num_people 10 \
        --start_condition age:18 \
        --stop_condition age:30 \
        --num_generations 20
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


def load_model_and_config(model_name: str, experiment: str) -> Dict[str, Any]:
    """
    Load model checkpoint and configuration from model directory.

    Args:
        model_name: Model name (e.g., "029")
        experiment: Experiment name (e.g., "destiny")

    Returns:
        Dictionary with checkpoint_path, hparams_path, hparams, dataset_dir, vocab_path
    """
    # Load hparams
    hparams_path = FPATH.TB_LOGS / experiment / model_name / "hparams.yaml"

    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.yaml not found: {hparams_path}")

    print(f"Loading configuration from: {hparams_path}")
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)

    # Find checkpoint
    ckpt_dir = FPATH.CHECKPOINTS_TRANSFORMER / experiment / model_name

    # Check for standard checkpoint names in order of preference
    checkpoint_candidates = [
        ckpt_dir / "best.ckpt",
        ckpt_dir / "last.ckpt",
    ]

    # Try standard checkpoint names first
    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if candidate.exists():
            checkpoint_path = candidate
            break

    # If no standard checkpoints found, look for any .ckpt files
    if checkpoint_path is None:
        if ckpt_dir.exists():
            ckpt_files = list(ckpt_dir.glob("*.ckpt"))
            if ckpt_files:
                # Use the latest checkpoint
                checkpoint_path = max(ckpt_files, key=lambda x: x.stat().st_mtime)

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint files found in {ckpt_dir}. "
            f"Expected locations: {[str(c) for c in checkpoint_candidates]}"
        )

    print(f"Loading model checkpoint from: {checkpoint_path}")

    # Extract dataset directory and vocab path
    dataset_dir = FPATH.DATA / hparams["dir_path"]
    vocab_path = dataset_dir / "vocab.json"

    print(f"  Dataset directory: {dataset_dir}")
    print(f"  Vocabulary path: {vocab_path}")

    return {
        'checkpoint_path': str(checkpoint_path),
        'hparams_path': str(hparams_path),
        'hparams': hparams,
        'dataset_dir': dataset_dir,
        'vocab_path': vocab_path
    }


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


def validate_conditions(start_condition: Dict[str, Any], stop_condition: Dict[str, Any]) -> bool:
    """
    Validate that stop condition is temporally after start condition.

    Args:
        start_condition: Start condition dictionary
        stop_condition: Stop condition dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If conditions are invalid or illogical
    """
    start_type = start_condition['type']
    stop_type = stop_condition['type']
    start_value = start_condition['value']
    stop_value = stop_condition['value']

    # Validate that types are recognized
    valid_start_types = ['index', 'age', 'year']
    valid_stop_types = ['tokens', 'age', 'year']

    if start_type not in valid_start_types:
        raise ValueError(
            f"Invalid start condition type: '{start_type}'. "
            f"Must be one of: {valid_start_types}"
        )

    if stop_type not in valid_stop_types:
        raise ValueError(
            f"Invalid stop condition type: '{stop_type}'. "
            f"Must be one of: {valid_stop_types}"
        )

    # If both are same type (age or year), can compare directly
    if start_type == stop_type and start_type in ['age', 'year']:
        if stop_value <= start_value:
            raise ValueError(
                f"Stop {stop_type} ({stop_value}) must be after "
                f"start {start_type} ({start_value})"
            )
        print(f"✓ Valid temporal range: {start_type} {start_value} → {stop_value}")

    # If start is age and stop is year (or vice versa), warn but allow
    elif (start_type == 'age' and stop_type == 'year') or \
         (start_type == 'year' and stop_type == 'age'):
        print(
            f"⚠ WARNING: Start condition is {start_type}:{start_value} "
            f"but stop is {stop_type}:{stop_value}. "
            f"Ensure this makes sense for your use case."
        )

    # If stop is tokens, just validate it's positive
    elif stop_type == 'tokens':
        if stop_value <= 0:
            raise ValueError(f"Stop tokens must be > 0, got {stop_value}")
        print(f"✓ Will generate {stop_value} tokens starting from {start_type}:{start_value}")

    # If start is index, less validation needed
    elif start_type == 'index':
        if start_value < 0:
            raise ValueError(f"Start index must be >= 0, got {start_value}")
        print(f"✓ Starting at index {start_value}, stopping at {stop_type}:{stop_value}")

    return True


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
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., '029')"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (e.g., 'destiny')"
    )

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

    print("=" * 80)
    print("TIME TOKEN SEQUENCE GENERATION")
    print("=" * 80)

    # Load model and configuration
    print(f"\nModel: {args.model_name}")
    print(f"Experiment: {args.experiment}\n")

    paths_info = load_model_and_config(args.model_name, args.experiment)

    print(f"\nLoading time token model...")
    model, hparams = load_time_token_model(
        paths_info['checkpoint_path'],
        paths_info['hparams_path']
    )

    print(f"Setting up datamodule...")
    datamodule, hparams = setup_time_token_datamodule(hparams)

    # Get vocabulary for decoding
    vocab = {i: token for token, i in datamodule.pipeline.vocab.items()}
    time_tokens = extract_time_tokens(vocab)

    print(f"Found {len(time_tokens['year_tokens'])} year tokens and {len(time_tokens['age_tokens'])} age tokens")

    # Parse and validate conditions
    print(f"\n" + "-" * 80)
    print("GENERATION CONDITIONS")
    print("-" * 80)

    start_condition = parse_condition(args.start_condition)
    stop_condition = parse_condition(args.stop_condition)

    print(f"Start condition: {start_condition}")
    print(f"Stop condition: {stop_condition}")

    # Validate conditions
    try:
        validate_conditions(start_condition, stop_condition)
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return

    # Generate sequences
    print(f"\n" + "-" * 80)
    print("GENERATION")
    print("-" * 80)
    print(f"Generating for {args.num_people} people with {args.num_generations} sequences each...")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Device: {args.device}\n")

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
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)

    total_generations = sum(len(r['generated_tokens']) for r in results)
    avg_length = np.mean([np.mean(r['generation_lengths']) for r in results])

    print(f"\nSummary:")
    print(f"  People processed: {len(results)}")
    print(f"  Generations per person: {args.num_generations}")
    print(f"  Total generations: {total_generations}")
    print(f"  Average generation length: {avg_length:.1f} tokens")
    print(f"\nResults saved to:")
    print(f"  {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()