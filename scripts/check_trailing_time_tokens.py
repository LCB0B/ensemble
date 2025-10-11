#!/usr/bin/env python3
"""
Check if sequences in LMDB dataset end with time tokens.

This script validates that the _remove_trailing_time_tokens() function
worked correctly during dataset creation.

Usage:
    python scripts/check_trailing_time_tokens.py
    python scripts/check_trailing_time_tokens.py --num-samples 500
    python scripts/check_trailing_time_tokens.py --check-all
    python scripts/check_trailing_time_tokens.py --show-last-tokens
    python scripts/check_trailing_time_tokens.py --dataset-path data/custom_dataset
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
from tqdm import tqdm

from src.dataset import LMDBDataset
from src.paths import FPATH


def load_vocabulary(dataset_path: Path) -> Dict[str, int]:
    """Load vocabulary from dataset directory."""
    vocab_path = dataset_path / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")

    with open(vocab_path, 'r') as f:
        return json.load(f)


def get_time_token_ids(vocab: Dict[str, int]) -> set:
    """Extract all time token IDs (YEAR_XXXX and AGE_YY tokens)."""
    time_token_ids = set()
    for token_name, token_id in vocab.items():
        if token_name.startswith(('YEAR_', 'AGE_')):
            time_token_ids.add(token_id)
    return time_token_ids


def decode_sequence(event_tokens: List, vocab: Dict[str, int]) -> List[str]:
    """Decode token IDs back to token strings."""
    id_to_token = {v: k for k, v in vocab.items()}

    # Handle both flat and nested event structures
    flat_tokens = []
    for item in event_tokens:
        if isinstance(item, (list, tuple)):
            flat_tokens.extend(item)
        else:
            flat_tokens.append(item)

    return [id_to_token.get(int(token_id), f"<UNK:{token_id}>") for token_id in flat_tokens]


def check_trailing_tokens(events: List, time_token_ids: set) -> Tuple[bool, List[int]]:
    """
    Check if sequence ends with time tokens.

    Returns:
        (has_trailing_time_tokens, trailing_token_ids)
    """
    if not events:
        return False, []

    trailing_tokens = []

    # Handle nested structure (list of lists) or flat list
    if isinstance(events[-1], (list, tuple)):
        # Nested structure - check last event
        if len(events[-1]) == 1 and events[-1][0] in time_token_ids:
            trailing_tokens.append(events[-1][0])
            # Check more events backwards
            for event in reversed(events[:-1]):
                if isinstance(event, (list, tuple)) and len(event) == 1 and event[0] in time_token_ids:
                    trailing_tokens.append(event[0])
                else:
                    break
    else:
        # Flat structure - check last token
        if events[-1] in time_token_ids:
            trailing_tokens.append(events[-1])
            # Check more tokens backwards
            for token in reversed(events[:-1]):
                if token in time_token_ids:
                    trailing_tokens.append(token)
                else:
                    break

    trailing_tokens.reverse()  # Put back in chronological order
    return len(trailing_tokens) > 0, trailing_tokens


def analyze_dataset(dataset_path: Path, num_samples: int = None, verbose: bool = False, show_last_tokens: bool = False):
    """Analyze dataset for trailing time tokens."""

    print(f"📂 Loading dataset from: {dataset_path}")

    # Load vocabulary
    vocab = load_vocabulary(dataset_path)
    time_token_ids = get_time_token_ids(vocab)
    id_to_token = {v: k for k, v in vocab.items()}

    print(f"📖 Vocabulary size: {len(vocab)}")
    print(f"🕐 Time token IDs: {len(time_token_ids)}")

    # Load LMDB dataset
    lmdb_path = dataset_path / "dataset.lmdb"
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB not found at {lmdb_path}")

    dataset = LMDBDataset(data=None, lmdb_path=lmdb_path)
    dataset._init_db()

    total_sequences = len(dataset)
    print(f"📊 Total sequences in dataset: {total_sequences}")

    # Determine how many to check
    if num_samples is None:
        num_to_check = total_sequences
        print(f"🔍 Checking ALL sequences")
    else:
        num_to_check = min(num_samples, total_sequences)
        print(f"🔍 Checking {num_to_check} sequences")

    if show_last_tokens:
        print(f"📝 Will display last token of each sequence\n")

    # Statistics
    sequences_with_trailing = []
    total_trailing_tokens = 0

    # Check sequences
    iterator = range(num_to_check)
    if not show_last_tokens:
        iterator = tqdm(iterator, desc="Checking sequences")

    for idx in iterator:
        try:
            data = dataset[idx]["data"]
            events = data.get("event", [])

            if not events:
                if show_last_tokens:
                    print(f"Seq {idx:5d}: <EMPTY>")
                continue

            # Get decoded sequence
            decoded = decode_sequence(events, vocab)
            last_token = decoded[-1] if decoded else "<EMPTY>"

            if show_last_tokens:
                is_time_token = "🕐" if last_token.startswith(('YEAR_', 'AGE_')) else "  "
                print(f"Seq {idx:5d}: {is_time_token} {last_token}")

            has_trailing, trailing_ids = check_trailing_tokens(events, time_token_ids)

            if has_trailing:
                # Get last few tokens for display
                last_tokens = decoded[-10:] if len(decoded) > 10 else decoded

                sequences_with_trailing.append({
                    'idx': idx,
                    'total_length': len(decoded),
                    'trailing_count': len(trailing_ids),
                    'trailing_tokens': [id_to_token.get(tid, f"<UNK:{tid}>") for tid in trailing_ids],
                    'last_10_tokens': last_tokens
                })
                total_trailing_tokens += len(trailing_ids)

        except Exception as e:
            if verbose:
                print(f"\n⚠️  Error processing sequence {idx}: {e}")
            if show_last_tokens:
                print(f"Seq {idx:5d}: <ERROR: {e}>")
            continue

    # Print results
    print("\n" + "=" * 80)
    print("📊 TRAILING TIME TOKEN ANALYSIS")
    print("=" * 80)
    print(f"\nSequences checked:                {num_to_check:,}")
    print(f"Sequences with trailing tokens:   {len(sequences_with_trailing):,}")
    print(f"Percentage with trailing tokens:  {len(sequences_with_trailing)/num_to_check*100:.2f}%")
    print(f"Total trailing time tokens found: {total_trailing_tokens:,}")

    if sequences_with_trailing:
        print(f"\n⚠️  WARNING: Found {len(sequences_with_trailing)} sequences ending with time tokens!")
        print("\n🔍 Examples of problematic sequences:")
        print("-" * 80)

        for i, seq_info in enumerate(sequences_with_trailing[:10]):  # Show first 10
            print(f"\nSequence {seq_info['idx']} (length: {seq_info['total_length']} tokens):")
            print(f"  Trailing time tokens ({seq_info['trailing_count']}): {' '.join(seq_info['trailing_tokens'])}")
            print(f"  Last 10 tokens: {' '.join(seq_info['last_10_tokens'])}")

        if len(sequences_with_trailing) > 10:
            print(f"\n... and {len(sequences_with_trailing) - 10} more sequences with trailing time tokens")
    else:
        print("\n✅ SUCCESS: No sequences end with time tokens!")
        print("   The _remove_trailing_time_tokens() function worked correctly.")

    print("\n" + "=" * 80)

    return sequences_with_trailing


def main():
    parser = argparse.ArgumentParser(
        description='Check if LMDB dataset sequences end with time tokens'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/destiny_dataset_timetoken',
        help='Path to dataset directory (default: data/destiny_dataset_timetoken)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of sequences to check (default: 100)'
    )
    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Check all sequences in the dataset'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed error messages'
    )
    parser.add_argument(
        '--show-last-tokens',
        action='store_true',
        help='Print the last token of each sequence (disables progress bar)'
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(f"❌ Error: Dataset path does not exist: {dataset_path}")
        return 1

    num_samples = None if args.check_all else args.num_samples

    try:
        sequences_with_trailing = analyze_dataset(
            dataset_path,
            num_samples=num_samples,
            verbose=args.verbose,
            show_last_tokens=args.show_last_tokens
        )

        # Return exit code based on results
        return 1 if sequences_with_trailing else 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
