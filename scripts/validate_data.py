#!/usr/bin/env python3
"""
Data validation script to check LMDB datasets for Time2Vec vs Time Tokens encoding.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --verbose
    python scripts/validate_data.py --samples 10 --save-report validation_report.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from src.dataset import LMDBDataset
from src.datamodule2 import BaseLightningDataModule


def load_vocabulary(dataset_path: Path) -> Optional[Dict]:
    """Load vocabulary from dataset directory."""
    vocab_path = dataset_path / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            return json.load(f)
    return None


def analyze_vocabulary(vocab: Dict, name: str) -> Dict[str, Any]:
    """Analyze vocabulary composition."""
    if not vocab:
        return {"error": "No vocabulary found"}

    # Convert to token_id -> token mapping
    id_to_token = {v: k for k, v in vocab.items()}

    # Count different token types
    year_tokens = [token for token in vocab.keys() if token.startswith("YEAR_")]
    age_tokens = [token for token in vocab.keys() if token.startswith("AGE_")]
    special_tokens = [token for token in vocab.keys() if token.startswith("[") and token.endswith("]")]
    regular_tokens = [token for token in vocab.keys()
                     if not (token.startswith("YEAR_") or token.startswith("AGE_")
                            or (token.startswith("[") and token.endswith("]")))]

    return {
        "name": name,
        "total_size": len(vocab),
        "year_tokens": len(year_tokens),
        "age_tokens": len(age_tokens),
        "special_tokens": len(special_tokens),
        "regular_tokens": len(regular_tokens),
        "year_token_range": (min(year_tokens, default=""), max(year_tokens, default="")),
        "age_token_range": (min(age_tokens, default=""), max(age_tokens, default="")),
        "sample_special": special_tokens[:10],
        "has_time_tokens": len(year_tokens) > 0 and len(age_tokens) > 0
    }


def load_dataset_info(dataset_path: Path, name: str) -> Dict[str, Any]:
    """Load basic information about a dataset."""
    info = {
        "name": name,
        "path": str(dataset_path),
        "exists": dataset_path.exists(),
        "lmdb_exists": (dataset_path / "dataset.lmdb").exists(),
        "vocab_exists": (dataset_path / "vocab.json").exists(),
    }

    if not info["exists"]:
        info["error"] = "Dataset directory does not exist"
        return info

    if not info["lmdb_exists"]:
        info["error"] = "LMDB file not found"
        return info

    try:
        # Load vocabulary
        vocab = load_vocabulary(dataset_path)
        info["vocabulary"] = analyze_vocabulary(vocab, name)

        # Load LMDB dataset
        lmdb_path = dataset_path / "dataset.lmdb"
        dataset = LMDBDataset(data=None, lmdb_path=lmdb_path)
        # Initialize the database connection
        dataset._init_db()

        info["num_sequences"] = len(dataset)
        info["dataset_loaded"] = True

        # Get sample data to check structure
        if len(dataset) > 0:
            sample = dataset[0]
            info["sample_keys"] = list(sample["data"].keys())

            # Check sequence lengths
            sample_lengths = []
            max_samples = min(100, len(dataset))
            for i in range(max_samples):
                data = dataset[i]["data"]
                if "event" in data and data["event"]:
                    sample_lengths.append(len(data["event"]))

            if sample_lengths:
                info["sequence_stats"] = {
                    "mean_length": np.mean(sample_lengths),
                    "median_length": np.median(sample_lengths),
                    "min_length": np.min(sample_lengths),
                    "max_length": np.max(sample_lengths),
                    "std_length": np.std(sample_lengths)
                }

    except Exception as e:
        info["error"] = f"Failed to load dataset: {str(e)}"
        info["dataset_loaded"] = False

    return info


def decode_sequence(event_tokens: List[int], vocab: Dict) -> List[str]:
    """Decode token IDs back to token strings."""
    id_to_token = {v: k for k, v in vocab.items()}
    return [id_to_token.get(token_id, f"<UNK:{token_id}>") for token_id in event_tokens]


def analyze_time_tokens_in_sequence(decoded_sequence: List[str]) -> Dict[str, Any]:
    """Analyze time token presence and positioning in a sequence."""
    year_positions = []
    age_positions = []

    for i, token in enumerate(decoded_sequence):
        if token.startswith("YEAR_"):
            year_positions.append(i)
        elif token.startswith("AGE_"):
            age_positions.append(i)

    return {
        "has_year_tokens": len(year_positions) > 0,
        "has_age_tokens": len(age_positions) > 0,
        "year_positions": year_positions[:10],  # First 10 positions
        "age_positions": age_positions[:10],
        "total_year_tokens": len(year_positions),
        "total_age_tokens": len(age_positions),
        "first_tokens": decoded_sequence[:10],  # First 10 tokens
        "tokens_are_time_prefixed": (
            len(decoded_sequence) >= 2 and
            decoded_sequence[0].startswith("YEAR_") and
            decoded_sequence[1].startswith("AGE_")
        )
    }


def sample_sequences(dataset: LMDBDataset, vocab: Dict, num_samples: int = 5, verbose: bool = False) -> List[Dict]:
    """Sample and analyze sequences from the dataset."""
    samples = []
    dataset_size = len(dataset)

    # Get diverse samples
    indices = np.linspace(0, dataset_size - 1, min(num_samples, dataset_size), dtype=int)

    for i, idx in enumerate(indices):
        try:
            data = dataset[idx]["data"]

            if "event" in data and data["event"]:
                event_tokens = data["event"]
                decoded = decode_sequence(event_tokens, vocab)
                time_analysis = analyze_time_tokens_in_sequence(decoded)

                sample_info = {
                    "sample_id": i,
                    "dataset_index": int(idx),
                    "sequence_length": len(event_tokens),
                    "available_keys": list(data.keys()),
                    "time_token_analysis": time_analysis,
                    "full_sequence": decoded if verbose else decoded[:20]  # Show first 20 tokens or full if verbose
                }

                # Add type-specific information
                if "abspos" in data and "age" in data:
                    sample_info["encoding_type"] = "time2vec"
                    sample_info["abspos_sample"] = data["abspos"][:5] if hasattr(data["abspos"], "__len__") else data["abspos"]
                    sample_info["age_sample"] = data["age"][:5] if hasattr(data["age"], "__len__") else data["age"]
                else:
                    sample_info["encoding_type"] = "time_tokens"

                samples.append(sample_info)

        except Exception as e:
            samples.append({
                "sample_id": i,
                "dataset_index": int(idx),
                "error": f"Failed to load sample: {str(e)}"
            })

    return samples


def compare_datasets(time2vec_info: Dict, timetoken_info: Dict) -> Dict[str, Any]:
    """Compare two datasets and highlight differences."""
    comparison = {
        "both_loaded": time2vec_info.get("dataset_loaded", False) and timetoken_info.get("dataset_loaded", False)
    }

    if not comparison["both_loaded"]:
        comparison["error"] = "Cannot compare - one or both datasets failed to load"
        return comparison

    # Compare vocabulary sizes
    tv_vocab = time2vec_info.get("vocabulary", {})
    tt_vocab = timetoken_info.get("vocabulary", {})

    comparison["vocabulary_comparison"] = {
        "time2vec_size": tv_vocab.get("total_size", 0),
        "timetoken_size": tt_vocab.get("total_size", 0),
        "size_difference": tt_vocab.get("total_size", 0) - tv_vocab.get("total_size", 0),
        "timetoken_has_time_tokens": tt_vocab.get("has_time_tokens", False),
        "time2vec_has_time_tokens": tv_vocab.get("has_time_tokens", False)
    }

    # Compare dataset sizes
    comparison["dataset_size_comparison"] = {
        "time2vec_sequences": time2vec_info.get("num_sequences", 0),
        "timetoken_sequences": timetoken_info.get("num_sequences", 0),
        "sequence_difference": timetoken_info.get("num_sequences", 0) - time2vec_info.get("num_sequences", 0)
    }

    # Compare sequence lengths
    tv_stats = time2vec_info.get("sequence_stats", {})
    tt_stats = timetoken_info.get("sequence_stats", {})

    if tv_stats and tt_stats:
        comparison["sequence_length_comparison"] = {
            "time2vec_mean": tv_stats.get("mean_length", 0),
            "timetoken_mean": tt_stats.get("mean_length", 0),
            "mean_difference": tt_stats.get("mean_length", 0) - tv_stats.get("mean_length", 0),
            "relative_increase_percent": ((tt_stats.get("mean_length", 0) - tv_stats.get("mean_length", 0)) / tv_stats.get("mean_length", 1)) * 100
        }

    return comparison


def print_summary_report(time2vec_info: Dict, timetoken_info: Dict, comparison: Dict, samples: Dict):
    """Print a comprehensive summary report."""
    print("=" * 80)
    print("📊 DATASET VALIDATION REPORT")
    print("=" * 80)

    # Dataset existence and loading
    print("\n🔍 Dataset Status:")
    for dataset_name, info in [("Time2Vec", time2vec_info), ("Time Tokens", timetoken_info)]:
        status = "✅ LOADED" if info.get("dataset_loaded", False) else "❌ FAILED"
        print(f"  {dataset_name:12}: {status}")
        if "error" in info:
            print(f"  {'':12}  Error: {info['error']}")

    # Vocabulary comparison
    print("\n📖 Vocabulary Analysis:")
    if comparison.get("both_loaded"):
        vocab_comp = comparison["vocabulary_comparison"]
        print(f"  Time2Vec size:     {vocab_comp['time2vec_size']:,}")
        print(f"  Time Tokens size:  {vocab_comp['timetoken_size']:,}")
        print(f"  Difference:        +{vocab_comp['size_difference']:,} tokens")

        if vocab_comp['timetoken_has_time_tokens']:
            tt_vocab = timetoken_info["vocabulary"]
            print(f"  Time tokens added: {tt_vocab['year_tokens']} years + {tt_vocab['age_tokens']} ages")
            print(f"  Year range:        {tt_vocab['year_token_range'][0]} to {tt_vocab['year_token_range'][1]}")
        else:
            print("  ⚠️  Time tokens NOT found in time tokens dataset!")

    # Dataset size comparison
    print("\n📈 Dataset Size:")
    if comparison.get("both_loaded"):
        size_comp = comparison["dataset_size_comparison"]
        print(f"  Time2Vec sequences:    {size_comp['time2vec_sequences']:,}")
        print(f"  Time Tokens sequences: {size_comp['timetoken_sequences']:,}")
        if size_comp['sequence_difference'] != 0:
            print(f"  Difference:            {size_comp['sequence_difference']:+,}")

    # Sequence length analysis
    print("\n📏 Sequence Length Analysis:")
    if comparison.get("both_loaded") and "sequence_length_comparison" in comparison:
        length_comp = comparison["sequence_length_comparison"]
        print(f"  Time2Vec mean length:    {length_comp['time2vec_mean']:.1f}")
        print(f"  Time Tokens mean length: {length_comp['timetoken_mean']:.1f}")
        print(f"  Mean increase:           +{length_comp['mean_difference']:.1f} tokens")
        print(f"  Relative increase:       {length_comp['relative_increase_percent']:.1f}%")

    # Sample analysis
    print("\n🔬 Sample Analysis:")
    for dataset_name, sample_list in samples.items():
        print(f"\n  {dataset_name}:")
        if sample_list:
            sample = sample_list[0]  # Show first sample
            if "error" not in sample:
                print(f"    Encoding type:    {sample.get('encoding_type', 'unknown')}")
                print(f"    Available keys:   {', '.join(sample['available_keys'])}")
                print(f"    Sample length:    {sample['sequence_length']}")

                time_analysis = sample.get("time_token_analysis", {})
                if time_analysis.get("has_year_tokens") or time_analysis.get("has_age_tokens"):
                    print(f"    Year tokens:      {time_analysis['total_year_tokens']}")
                    print(f"    Age tokens:       {time_analysis['total_age_tokens']}")
                    print(f"    Time-prefixed:    {time_analysis['tokens_are_time_prefixed']}")

                print(f"    First tokens:     {' '.join(time_analysis.get('first_tokens', [])[:8])}")
        else:
            print("    No samples available")

    # Issues and recommendations
    print("\n⚠️  Issues Found:")
    issues = []

    if not time2vec_info.get("dataset_loaded"):
        issues.append("Time2Vec dataset failed to load")
    if not timetoken_info.get("dataset_loaded"):
        issues.append("Time Tokens dataset failed to load")

    if comparison.get("both_loaded"):
        vocab_comp = comparison["vocabulary_comparison"]
        if not vocab_comp['timetoken_has_time_tokens']:
            issues.append("Time tokens vocabulary does not contain YEAR_/AGE_ tokens")
        if vocab_comp['size_difference'] < 200:
            issues.append(f"Vocabulary increase too small ({vocab_comp['size_difference']} tokens, expected ~232)")

    if samples.get("Time Tokens"):
        tt_samples = samples["Time Tokens"]
        if tt_samples and "error" not in tt_samples[0]:
            sample = tt_samples[0]
            if not sample.get("time_token_analysis", {}).get("tokens_are_time_prefixed"):
                issues.append("Time tokens sequences don't start with YEAR_/AGE_ tokens")

    if not issues:
        print("    ✅ No issues detected!")
    else:
        for issue in issues:
            print(f"    ❌ {issue}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Validate LMDB datasets for Time2Vec vs Time Tokens')
    parser.add_argument('--time2vec-path', type=str, default='data/data_reverted/destiny_dataset',
                       help='Path to Time2Vec dataset directory')
    parser.add_argument('--timetoken-path', type=str, default='data/destiny_dataset_timetoken',
                       help='Path to Time Tokens dataset directory')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of sample sequences to analyze')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed sample sequences')
    parser.add_argument('--save-report', type=str,
                       help='Save detailed report to file')

    args = parser.parse_args()

    time2vec_path = Path(args.time2vec_path)
    timetoken_path = Path(args.timetoken_path)

    print("🔍 Loading dataset information...")

    # Load dataset information
    time2vec_info = load_dataset_info(time2vec_path, "Time2Vec")
    timetoken_info = load_dataset_info(timetoken_path, "Time Tokens")

    # Compare datasets
    comparison = compare_datasets(time2vec_info, timetoken_info)

    # Sample sequences
    samples = {}

    if time2vec_info.get("dataset_loaded"):
        print(f"📝 Sampling {args.samples} sequences from Time2Vec dataset...")
        dataset = LMDBDataset(data=None, lmdb_path=time2vec_path / "dataset.lmdb")
        dataset._init_db()
        vocab = load_vocabulary(time2vec_path)
        samples["Time2Vec"] = sample_sequences(dataset, vocab, args.samples, args.verbose)

    if timetoken_info.get("dataset_loaded"):
        print(f"📝 Sampling {args.samples} sequences from Time Tokens dataset...")
        dataset = LMDBDataset(data=None, lmdb_path=timetoken_path / "dataset.lmdb")
        dataset._init_db()
        vocab = load_vocabulary(timetoken_path)
        samples["Time Tokens"] = sample_sequences(dataset, vocab, args.samples, args.verbose)

    # Print summary report
    print_summary_report(time2vec_info, timetoken_info, comparison, samples)

    # Print detailed samples if verbose
    if args.verbose:
        print("\n" + "=" * 80)
        print("📋 DETAILED SAMPLE SEQUENCES")
        print("=" * 80)

        for dataset_name, sample_list in samples.items():
            print(f"\n🔍 {dataset_name} Samples:")
            for sample in sample_list[:3]:  # Show first 3 samples in detail
                if "error" not in sample:
                    print(f"\n  Sample {sample['sample_id']} (length: {sample['sequence_length']}):")
                    tokens = sample['full_sequence'][:50]  # First 50 tokens
                    for i in range(0, len(tokens), 10):
                        token_group = ' '.join(tokens[i:i+10])
                        print(f"    {i:3d}: {token_group}")
                    if len(sample['full_sequence']) > 50:
                        print(f"    ... ({len(sample['full_sequence']) - 50} more tokens)")

    # Save detailed report if requested
    if args.save_report:
        print(f"\n💾 Saving detailed report to {args.save_report}...")
        report_data = {
            "time2vec_info": time2vec_info,
            "timetoken_info": timetoken_info,
            "comparison": comparison,
            "samples": samples
        }

        with open(args.save_report, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"✅ Report saved successfully!")


if __name__ == "__main__":
    main()