"""
Data analysis and reporting utilities for time encoding comparison and dataset analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import polars as pl
from datetime import datetime

from src.log_data import (
    calculate_sequence_length_statistics,
    compare_time_encoding_modes,
    summarize_data_creation
)


def analyze_sequence_length_impact(
    time2vec_log_dir: Optional[Path],
    time_tokens_log_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze the impact of switching from time2vec to time_tokens encoding

    Args:
        time2vec_log_dir: Directory containing time2vec encoding logs (None if not available)
        time_tokens_log_dir: Directory containing time_tokens encoding logs
        output_dir: Directory to save analysis results

    Returns:
        Dictionary containing analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load time_tokens stats
    time_tokens_stats = load_dataset_stats(time_tokens_log_dir)

    # Load time2vec stats if available
    time2vec_stats = None
    if time2vec_log_dir and time2vec_log_dir.exists():
        time2vec_stats = load_dataset_stats(time2vec_log_dir)

    # Create comparison report
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "sequence_length_impact",
        "time2vec_available": time2vec_stats is not None,
        "time_tokens_stats": time_tokens_stats,
        "time2vec_stats": time2vec_stats,
    }

    # Calculate impact metrics if both are available
    if time2vec_stats and time_tokens_stats:
        analysis["impact_metrics"] = calculate_impact_metrics(time2vec_stats, time_tokens_stats)

    # Save analysis
    analysis_file = output_dir / "sequence_length_impact_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


def load_dataset_stats(log_dir: Path) -> Dict[str, Any]:
    """
    Load statistics from a dataset log directory

    Args:
        log_dir: Directory containing log files

    Returns:
        Dictionary containing all available statistics
    """
    stats = {}

    # Load vocabulary stats
    vocab_file = log_dir / "vocabulary_stats.json"
    if vocab_file.exists():
        with open(vocab_file, "r", encoding="utf-8") as f:
            stats["vocabulary"] = json.load(f)

    # Load sequence length analysis
    seq_file = log_dir / "sequence_length_analysis.json"
    if seq_file.exists():
        with open(seq_file, "r", encoding="utf-8") as f:
            stats["sequence_length_analysis"] = json.load(f)

    # Load LMDB sequence length stats
    lmdb_file = log_dir / "lmdb_sequence_length_stats.json"
    if lmdb_file.exists():
        with open(lmdb_file, "r", encoding="utf-8") as f:
            stats["lmdb_sequence_stats"] = json.load(f)

    # Load time token insertion stats
    time_token_file = log_dir / "time_token_insertion.jsonl"
    if time_token_file.exists():
        time_token_stats = []
        with open(time_token_file, "r", encoding="utf-8") as f:
            for line in f:
                time_token_stats.append(json.loads(line))
        stats["time_token_insertion"] = time_token_stats

    # Load detailed time token insertion stats
    detailed_phases_file = log_dir / "processing_phases.jsonl"
    if detailed_phases_file.exists():
        detailed_stats = []
        with open(detailed_phases_file, "r", encoding="utf-8") as f:
            for line in f:
                phase_data = json.loads(line)
                if phase_data.get("phase") == "time_token_insertion_detailed":
                    detailed_stats.append(phase_data)
        if detailed_stats:
            stats["time_token_detailed"] = detailed_stats

    # Load sequence length comparisons
    comparison_file = log_dir / "sequence_length_comparison_tokenized_events.jsonl"
    if comparison_file.exists():
        comparisons = []
        with open(comparison_file, "r", encoding="utf-8") as f:
            for line in f:
                comparisons.append(json.loads(line))
        stats["sequence_length_comparisons"] = comparisons

    return stats


def calculate_impact_metrics(
    time2vec_stats: Dict[str, Any],
    time_tokens_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate impact metrics when switching from time2vec to time_tokens

    Args:
        time2vec_stats: Statistics from time2vec encoding
        time_tokens_stats: Statistics from time_tokens encoding

    Returns:
        Dictionary containing impact metrics
    """
    metrics = {}

    # Vocabulary impact
    if "vocabulary" in both_stats(time2vec_stats, time_tokens_stats):
        t2v_vocab_size = time2vec_stats["vocabulary"]["new_vocab_size"]
        tt_vocab_size = time_tokens_stats["vocabulary"]["new_vocab_size"]

        metrics["vocabulary_impact"] = {
            "time2vec_vocab_size": t2v_vocab_size,
            "time_tokens_vocab_size": tt_vocab_size,
            "vocab_size_increase": tt_vocab_size - t2v_vocab_size,
            "vocab_size_percent_increase": ((tt_vocab_size - t2v_vocab_size) / t2v_vocab_size * 100)
        }

    # Sequence length impact
    if "lmdb_sequence_stats" in both_stats(time2vec_stats, time_tokens_stats):
        t2v_mean = time2vec_stats["lmdb_sequence_stats"]["mean"]
        tt_mean = time_tokens_stats["lmdb_sequence_stats"]["mean"]

        metrics["sequence_length_impact"] = {
            "time2vec_mean_length": t2v_mean,
            "time_tokens_mean_length": tt_mean,
            "mean_length_increase": tt_mean - t2v_mean,
            "mean_length_percent_increase": ((tt_mean - t2v_mean) / t2v_mean * 100)
        }

    # Time token insertion metrics
    if "time_token_insertion" in time_tokens_stats:
        insertion_stats = time_tokens_stats["time_token_insertion"]
        if insertion_stats:
            total_events = sum(stat["events_processed"] for stat in insertion_stats)
            total_tokens = sum(stat["time_tokens_added"] for stat in insertion_stats)

            metrics["time_token_insertion"] = {
                "total_events_processed": total_events,
                "total_time_tokens_added": total_tokens,
                "average_tokens_per_event": total_tokens / total_events if total_events > 0 else 0
            }

    return metrics


def both_stats(stats1: Dict, stats2: Dict) -> List[str]:
    """Helper function to find keys present in both statistics dictionaries"""
    return list(set(stats1.keys()) & set(stats2.keys()))


def generate_dataset_report(
    log_dir: Path,
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive report for a dataset

    Args:
        log_dir: Directory containing dataset logs
        output_file: Optional file path to save the report

    Returns:
        Dictionary containing the report
    """
    if output_file is None:
        output_file = log_dir / "dataset_report.json"

    # Load all available statistics
    stats = load_dataset_stats(log_dir)

    # Create report
    report = {
        "timestamp": datetime.now().isoformat(),
        "log_directory": str(log_dir),
        "report_type": "dataset_analysis",
        "statistics": stats,
        "summary": {}
    }

    # Add summary metrics
    if "vocabulary" in stats:
        report["summary"]["vocabulary_size"] = stats["vocabulary"]["new_vocab_size"]
        if "added_tokens_breakdown" in stats["vocabulary"]:
            report["summary"]["time_tokens_added"] = stats["vocabulary"]["added_tokens_breakdown"]

    if "lmdb_sequence_stats" in stats:
        seq_stats = stats["lmdb_sequence_stats"]
        report["summary"]["sequence_length_summary"] = {
            "mean": seq_stats["mean"],
            "median": seq_stats["median"],
            "std": seq_stats["std"],
            "min": seq_stats["min"],
            "max": seq_stats["max"],
            "total_sequences": seq_stats["total_sequences"]
        }

    # Save report
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def compare_datasets(
    dataset1_log_dir: Path,
    dataset2_log_dir: Path,
    dataset1_name: str = "Dataset 1",
    dataset2_name: str = "Dataset 2",
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Compare two datasets and generate a comparison report

    Args:
        dataset1_log_dir: Log directory for first dataset
        dataset2_log_dir: Log directory for second dataset
        dataset1_name: Name for first dataset
        dataset2_name: Name for second dataset
        output_dir: Directory to save comparison results

    Returns:
        Dictionary containing comparison results
    """
    if output_dir is None:
        output_dir = Path("logs/dataset_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load statistics for both datasets
    stats1 = load_dataset_stats(dataset1_log_dir)
    stats2 = load_dataset_stats(dataset2_log_dir)

    # Create comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "comparison_type": "dataset_comparison",
        "datasets": {
            dataset1_name: {
                "log_directory": str(dataset1_log_dir),
                "statistics": stats1
            },
            dataset2_name: {
                "log_directory": str(dataset2_log_dir),
                "statistics": stats2
            }
        },
        "differences": calculate_dataset_differences(stats1, stats2, dataset1_name, dataset2_name)
    }

    # Save comparison
    comparison_file = output_dir / f"dataset_comparison_{dataset1_name}_{dataset2_name}.json"
    comparison_file = Path(str(comparison_file).replace(" ", "_"))
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    return comparison


def calculate_dataset_differences(
    stats1: Dict[str, Any],
    stats2: Dict[str, Any],
    name1: str,
    name2: str
) -> Dict[str, Any]:
    """
    Calculate differences between two datasets

    Args:
        stats1: Statistics from first dataset
        stats2: Statistics from second dataset
        name1: Name of first dataset
        name2: Name of second dataset

    Returns:
        Dictionary containing differences
    """
    differences = {}

    # Compare vocabulary sizes
    if "vocabulary" in both_stats(stats1, stats2):
        vocab1 = stats1["vocabulary"]["new_vocab_size"]
        vocab2 = stats2["vocabulary"]["new_vocab_size"]

        differences["vocabulary"] = {
            f"{name1}_vocab_size": vocab1,
            f"{name2}_vocab_size": vocab2,
            "absolute_difference": abs(vocab2 - vocab1),
            "relative_difference_percent": ((vocab2 - vocab1) / vocab1 * 100) if vocab1 > 0 else 0
        }

    # Compare sequence lengths
    if "lmdb_sequence_stats" in both_stats(stats1, stats2):
        mean1 = stats1["lmdb_sequence_stats"]["mean"]
        mean2 = stats2["lmdb_sequence_stats"]["mean"]

        differences["sequence_length"] = {
            f"{name1}_mean_length": mean1,
            f"{name2}_mean_length": mean2,
            "absolute_difference": abs(mean2 - mean1),
            "relative_difference_percent": ((mean2 - mean1) / mean1 * 100) if mean1 > 0 else 0
        }

    return differences


def analyze_time_encoding_performance(
    time2vec_dir: Optional[Path],
    time_tokens_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze performance implications of different time encoding methods

    Args:
        time2vec_dir: Directory with time2vec logs
        time_tokens_dir: Directory with time_tokens logs
        output_dir: Directory to save analysis

    Returns:
        Performance analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "time_encoding_performance",
        "recommendations": []
    }

    # Load stats
    tt_stats = load_dataset_stats(time_tokens_dir)
    t2v_stats = load_dataset_stats(time2vec_dir) if time2vec_dir else None

    # Analyze sequence length impact
    if t2v_stats and "lmdb_sequence_stats" in both_stats(t2v_stats, tt_stats):
        t2v_mean = t2v_stats["lmdb_sequence_stats"]["mean"]
        tt_mean = tt_stats["lmdb_sequence_stats"]["mean"]
        increase_percent = ((tt_mean - t2v_mean) / t2v_mean * 100)

        analysis["sequence_length_analysis"] = {
            "time2vec_mean": t2v_mean,
            "time_tokens_mean": tt_mean,
            "increase_percent": increase_percent
        }

        # Add recommendations based on increase
        if increase_percent < 5:
            analysis["recommendations"].append("Low sequence length increase - time_tokens encoding is feasible")
        elif increase_percent < 15:
            analysis["recommendations"].append("Moderate sequence length increase - consider memory/compute trade-offs")
        else:
            analysis["recommendations"].append("High sequence length increase - may significantly impact performance")

    # Analyze vocabulary impact
    if t2v_stats and "vocabulary" in both_stats(t2v_stats, tt_stats):
        t2v_vocab = t2v_stats["vocabulary"]["new_vocab_size"]
        tt_vocab = tt_stats["vocabulary"]["new_vocab_size"]
        vocab_increase = tt_vocab - t2v_vocab

        analysis["vocabulary_analysis"] = {
            "time2vec_vocab_size": t2v_vocab,
            "time_tokens_vocab_size": tt_vocab,
            "tokens_added": vocab_increase
        }

        if vocab_increase > 0:
            analysis["recommendations"].append(f"Vocabulary increased by {vocab_increase} tokens - consider embedding memory impact")

    # Save analysis
    analysis_file = output_dir / "time_encoding_performance_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


def analyze_time_tokens_per_sequence(log_dir: Path) -> Dict[str, Any]:
    """
    Analyze how many time tokens are added per sequence on average

    Args:
        log_dir: Directory containing time_tokens encoding logs

    Returns:
        Dictionary containing detailed time token per sequence analysis
    """
    stats = load_dataset_stats(log_dir)

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "time_tokens_per_sequence",
        "log_directory": str(log_dir)
    }

    # Get detailed time token stats
    if "time_token_detailed" in stats:
        detailed_stats = stats["time_token_detailed"][-1]["stats"]  # Get most recent

        analysis["per_sequence_metrics"] = {
            "average_tokens_added_per_sequence": detailed_stats.get("average_tokens_per_sequence", 0),
            "average_length_increase_per_sequence": detailed_stats.get("average_length_increase_per_sequence", 0),
            "total_sequences_processed": detailed_stats.get("total_sequences", 0),
            "total_tokens_added": detailed_stats.get("tokens_added", 0),
            "total_events_processed": detailed_stats.get("events_processed", 0)
        }

        # Length increase distribution
        if "length_increase_details" in detailed_stats:
            analysis["length_increase_distribution"] = detailed_stats["length_increase_details"]

    # Calculate from basic stats if detailed not available
    elif "time_token_insertion" in stats:
        insertion_stats = stats["time_token_insertion"]
        if insertion_stats:
            latest_stats = insertion_stats[-1]

            analysis["per_sequence_metrics"] = {
                "average_tokens_added_per_sequence": latest_stats.get("tokens_per_event", 0),
                "total_events_processed": latest_stats.get("events_processed", 0),
                "total_tokens_added": latest_stats.get("time_tokens_added", 0)
            }

    # Add theoretical expectation
    analysis["theoretical_expectation"] = {
        "expected_tokens_per_event": 2,  # year + age
        "explanation": "Each event should get exactly 2 time tokens: 1 year token + 1 age token"
    }

    # Calculate efficiency metrics
    if "per_sequence_metrics" in analysis:
        metrics = analysis["per_sequence_metrics"]
        expected = 2.0
        actual = metrics.get("average_tokens_added_per_sequence", 0)

        analysis["efficiency_metrics"] = {
            "expected_tokens_per_sequence": expected,
            "actual_tokens_per_sequence": actual,
            "efficiency_ratio": actual / expected if expected > 0 else 0,
            "is_performing_as_expected": abs(actual - expected) < 0.1  # Within 10% tolerance
        }

    return analysis