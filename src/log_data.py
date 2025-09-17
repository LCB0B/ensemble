"""
Data creation logging utilities for tracking sequence lengths, vocabulary changes,
and time encoding comparisons during dataset processing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


def create_data_creation_directory(dataset_name: str) -> Path:
    """
    Create logs/data_creation/{dataset_name}/ directory structure

    Args:
        dataset_name: Name of the dataset being processed

    Returns:
        Path to the created log directory
    """
    log_dir = Path("logs") / "data_creation" / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_data_creation_logger(log_dir: Path, dataset_name: str) -> logging.Logger:
    """
    Setup logging for data creation process

    Args:
        log_dir: Directory where logs will be saved
        dataset_name: Name of the dataset being processed

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"data_creation_{dataset_name}")
    logger.setLevel(logging.INFO)

    # Create file handler
    log_file = log_dir / "data_creation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger if not already added
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


def log_sequence_length_stats(stats: Dict[str, Any], log_dir: Path, filename: str = "sequence_length_analysis.json"):
    """
    Save sequence length statistics to JSON

    Args:
        stats: Dictionary containing sequence length statistics
        log_dir: Directory where logs will be saved
        filename: Name of the output file
    """
    log_file = log_dir / filename

    # Add timestamp to stats
    stats_with_timestamp = {
        "timestamp": datetime.now().isoformat(),
        **stats
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(stats_with_timestamp, f, indent=2, default=str)


def log_vocabulary_changes(
    original_size: int,
    new_size: int,
    added_tokens: Dict[str, int],
    log_dir: Path
):
    """
    Log vocabulary size changes when adding time tokens

    Args:
        original_size: Original vocabulary size
        new_size: New vocabulary size after adding time tokens
        added_tokens: Dictionary of token type -> count added
        log_dir: Directory where logs will be saved
    """
    vocab_stats = {
        "timestamp": datetime.now().isoformat(),
        "original_vocab_size": original_size,
        "new_vocab_size": new_size,
        "total_tokens_added": new_size - original_size,
        "added_tokens_breakdown": added_tokens,
        "percent_increase": ((new_size - original_size) / original_size) * 100
    }

    log_file = log_dir / "vocabulary_stats.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(vocab_stats, f, indent=2)


def calculate_sequence_length_statistics(lengths: List[int]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for sequence lengths

    Args:
        lengths: List of sequence lengths

    Returns:
        Dictionary containing various statistics
    """
    if not lengths:
        return {}

    lengths_array = np.array(lengths)

    return {
        "count": len(lengths),
        "mean": float(np.mean(lengths_array)),
        "median": float(np.median(lengths_array)),
        "std": float(np.std(lengths_array)),
        "min": int(np.min(lengths_array)),
        "max": int(np.max(lengths_array)),
        "q25": float(np.percentile(lengths_array, 25)),
        "q75": float(np.percentile(lengths_array, 75)),
        "q90": float(np.percentile(lengths_array, 90)),
        "q95": float(np.percentile(lengths_array, 95)),
        "q99": float(np.percentile(lengths_array, 99))
    }


def log_sequence_length_comparison(
    original_lengths: List[int],
    new_lengths: List[int],
    log_dir: Path,
    phase: str = "batch"
):
    """
    Log comparison between original and new sequence lengths

    Args:
        original_lengths: List of original sequence lengths
        new_lengths: List of new sequence lengths (with time tokens)
        log_dir: Directory where logs will be saved
        phase: Processing phase (e.g., "batch", "final")
    """
    original_stats = calculate_sequence_length_statistics(original_lengths)
    new_stats = calculate_sequence_length_statistics(new_lengths)

    if original_lengths and new_lengths:
        increases = [new - orig for orig, new in zip(original_lengths, new_lengths)]
        increase_stats = calculate_sequence_length_statistics(increases)
    else:
        increase_stats = {}

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "original_stats": original_stats,
        "new_stats": new_stats,
        "increase_stats": increase_stats,
        "samples_processed": len(original_lengths)
    }

    # Append to existing file or create new one
    log_file = log_dir / f"sequence_length_comparison_{phase}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(comparison, default=str) + "\n")


def compare_time_encoding_modes(
    time2vec_stats: Optional[Dict[str, Any]],
    time_tokens_stats: Dict[str, Any],
    log_dir: Path
):
    """
    Generate comparison report between time2vec and time_tokens encoding modes

    Args:
        time2vec_stats: Statistics from time2vec encoding (None if not available)
        time_tokens_stats: Statistics from time_tokens encoding
        log_dir: Directory where logs will be saved
    """
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "encoding_modes": {
            "time2vec": time2vec_stats,
            "time_tokens": time_tokens_stats
        }
    }

    if time2vec_stats and time_tokens_stats:
        # Calculate differences if both are available
        if "sequence_stats" in time2vec_stats and "sequence_stats" in time_tokens_stats:
            t2v_mean = time2vec_stats["sequence_stats"].get("mean", 0)
            tt_mean = time_tokens_stats["sequence_stats"].get("mean", 0)

            comparison["comparison_metrics"] = {
                "mean_length_increase": tt_mean - t2v_mean,
                "percent_length_increase": ((tt_mean - t2v_mean) / t2v_mean * 100) if t2v_mean > 0 else 0
            }

    log_file = log_dir / "time_encoding_comparison.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)


def log_processing_phase(
    phase_name: str,
    stats: Dict[str, Any],
    log_dir: Path,
    logger: Optional[logging.Logger] = None
):
    """
    Log statistics for a specific processing phase

    Args:
        phase_name: Name of the processing phase
        stats: Statistics dictionary for this phase
        log_dir: Directory where logs will be saved
        logger: Optional logger instance
    """
    if logger:
        logger.info(f"Completed {phase_name}: {stats}")

    phase_data = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase_name,
        "stats": stats
    }

    # Append to processing log
    log_file = log_dir / "processing_phases.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(phase_data, default=str) + "\n")


def summarize_data_creation(log_dir: Path) -> Dict[str, Any]:
    """
    Create a summary of the entire data creation process

    Args:
        log_dir: Directory containing all log files

    Returns:
        Summary dictionary
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "log_directory": str(log_dir),
        "files_created": []
    }

    # List all log files created
    for log_file in log_dir.glob("*.json*"):
        summary["files_created"].append(str(log_file.name))

    # Try to read key statistics
    vocab_file = log_dir / "vocabulary_stats.json"
    if vocab_file.exists():
        with open(vocab_file, "r", encoding="utf-8") as f:
            summary["vocabulary_summary"] = json.load(f)

    seq_file = log_dir / "sequence_length_analysis.json"
    if seq_file.exists():
        with open(seq_file, "r", encoding="utf-8") as f:
            summary["sequence_length_summary"] = json.load(f)

    # Save summary
    summary_file = log_dir / "data_creation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def log_time_token_insertion_stats(
    events_processed: int,
    tokens_added: int,
    log_dir: Path,
    batch_id: Optional[str] = None
):
    """
    Log statistics about time token insertion

    Args:
        events_processed: Number of events processed
        tokens_added: Total number of time tokens added
        log_dir: Directory where logs will be saved
        batch_id: Optional batch identifier
    """
    stats = {
        "timestamp": datetime.now().isoformat(),
        "batch_id": batch_id,
        "events_processed": events_processed,
        "time_tokens_added": tokens_added,
        "tokens_per_event": tokens_added / events_processed if events_processed > 0 else 0
    }

    # Append to time token insertion log
    log_file = log_dir / "time_token_insertion.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats, default=str) + "\n")