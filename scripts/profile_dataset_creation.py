"""
Profile dataset creation pipeline for memory and time usage.

This script profiles the complete dataset creation process including:
- Event tokenization and processing
- Time token insertion
- LMDB database creation
- Sorting and grouping operations

Usage:
    python scripts/profile_dataset_creation.py

Output:
    - Console report with nested timing and memory usage
    - JSON file with detailed profiling data
    - Summary statistics
"""

from pathlib import Path
import yaml
import polars as pl
import pyarrow.dataset as ds
from src.profiler import profiler
from src.datamodule2 import PretrainDataModule
from src.paths import FPATH, check_and_copy_file_or_dir


def main():
    """Profile dataset creation with a test configuration"""

    print("="*80)
    print("DATASET CREATION PROFILING")
    print("="*80)
    print("\nThis will create a small test dataset and profile:")
    print("  - Memory usage (start, end, delta, peak)")
    print("  - Execution time for each operation")
    print("  - Nested call hierarchy")
    print()

    # Load hparams from existing config
    with open(
        FPATH.CONFIGS / "destiny" / "hparams_destiny_pretrain.yaml",
        "r",
        encoding="utf-8",
    ) as stream:
        hparams = yaml.safe_load(stream)

    # Override with test settings
    test_config = {
        "dir_path": FPATH.DATA / "destiny_profile_test",
        "time_encoding": "time_tokens",  # or "time2vec"
        "pretrain_style": hparams.get("pretrain_style", "CLM"),
        "n_tokens": 1e5,  # Smaller for testing
        "max_seq_len": hparams.get("max_seq_len", 512),
    }

    print("Configuration:")
    print(f"  dir_path: {test_config['dir_path']}")
    print(f"  time_encoding: {test_config['time_encoding']}")
    print(f"  pretrain_style: {test_config['pretrain_style']}")
    print(f"  n_tokens: {test_config['n_tokens']}")
    print()

    # Load data sources
    print("Loading data sources...")
    source_paths = [
        (FPATH.DATA / hparams["source_dir"] / path).with_suffix(".parquet")
        for path in hparams["sources"]
    ]
    background_path = (
        FPATH.DATA / hparams["source_dir"] / hparams["background"]
    ).with_suffix(".parquet")

    for s in source_paths + [background_path]:
        check_and_copy_file_or_dir(s, verbosity=0)

    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    print(f"Loaded {len(sources)} sources and background data\n")

    # Create PretrainDataModule instance
    with profiler.profile("full_dataset_creation"):
        dm = PretrainDataModule(
            dir_path=test_config["dir_path"],
            sources=sources,
            background=background,
            subset_background=hparams["subset_background"],
            n_tokens=test_config["n_tokens"],
            lengths=hparams["lengths"],
            num_workers=0,  # Use 0 for profiling
            max_seq_len=test_config["max_seq_len"],
            source_dir=hparams["source_dir"],
            pretrain_style=test_config["pretrain_style"],
            time_encoding=test_config["time_encoding"],
            masking_ratio=hparams.get("masking_ratio", 0.15),
        )

        # Prepare data triggers the full pipeline
        dm.prepare_data()

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)

    # Print detailed report
    profiler.print_report(min_duration=0.1)  # Show operations > 0.1 seconds

    # Save to JSON for later analysis
    output_file = "profiling_results.json"
    profiler.save_json(output_file)

    # Print summary statistics
    summary = profiler.get_summary()
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total Time:          {summary['total_time_seconds']:>12.2f} seconds")
    print(f"Peak Memory:         {summary['peak_memory_gb']:>12.3f} GB")
    print(f"Memory Allocated:    {summary['total_memory_allocated_gb']:>12.3f} GB")
    print(f"Memory Freed:        {summary['total_memory_freed_gb']:>12.3f} GB")
    print(f"Profiled Blocks:     {summary['num_profiled_blocks']:>12d}")
    print("="*80)

    # Print top bottlenecks
    print("\n" + "="*80)
    print("TOP 5 TIME BOTTLENECKS")
    print("="*80)
    hotspots_time = profiler.get_hotspots(top_n=5, by="time")
    for i, entry in enumerate(hotspots_time, 1):
        print(f"{i}. {entry.name}")
        print(f"   Time: {entry.duration_seconds:.2f}s")
        print(f"   Memory Peak: {entry.memory_peak_gb:.3f} GB")
        print()

    print("="*80)
    print("TOP 5 MEMORY BOTTLENECKS")
    print("="*80)
    hotspots_memory = profiler.get_hotspots(top_n=5, by="memory")
    for i, entry in enumerate(hotspots_memory, 1):
        print(f"{i}. {entry.name}")
        print(f"   Memory Peak: {entry.memory_peak_gb:.3f} GB")
        print(f"   Time: {entry.duration_seconds:.2f}s")
        print()

    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. Check the operations with highest memory peaks")
    print("2. Look for operations with large memory deltas (allocations)")
    print("3. Identify long-running operations that could be optimized")
    print("4. Review the nested structure to understand call hierarchy")
    print("\nFor detailed analysis, review: " + output_file)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
