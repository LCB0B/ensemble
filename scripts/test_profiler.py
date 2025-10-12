"""
Simple test script to verify the profiler works correctly.

Usage:
    python scripts/test_profiler.py
"""

import time
import numpy as np
from src.profiler import profiler


def allocate_memory(size_mb: int):
    """Allocate memory for testing"""
    # Allocate size_mb megabytes of memory
    arr = np.zeros((size_mb * 1024 * 1024 // 8,), dtype=np.float64)
    time.sleep(0.1)  # Simulate work
    return arr


def slow_function():
    """Simulate a slow function"""
    time.sleep(0.5)
    return "done"


def nested_operations():
    """Test nested profiling"""
    with profiler.profile("nested_operations"):
        with profiler.profile("operation_1"):
            arr1 = allocate_memory(100)  # 100 MB

        with profiler.profile("operation_2"):
            arr2 = allocate_memory(200)  # 200 MB
            result = slow_function()

        with profiler.profile("operation_3"):
            # This should show memory freed
            del arr1
            del arr2
            time.sleep(0.2)


def main():
    print("="*80)
    print("PROFILER TEST")
    print("="*80)
    print("\nRunning test operations...\n")

    with profiler.profile("test_main"):
        nested_operations()

    # Print report
    profiler.print_report()

    # Save to JSON
    profiler.save_json("test_profiler_results.json")

    # Get summary
    summary = profiler.get_summary()
    print("\n📊 Test Summary:")
    print(f"   Total Time: {summary['total_time_seconds']:.2f}s")
    print(f"   Peak Memory: {summary['peak_memory_gb']:.3f} GB")
    print(f"   Total Allocated: {summary['total_memory_allocated_gb']:.3f} GB")
    print(f"   Total Freed: {summary['total_memory_freed_gb']:.3f} GB")

    print("\n✅ Profiler test complete!")


if __name__ == "__main__":
    main()
