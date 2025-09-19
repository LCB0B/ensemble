#!/usr/bin/env python3
"""
Plot training loss from TensorBoard logs for transformer experiments.

Usage:
    python scripts/plot_training_loss.py EXPERIMENT_NAME
    python scripts/plot_training_loss.py --list  # List available experiments
    python scripts/plot_training_loss.py --compare EXP1 EXP2 EXP3  # Compare multiple experiments

Examples:
    python scripts/plot_training_loss.py 001_shrewd_otter-pretrain-lr0.0003
    python scripts/plot_training_loss.py --compare 001_shrewd_otter 002_scary_chameleon
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_experiments(log_dir: Path, pattern: Optional[str] = None) -> List[str]:
    """Find all available experiments in the log directory."""
    if not log_dir.exists():
        return []

    experiments = []
    for exp_dir in log_dir.iterdir():
        if exp_dir.is_dir():
            if pattern is None or pattern.lower() in exp_dir.name.lower():
                experiments.append(exp_dir.name)

    return sorted(experiments)


def load_tensorboard_logs(log_path: Path, verbose: bool = False) -> Dict[str, List[Tuple[int, float]]]:
    """Load scalar data from TensorBoard logs."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log path does not exist: {log_path}")

    # Find tfevents files
    tfevents_files = list(log_path.glob("*.tfevents*"))
    if not tfevents_files:
        raise FileNotFoundError(f"No tfevents files found in {log_path}")

    if verbose:
        print(f"Found {len(tfevents_files)} tfevents files:")
        for f in tfevents_files:
            print(f"  {f.name}")

    # Load data from all tfevents files
    all_scalars = {}

    for tfevents_file in tfevents_files:
        try:
            ea = EventAccumulator(str(tfevents_file))
            ea.Reload()

            # Get available scalar tags
            scalar_tags = ea.Tags()['scalars']

            if verbose:
                print(f"\nFile {tfevents_file.name} contains metrics:")
                for tag in scalar_tags:
                    scalar_events = ea.Scalars(tag)
                    print(f"  {tag}: {len(scalar_events)} data points")

            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                if tag not in all_scalars:
                    all_scalars[tag] = []

                for event in scalar_events:
                    all_scalars[tag].append((event.step, event.value))

        except Exception as e:
            print(f"Warning: Could not load {tfevents_file}: {e}")
            continue

    # Sort by step for each metric and remove duplicates
    for tag in all_scalars:
        # Remove duplicates by converting to dict and back
        unique_points = {}
        for step, value in all_scalars[tag]:
            unique_points[step] = value  # Later values overwrite earlier ones for same step
        all_scalars[tag] = sorted(unique_points.items())

    if verbose:
        print(f"\nCombined metrics across all files:")
        for tag in sorted(all_scalars.keys()):
            print(f"  {tag}: {len(all_scalars[tag])} data points")

    return all_scalars


def smooth_curve(steps: List[int], values: List[float], smoothing: float = 0.9) -> Tuple[List[int], List[float]]:
    """Apply exponential smoothing to the curve."""
    if not values:
        return steps, values

    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed_val = smoothing * smoothed[-1] + (1 - smoothing) * values[i]
        smoothed.append(smoothed_val)

    return steps, smoothed


def plot_single_experiment(experiment_name: str, log_dir: Path,
                          metrics: List[str] = None,
                          smoothing: float = 0.9,
                          save_path: Optional[Path] = None,
                          show_available: bool = False,
                          verbose: bool = False) -> None:
    """Plot training curves for a single experiment."""
    log_path = log_dir / experiment_name
    print(f"Loading logs from: {log_path}")

    try:
        scalars = load_tensorboard_logs(log_path, verbose=verbose)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Show available metrics if requested
    if show_available:
        print(f"\nAvailable metrics in {experiment_name}:")
        for metric in sorted(scalars.keys()):
            num_points = len(scalars[metric])
            print(f"  {metric} ({num_points} data points)")
        return

    # Auto-detect metrics if none specified
    if metrics is None:
        # Try common training metrics in order of preference
        candidate_metrics = [
            'train/loss', 'val/loss', 'train_loss', 'val_loss',
            'loss', 'validation_loss', 'training_loss'
        ]
        metrics = []

        # Add loss metrics that exist (prioritize training loss)
        for candidate in candidate_metrics:
            if candidate in scalars:
                metrics.append(candidate)
                # Always include both train and val loss if available
                if candidate == 'train/loss' and 'val/loss' in scalars:
                    if 'val/loss' not in metrics:
                        metrics.append('val/loss')
                elif candidate == 'val/loss' and 'train/loss' in scalars:
                    if 'train/loss' not in metrics:
                        metrics.insert(0, 'train/loss')  # Insert at beginning to prioritize

        # Add accuracy/MLM metrics if available and we have loss metrics
        if metrics:  # Only add other metrics if we have loss metrics
            for metric in sorted(scalars.keys()):
                if any(keyword in metric.lower() for keyword in ['acc', 'mlm', 'top']):
                    if metric not in metrics:
                        metrics.append(metric)

        # If still no loss metrics, use all available
        if not metrics:
            metrics = list(scalars.keys())

        if verbose:
            print(f"Auto-detected metrics: {metrics}")

    # Filter metrics that exist in the logs
    available_metrics = [m for m in metrics if m in scalars]
    missing_metrics = [m for m in metrics if m not in scalars]

    if missing_metrics:
        print(f"Warning: Metrics not found in logs: {missing_metrics}")

    if not available_metrics:
        print("Error: No requested metrics found in logs")
        print(f"Available metrics: {sorted(scalars.keys())}")
        print("\nTip: Use --show-metrics to see all available metrics first")
        return

    # Set up the plot
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(available_metrics)))

    for i, metric in enumerate(available_metrics):
        steps, values = zip(*scalars[metric])

        # Plot raw data (lighter)
        ax.plot(steps, values, alpha=0.3, color=colors[i], linewidth=0.5)

        # Plot smoothed data
        if smoothing > 0:
            smooth_steps, smooth_values = smooth_curve(list(steps), list(values), smoothing)
            ax.plot(smooth_steps, smooth_values, label=metric, color=colors[i], linewidth=2)
        else:
            ax.plot(steps, values, label=metric, color=colors[i], linewidth=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Curves: {experiment_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale for y-axis if losses are large
    if any('loss' in m.lower() for m in available_metrics):
        min_loss = min([min(dict(scalars[m]).values()) for m in available_metrics if 'loss' in m.lower()])
        if min_loss > 1:
            ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_comparison(experiment_names: List[str], log_dir: Path,
                   metric: str = 'train/loss',
                   smoothing: float = 0.9,
                   save_path: Optional[Path] = None) -> None:
    """Plot comparison of multiple experiments."""
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_names)))

    for i, exp_name in enumerate(experiment_names):
        log_path = log_dir / exp_name

        try:
            scalars = load_tensorboard_logs(log_path)

            if metric not in scalars:
                print(f"Warning: Metric '{metric}' not found in {exp_name}")
                print(f"Available metrics: {list(scalars.keys())}")
                continue

            steps, values = zip(*scalars[metric])

            # Plot raw data (lighter)
            ax.plot(steps, values, alpha=0.2, color=colors[i], linewidth=0.5)

            # Plot smoothed data
            if smoothing > 0:
                smooth_steps, smooth_values = smooth_curve(list(steps), list(values), smoothing)
                ax.plot(smooth_steps, smooth_values, label=exp_name, color=colors[i], linewidth=2)
            else:
                ax.plot(steps, values, label=exp_name, color=colors[i], linewidth=2)

        except Exception as e:
            print(f"Error loading {exp_name}: {e}")
            continue

    ax.set_xlabel('Training Step')
    ax.set_ylabel(metric)
    ax.set_title(f'Training Comparison: {metric}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Log scale for y-axis if losses are large
    if 'loss' in metric.lower():
        try:
            y_min, y_max = ax.get_ylim()
            if y_min > 1:
                ax.set_yscale('log')
        except:
            pass

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot training loss from TensorBoard logs')
    parser.add_argument('experiment', nargs='?', help='Experiment name to plot')
    parser.add_argument('--log-dir', type=str, default='logs/transformer_logs/destiny',
                       help='Directory containing experiment logs')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    parser.add_argument('--compare', nargs='+', help='Compare multiple experiments')
    parser.add_argument('--metric', type=str, default='train/loss',
                       help='Metric to plot (default: train/loss)')
    parser.add_argument('--metrics', nargs='+', default=None,
                       help='Metrics to plot for single experiment (auto-detected if not specified)')
    parser.add_argument('--smoothing', type=float, default=0.9,
                       help='Smoothing factor (0=no smoothing, 0.9=default)')
    parser.add_argument('--save', type=str, help='Save plot to file')
    parser.add_argument('--pattern', type=str, help='Filter experiments by pattern')
    parser.add_argument('--show-metrics', action='store_true',
                       help='Show available metrics for the experiment')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed debugging information about log file loading')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    # List available experiments
    if args.list:
        experiments = find_experiments(log_dir, args.pattern)
        if experiments:
            print(f"Available experiments in {log_dir}:")
            for exp in experiments:
                print(f"  {exp}")
        else:
            print(f"No experiments found in {log_dir}")
        return

    # Compare multiple experiments
    if args.compare:
        # Expand experiment names if they're partial matches
        all_experiments = find_experiments(log_dir)
        expanded_names = []

        for exp_pattern in args.compare:
            matches = [exp for exp in all_experiments if exp_pattern.lower() in exp.lower()]
            if matches:
                expanded_names.extend(matches)
            else:
                print(f"Warning: No experiments found matching '{exp_pattern}'")

        if expanded_names:
            save_path = Path(args.save) if args.save else None
            plot_comparison(expanded_names, log_dir, args.metric, args.smoothing, save_path)
        else:
            print("No valid experiments found for comparison")
        return

    # Plot single experiment
    if not args.experiment:
        print("Error: Please provide an experiment name or use --list to see available experiments")
        return

    # Check if experiment exists or try to find partial match
    all_experiments = find_experiments(log_dir)
    experiment_name = args.experiment

    if experiment_name not in all_experiments:
        matches = [exp for exp in all_experiments if experiment_name.lower() in exp.lower()]
        if matches:
            if len(matches) == 1:
                experiment_name = matches[0]
                print(f"Found matching experiment: {experiment_name}")
            else:
                print(f"Multiple experiments found matching '{args.experiment}':")
                for match in matches:
                    print(f"  {match}")
                print("Please specify the full experiment name.")
                return
        else:
            print(f"Experiment '{args.experiment}' not found")
            print("Available experiments:")
            for exp in all_experiments:
                print(f"  {exp}")
            return

    save_path = Path(args.save) if args.save else None

    # Show available metrics if requested
    if args.show_metrics:
        plot_single_experiment(experiment_name, log_dir, show_available=True, verbose=args.verbose)
        return

    plot_single_experiment(experiment_name, log_dir, args.metrics, args.smoothing, save_path, verbose=args.verbose)


if __name__ == "__main__":
    main()