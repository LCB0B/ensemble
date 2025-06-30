#!/usr/bin/env python3
"""
Analysis script for generated life sequences.
Provides basic event counting, statistics, and visualization.
"""

import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional
import argparse

# Import project paths
import sys
sys.path.append('.')
try:
    from src.paths import FPATH
except ImportError:
    print("Warning: Could not import FPATH, using relative paths")
    class FakeFPATH:
        DATA = Path("data")
    FPATH = FakeFPATH()


class SequenceAnalyzer:
    """Analyzes generated life sequences."""
    
    def __init__(self, generation_dir: Path):
        """
        Initialize analyzer with a generation directory.
        
        Args:
            generation_dir: Path to directory containing generated sequences
        """
        self.generation_dir = Path(generation_dir)
        self.vocab = None
        self.sequences = None
        self.prompts = None
        self.original_batches = None
        self.metadata = None
        
        # Load data
        self._load_generation_data()
        self._load_vocabulary()
    
    def _load_generation_data(self):
        """Load generated sequences and metadata."""
        print(f"Loading data from: {self.generation_dir}")
        
        # Load metadata
        metadata_path = self.generation_dir / "generation_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Metadata loaded: {self.metadata}")
        
        # Load sequences
        sequences_path = self.generation_dir / "generated_sequences.pkl"
        if sequences_path.exists():
            with open(sequences_path, 'rb') as f:
                self.sequences = pickle.load(f)
            print(f"Generated sequences loaded: {len(self.sequences)} batches")
        else:
            print("Warning: No generated_sequences.pkl found")
            return
        
        # Load prompts
        prompts_path = self.generation_dir / "prompts.pkl"
        if prompts_path.exists():
            with open(prompts_path, 'rb') as f:
                self.prompts = pickle.load(f)
            print(f"Prompts loaded: {len(self.prompts)} batches")
        
        # Load original batches
        batches_path = self.generation_dir / "original_batches.pkl"
        if batches_path.exists():
            with open(batches_path, 'rb') as f:
                self.original_batches = pickle.load(f)
            print(f"Original batches loaded: {len(self.original_batches)} batches")
    
    def _load_vocabulary(self):
        """Load vocabulary for token decoding."""
        # Try to find vocabulary file
        vocab_paths = [
            FPATH.DATA / "life_all_compiled" / "vocab.json",
            FPATH.DATA / "life_test_compiled" / "vocab.json"
        ]
        
        for vocab_path in vocab_paths:
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    self.vocab = json.load(f)
                print(f"Vocabulary loaded from: {vocab_path}")
                print(f"Vocabulary size: {len(self.vocab)}")
                break
        
        if self.vocab is None:
            print("Warning: No vocabulary file found. Token decoding will not be available.")
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the generated sequences."""
        if self.sequences is None:
            return {"error": "No sequences loaded"}
        
        stats = {}
        
        # Collect all sequences
        all_sequences = []
        for batch in self.sequences:
            if isinstance(batch, torch.Tensor):
                all_sequences.append(batch)
            elif isinstance(batch, list):
                all_sequences.extend(batch)
        
        if not all_sequences:
            return {"error": "No valid sequences found"}
        
        # Stack all sequences
        if isinstance(all_sequences[0], torch.Tensor):
            stacked = torch.cat(all_sequences, dim=0)
        else:
            stacked = torch.stack(all_sequences, dim=0)
        
        print(f"Stacked sequences shape: {stacked.shape}")
        
        # Basic shape statistics
        stats['total_sequences'] = stacked.shape[0]
        if len(stacked.shape) == 3:  # [batch, simulations, seq_len]
            stats['simulations_per_batch'] = stacked.shape[1]
            stats['sequence_length'] = stacked.shape[2]
            # Reshape to [total_sequences, seq_len]
            sequences_2d = stacked.view(-1, stacked.shape[-1])
        else:  # [total_sequences, seq_len]
            stats['sequence_length'] = stacked.shape[1]
            sequences_2d = stacked
        
        # Event statistics
        stats['unique_tokens'] = len(torch.unique(sequences_2d))
        stats['total_events'] = sequences_2d.numel()
        
        # Non-zero (non-padding) statistics
        non_zero_mask = sequences_2d != 0
        stats['total_non_padding_events'] = non_zero_mask.sum().item()
        stats['avg_events_per_sequence'] = stats['total_non_padding_events'] / sequences_2d.shape[0]
        
        # Token frequency
        token_counts = Counter(sequences_2d.flatten().tolist())
        stats['most_common_tokens'] = token_counts.most_common(10)
        
        return stats
    
    def count_events(self) -> pd.DataFrame:
        """Count event frequencies across all generated sequences."""
        if self.sequences is None:
            print("No sequences to analyze")
            return pd.DataFrame()
        
        # Collect all sequences into a single tensor
        all_sequences = []
        for batch in self.sequences:
            if isinstance(batch, torch.Tensor):
                all_sequences.append(batch)
        
        if not all_sequences:
            print("No valid sequences found")
            return pd.DataFrame()
        
        # Stack and flatten
        stacked = torch.cat(all_sequences, dim=0)
        if len(stacked.shape) == 3:
            sequences_flat = stacked.view(-1, stacked.shape[-1])
        else:
            sequences_flat = stacked
        
        # Count token frequencies
        all_tokens = sequences_flat.flatten()
        token_counts = Counter(all_tokens.tolist())
        
        # Remove padding tokens (0)
        if 0 in token_counts:
            del token_counts[0]
        
        # Create DataFrame
        count_data = []
        for token_id, count in token_counts.most_common():
            row = {
                'token_id': token_id,
                'count': count,
                'frequency': count / len(all_tokens.tolist())
            }
            
            # Add decoded token if vocabulary is available
            if self.vocab:
                # Vocabulary is typically {token: id}, so we need to reverse it
                id_to_token = {v: k for k, v in self.vocab.items()}
                row['event_name'] = id_to_token.get(token_id, f"UNK_{token_id}")
            else:
                row['event_name'] = f"token_{token_id}"
            
            count_data.append(row)
        
        df = pd.DataFrame(count_data)
        return df
    
    def analyze_sequence_lengths(self) -> Dict[str, Any]:
        """Analyze the distribution of non-padding sequence lengths."""
        if self.sequences is None:
            return {}
        
        # Collect all sequences
        all_sequences = []
        for batch in self.sequences:
            if isinstance(batch, torch.Tensor):
                all_sequences.append(batch)
        
        stacked = torch.cat(all_sequences, dim=0)
        if len(stacked.shape) == 3:
            sequences_flat = stacked.view(-1, stacked.shape[-1])
        else:
            sequences_flat = stacked
        
        # Calculate non-padding lengths for each sequence
        lengths = []
        for seq in sequences_flat:
            non_zero_count = (seq != 0).sum().item()
            lengths.append(non_zero_count)
        
        lengths_array = np.array(lengths)
        
        return {
            'mean_length': float(lengths_array.mean()),
            'std_length': float(lengths_array.std()),
            'min_length': int(lengths_array.min()),
            'max_length': int(lengths_array.max()),
            'median_length': float(np.median(lengths_array)),
            'percentiles': {
                '25th': float(np.percentile(lengths_array, 25)),
                '75th': float(np.percentile(lengths_array, 75)),
                '90th': float(np.percentile(lengths_array, 90)),
                '95th': float(np.percentile(lengths_array, 95))
            }
        }
    
    def plot_event_distribution(self, top_n: int = 20, save_path: Optional[Path] = None):
        """Plot the distribution of most common events."""
        event_counts = self.count_events()
        
        if event_counts.empty:
            print("No events to plot")
            return
        
        # Plot top N events
        top_events = event_counts.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(top_events)), top_events['count'])
        plt.xlabel('Events')
        plt.ylabel('Count')
        plt.title(f'Top {top_n} Most Common Events in Generated Sequences')
        plt.xticks(range(len(top_events)), top_events['event_name'], rotation=45, ha='right')
        
        # Add count labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_sequence_length_distribution(self, save_path: Optional[Path] = None):
        """Plot the distribution of sequence lengths."""
        length_stats = self.analyze_sequence_lengths()
        
        if not length_stats:
            print("No sequence length data to plot")
            return
        
        # Get individual sequence lengths for histogram
        all_sequences = []
        for batch in self.sequences:
            if isinstance(batch, torch.Tensor):
                all_sequences.append(batch)
        
        stacked = torch.cat(all_sequences, dim=0)
        if len(stacked.shape) == 3:
            sequences_flat = stacked.view(-1, stacked.shape[-1])
        else:
            sequences_flat = stacked
        
        lengths = [(seq != 0).sum().item() for seq in sequences_flat]
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(length_stats['mean_length'], color='red', linestyle='--', 
                   label=f"Mean: {length_stats['mean_length']:.1f}")
        plt.axvline(length_stats['median_length'], color='green', linestyle='--', 
                   label=f"Median: {length_stats['median_length']:.1f}")
        
        plt.xlabel('Sequence Length (non-padding tokens)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Generated Sequence Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 60)
        report.append("GENERATED LIFE SEQUENCES ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        stats = self.get_basic_stats()
        report.append("BASIC STATISTICS:")
        for key, value in stats.items():
            if key != 'most_common_tokens':
                report.append(f"  {key}: {value}")
        report.append("")
        
        # Sequence length analysis
        length_stats = self.analyze_sequence_lengths()
        if length_stats:
            report.append("SEQUENCE LENGTH ANALYSIS:")
            report.append(f"  Mean length: {length_stats['mean_length']:.2f}")
            report.append(f"  Std deviation: {length_stats['std_length']:.2f}")
            report.append(f"  Min/Max length: {length_stats['min_length']}/{length_stats['max_length']}")
            report.append(f"  Median length: {length_stats['median_length']:.2f}")
            report.append("  Percentiles:")
            for pct, value in length_stats['percentiles'].items():
                report.append(f"    {pct}: {value:.2f}")
            report.append("")
        
        # Event frequency analysis
        event_counts = self.count_events()
        if not event_counts.empty:
            report.append("TOP 15 MOST COMMON EVENTS:")
            for _, row in event_counts.head(15).iterrows():
                report.append(f"  {row['event_name']}: {row['count']} ({row['frequency']:.4f})")
            report.append("")
        
        # Metadata
        if self.metadata:
            report.append("GENERATION METADATA:")
            for key, value in self.metadata.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze generated life sequences")
    parser.add_argument("--generation_dir", type=str, required=True, 
                       help="Path to generation directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save analysis outputs")
    parser.add_argument("--top_events", type=int, default=20,
                       help="Number of top events to show in plots")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SequenceAnalyzer(args.generation_dir)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.generation_dir) / "analysis"
    
    output_dir.mkdir(exist_ok=True)
    
    # Generate report
    print("Generating analysis report...")
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Event distribution plot
    event_plot_path = output_dir / "event_distribution.png"
    analyzer.plot_event_distribution(top_n=args.top_events, save_path=event_plot_path)
    
    # Sequence length distribution plot
    length_plot_path = output_dir / "sequence_length_distribution.png"
    analyzer.plot_sequence_length_distribution(save_path=length_plot_path)
    
    # Save event counts as CSV
    event_counts = analyzer.count_events()
    if not event_counts.empty:
        csv_path = output_dir / "event_counts.csv"
        event_counts.to_csv(csv_path, index=False)
        print(f"Event counts saved to: {csv_path}")
    
    print(f"\nAnalysis complete! All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()