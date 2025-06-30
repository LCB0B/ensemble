#!/usr/bin/env python3
"""
Test and analyze sequence-level embeddings for life trajectories.
"""

import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import umap
import pandas as pd
from typing import Dict, List, Any
import argparse

# Set up paths
import sys
sys.path.append('.')
from src.datamodule4 import LifeLightningDataModule
from src.encoder_nano_risk import CausalEncoder
from src.paths import FPATH


def load_model_and_data(checkpoint_path: Path, hparams_path: Path, data_dir: Path, device: str = "cuda"):
    """Load model and data for embedding analysis."""
    
    # Load hyperparameters
    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)
    
    print(f"Loading model from: {checkpoint_path}")
    model = CausalEncoder.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(device),
        strict=False,
        **hparams
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Set up data module
    print(f"Setting up data module with: {data_dir}")
    
    # Set up paths for pre-compiled data
    lmdb_path = data_dir / "dataset.lmdb"
    vocab_path = data_dir / "vocab.json"
    pnr_to_idx_path = data_dir / "pnr_to_database_idx.json"
    
    datamodule = LifeLightningDataModule(
        dir_path=data_dir,
        lmdb_path=lmdb_path,
        vocab_path=vocab_path,
        pnr_to_idx_path=pnr_to_idx_path,
        background=None,
        cls_token=hparams.get('include_cls', False),
        sep_token=hparams.get('include_sep', True),
        segment=hparams.get('include_segment', True),
        batch_size=16,  # Smaller batch for analysis
        num_workers=4,
        max_seq_len=hparams.get('max_seq_len', 2048),
        cutoff=hparams.get('token_freq_cutoff', 100)
    )
    
    datamodule.setup('predict')
    dataloader = datamodule.predict_dataloader()
    
    return model, dataloader, datamodule


def extract_embeddings(model, dataloader, datamodule, device: str, max_batches: int = 10):
    """Extract sequence embeddings using different pooling methods."""
    
    embeddings_data = {
        'mean_pool': [],
        'attention_pool': [], 
        'attention_weights': [],
        'cls': [],
        'max_pool': [],
        'sequences': [],
        'sequence_info': []
    }
    
    print(f"Extracting embeddings from {max_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            print(f"Processing batch {batch_idx + 1}/{max_batches}")
            
            # Process batch the same way as the working embedding example
            try:
                batch = datamodule.transfer_batch_to_device(batch, device, 0)
                batch = datamodule.on_after_batch_transfer(batch, 0)
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
            
            try:
                # Get all embedding types
                embeddings = model.compare_sequence_embeddings(batch)
                
                # Store embeddings
                for key in ['mean_pool', 'attention_pool', 'cls', 'max_pool']:
                    embeddings_data[key].append(embeddings[key].cpu())
                
                embeddings_data['attention_weights'].append(embeddings['attention_weights'].cpu())
                
                # Store sequence information
                if 'event' in batch:
                    embeddings_data['sequences'].append(batch['event'].cpu())
                    
                    # Store sequence metadata
                    batch_size = batch['event'].size(0)
                    for i in range(batch_size):
                        seq_length = (batch['event'][i] != 0).sum().item()
                        embeddings_data['sequence_info'].append({
                            'batch_idx': batch_idx,
                            'seq_idx': i,
                            'seq_length': seq_length
                        })
                        
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Concatenate all embeddings
    for key in ['mean_pool', 'attention_pool', 'cls', 'max_pool']:
        if embeddings_data[key]:
            embeddings_data[key] = torch.cat(embeddings_data[key], dim=0)
    
    if embeddings_data['attention_weights']:
        embeddings_data['attention_weights'] = torch.cat(embeddings_data['attention_weights'], dim=0)
    
    if embeddings_data['sequences']:
        embeddings_data['sequences'] = torch.cat(embeddings_data['sequences'], dim=0)
    
    print(f"Extracted embeddings for {len(embeddings_data['sequence_info'])} sequences")
    return embeddings_data


def analyze_embedding_similarities(embeddings_data: Dict[str, torch.Tensor]):
    """Analyze similarities between different embedding methods."""
    
    methods = ['mean_pool', 'attention_pool', 'cls', 'max_pool']
    similarities = {}
    
    print("Computing embedding similarities...")
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i <= j:  # Only compute upper triangle + diagonal
                emb1 = embeddings_data[method1].numpy()
                emb2 = embeddings_data[method2].numpy()
                
                # Compute cosine similarity between methods
                if method1 == method2:
                    # Self-similarity (should be 1.0)
                    sim = 1.0
                else:
                    # Cross-similarity: how similar are the embeddings from different methods?
                    similarities_matrix = cosine_similarity(emb1, emb2)
                    # Take diagonal (same sequence, different methods)
                    sim = np.diag(similarities_matrix).mean()
                
                similarities[f"{method1}_vs_{method2}"] = sim
                print(f"Similarity {method1} vs {method2}: {sim:.4f}")
    
    return similarities


def visualize_embeddings(embeddings_data: Dict[str, torch.Tensor], output_dir: Path):
    """Create visualizations of the embeddings."""
    
    output_dir.mkdir(exist_ok=True)
    methods = ['mean_pool', 'attention_pool', 'cls', 'max_pool']
    
    # 1. PCA visualization
    print("Creating PCA visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        embeddings = embeddings_data[method].numpy()
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Color by sequence length
        seq_lengths = [info['seq_length'] for info in embeddings_data['sequence_info']]
        
        scatter = axes[idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                  c=seq_lengths, cmap='viridis', alpha=0.6)
        axes[idx].set_title(f'{method.replace("_", " ").title()} Embeddings (PCA)')
        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[idx])
        cbar.set_label('Sequence Length')
    
    plt.tight_layout()
    plt.savefig(output_dir / "embeddings_pca.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. UMAP visualizations (clean, minimal style)
    print("Creating UMAP visualizations...")
    seq_lengths = [info['seq_length'] for info in embeddings_data['sequence_info']]
    
    # Create UMAP for each embedding method
    for method_name in ['mean_pool', 'attention_pool', 'cls', 'max_pool']:
        print(f"  Creating UMAP for {method_name}...")
        embeddings = embeddings_data[method_name].numpy()
        
        # Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            n_components=2, 
            random_state=42,
            metric='cosine'
        )
        embeddings_umap = reducer.fit_transform(embeddings)
        
        # Create clean minimal plot
        plt.figure(figsize=(12, 12), facecolor='white')
        
        # Create scatter plot with minimal styling
        scatter = plt.scatter(
            embeddings_umap[:, 0], 
            embeddings_umap[:, 1], 
            c=seq_lengths, 
            cmap='viridis', 
            alpha=0.7,
            s=15,
            edgecolors='none'
        )
        
        # Remove all axes, labels, ticks
        plt.axis('off')
        plt.gca().set_aspect('equal')
        
        # Remove any remaining whitespace/margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        
        # Save with no padding
        plt.savefig(output_dir / f"embeddings_umap_{method_name}_clean.png", 
                    dpi=300, 
                    bbox_inches='tight', 
                    pad_inches=0,
                    facecolor='white',
                    edgecolor='none')
        plt.close()
    
    # 3. t-SNE visualization (for comparison)
    print("Creating t-SNE visualization...")
    embeddings = embeddings_data['attention_pool'].numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    seq_lengths = [info['seq_length'] for info in embeddings_data['sequence_info']]
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                         c=seq_lengths, cmap='viridis', alpha=0.6)
    plt.title('Attention-Pooled Embeddings (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sequence Length')
    plt.savefig(output_dir / "embeddings_tsne.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Attention weights visualization
    if 'attention_weights' in embeddings_data:
        print("Creating attention weights visualization...")
        attention_weights = embeddings_data['attention_weights'].numpy()
        
        # Plot average attention weights across positions
        mean_attention = np.mean(attention_weights, axis=0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(mean_attention[:200])  # Plot first 200 positions
        plt.title('Average Attention Weights Across Sequence Positions')
        plt.xlabel('Sequence Position')
        plt.ylabel('Average Attention Weight')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "attention_weights.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot attention weights for individual sequences
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i in range(min(6, len(attention_weights))):
            seq_attention = attention_weights[i]
            seq_length = embeddings_data['sequence_info'][i]['seq_length']
            
            axes[i].plot(seq_attention[:seq_length])
            axes[i].set_title(f'Sequence {i+1} (Length: {seq_length})')
            axes[i].set_xlabel('Position')
            axes[i].set_ylabel('Attention Weight')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(6, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "individual_attention_weights.png", dpi=300, bbox_inches='tight')
        plt.close()


def analyze_embedding_diversity(embeddings_data: Dict[str, torch.Tensor]):
    """Analyze the diversity and distribution of embeddings."""
    
    methods = ['mean_pool', 'attention_pool', 'cls', 'max_pool']
    analysis = {}
    
    for method in methods:
        embeddings = embeddings_data[method].numpy()
        
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(embeddings)
        
        # Remove diagonal (self-similarities)
        similarities_no_diag = similarities[~np.eye(similarities.shape[0], dtype=bool)]
        
        analysis[method] = {
            'mean_similarity': similarities_no_diag.mean(),
            'std_similarity': similarities_no_diag.std(),
            'min_similarity': similarities_no_diag.min(),
            'max_similarity': similarities_no_diag.max(),
            'embedding_norm_mean': np.linalg.norm(embeddings, axis=1).mean(),
            'embedding_norm_std': np.linalg.norm(embeddings, axis=1).std()
        }
    
    return analysis


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Test sequence embeddings")
    parser.add_argument("--experiment_name", type=str, default="stable_pretrain")
    parser.add_argument("--experiment_subdir", type=str, default="8192")
    parser.add_argument("--data_dir", type=str, default="life_all_compiled")
    parser.add_argument("--max_batches", type=int, default=750)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set up paths
    hparams_path = FPATH.TB_LOGS / args.experiment_name / args.experiment_subdir / 'hparams.yaml'
    checkpoint_path = FPATH.CHECKPOINTS / args.experiment_name / args.experiment_subdir / 'last.ckpt'
    data_dir = FPATH.DATA / args.data_dir
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = FPATH.GENERATED / "embedding_analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SEQUENCE EMBEDDING ANALYSIS")
    print("=" * 60)
    print(f"Model: {checkpoint_path}")
    print(f"Data: {data_dir}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load model and data
    model, dataloader, datamodule = load_model_and_data(
        checkpoint_path, hparams_path, data_dir, args.device
    )
    
    # Extract embeddings
    embeddings_data = extract_embeddings(model, dataloader, datamodule, args.device, args.max_batches)
    
    # Analyze similarities between methods
    similarities = analyze_embedding_similarities(embeddings_data)
    
    # Analyze embedding diversity
    diversity_analysis = analyze_embedding_diversity(embeddings_data)
    
    # Create visualizations
    visualize_embeddings(embeddings_data, output_dir)
    
    # Generate report
    report = []
    report.append("SEQUENCE EMBEDDING ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")
    
    report.append("EMBEDDING SIMILARITIES (between methods):")
    for pair, sim in similarities.items():
        report.append(f"  {pair}: {sim:.4f}")
    report.append("")
    
    report.append("EMBEDDING DIVERSITY ANALYSIS:")
    for method, stats in diversity_analysis.items():
        report.append(f"  {method.upper()}:")
        for stat_name, value in stats.items():
            report.append(f"    {stat_name}: {value:.4f}")
        report.append("")
    
    # Save report
    report_text = "\n".join(report)
    print(report_text)
    
    with open(output_dir / "embedding_analysis_report.txt", 'w') as f:
        f.write(report_text)
    
    # Save embeddings for further analysis
    torch.save(embeddings_data, output_dir / "embeddings_data.pt")
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()