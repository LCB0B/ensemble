#!/usr/bin/env python3
"""
Simple example of how to use sequence embeddings.
"""

import torch
import yaml
from pathlib import Path
import numpy as np

# Set up paths
import sys
sys.path.append('.')
from src.datamodule4 import LifeLightningDataModule
from src.encoder_nano_risk import CausalEncoder
from src.paths import FPATH


def main():
    """Simple example of sequence embedding usage."""
    
    print("🧬 Life Sequence Embedding Example")
    print("=" * 50)
    
    # Load model and data (using your existing paths)
    hparams_path = FPATH.TB_LOGS / 'stable_pretrain' / '8192' / 'hparams.yaml'
    checkpoint_path = FPATH.CHECKPOINTS / 'stable_pretrain' / '8192' / 'last.ckpt'
    data_dir = FPATH.DATA / 'life_all_compiled'
    
    # Load hyperparameters
    with open(hparams_path, "r") as f:
        hparams = yaml.safe_load(f)
    
    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CausalEncoder.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(device),
        strict=False,
        **hparams
    )
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    # Set up data
    print("Setting up data...")
    
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
        batch_size=4,  # Small batch for demo
        num_workers=2,
        max_seq_len=hparams.get('max_seq_len', 2048),
        cutoff=hparams.get('token_freq_cutoff', 100)
    )
    
    datamodule.setup('predict')
    dataloader = datamodule.predict_dataloader()
    
    # Get a sample batch
    print("Getting sample batch...")
    batch = next(iter(dataloader))
    
    # Move to device and process batch (important for attention mask creation)
    batch = datamodule.transfer_batch_to_device(batch, device, 0)
    batch = datamodule.on_after_batch_transfer(batch, 0)
    
    print(f"✓ Batch loaded with {batch['event'].shape[0]} sequences")
    print(f"✓ Sequence length: {batch['event'].shape[1]}")
    
    # Test different embedding methods
    print("\n🔍 Computing sequence embeddings...")
    
    with torch.no_grad():
        # 1. Mean pooling
        print("1. Mean pooling...")
        mean_embeddings = model.get_sequence_embedding_mean_pool(batch)
        print(f"   Shape: {mean_embeddings.shape}")
        print(f"   Mean norm: {torch.norm(mean_embeddings, dim=1).mean().item():.4f}")
        
        # 2. Attention-weighted pooling
        print("2. Attention-weighted pooling...")
        attention_embeddings, attention_weights = model.get_sequence_embedding_attention_pool(batch)
        print(f"   Shape: {attention_embeddings.shape}")
        print(f"   Mean norm: {torch.norm(attention_embeddings, dim=1).mean().item():.4f}")
        print(f"   Attention weights shape: {attention_weights.shape}")
        
        # 3. CLS token
        print("3. CLS token embedding...")
        cls_embeddings = model.get_sequence_embedding_cls(batch)
        print(f"   Shape: {cls_embeddings.shape}")
        print(f"   Mean norm: {torch.norm(cls_embeddings, dim=1).mean().item():.4f}")
        
        # 4. Max pooling
        print("4. Max pooling...")
        max_embeddings = model.get_sequence_embedding_max_pool(batch)
        print(f"   Shape: {max_embeddings.shape}")
        print(f"   Mean norm: {torch.norm(max_embeddings, dim=1).mean().item():.4f}")
    
    # Compare similarities between methods
    print("\n📊 Comparing embedding methods...")
    
    # Compute cosine similarities between methods
    def cosine_similarity(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
    
    similarities = {
        'mean_vs_attention': cosine_similarity(mean_embeddings, attention_embeddings),
        'mean_vs_cls': cosine_similarity(mean_embeddings, cls_embeddings),
        'mean_vs_max': cosine_similarity(mean_embeddings, max_embeddings),
        'attention_vs_cls': cosine_similarity(attention_embeddings, cls_embeddings),
        'attention_vs_max': cosine_similarity(attention_embeddings, max_embeddings),
        'cls_vs_max': cosine_similarity(cls_embeddings, max_embeddings),
    }
    
    for pair, sim in similarities.items():
        print(f"   {pair}: {sim:.4f}")
    
    # Show attention patterns
    print("\n🎯 Attention analysis...")
    for i in range(min(2, batch['event'].shape[0])):
        seq_length = (batch['event'][i] != 0).sum().item()
        seq_attention = attention_weights[i, :seq_length]
        
        print(f"   Sequence {i+1} (length {seq_length}):")
        print(f"     Max attention position: {seq_attention.argmax().item()}")
        print(f"     Max attention weight: {seq_attention.max().item():.4f}")
        print(f"     Min attention weight: {seq_attention.min().item():.4f}")
        print(f"     Attention entropy: {(-seq_attention * torch.log(seq_attention + 1e-8)).sum().item():.4f}")
    
    print("\n✨ Example complete!")
    print("\nWhat the embeddings represent:")
    print("• Mean pooling: Average of all token representations")
    print("• Attention pooling: Learned weighted average (focuses on important events)")
    print("• CLS token: First position representation (if using CLS tokens)")
    print("• Max pooling: Element-wise maximum across sequence")
    print("\nUse these embeddings for:")
    print("• Clustering similar life trajectories")
    print("• Measuring life path similarity")
    print("• Predicting life outcomes")
    print("• Analyzing population patterns")


if __name__ == "__main__":
    main()