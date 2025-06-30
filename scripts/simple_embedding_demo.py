#!/usr/bin/env python3
"""
Simple demonstration of sequence embedding methods without requiring datamodule.
This creates synthetic data to show how the embedding methods work.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Simple mock of the embedding functionality for demonstration
class SimpleSequenceEmbedder(nn.Module):
    """Simplified version demonstrating the embedding concepts."""
    
    def __init__(self, vocab_size=1000, d_model=128, seq_len=100):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Simple transformer components
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        
        # Attention pooling layer
        self.attention_pool = nn.Linear(d_model, 1, bias=False)
        
    def forward(self, sequences):
        """Forward pass to get hidden representations."""
        # Embed sequences
        embedded = self.embedding(sequences)  # [batch, seq_len, d_model]
        
        # Apply transformer
        hidden = self.transformer(embedded)  # [batch, seq_len, d_model]
        
        return hidden
    
    def get_sequence_embedding_mean_pool(self, sequences):
        """Mean pooling embedding."""
        hidden = self.forward(sequences)
        
        # Create padding mask (0 = padding)
        padding_mask = (sequences != 0).float()
        seq_lengths = padding_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        # Masked mean pooling
        masked_hidden = hidden * padding_mask.unsqueeze(-1)
        mean_pooled = masked_hidden.sum(dim=1) / seq_lengths
        
        return mean_pooled
    
    def get_sequence_embedding_attention_pool(self, sequences):
        """Attention-weighted pooling embedding."""
        hidden = self.forward(sequences)
        
        # Compute attention scores
        attention_scores = self.attention_pool(hidden).squeeze(-1)
        
        # Mask padding tokens
        padding_mask = (sequences != 0).float()
        masked_scores = attention_scores.masked_fill(padding_mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(masked_scores, dim=1)
        
        # Apply attention weights
        attended = torch.sum(hidden * attention_weights.unsqueeze(-1), dim=1)
        
        return attended, attention_weights


def create_synthetic_life_sequences(batch_size=8, seq_len=50, vocab_size=100):
    """Create synthetic life sequence data for demonstration."""
    sequences = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    for i in range(batch_size):
        # Create a realistic-looking life sequence
        # Start with childhood events (tokens 1-20)
        # Move to adult events (tokens 21-60)
        # End with later life events (tokens 61-99)
        
        length = np.random.randint(20, seq_len)
        age_progression = np.linspace(1, 99, length)
        
        # Generate events based on "age"
        for j, age in enumerate(age_progression):
            if age < 20:  # Childhood
                token = np.random.randint(1, 21)
            elif age < 60:  # Adulthood
                token = np.random.randint(21, 61)
            else:  # Later life
                token = np.random.randint(61, 100)
            
            sequences[i, j] = token
    
    return sequences


def demo_embedding_methods():
    """Demonstrate different sequence embedding methods."""
    print("🧬 Sequence Embedding Methods Demo")
    print("=" * 50)
    
    # Create model and synthetic data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleSequenceEmbedder(vocab_size=100, d_model=64, seq_len=50)
    model.to(device)
    model.eval()
    
    # Create synthetic life sequences
    print("\n📊 Creating synthetic life sequences...")
    sequences = create_synthetic_life_sequences(batch_size=8, seq_len=50, vocab_size=100)
    sequences = sequences.to(device)
    
    print(f"Created {sequences.shape[0]} sequences of length {sequences.shape[1]}")
    
    # Analyze sequence lengths
    seq_lengths = [(seq != 0).sum().item() for seq in sequences]
    print(f"Sequence lengths: {seq_lengths}")
    
    print("\n🔍 Computing embeddings...")
    
    with torch.no_grad():
        # 1. Mean pooling
        mean_embeddings = model.get_sequence_embedding_mean_pool(sequences)
        print(f"Mean pooling embeddings shape: {mean_embeddings.shape}")
        print(f"Mean embedding norms: {torch.norm(mean_embeddings, dim=1).cpu().numpy()}")
        
        # 2. Attention-weighted pooling
        attention_embeddings, attention_weights = model.get_sequence_embedding_attention_pool(sequences)
        print(f"Attention pooling embeddings shape: {attention_embeddings.shape}")
        print(f"Attention embedding norms: {torch.norm(attention_embeddings, dim=1).cpu().numpy()}")
        print(f"Attention weights shape: {attention_weights.shape}")
        
        # Compare similarities
        print("\n📈 Comparing embedding methods...")
        similarity = torch.nn.functional.cosine_similarity(mean_embeddings, attention_embeddings, dim=1)
        print(f"Cosine similarity between methods: {similarity.cpu().numpy()}")
        print(f"Average similarity: {similarity.mean().item():.4f}")
        
        # Analyze attention patterns
        print("\n🎯 Attention Analysis...")
        for i in range(min(3, len(sequences))):
            seq_len = seq_lengths[i]
            seq_attention = attention_weights[i, :seq_len].cpu().numpy()
            
            print(f"\nSequence {i+1} (length {seq_len}):")
            print(f"  Max attention at position: {seq_attention.argmax()}")
            print(f"  Max attention weight: {seq_attention.max():.4f}")
            print(f"  Attention entropy: {-np.sum(seq_attention * np.log(seq_attention + 1e-8)):.4f}")
            
            # Show top-3 attended positions
            top_positions = seq_attention.argsort()[-3:][::-1]
            print(f"  Top 3 attended positions: {top_positions} with weights {seq_attention[top_positions]}")
    
    # Visualization
    print("\n📊 Creating visualizations...")
    
    # Plot attention weights for first few sequences
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(4, len(sequences))):
        seq_len = seq_lengths[i]
        seq_attention = attention_weights[i, :seq_len].cpu().numpy()
        
        axes[i].bar(range(seq_len), seq_attention)
        axes[i].set_title(f'Sequence {i+1} Attention Weights')
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Attention Weight')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_weights_demo.png', dpi=300, bbox_inches='tight')
    print("✓ Attention weights plot saved as 'attention_weights_demo.png'")
    
    # Plot embedding comparison
    plt.figure(figsize=(10, 6))
    
    # Project embeddings to 2D for visualization (simple projection)
    mean_2d = mean_embeddings[:, :2].cpu().numpy()
    attention_2d = attention_embeddings[:, :2].cpu().numpy()
    
    plt.scatter(mean_2d[:, 0], mean_2d[:, 1], label='Mean Pooling', alpha=0.7, s=100)
    plt.scatter(attention_2d[:, 0], attention_2d[:, 1], label='Attention Pooling', alpha=0.7, s=100)
    
    # Draw lines connecting same sequences
    for i in range(len(mean_2d)):
        plt.plot([mean_2d[i, 0], attention_2d[i, 0]], 
                [mean_2d[i, 1], attention_2d[i, 1]], 
                'k--', alpha=0.3)
    
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('Sequence Embeddings: Mean vs Attention Pooling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('embedding_comparison_demo.png', dpi=300, bbox_inches='tight')
    print("✓ Embedding comparison plot saved as 'embedding_comparison_demo.png'")
    
    print("\n✨ Demo complete!")
    print("\nKey Insights:")
    print("• Mean pooling treats all positions equally")
    print("• Attention pooling learns to focus on important positions")
    print("• Both methods create fixed-size representations of variable-length sequences")
    print("• Attention weights provide interpretability about which events matter most")
    print("\nApplications for life sequences:")
    print("• Cluster people with similar life trajectories")
    print("• Predict life outcomes from early events")
    print("• Find people with similar attention patterns")
    print("• Analyze which life events are most predictive")


if __name__ == "__main__":
    demo_embedding_methods()