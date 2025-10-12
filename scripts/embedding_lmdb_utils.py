#!/usr/bin/env python3
"""
LMDB Utilities for Embedding Storage and Retrieval

Provides efficient LMDB-based storage and retrieval functions for large-scale
embedding datasets. Compatible with both Time2Vec and Time Token embeddings.

Functions:
- save_embeddings_to_lmdb: Convert numpy embeddings to LMDB format
- load_and_analyze_embeddings_lmdb: Load and analyze LMDB embeddings
- find_person_embedding_lmdb: Find embedding by person ID
- get_embedding_statistics_lmdb: Detailed embedding statistics
- get_embedding_by_index_lmdb: Retrieve specific embedding by index

Usage:
    from scripts.embedding_lmdb_utils import save_embeddings_to_lmdb

    # After extracting embeddings
    lmdb_results = save_embeddings_to_lmdb(results, output_dir)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import lmdb
import numpy as np


def save_embeddings_to_lmdb(
    results: Dict[str, Any],
    output_dir: str,
    model_name: Optional[str] = None,
    exclude_time_tokens: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Convert numpy embeddings to LMDB format for efficient storage and retrieval.

    Args:
        results: Dictionary containing 'embeddings', 'metadata', 'pooling_strategy'
        output_dir: Directory to save LMDB database
        model_name: Optional model name for config
        exclude_time_tokens: Optional flag for config

    Returns:
        Dictionary with LMDB path and statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lmdb_path = output_path / "embeddings.lmdb"

    # Create LMDB environment
    embeddings = results['embeddings']
    map_size = embeddings.nbytes * 2  # Double for safety

    env = lmdb.open(str(lmdb_path), map_size=map_size, max_dbs=3)

    # Create sub-databases
    embeddings_db = env.open_db(b'embeddings')
    metadata_db = env.open_db(b'metadata')
    config_db = env.open_db(b'config')

    print(f"Saving {len(embeddings):,} embeddings to LMDB...")

    with env.begin(write=True) as txn:
        # Save embeddings with integer keys
        for i, embedding in enumerate(embeddings):
            key = str(i).encode('utf-8')
            value = embedding.astype(np.float32).tobytes()
            txn.put(key, value, db=embeddings_db)

            if (i + 1) % 10000 == 0:
                print(f"   Saved {i+1:,}/{len(embeddings):,} embeddings")

        # Save metadata
        if 'metadata' in results:
            metadata_bytes = pickle.dumps(results['metadata'])
            txn.put(b'metadata', metadata_bytes, db=metadata_db)

        # Load and save person mapping CSV if it exists
        person_mapping_csv_path = output_path / "person_id_mapping.csv"
        if person_mapping_csv_path.exists():
            person_mapping_bytes = person_mapping_csv_path.read_bytes()
            txn.put(b'person_mapping', person_mapping_bytes, db=metadata_db)

        # Save config
        config = {
            'total_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'dtype': str(embeddings.dtype),
            'pooling_strategy': results.get('pooling_strategy', 'unknown'),
            'output_dir': str(output_dir),
        }

        # Add optional config fields
        if model_name:
            config['model_name'] = model_name
        if exclude_time_tokens is not None:
            config['exclude_time_tokens'] = exclude_time_tokens
        if 'model_d_model' in results:
            config['model_d_model'] = results['model_d_model']

        config_bytes = json.dumps(config).encode('utf-8')
        txn.put(b'config', config_bytes, db=config_db)

    env.close()

    print(f"LMDB database created: {lmdb_path}")
    print(f"Database size: {lmdb_path.stat().st_size / 1024**2:.1f} MB")

    return {
        'lmdb_path': lmdb_path,
        'total_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'pooling_strategy': results.get('pooling_strategy', 'unknown')
    }


def load_and_analyze_embeddings_lmdb(embeddings_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load previously saved embeddings from LMDB and analyze them.

    Args:
        embeddings_dir: Directory containing embeddings.lmdb

    Returns:
        Dictionary with config, sample embeddings, and metadata
    """
    embeddings_path = Path(embeddings_dir)
    lmdb_path = embeddings_path / "embeddings.lmdb"

    if not lmdb_path.exists():
        print(f"LMDB database not found: {lmdb_path}")
        return None

    print(f"Loading embeddings from LMDB: {lmdb_path}")

    env = lmdb.open(str(lmdb_path), readonly=True, max_dbs=3)
    embeddings_db = env.open_db(b'embeddings')
    metadata_db = env.open_db(b'metadata')
    config_db = env.open_db(b'config')

    # Load config
    with env.begin() as txn:
        config_bytes = txn.get(b'config', db=config_db)
        if config_bytes:
            config = json.loads(config_bytes.decode('utf-8'))
        else:
            config = {}

    print(f"Embedding Analysis:")
    print(f"   - Total sequences: {config.get('total_embeddings', 'Unknown'):,}")
    print(f"   - Dimensions: {config.get('embedding_dim', 'Unknown')}")
    print(f"   - Pooling strategy: {config.get('pooling_strategy', 'Unknown')}")
    if 'model_d_model' in config:
        print(f"   - Model d_model: {config.get('model_d_model', 'Unknown')}")

    # Load a sample of embeddings for statistics
    sample_size = min(1000, config.get('total_embeddings', 0))
    sample_embeddings = []

    with env.begin() as txn:
        cursor = txn.cursor(db=embeddings_db)
        for i, (key, value) in enumerate(cursor):
            if i >= sample_size:
                break
            embedding = np.frombuffer(value, dtype=np.float32)
            sample_embeddings.append(embedding)

    if sample_embeddings:
        sample_array = np.array(sample_embeddings)
        print(f"   - Sample embedding stats (n={len(sample_embeddings)}):")
        print(f"     • Mean: {sample_array.mean():.4f}")
        print(f"     • Std: {sample_array.std():.4f}")
        print(f"     • Min: {sample_array.min():.4f}")
        print(f"     • Max: {sample_array.max():.4f}")

    # Load person mapping if available
    with env.begin() as txn:
        person_mapping_bytes = txn.get(b'person_mapping', db=metadata_db)
        if person_mapping_bytes:
            print(f"   - Person mapping available")
        else:
            print(f"   - No person mapping found")

        metadata_bytes = txn.get(b'metadata', db=metadata_db)
        if metadata_bytes:
            metadata = pickle.loads(metadata_bytes)
            print(f"   - Additional metadata available")
        else:
            metadata = None

    env.close()

    return {
        'config': config,
        'lmdb_path': lmdb_path,
        'sample_embeddings': sample_array if sample_embeddings else None,
        'metadata': metadata
    }


def find_person_embedding_lmdb(
    embeddings_dir: str,
    person_id: str
) -> Optional[Dict[str, Any]]:
    """
    Find the embedding for a specific person ID from LMDB storage.

    Args:
        embeddings_dir: Directory containing embeddings.lmdb
        person_id: Person ID to lookup

    Returns:
        Dictionary with person_id, embedding, and embedding_idx
    """
    embeddings_path = Path(embeddings_dir)
    lmdb_path = embeddings_path / "embeddings.lmdb"

    if not lmdb_path.exists():
        print(f"LMDB database not found: {lmdb_path}")
        return None

    env = lmdb.open(str(lmdb_path), readonly=True, max_dbs=3)
    embeddings_db = env.open_db(b'embeddings')
    metadata_db = env.open_db(b'metadata')

    # Load person mapping
    with env.begin() as txn:
        person_mapping_bytes = txn.get(b'person_mapping', db=metadata_db)
        if not person_mapping_bytes:
            print("Person mapping not found in LMDB")
            env.close()
            return None

    # Parse person mapping
    person_mapping_csv = person_mapping_bytes.decode('utf-8')

    # Find embedding index for person_id
    person_rows = []
    for line in person_mapping_csv.split('\n')[1:]:  # Skip header
        if line.strip() and person_id in line:
            person_rows.append(line.split(','))

    if not person_rows:
        print(f"Person ID '{person_id}' not found in embeddings")
        env.close()
        return None

    # Get first match (assuming person_id is unique)
    row = person_rows[0]
    embedding_idx = int(row[0])  # Assuming first column is embedding_idx

    # Load the specific embedding
    with env.begin() as txn:
        key = str(embedding_idx).encode('utf-8')
        value = txn.get(key, db=embeddings_db)
        if value:
            person_embedding = np.frombuffer(value, dtype=np.float32)
        else:
            print(f"Embedding not found for index {embedding_idx}")
            env.close()
            return None

    env.close()

    print(f"Found embedding for person '{person_id}':")
    print(f"   - Embedding index: {embedding_idx}")
    print(f"   - Embedding shape: {person_embedding.shape}")

    return {
        'person_id': person_id,
        'embedding': person_embedding,
        'embedding_idx': embedding_idx,
    }


def get_embedding_statistics_lmdb(
    embeddings_dir: str,
    sample_size: int = 10000
) -> None:
    """
    Get detailed statistics about the LMDB embeddings.

    Args:
        embeddings_dir: Directory containing embeddings.lmdb
        sample_size: Number of embeddings to sample for statistics
    """
    embeddings_path = Path(embeddings_dir)
    lmdb_path = embeddings_path / "embeddings.lmdb"

    if not lmdb_path.exists():
        print(f"LMDB database not found: {lmdb_path}")
        return None

    print("DETAILED LMDB EMBEDDING STATISTICS")
    print("=" * 50)

    env = lmdb.open(str(lmdb_path), readonly=True, max_dbs=3)
    embeddings_db = env.open_db(b'embeddings')
    config_db = env.open_db(b'config')

    # Load config
    with env.begin() as txn:
        config_bytes = txn.get(b'config', db=config_db)
        if config_bytes:
            config = json.loads(config_bytes.decode('utf-8'))
        else:
            config = {}

    total_embeddings = config.get('total_embeddings', 0)
    embedding_dim = config.get('embedding_dim', 0)

    print(f"Shape: ({total_embeddings:,}, {embedding_dim})")
    print(f"LMDB file size: {lmdb_path.stat().st_size / 1024**2:.1f} MB")

    # Sample embeddings for statistics
    print(f"\nSampling {min(sample_size, total_embeddings):,} embeddings for statistics...")

    embeddings_sample = []
    with env.begin() as txn:
        cursor = txn.cursor(db=embeddings_db)
        for i, (key, value) in enumerate(cursor):
            if i >= sample_size:
                break
            embedding = np.frombuffer(value, dtype=np.float32)
            embeddings_sample.append(embedding)

            if (i + 1) % 1000 == 0:
                print(f"   Sampled {i+1:,} embeddings")

    if embeddings_sample:
        embeddings_array = np.array(embeddings_sample)

        # Statistical summaries
        print(f"\nValue Distribution (sample of {len(embeddings_sample):,}):")
        print(f"  Mean: {embeddings_array.mean():.6f}")
        print(f"  Std:  {embeddings_array.std():.6f}")
        print(f"  Min:  {embeddings_array.min():.6f}")
        print(f"  Max:  {embeddings_array.max():.6f}")

        # Percentiles
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = np.percentile(embeddings_array, p)
            print(f"  {p:2d}%: {val:.6f}")

        # Per-dimension statistics
        dim_means = embeddings_array.mean(axis=0)
        dim_stds = embeddings_array.std(axis=0)

        print(f"\nPer-dimension statistics:")
        print(f"  Dimension means - Min: {dim_means.min():.6f}, Max: {dim_means.max():.6f}")
        print(f"  Dimension stds  - Min: {dim_stds.min():.6f}, Max: {dim_stds.max():.6f}")

    env.close()


def get_embedding_by_index_lmdb(
    embeddings_dir: str,
    embedding_idx: int
) -> Optional[np.ndarray]:
    """
    Get a specific embedding by its index from LMDB.

    Args:
        embeddings_dir: Directory containing embeddings.lmdb
        embedding_idx: Index of the embedding to retrieve

    Returns:
        Numpy array of the embedding, or None if not found
    """
    embeddings_path = Path(embeddings_dir)
    lmdb_path = embeddings_path / "embeddings.lmdb"

    if not lmdb_path.exists():
        print(f"LMDB database not found: {lmdb_path}")
        return None

    env = lmdb.open(str(lmdb_path), readonly=True, max_dbs=3)
    embeddings_db = env.open_db(b'embeddings')

    with env.begin() as txn:
        key = str(embedding_idx).encode('utf-8')
        value = txn.get(key, db=embeddings_db)

        if value:
            embedding = np.frombuffer(value, dtype=np.float32)
            env.close()
            return embedding
        else:
            print(f"Embedding not found for index {embedding_idx}")
            env.close()
            return None


# ==========================================
# USAGE EXAMPLES
# ==========================================
if __name__ == "__main__":
    """
    Example usage of LMDB utility functions:

    # Save embeddings to LMDB
    from src.embedding_encoder_class import TimeTokenEmbeddingEncoder
    results = model.extract_and_save_embeddings(...)
    lmdb_results = save_embeddings_to_lmdb(results, output_dir)

    # Load and analyze embeddings
    analysis = load_and_analyze_embeddings_lmdb("path/to/embeddings")

    # Find specific person's embedding
    person_result = find_person_embedding_lmdb("path/to/embeddings", "12345")

    # Get detailed statistics
    get_embedding_statistics_lmdb("path/to/embeddings", sample_size=5000)

    # Get specific embedding by index
    embedding = get_embedding_by_index_lmdb("path/to/embeddings", 0)
    """
    print("Import this module to use LMDB utility functions.")
    print("See docstrings for usage examples.")
