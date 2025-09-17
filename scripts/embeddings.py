#!/usr/bin/env python3
"""
Usage script for the integrated EmbeddingNanoEncoder
Extracts and saves high-dimensional embeddings with person ID tracking using LMDB storage
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import yaml
import polars as pl
import pyarrow.dataset as ds
import numpy as np
import lmdb
import pickle
import json
from pathlib import Path

# Import your modules
from src.datamodule2 import PretrainDataModule
from src.paths import FPATH
from src.embedding_encoder_class import EmbeddingNanoEncoder


def main():
    """Main extraction pipeline using integrated EmbeddingNanoEncoder"""
    
    # Configuration - Update these for your specific model
    MODEL_NAME = "021_muddy_cobra-pretrain-lr0.0003"
    EXPERIMENT_NAME = "destiny"
    
    # Paths using FPATH
    CHECKPOINT_PATH = FPATH.CHECKPOINTS_TRANSFORMER / EXPERIMENT_NAME / MODEL_NAME / "best.ckpt"
    OUTPUT_DIR = FPATH.EMBEDDINGS / MODEL_NAME
    
    print("INTEGRATED EMBEDDING EXTRACTION")
    print("=" * 50)
    print(f"Model: {CHECKPOINT_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Load model from checkpoint
    model = EmbeddingNanoEncoder.from_checkpoint(str(CHECKPOINT_PATH))
    
    # Load the same hparams used for training to set up datamodule
    hparams_path = FPATH.CONFIGS / "destiny" / "hparams_destiny_pretrain.yaml"
    with open(hparams_path, "r", encoding="utf-8") as stream:
        hparams = yaml.safe_load(stream)
    
    print(f"\nSetting up datamodule...")
    print(f"   - Source dir: {hparams['source_dir']}")
    print(f"   - Sources: {hparams['sources']}")
    
    # Set up datamodule (same as training)
    source_paths = [
        (FPATH.DATA / hparams["source_dir"] / path).with_suffix(".parquet")
        for path in hparams["sources"]
    ]
    background_path = (
        FPATH.DATA / hparams["source_dir"] / hparams["background"]
    ).with_suffix(".parquet")
    
    sources = [ds.dataset(s, format="parquet") for s in source_paths]
    background = pl.read_parquet(background_path)
    
    datamodule = PretrainDataModule(
        dir_path=FPATH.DATA / hparams["dir_path"],
        sources=sources,
        background=background,
        subset_background=hparams["subset_background"],
        n_tokens=hparams["n_tokens"],
        lengths=hparams["lengths"],
        num_workers=8,  # Set to 0 for embedding extraction
        max_seq_len=hparams["max_seq_len"],
        source_dir=hparams["source_dir"],
        pretrain_style=hparams["pretrain_style"],
        masking_ratio=hparams.get("masking_ratio"),
    )
    
    # Prepare data
    datamodule.prepare_data()
    
    # Extract embeddings using the integrated method
    results = model.extract_and_save_embeddings(
        datamodule=datamodule,
        output_dir=str(OUTPUT_DIR),
        pooling_strategy="mean",          # Options: "mean", "last", "max", "cls"
        max_seq_len=None,                 # Use model default, or override with int
        include_datasets=["train", "val", "predict"],  # Which datasets to process
        max_sequences=None               # Limit number of sequences (None for all)
    )
    
    # Convert to LMDB format
    lmdb_results = save_embeddings_to_lmdb(results, OUTPUT_DIR)
    
    print(f"\nEXTRACTION COMPLETE!")
    print(f"Results:")
    print(f"   - Total embeddings: {lmdb_results['total_embeddings']:,}")
    print(f"   - Embedding dimensions: {lmdb_results['embedding_dim']}")
    print(f"   - Pooling strategy: {lmdb_results['pooling_strategy']}")
    print(f"   - Saved to LMDB: {lmdb_results['lmdb_path']}")
    
    return lmdb_results


def save_embeddings_to_lmdb(results, output_dir):
    """Convert numpy embeddings to LMDB format"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lmdb_path = output_path / "embeddings.lmdb"
    
    # Create LMDB environment
    # Calculate map_size based on data size (embeddings + overhead)
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
        
        # Save person mapping
        if 'person_mapping' in results:
            person_mapping_bytes = results['person_mapping'].write_csv().encode('utf-8')
            txn.put(b'person_mapping', person_mapping_bytes, db=metadata_db)
        
        # Save config
        config = {
            'total_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'dtype': str(embeddings.dtype),
            'pooling_strategy': results.get('pooling_strategy', 'unknown'),
            'output_dir': str(output_dir),
            'model_d_model': results.get('model_d_model', None)
        }
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


def extract_with_different_pooling_strategies():
    """Extract embeddings with different pooling strategies for comparison"""
    
    MODEL_NAME = "your_model_name_here"
    EXPERIMENT_NAME = "destiny"
    CHECKPOINT_PATH = FPATH.CHECKPOINTS_TRANSFORMER / EXPERIMENT_NAME / MODEL_NAME / "best.ckpt"
    
    # Load model once
    model = EmbeddingNanoEncoder.from_checkpoint(str(CHECKPOINT_PATH))
    
    # Set up datamodule (you'd need to do this setup as in main())
    # datamodule = setup_datamodule()  # Implement this based on main()
    
    pooling_strategies = ["mean", "last", "max"]
    
    for strategy in pooling_strategies:
        print(f"Extracting with {strategy} pooling...")
        
        output_dir = FPATH.DATA / "embeddings" / f"{MODEL_NAME}_{strategy}_pooling"
        
        # results = model.extract_and_save_embeddings(
        #     datamodule=datamodule,
        #     output_dir=str(output_dir),
        #     pooling_strategy=strategy
        # )
        # save_embeddings_to_lmdb(results, output_dir)
        
        print(f"Completed {strategy} pooling: {output_dir}")


def extract_limited_sequences():
    """Extract embeddings from a limited number of sequences for testing or sampling"""
    
    MODEL_NAME = "your_model_name_here"
    EXPERIMENT_NAME = "destiny"
    CHECKPOINT_PATH = FPATH.CHECKPOINTS_TRANSFORMER / EXPERIMENT_NAME / MODEL_NAME / "best.ckpt"
    
    model = EmbeddingNanoEncoder.from_checkpoint(str(CHECKPOINT_PATH))
    
    # Set up datamodule (implement based on main())
    # datamodule = setup_datamodule()
    
    # Extract only first 10,000 sequences
    output_dir = FPATH.DATA / "embeddings" / f"{MODEL_NAME}_10k_sample"
    
    # results = model.extract_and_save_embeddings(
    #     datamodule=datamodule,
    #     output_dir=str(output_dir),
    #     pooling_strategy="mean",
    #     max_sequences=10000  # Only process first 10,000 sequences
    # )
    # save_embeddings_to_lmdb(results, output_dir)
    
    print(f"Limited sequence extraction complete: {output_dir}")


def extract_subset_for_testing():
    """Extract embeddings from only validation set for quick testing"""
    
    MODEL_NAME = "your_model_name_here"
    EXPERIMENT_NAME = "destiny" 
    CHECKPOINT_PATH = FPATH.CHECKPOINTS_TRANSFORMER / EXPERIMENT_NAME / MODEL_NAME / "best.ckpt"
    
    model = EmbeddingNanoEncoder.from_checkpoint(str(CHECKPOINT_PATH))
    
    # Set up datamodule (implement based on main())
    # datamodule = setup_datamodule()
    
    # Quick test - only validation set
    output_dir = FPATH.DATA / "embeddings" / f"{MODEL_NAME}_val_only"
    
    # results = model.extract_and_save_embeddings(
    #     datamodule=datamodule,
    #     output_dir=str(output_dir),
    #     pooling_strategy="mean",
    #     include_datasets=["val"]  # Only validation set
    # )
    # save_embeddings_to_lmdb(results, output_dir)
    
    print(f"Quick validation embedding test complete: {output_dir}")


def extract_with_custom_sequence_length():
    """Extract embeddings with a different sequence length than training"""
    
    MODEL_NAME = "your_model_name_here"
    EXPERIMENT_NAME = "destiny"
    CHECKPOINT_PATH = FPATH.CHECKPOINTS_TRANSFORMER / EXPERIMENT_NAME / MODEL_NAME / "best.ckpt"
    
    model = EmbeddingNanoEncoder.from_checkpoint(str(CHECKPOINT_PATH))
    
    # Set up datamodule (implement based on main())
    # datamodule = setup_datamodule()
    
    # Extract with shorter sequences
    output_dir = FPATH.DATA / "embeddings" / f"{MODEL_NAME}_short_seq"
    
    # results = model.extract_and_save_embeddings(
    #     datamodule=datamodule,
    #     output_dir=str(output_dir),
    #     pooling_strategy="mean",
    #     max_seq_len=256  # Shorter than training length
    # )
    # save_embeddings_to_lmdb(results, output_dir)
    
    print(f"Custom sequence length extraction complete: {output_dir}")


def load_and_analyze_embeddings_lmdb(embeddings_dir: str):
    """Load previously saved embeddings from LMDB and analyze them"""
    
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
            person_mapping_csv = person_mapping_bytes.decode('utf-8')
            # You might want to save this to a temporary file and load with polars
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


def find_person_embedding_lmdb(embeddings_dir: str, person_id: str):
    """Find the embedding for a specific person ID from LMDB storage"""
    
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
    
    # Parse person mapping (you might want to optimize this)
    person_mapping_csv = person_mapping_bytes.decode('utf-8')
    # For now, we'll do a simple search - in practice you might want to 
    # store person mappings more efficiently in LMDB
    
    # Find embedding index for person_id
    # This is a simplified implementation - you'd want to optimize this
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


def get_embedding_statistics_lmdb(embeddings_dir: str, sample_size: int = 10000):
    """Get detailed statistics about the LMDB embeddings"""
    
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


def get_embedding_by_index_lmdb(embeddings_dir: str, embedding_idx: int):
    """Get a specific embedding by its index from LMDB"""
    
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


if __name__ == "__main__":
    # Update MODEL_NAME before running!
    
    # Main extraction
    results = main()
    
    # Or run specific variants:
    # extract_with_different_pooling_strategies()
    # extract_subset_for_testing()
    # extract_with_custom_sequence_length()
    
    # Analyze results
    # analysis = load_and_analyze_embeddings_lmdb("path/to/embeddings/directory")
    
    # Find specific person's embedding
    # person_result = find_person_embedding_lmdb("path/to/embeddings/directory", "12345")
    
    # Get detailed statistics
    # get_embedding_statistics_lmdb("path/to/embeddings/directory")
    
    # Get specific embedding by index
    # embedding = get_embedding_by_index_lmdb("path/to/embeddings/directory", 0)