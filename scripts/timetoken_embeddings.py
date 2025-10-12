#!/usr/bin/env python3
"""
Time Token Sentence Embeddings Extraction Script

Extract sentence-level embeddings from time token transformer models.
Compatible with models trained using encoder_nano_timetoken.py architecture.

Usage: python scripts/timetoken_embeddings.py
"""

import yaml
import polars as pl
import pyarrow.dataset as ds

from src.embedding_encoder_class import TimeTokenEmbeddingEncoder
from src.datamodule2 import PretrainDataModule
from src.paths import FPATH
from src.generation_utils import load_vocab


# ——— Configuration ———
MODEL_NAME = "029"
EXPERIMENT_NAME = "destiny"
POOLING_METHOD = "mean"  # Options: "mean", "last", "max", "cls"
EXCLUDE_TIME_TOKENS = True  # True to exclude YEAR_XXXX/AGE_YY tokens, False to include all
MAX_SEQUENCES = None  # Set to int to limit, None for all
INCLUDE_DATASETS = ["train", "val"]
# ————————————————

print("TIME TOKEN SENTENCE EMBEDDINGS EXTRACTION")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Pooling method: {POOLING_METHOD}")
print(f"Exclude time tokens: {EXCLUDE_TIME_TOKENS}")
print(f"Datasets: {INCLUDE_DATASETS}")
print(f"Max sequences: {MAX_SEQUENCES or 'All'}")

# 1) Load model from checkpoint
checkpoint_path = FPATH.CHECKPOINTS_TRANSFORMER / EXPERIMENT_NAME / MODEL_NAME / "best.ckpt"
if not checkpoint_path.exists():
    checkpoint_path = FPATH.CHECKPOINTS_TRANSFORMER / EXPERIMENT_NAME / MODEL_NAME / "last.ckpt"

if not checkpoint_path.exists():
    raise FileNotFoundError(f"No checkpoint found for {checkpoint_path}")

print(f"\\nLoading model from: {checkpoint_path}")
model = TimeTokenEmbeddingEncoder.from_checkpoint(str(checkpoint_path))

# 2) Load hparams and set up datamodule
hparams_path = FPATH.TB_LOGS / EXPERIMENT_NAME / MODEL_NAME / "hparams.yaml"
if not hparams_path.exists():
    raise FileNotFoundError(f"hparams.yaml not found: {hparams_path}")

with open(hparams_path, "r", encoding="utf-8") as f:
    hparams = yaml.safe_load(f)

print(f"\\nSetting up datamodule...")
print(f"   - Source dir: {hparams['source_dir']}")
print(f"   - Sources: {hparams['sources']}")

# Set up datamodule paths
source_paths = [
    (FPATH.DATA / hparams["source_dir"] / path).with_suffix(".parquet")
    for path in hparams["sources"]
]
background_path = (
    FPATH.DATA / hparams["source_dir"] / hparams["background"]
).with_suffix(".parquet")

sources = [ds.dataset(s, format="parquet") for s in source_paths]
background = pl.read_parquet(background_path)

# Create datamodule
datamodule = PretrainDataModule(
    dir_path=FPATH.DATA / hparams["dir_path"],
    sources=sources,
    background=background,
    subset_background=hparams["subset_background"],
    n_tokens=hparams["n_tokens"],
    lengths=hparams["lengths"],
    num_workers=0,  # Set to 0 for embedding extraction
    max_seq_len=hparams["max_seq_len"],
    source_dir=hparams["source_dir"],
    pretrain_style=hparams["pretrain_style"],
    masking_ratio=hparams.get("masking_ratio"),
)

# Prepare data
print("Preparing data...")
datamodule.prepare_data()

# 3) Load vocabulary for time token filtering
vocab = None
if EXCLUDE_TIME_TOKENS:
    vocab_path = FPATH.DATA / hparams["dir_path"] / "vocab.json"
    print(f"Loading vocabulary for time token filtering: {vocab_path}")
    vocab = load_vocab(str(vocab_path))
    print(f"Loaded vocabulary with {len(vocab)} tokens")

# 4) Set output directory
output_suffix = f"{POOLING_METHOD}"
if EXCLUDE_TIME_TOKENS:
    output_suffix += "_filtered"
output_dir = FPATH.EMBEDDINGS / f"{MODEL_NAME}_{output_suffix}"
if MAX_SEQUENCES:
    output_dir = output_dir.with_name(f"{output_dir.name}_{MAX_SEQUENCES}seqs")

print(f"\\nOutput directory: {output_dir}")

# 5) Extract embeddings
results = model.extract_and_save_embeddings(
    datamodule=datamodule,
    output_dir=str(output_dir),
    pooling_strategy=POOLING_METHOD,
    exclude_time_tokens=EXCLUDE_TIME_TOKENS,
    max_seq_len=None,  # Use model default
    include_datasets=INCLUDE_DATASETS,
    max_sequences=MAX_SEQUENCES,
    vocab=vocab
)

# 6) Summary
print(f"\\n🎉 EXTRACTION COMPLETE!")
print(f"Results:")
print(f"   - Total embeddings: {results['embeddings'].shape[0]:,}")
print(f"   - Embedding dimensions: {results['embeddings'].shape[1]}")
print(f"   - Pooling method: {POOLING_METHOD}")
print(f"   - Exclude time tokens: {EXCLUDE_TIME_TOKENS}")
print(f"   - Output directory: {results['output_dir']}")

# 7) Quick embedding statistics
embeddings = results['embeddings']
print(f"\\nEmbedding Statistics:")
print(f"   - Mean: {embeddings.mean():.6f}")
print(f"   - Std:  {embeddings.std():.6f}")
print(f"   - Min:  {embeddings.min():.6f}")
print(f"   - Max:  {embeddings.max():.6f}")

print(f"\\nFiles saved:")
print(f"   - embeddings.npy: {embeddings.nbytes / 1024**2:.1f} MB")
print(f"   - person_id_mapping.csv: Person ID to embedding index mapping")
print(f"   - metadata.pkl: Batch metadata and processing info")
print(f"   - config.json: Extraction configuration and model info")

print(f"\\n✅ Time token sentence embeddings ready for analysis!")

# Show configuration summary
print(f"\\nConfiguration Summary:")
print(f"   - Pooling method: {POOLING_METHOD}")
print(f"   - Time tokens excluded: {'Yes' if EXCLUDE_TIME_TOKENS else 'No'}")
print(f"   - Vocabulary used: {'Yes' if vocab else 'No'}")
if vocab and EXCLUDE_TIME_TOKENS:
    # Count time tokens in vocabulary
    time_tokens = [token for token in vocab.values() if token.startswith(('YEAR_', 'AGE_'))]
    print(f"   - Time tokens in vocab: {len(time_tokens):,} / {len(vocab):,}")