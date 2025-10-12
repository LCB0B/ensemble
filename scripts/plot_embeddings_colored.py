#!/usr/bin/env python3
"""
Token-Based Embedding Visualization Script

Visualize transformer embeddings in 2D using various projection methods,
with colors based on token-level features from sequences.

Usage:
    python scripts/plot_embeddings_colored.py \
        --embeddings_dir embeddings/029_mean_filtered \
        --model_name 029 \
        --experiment destiny \
        --method umap \
        --color-by birth_year gender most_common_token \
        --output figures/embeddings/
"""

import argparse
import json
import lzma
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import polars as pl
import yaml
import matplotlib.pyplot as plt
import lmdb
import msgpack

from src.paths import FPATH
from src.generation_utils import load_vocab


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_hparams_and_paths(model_name: str, experiment: str) -> Dict:
    """
    Load model hyperparameters and extract dataset paths.

    Args:
        model_name: Model name (e.g., "029")
        experiment: Experiment name (e.g., "destiny")

    Returns:
        Dictionary with hparams and derived paths
    """
    hparams_path = FPATH.TB_LOGS / experiment / model_name / "hparams.yaml"

    if not hparams_path.exists():
        raise FileNotFoundError(f"hparams.yaml not found: {hparams_path}")

    print(f"Loading hparams from {hparams_path}")
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)

    # Extract dataset directory
    dataset_dir = FPATH.DATA / hparams["dir_path"]
    vocab_path = dataset_dir / "vocab.json"

    print(f"  Dataset dir: {dataset_dir}")
    print(f"  Vocab path: {vocab_path}")

    return {
        'hparams': hparams,
        'dataset_dir': dataset_dir,
        'vocab_path': vocab_path
    }


def load_embeddings_and_mapping(embeddings_dir: Path) -> Tuple[np.ndarray, pl.DataFrame]:
    """
    Load embeddings and person ID mapping.

    Args:
        embeddings_dir: Directory containing embeddings.npy and person_id_mapping.csv

    Returns:
        Tuple of (embeddings array, mapping dataframe)
    """
    embeddings_path = embeddings_dir / "embeddings.npy"
    mapping_path = embeddings_dir / "person_id_mapping.csv"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")

    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)

    print(f"Loading mapping from {mapping_path}")
    mapping = pl.read_csv(mapping_path)

    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"Loaded {len(mapping)} person ID mappings")

    return embeddings, mapping


def load_sequences_from_lmdb(dataset_dir: Path, person_ids: List[int]) -> Dict[int, List[int]]:
    """
    Load sequences from LMDB dataset.

    Args:
        dataset_dir: Path to dataset directory containing LMDB files
        person_ids: List of person IDs to load

    Returns:
        Dictionary mapping person_id -> token sequence
    """
    # Load person_id to database_idx mapping
    pnr_mapping_path = dataset_dir / "pnr_to_database_idx.json"
    if not pnr_mapping_path.exists():
        print(f"WARNING: pnr_to_database_idx.json not found at {pnr_mapping_path}")
        return {}

    with open(pnr_mapping_path, "r", encoding="utf-8") as f:
        pnr_to_database_idx = json.load(f)

    # Open LMDB database
    lmdb_path = dataset_dir / "dataset.lmdb"
    if not lmdb_path.exists():
        print(f"WARNING: LMDB database not found at {lmdb_path}")
        return {}

    print(f"Loading sequences from {lmdb_path}")

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        create=False,
    )

    sequences = {}

    with env.begin() as txn:
        for person_id in person_ids:
            # Skip if person_id is None
            if person_id is None:
                continue

            # Look up database index
            database_idx = pnr_to_database_idx.get(str(person_id))
            if database_idx is None:
                continue

            # Get data from LMDB using database index as key
            key = str(database_idx).encode('utf-8')
            value = txn.get(key)

            if value is None:
                continue

            try:
                # Decode using lzma + msgpack (same as dataset.py)
                data = msgpack.unpackb(lzma.decompress(value), raw=False)

                # Extract event tokens
                if isinstance(data, dict) and 'event' in data:
                    events = data['event']

                    # Flatten if nested (events can be list of lists)
                    if events and isinstance(events[0], list):
                        # Flatten nested lists
                        sequence = [token for sublist in events for token in sublist]
                    else:
                        sequence = events

                    # Remove padding (0 tokens)
                    if isinstance(sequence, np.ndarray):
                        sequence = sequence[sequence != 0].tolist()
                    elif isinstance(sequence, list):
                        sequence = [t for t in sequence if t != 0]

                    sequences[person_id] = sequence

            except Exception as e:
                print(f"  Error decoding person_id {person_id}: {e}")
                continue

    env.close()

    print(f"Loaded {len(sequences)} sequences from LMDB")
    return sequences


# ============================================================================
# TOKEN FEATURE EXTRACTION
# ============================================================================

def extract_token_features(
    sequences: Dict[int, List[int]],
    vocab: Dict[int, str],
    mapping: pl.DataFrame
) -> pl.DataFrame:
    """
    Extract token-based features for each embedding.

    Args:
        sequences: Dictionary mapping person_id -> token sequence
        vocab: Dictionary mapping token_id -> token_name
        mapping: Person ID mapping dataframe

    Returns:
        DataFrame with extracted features
    """
    features = []

    for row in mapping.iter_rows(named=True):
        person_id = row['person_id']
        embedding_idx = row['embedding_idx']

        if person_id not in sequences or person_id is None:
            # No sequence available
            features.append({
                'embedding_idx': embedding_idx,
                'person_id': person_id,
                'sequence_length': 0,
                'token_diversity': 0,
                'most_common_token': 'UNKNOWN',
                'first_token': 'UNKNOWN',
                'last_token': 'UNKNOWN',
                'has_year_token': False,
                'has_age_token': False,
                'dominant_category': 'UNKNOWN',
                'birth_year': None,
                'gender': 'UNKNOWN'
            })
            continue

        sequence = sequences[person_id]

        if len(sequence) == 0:
            features.append({
                'embedding_idx': embedding_idx,
                'person_id': person_id,
                'sequence_length': 0,
                'token_diversity': 0,
                'most_common_token': 'UNKNOWN',
                'first_token': 'UNKNOWN',
                'last_token': 'UNKNOWN',
                'has_year_token': False,
                'has_age_token': False,
                'dominant_category': 'UNKNOWN',
                'birth_year': None,
                'gender': 'UNKNOWN'
            })
            continue

        # Decode tokens
        decoded_tokens = [vocab.get(tid, f"UNK_{tid}") for tid in sequence]

        # Calculate features
        token_counts = Counter(decoded_tokens)
        most_common = token_counts.most_common(1)[0][0] if token_counts else 'UNKNOWN'

        # Check for time tokens
        has_year = any(t.startswith('YEAR_') for t in decoded_tokens)
        has_age = any(t.startswith('AGE_') for t in decoded_tokens)

        # Determine dominant category
        year_count = sum(1 for t in decoded_tokens if t.startswith('YEAR_'))
        age_count = sum(1 for t in decoded_tokens if t.startswith('AGE_'))
        event_count = len(decoded_tokens) - year_count - age_count

        if year_count > age_count and year_count > event_count:
            dominant = 'YEAR'
        elif age_count > year_count and age_count > event_count:
            dominant = 'AGE'
        elif event_count > 0:
            dominant = 'EVENT'
        else:
            dominant = 'UNKNOWN'

        # Extract demographic features
        birth_year = None
        gender = 'UNKNOWN'

        for token in decoded_tokens:
            # Extract birth year (first occurrence)
            if birth_year is None and token.startswith('DEM_birthyear_'):
                try:
                    year_str = token.split('_')[-1]
                    birth_year = int(year_str)
                except (ValueError, IndexError):
                    pass

            # Extract gender (first occurrence)
            if gender == 'UNKNOWN' and token.startswith('DEM_female_'):
                try:
                    gender_value = token.split('_')[-1]
                    if gender_value == '0':
                        gender = 'Male'
                    elif gender_value == '1':
                        gender = 'Female'
                except IndexError:
                    pass

            # Stop if we've found both
            if birth_year is not None and gender != 'UNKNOWN':
                break

        features.append({
            'embedding_idx': embedding_idx,
            'person_id': person_id,
            'sequence_length': len(sequence),
            'token_diversity': len(set(sequence)),
            'most_common_token': most_common,
            'first_token': decoded_tokens[0] if decoded_tokens else 'UNKNOWN',
            'last_token': decoded_tokens[-1] if decoded_tokens else 'UNKNOWN',
            'has_year_token': has_year,
            'has_age_token': has_age,
            'dominant_category': dominant,
            'birth_year': birth_year,
            'gender': gender
        })

    return pl.DataFrame(features)


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def apply_projection(embeddings: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Apply dimensionality reduction to embeddings.

    Args:
        embeddings: High-dimensional embeddings
        method: Projection method ('pca', 'umap', 'tsne', 'pacmap')
        **kwargs: Additional arguments for the projection method

    Returns:
        2D projected embeddings
    """
    print(f"Applying {method.upper()} projection...")

    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, **kwargs)
        projected = reducer.fit_transform(embeddings)
        print(f"  Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")

    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                random_state=kwargs.get('random_state', 42),
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                metric=kwargs.get('metric', 'euclidean')
            )
            projected = reducer.fit_transform(embeddings)
        except ImportError:
            print("  ERROR: umap-learn not installed. Install with: pip install umap-learn")
            raise

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            random_state=kwargs.get('random_state', 42),
            perplexity=kwargs.get('perplexity', 30),
            n_iter=kwargs.get('n_iter', 1000)
        )
        projected = reducer.fit_transform(embeddings)

    elif method == 'pacmap':
        try:
            import pacmap
            reducer = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=kwargs.get('n_neighbors', 10),
                MN_ratio=kwargs.get('MN_ratio', 0.5),
                FP_ratio=kwargs.get('FP_ratio', 2.0)
            )
            projected = reducer.fit_transform(embeddings)
        except ImportError:
            print("  ERROR: pacmap not installed. Install with: pip install pacmap")
            raise

    else:
        raise ValueError(f"Unknown projection method: {method}")

    print(f"  Projection complete: {projected.shape}")
    return projected


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_continuous_feature(
    projected: np.ndarray,
    feature_values: np.ndarray,
    feature_name: str,
    method: str,
    output_path: Path
):
    """Plot embeddings colored by continuous feature."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=feature_values,
        cmap='viridis',
        s=5,
        alpha=0.6,
        edgecolors='none'
    )

    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title(f'Embeddings colored by {feature_name}', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(feature_name, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_categorical_feature(
    projected: np.ndarray,
    feature_values: List[str],
    feature_name: str,
    method: str,
    output_path: Path,
    max_categories: int = 20
):
    """Plot embeddings colored by categorical feature."""
    # Get unique categories
    unique_categories = list(set(feature_values))

    # If too many categories, keep only top N by frequency
    if len(unique_categories) > max_categories:
        category_counts = Counter(feature_values)
        top_categories = [cat for cat, _ in category_counts.most_common(max_categories)]

        # Map others to 'OTHER'
        feature_values = [
            val if val in top_categories else 'OTHER'
            for val in feature_values
        ]
        unique_categories = top_categories + ['OTHER']

    # Create color map
    cmap_name = 'tab20' if len(unique_categories) <= 20 else 'tab20b'
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = {cat: cmap(i / len(unique_categories)) for i, cat in enumerate(unique_categories)}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each category separately for legend
    for category in unique_categories:
        mask = np.array([val == category for val in feature_values])
        if mask.sum() == 0:
            continue

        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[colors[category]],
            label=category[:50],  # Truncate long labels
            s=5,
            alpha=0.6,
            edgecolors='none'
        )

    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title(f'Embeddings colored by {feature_name}', fontsize=14, fontweight='bold')

    # Legend
    if len(unique_categories) <= 15:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=8,
            markerscale=2,
            frameon=True
        )
    else:
        # Too many categories for legend
        ax.text(
            1.02, 0.5,
            f'{len(unique_categories)} categories',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='center'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_boolean_feature(
    projected: np.ndarray,
    feature_values: np.ndarray,
    feature_name: str,
    method: str,
    output_path: Path
):
    """Plot embeddings colored by boolean feature."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot False values
    mask_false = ~feature_values
    if mask_false.sum() > 0:
        ax.scatter(
            projected[mask_false, 0],
            projected[mask_false, 1],
            c='lightgray',
            label='False',
            s=5,
            alpha=0.4,
            edgecolors='none'
        )

    # Plot True values
    mask_true = feature_values
    if mask_true.sum() > 0:
        ax.scatter(
            projected[mask_true, 0],
            projected[mask_true, 1],
            c='red',
            label='True',
            s=5,
            alpha=0.7,
            edgecolors='none'
        )

    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title(f'Embeddings colored by {feature_name}', fontsize=14, fontweight='bold')

    ax.legend(fontsize=10, markerscale=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize embeddings with token-based coloring"
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        required=True,
        help='Directory containing embeddings.npy and person_id_mapping.csv'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name (e.g., "029")'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Experiment name (e.g., "destiny")'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='umap',
        choices=['pca', 'umap', 'tsne', 'pacmap'],
        help='Projection method'
    )
    parser.add_argument(
        '--color-by',
        type=str,
        nargs='+',
        default=['birth_year', 'gender', 'sequence_length'],
        help='Features to color by (birth_year, gender, sequence_length, token_diversity, etc.)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/embeddings_colored',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to plot (for speed)'
    )

    args = parser.parse_args()

    # Setup paths
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TOKEN-BASED EMBEDDING VISUALIZATION")
    print("=" * 80)

    # 1. Load hparams and extract dataset paths
    print(f"\nModel: {args.model_name}")
    print(f"Experiment: {args.experiment}")
    paths_info = load_hparams_and_paths(args.model_name, args.experiment)
    dataset_dir = paths_info['dataset_dir']
    vocab_path = paths_info['vocab_path']

    # 2. Load embeddings and mapping
    embeddings, mapping = load_embeddings_and_mapping(embeddings_dir)

    # Subsample if requested
    if args.max_samples and len(embeddings) > args.max_samples:
        print(f"\nSubsampling to {args.max_samples} embeddings...")
        indices = np.random.choice(len(embeddings), args.max_samples, replace=False)
        embeddings = embeddings[indices]
        mapping = mapping.filter(pl.col('embedding_idx').is_in(indices))

    # 3. Load vocabulary
    print(f"\nLoading vocabulary from {vocab_path}")
    vocab = load_vocab(str(vocab_path))
    print(f"Loaded vocabulary with {len(vocab)} tokens")

    # 4. Load sequences
    print("\nLoading sequences from dataset...")
    person_ids = mapping['person_id'].to_list()
    sequences = load_sequences_from_lmdb(dataset_dir, person_ids)

    # 5. Extract token features
    print("\nExtracting token-based features...")
    features_df = extract_token_features(sequences, vocab, mapping)
    print(f"Extracted features for {len(features_df)} embeddings")

    # 6. Apply projection
    print(f"\nApplying {args.method.upper()} projection...")
    projected = apply_projection(embeddings, args.method)

    # 7. Create plots for each feature
    print(f"\nCreating plots...")

    # Prepare birth_year as numeric array (convert None to NaN)
    birth_years = []
    for val in features_df['birth_year'].to_list():
        if val is None:
            birth_years.append(np.nan)
        else:
            birth_years.append(float(val))
    birth_year_array = np.array(birth_years)

    available_features = {
        'sequence_length': ('continuous', features_df['sequence_length'].to_numpy()),
        'token_diversity': ('continuous', features_df['token_diversity'].to_numpy()),
        'most_common_token': ('categorical', features_df['most_common_token'].to_list()),
        'first_token': ('categorical', features_df['first_token'].to_list()),
        'last_token': ('categorical', features_df['last_token'].to_list()),
        'has_year_token': ('boolean', features_df['has_year_token'].to_numpy()),
        'has_age_token': ('boolean', features_df['has_age_token'].to_numpy()),
        'dominant_category': ('categorical', features_df['dominant_category'].to_list()),
        'birth_year': ('continuous', birth_year_array),
        'gender': ('categorical', features_df['gender'].to_list()),
    }

    for feature_name in args.color_by:
        if feature_name not in available_features:
            print(f"  WARNING: Unknown feature '{feature_name}', skipping")
            continue

        feature_type, feature_values = available_features[feature_name]
        output_path = output_dir / f"{args.method}_{feature_name}.png"

        print(f"\n  Creating plot for '{feature_name}' ({feature_type})...")

        if feature_type == 'continuous':
            plot_continuous_feature(
                projected, feature_values, feature_name, args.method, output_path
            )
        elif feature_type == 'categorical':
            plot_categorical_feature(
                projected, feature_values, feature_name, args.method, output_path
            )
        elif feature_type == 'boolean':
            plot_boolean_feature(
                projected, feature_values, feature_name, args.method, output_path
            )

    print("\n" + "=" * 80)
    print(f"COMPLETE! Plots saved to {output_dir}")
    print("=" * 80)

    # Print summary
    print("\nFeature Statistics:")
    print(f"  Total embeddings: {len(embeddings)}")
    print(f"  Sequences loaded: {len(sequences)}")
    print(f"  Avg sequence length: {features_df['sequence_length'].mean():.1f}")
    print(f"  Avg token diversity: {features_df['token_diversity'].mean():.1f}")
    print(f"  Has YEAR tokens: {features_df['has_year_token'].sum()} / {len(features_df)}")
    print(f"  Has AGE tokens: {features_df['has_age_token'].sum()} / {len(features_df)}")

    # Birth year statistics
    birth_year_count = features_df.filter(pl.col('birth_year').is_not_null()).height
    if birth_year_count > 0:
        birth_year_values = features_df.filter(pl.col('birth_year').is_not_null())['birth_year']
        print(f"  Birth year available: {birth_year_count} / {len(features_df)}")
        print(f"    Birth year range: {birth_year_values.min():.0f} - {birth_year_values.max():.0f}")
    else:
        print(f"  Birth year available: 0 / {len(features_df)}")

    # Gender statistics
    gender_counts = features_df['gender'].value_counts()
    print(f"  Gender distribution:")
    for row in gender_counts.iter_rows(named=True):
        print(f"    {row['gender']}: {row['count']}")

    print("\nAvailable features for --color-by:")
    for feat in available_features.keys():
        print(f"  - {feat}")


if __name__ == "__main__":
    main()
