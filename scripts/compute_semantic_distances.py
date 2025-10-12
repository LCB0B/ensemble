#!/usr/bin/env python3
"""
Semantic Distance Computation Script

Compute semantic distances between life sequences using trajectory-based similarity.
Uses the SemanticDistanceComputer class to implement the distance metric:

    d(M_u, M_v) = E_{t ~ ½(M_u + M_v)} [ | log M_u(t) − log M_v(t) | ]

Usage: python scripts/compute_semantic_distances.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.semantic_distance import SemanticDistanceComputer
from src.paths import FPATH


# ——— Configuration ———
MODEL_NAME = "108_mature_lynx-pretrain-lr0.2"
EXPERIMENT_NAME = "destiny"
DEVICE = "cuda"

# Person IDs to compare (example - adjust based on your data)
PERSON_IDS = list(range(10))  # First 10 persons

# Distance computation parameters (from the mathematical formulation)
N_TRAJECTORIES = 20  # Number of trajectories per prompt
MAX_TRAJECTORY_LENGTH = 20  # Maximum tokens per trajectory
TEMPERATURE = 1.0  # Sampling temperature (λ)
TOP_P = 0.9  # Top-p sampling (optional)

# Computation mode
COMPUTE_FULL_MATRIX = True  # If False, only compute distances to first person
CACHE_TRAJECTORIES = True  # Pre-cache trajectories for efficiency

# Output
OUTPUT_DIR = FPATH.DATA / "semantic_distances"
SAVE_MATRIX = True
SAVE_VISUALIZATION = True
# ————————————————


def main():
    """Main execution"""
    print("=" * 60)
    print("SEMANTIC DISTANCE COMPUTATION")
    print("=" * 60)

    # Initialize computer
    print("\n1. Initializing SemanticDistanceComputer...")
    computer = SemanticDistanceComputer(
        model_name=MODEL_NAME,
        experiment_name=EXPERIMENT_NAME,
        device=DEVICE,
        load_sequences=True,
        max_sequences=max(PERSON_IDS) + 10 if PERSON_IDS else 100
    )

    # Show statistics
    stats = computer.get_statistics()
    print(f"\nLoaded {stats['num_persons_loaded']} person sequences")
    print(f"Available person IDs: {min(stats['person_ids'])} to {max(stats['person_ids'])}")

    # Filter person IDs to those actually loaded
    available_person_ids = [pid for pid in PERSON_IDS if pid in stats['person_ids']]
    if len(available_person_ids) < len(PERSON_IDS):
        print(f"\nWarning: Only {len(available_person_ids)}/{len(PERSON_IDS)} requested persons found")
        PERSON_IDS[:] = available_person_ids

    if not PERSON_IDS:
        print("Error: No valid person IDs found!")
        return

    # Show sample sequences
    print(f"\n2. Sample sequences:")
    for person_id in PERSON_IDS[:3]:  # Show first 3
        seq = computer.get_person_sequence(person_id)
        decoded = computer.decode_trajectory(seq[:10])  # First 10 tokens
        print(f"  Person {person_id}: {' '.join(decoded[:10])}... ({len(seq)} tokens total)")

    # Optionally pre-cache trajectories
    if CACHE_TRAJECTORIES:
        print(f"\n3. Pre-caching trajectories for efficiency...")
        computer.cache_trajectories(
            person_ids=PERSON_IDS,
            n=N_TRAJECTORIES,
            m=MAX_TRAJECTORY_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )

    # Compute distances
    if COMPUTE_FULL_MATRIX:
        print(f"\n4. Computing full distance matrix...")
        distance_matrix = computer.compute_distance_matrix(
            person_ids=PERSON_IDS,
            n=N_TRAJECTORIES,
            m=MAX_TRAJECTORY_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            cache_trajectories=CACHE_TRAJECTORIES,
            symmetric=True
        )

        # Show results
        print(f"\nDistance Matrix ({len(PERSON_IDS)}x{len(PERSON_IDS)}):")
        print(distance_matrix)

        print(f"\nStatistics:")
        # Get upper triangle (excluding diagonal)
        upper_tri = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        print(f"  Mean distance: {upper_tri.mean():.6f}")
        print(f"  Std distance:  {upper_tri.std():.6f}")
        print(f"  Min distance:  {upper_tri.min():.6f}")
        print(f"  Max distance:  {upper_tri.max():.6f}")

        # Find most similar and most different pairs
        min_idx = np.unravel_index(np.argmin(distance_matrix + np.eye(len(PERSON_IDS)) * 1e9), distance_matrix.shape)
        max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

        print(f"\nMost similar pair:")
        print(f"  Persons {PERSON_IDS[min_idx[0]]} and {PERSON_IDS[min_idx[1]]}: distance = {distance_matrix[min_idx]:.6f}")

        print(f"\nMost different pair:")
        print(f"  Persons {PERSON_IDS[max_idx[0]]} and {PERSON_IDS[max_idx[1]]}: distance = {distance_matrix[max_idx]:.6f}")

        # Save results
        if SAVE_MATRIX:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"distance_matrix_{MODEL_NAME}_n{N_TRAJECTORIES}_m{MAX_TRAJECTORY_LENGTH}.npy"
            np.save(output_file, distance_matrix)
            print(f"\n✓ Distance matrix saved to: {output_file}")

            # Save metadata
            metadata = {
                'model_name': MODEL_NAME,
                'experiment_name': EXPERIMENT_NAME,
                'person_ids': PERSON_IDS,
                'n_trajectories': N_TRAJECTORIES,
                'max_trajectory_length': MAX_TRAJECTORY_LENGTH,
                'temperature': TEMPERATURE,
                'top_p': TOP_P,
                'distance_matrix_shape': distance_matrix.shape,
                'mean_distance': float(upper_tri.mean()),
                'std_distance': float(upper_tri.std()),
                'min_distance': float(upper_tri.min()),
                'max_distance': float(upper_tri.max())
            }

            import json
            metadata_file = OUTPUT_DIR / f"distance_matrix_{MODEL_NAME}_n{N_TRAJECTORIES}_m{MAX_TRAJECTORY_LENGTH}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✓ Metadata saved to: {metadata_file}")

        # Visualize
        if SAVE_VISUALIZATION:
            visualize_distance_matrix(distance_matrix, PERSON_IDS, OUTPUT_DIR)

    else:
        # Compute distances to first person only
        target_person_id = PERSON_IDS[0]
        candidate_person_ids = PERSON_IDS[1:]

        print(f"\n4. Computing distances from person {target_person_id} to {len(candidate_person_ids)} candidates...")
        distances = computer.compute_distances_to_target(
            target_person_id=target_person_id,
            candidate_person_ids=candidate_person_ids,
            n=N_TRAJECTORIES,
            m=MAX_TRAJECTORY_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )

        # Show results
        print(f"\nDistances from person {target_person_id}:")
        for candidate_id, distance in zip(candidate_person_ids, distances):
            print(f"  Person {candidate_id}: {distance:.6f}")

        print(f"\nStatistics:")
        print(f"  Mean distance: {distances.mean():.6f}")
        print(f"  Std distance:  {distances.std():.6f}")
        print(f"  Min distance:  {distances.min():.6f}")
        print(f"  Max distance:  {distances.max():.6f}")

        # Find closest and farthest
        closest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)

        print(f"\nClosest person: {candidate_person_ids[closest_idx]} (distance = {distances[closest_idx]:.6f})")
        print(f"Farthest person: {candidate_person_ids[farthest_idx]} (distance = {distances[farthest_idx]:.6f})")

        # Save results
        if SAVE_MATRIX:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = OUTPUT_DIR / f"distances_from_{target_person_id}_{MODEL_NAME}.npy"
            np.save(output_file, distances)
            print(f"\n✓ Distances saved to: {output_file}")

    print(f"\n✅ Semantic distance computation complete!")


def visualize_distance_matrix(distance_matrix, person_ids, output_dir):
    """Visualize distance matrix as a heatmap"""
    print(f"\nCreating visualization...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        distance_matrix,
        xticklabels=person_ids,
        yticklabels=person_ids,
        cmap='viridis',
        annot=True if len(person_ids) <= 10 else False,  # Annotate if small enough
        fmt='.3f',
        cbar_kws={'label': 'Semantic Distance'},
        ax=ax
    )

    ax.set_title(f'Semantic Distance Matrix\n({MODEL_NAME}, n={N_TRAJECTORIES}, m={MAX_TRAJECTORY_LENGTH})')
    ax.set_xlabel('Person ID')
    ax.set_ylabel('Person ID')

    plt.tight_layout()

    # Save
    output_file = output_dir / f"distance_matrix_{MODEL_NAME}_n{N_TRAJECTORIES}_m{MAX_TRAJECTORY_LENGTH}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}")

    plt.close()


def example_pairwise_comparison():
    """Example: Detailed comparison of two specific persons"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Detailed Pairwise Comparison")
    print("=" * 60)

    # Initialize
    computer = SemanticDistanceComputer(
        model_name=MODEL_NAME,
        experiment_name=EXPERIMENT_NAME,
        device=DEVICE,
        load_sequences=True,
        max_sequences=10
    )

    # Pick two persons
    person_id_u = 0
    person_id_v = 1

    # Get sequences
    seq_u = computer.get_person_sequence(person_id_u)
    seq_v = computer.get_person_sequence(person_id_v)

    print(f"\nComparing persons {person_id_u} and {person_id_v}:")
    print(f"  Person {person_id_u}: {len(seq_u)} tokens")
    print(f"  Person {person_id_v}: {len(seq_v)} tokens")

    # Decode first few tokens
    decoded_u = computer.decode_trajectory(seq_u[:15])
    decoded_v = computer.decode_trajectory(seq_v[:15])
    print(f"\n  Person {person_id_u} (first 15 tokens): {' '.join(decoded_u)}")
    print(f"  Person {person_id_v} (first 15 tokens): {' '.join(decoded_v)}")

    # Compute distance
    print(f"\nComputing distance with n={N_TRAJECTORIES}, m={MAX_TRAJECTORY_LENGTH}...")
    distance = computer.compute_distance(
        person_id_u=person_id_u,
        person_id_v=person_id_v,
        n=N_TRAJECTORIES,
        m=MAX_TRAJECTORY_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )

    print(f"\nSemantic distance: {distance:.6f}")

    # Show sample trajectories
    print(f"\nSample trajectory from person {person_id_u}:")
    trajectories_u = computer.sample_trajectories(seq_u, n=1, m=10)
    decoded = computer.decode_trajectory(trajectories_u[0])
    print(f"  {' '.join(decoded)}")

    print(f"\nSample trajectory from person {person_id_v}:")
    trajectories_v = computer.sample_trajectories(seq_v, n=1, m=10)
    decoded = computer.decode_trajectory(trajectories_v[0])
    print(f"  {' '.join(decoded)}")


if __name__ == "__main__":
    # Run main computation
    main()

    # Optionally run detailed pairwise example
    # example_pairwise_comparison()
