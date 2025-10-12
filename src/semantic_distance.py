"""
Semantic Distance Computation for Life Sequences

Implements trajectory-based semantic distance metric for comparing life sequences.

Based on the distance metric:
    d(M_u, M_v) = E_{t ~ ½(M_u + M_v)} [ | log M_u(t) − log M_v(t) | ]

Where:
    - M_u(t) = (∏ᵢ P_M(aᵢ | u, a_<i))^(1/m') is the sequence score
    - We use log-space: log M_u(t) = (1/m') * Σᵢ log P_M(aᵢ | u, a_<i)

Uses Monte Carlo estimation:
    1. Sample n trajectories from M conditioned on u → T_u
    2. Sample n trajectories from M conditioned on v → T_v
    3. Let U = T_u ∪ T_v (multiset union; size 2n)
    4. Estimate: d̂ = (1/(2n)) * Σ_{t∈U} | log M_u(t) − log M_v(t) |
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from tqdm import tqdm

from src.embedding_encoder_class import TimeTokenEmbeddingEncoder
from src.generation_utils import load_vocab, get_sequences_from_dataloader
from src.datamodule2 import PretrainDataModule
from src.paths import FPATH
import polars as pl
import pyarrow.dataset as ds


class SemanticDistanceComputer:
    """
    Compute semantic distances between life sequences using trajectory-based similarity.

    This class implements the trajectory-based distance metric where we compare
    how two prompts (life sequences) score future continuations.

    Args:
        model_name: Name of the model (e.g., "108_mature_lynx-pretrain-lr0.2")
        experiment_name: Experiment name (e.g., "destiny")
        device: Device to run on ('cuda' or 'cpu')
        load_sequences: Whether to load sequences immediately (default True)
        max_sequences: Maximum number of sequences to load (None for all)
    """

    def __init__(
        self,
        model_name: str,
        experiment_name: str,
        device: str = 'cuda',
        load_sequences: bool = True,
        max_sequences: Optional[int] = None
    ):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.device = device

        print(f"Initializing SemanticDistanceComputer")
        print(f"  Model: {model_name}")
        print(f"  Experiment: {experiment_name}")
        print(f"  Device: {device}")

        # Load model
        checkpoint_path = FPATH.CHECKPOINTS_TRANSFORMER / experiment_name / model_name / "best.ckpt"
        if not checkpoint_path.exists():
            checkpoint_path = FPATH.CHECKPOINTS_TRANSFORMER / experiment_name / model_name / "last.ckpt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found for {model_name}")

        print(f"\nLoading model from: {checkpoint_path}")
        self.model = TimeTokenEmbeddingEncoder.from_checkpoint(str(checkpoint_path))
        self.model.to(device)
        self.model.eval()

        # Store original dtype
        self.original_dtype = next(self.model.parameters()).dtype
        if self.original_dtype == torch.float16:
            self.model._mixed_precision = False

        # Load hparams and setup datamodule
        hparams_path = FPATH.TB_LOGS / experiment_name / model_name / "hparams.yaml"
        if not hparams_path.exists():
            raise FileNotFoundError(f"hparams.yaml not found: {hparams_path}")

        with open(hparams_path, "r", encoding="utf-8") as f:
            self.hparams = yaml.safe_load(f)

        print(f"Setting up datamodule...")
        self._setup_datamodule()

        # Load vocabulary
        vocab_path = FPATH.DATA / self.hparams["dir_path"] / "vocab.json"
        print(f"Loading vocabulary: {vocab_path}")
        self.vocab = load_vocab(str(vocab_path))
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")

        # Storage for sequences and caches
        self.person_sequences = {}  # person_id -> token sequence
        self.person_id_to_idx = {}  # person_id -> dataset index
        self._trajectory_cache = {}  # (person_id, n, m, temp, top_p) -> trajectories
        self._score_cache = {}  # (prompt_person_id, trajectory_hash) -> score

        # Load sequences if requested
        if load_sequences:
            self.load_person_sequences(max_sequences=max_sequences)

        print(f"\n✓ SemanticDistanceComputer initialized")

    def _setup_datamodule(self):
        """Setup datamodule from hparams"""
        source_paths = [
            (FPATH.DATA / self.hparams["source_dir"] / path).with_suffix(".parquet")
            for path in self.hparams["sources"]
        ]
        background_path = (
            FPATH.DATA / self.hparams["source_dir"] / self.hparams["background"]
        ).with_suffix(".parquet")

        sources = [ds.dataset(s, format="parquet") for s in source_paths]
        background = pl.read_parquet(background_path)

        self.datamodule = PretrainDataModule(
            dir_path=FPATH.DATA / self.hparams["dir_path"],
            sources=sources,
            background=background,
            subset_background=self.hparams["subset_background"],
            n_tokens=self.hparams["n_tokens"],
            lengths=self.hparams["lengths"],
            num_workers=0,
            max_seq_len=self.hparams["max_seq_len"],
            source_dir=self.hparams["source_dir"],
            pretrain_style=self.hparams["pretrain_style"],
            masking_ratio=self.hparams.get("masking_ratio"),
        )

        self.datamodule.prepare_data()

    def load_person_sequences(
        self,
        person_ids: Optional[List[int]] = None,
        max_sequences: Optional[int] = None
    ):
        """
        Load sequences from datamodule for specified person_ids.

        Args:
            person_ids: List of person IDs to load (None for all)
            max_sequences: Maximum number of sequences to load
        """
        print(f"\nLoading person sequences...")

        # Get sequences from dataloader
        num_to_load = max_sequences if max_sequences is not None else 1000
        sequences = get_sequences_from_dataloader(self.datamodule, num_to_load, self.device)

        if not sequences:
            raise ValueError("No sequences found in dataset")

        # Store sequences
        for idx, person_data in enumerate(sequences):
            person_id = person_data["synthetic_person_id"]

            # Filter by person_ids if specified
            if person_ids is not None and person_id not in person_ids:
                continue

            # Extract sequence tokens (remove padding)
            sequence = person_data['event'][0].cpu().tolist()
            actual_length = (person_data['event'][0] != 0).sum().item()
            sequence = sequence[:actual_length]

            self.person_sequences[person_id] = sequence
            self.person_id_to_idx[person_id] = idx

        print(f"Loaded {len(self.person_sequences)} person sequences")
        if person_ids is not None:
            missing = set(person_ids) - set(self.person_sequences.keys())
            if missing:
                print(f"Warning: Could not find sequences for person_ids: {missing}")

    def get_person_sequence(self, person_id: int) -> List[int]:
        """
        Get token sequence for a specific person_id.

        Args:
            person_id: Person ID to lookup

        Returns:
            List of token IDs for the person's sequence
        """
        if person_id not in self.person_sequences:
            raise ValueError(f"Person ID {person_id} not found. Load sequences first.")
        return self.person_sequences[person_id]

    @torch.no_grad()
    def sample_trajectories(
        self,
        prompt_tokens: List[int],
        n: int = 20,
        m: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token: Optional[int] = None
    ) -> List[List[int]]:
        """
        Sample n trajectories from prompt, each max m tokens.

        Implements ancestral sampling with temperature and top-p.

        Args:
            prompt_tokens: List of token IDs for the prompt
            n: Number of trajectories to sample
            m: Maximum length of each trajectory
            temperature: Sampling temperature (λ in the pseudocode)
            top_p: Top-p (nucleus) sampling parameter
            eos_token: Optional end-of-sequence token ID

        Returns:
            List of n trajectories, each a list of token IDs
        """
        trajectories = []

        for traj_idx in range(n):
            # Set random seed for reproducibility
            torch.manual_seed(42 + traj_idx)

            trajectory = []
            context = prompt_tokens.copy()

            for step in range(m):
                # Prepare batch
                batch = {
                    'event': torch.tensor([context], dtype=torch.long, device=self.device),
                    'attn_mask': torch.ones(1, len(context), dtype=torch.long, device=self.device)
                }

                # Get next-token logits
                if self.original_dtype == torch.float16:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model.forward_generation(batch)
                else:
                    logits = self.model.forward_generation(batch)

                # Get logits for last position
                next_token_logits = logits[0, -1, :].float()  # [vocab_size]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-p filtering if specified
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[0] = False

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('inf')

                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Add to trajectory
                trajectory.append(next_token)
                context.append(next_token)

                # Check for EOS
                if eos_token is not None and next_token == eos_token:
                    break

            trajectories.append(trajectory)

        return trajectories

    @torch.no_grad()
    def sequence_score_log(
        self,
        prompt_tokens: List[int],
        trajectory_tokens: List[int]
    ) -> float:
        """
        Compute log M_u(t) = (1/m') * Σ log P(ai | u, a<i)

        Returns geometric mean score in log space.

        Args:
            prompt_tokens: List of token IDs for the prompt
            trajectory_tokens: List of token IDs for the trajectory

        Returns:
            Log-score (normalized by trajectory length)
        """
        if len(trajectory_tokens) == 0:
            return 0.0  # Handle empty trajectory

        # Build full context (prompt + trajectory)
        context = prompt_tokens.copy()
        acc_log = 0.0

        for token_idx, token in enumerate(trajectory_tokens):
            # Prepare batch
            batch = {
                'event': torch.tensor([context], dtype=torch.long, device=self.device),
                'attn_mask': torch.ones(1, len(context), dtype=torch.long, device=self.device)
            }

            # Get logits
            if self.original_dtype == torch.float16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model.forward_generation(batch)
            else:
                logits = self.model.forward_generation(batch)

            # Get log-probabilities for the last position
            next_token_logits = logits[0, -1, :].float()
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            # Get log-prob for the actual token
            token_log_prob = log_probs[token].item()
            acc_log += token_log_prob

            # Add token to context for next iteration
            context.append(token)

        # Return normalized log-score (geometric mean in log space)
        return acc_log / len(trajectory_tokens)

    def compute_distance(
        self,
        person_id_u: int,
        person_id_v: int,
        n: int = 20,
        m: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        use_cache: bool = True
    ) -> float:
        """
        Compute semantic distance between two person sequences.

        Algorithm:
        1. Get sequences for person_id_u and person_id_v
        2. Sample n trajectories from each (T_u and T_v)
        3. Union U = T_u ∪ T_v (size 2n)
        4. For each t in U: compute |log M_u(t) - log M_v(t)|
        5. Return average: d̂ = (1/2n) * Σ |differences|

        Args:
            person_id_u: First person ID
            person_id_v: Second person ID
            n: Number of trajectories per prompt
            m: Maximum trajectory length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_cache: Whether to use cached trajectories

        Returns:
            Distance estimate (non-negative float)
        """
        # Get sequences
        seq_u = self.get_person_sequence(person_id_u)
        seq_v = self.get_person_sequence(person_id_v)

        # Sample or retrieve trajectories
        cache_key_u = (person_id_u, n, m, temperature, top_p)
        cache_key_v = (person_id_v, n, m, temperature, top_p)

        if use_cache and cache_key_u in self._trajectory_cache:
            trajectories_u = self._trajectory_cache[cache_key_u]
        else:
            trajectories_u = self.sample_trajectories(seq_u, n, m, temperature, top_p)
            if use_cache:
                self._trajectory_cache[cache_key_u] = trajectories_u

        if use_cache and cache_key_v in self._trajectory_cache:
            trajectories_v = self._trajectory_cache[cache_key_v]
        else:
            trajectories_v = self.sample_trajectories(seq_v, n, m, temperature, top_p)
            if use_cache:
                self._trajectory_cache[cache_key_v] = trajectories_v

        # Union of trajectories
        all_trajectories = trajectories_u + trajectories_v

        # Compute score differences
        differences = []
        for trajectory in all_trajectories:
            score_u = self.sequence_score_log(seq_u, trajectory)
            score_v = self.sequence_score_log(seq_v, trajectory)
            diff = abs(score_u - score_v)
            differences.append(diff)

        # Return average
        distance = sum(differences) / (2 * n)
        return distance

    def compute_distance_matrix(
        self,
        person_ids: List[int],
        n: int = 20,
        m: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        cache_trajectories: bool = True,
        symmetric: bool = True
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix for list of person_ids.

        Args:
            person_ids: List of person IDs
            n: Number of trajectories per prompt
            m: Maximum trajectory length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            cache_trajectories: Whether to cache trajectories
            symmetric: If True, only compute upper triangle and mirror

        Returns:
            NxN distance matrix (numpy array)
        """
        n_persons = len(person_ids)
        distance_matrix = np.zeros((n_persons, n_persons))

        print(f"\nComputing {n_persons}x{n_persons} distance matrix...")

        # Optionally pre-cache all trajectories
        if cache_trajectories:
            print("Pre-caching trajectories...")
            for person_id in tqdm(person_ids, desc="Caching"):
                cache_key = (person_id, n, m, temperature, top_p)
                if cache_key not in self._trajectory_cache:
                    seq = self.get_person_sequence(person_id)
                    trajectories = self.sample_trajectories(seq, n, m, temperature, top_p)
                    self._trajectory_cache[cache_key] = trajectories

        # Compute distances
        total_pairs = n_persons * (n_persons - 1) // 2 if symmetric else n_persons * n_persons
        pbar = tqdm(total=total_pairs, desc="Computing distances")

        for i, person_id_u in enumerate(person_ids):
            start_j = i + 1 if symmetric else 0
            for j in range(start_j, n_persons):
                person_id_v = person_ids[j]

                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    distance = self.compute_distance(
                        person_id_u, person_id_v, n, m, temperature, top_p, use_cache=True
                    )
                    distance_matrix[i, j] = distance

                    # Mirror if symmetric
                    if symmetric and i != j:
                        distance_matrix[j, i] = distance

                pbar.update(1)

        pbar.close()
        print(f"Distance matrix computed: {distance_matrix.shape}")

        return distance_matrix

    def compute_distances_to_target(
        self,
        target_person_id: int,
        candidate_person_ids: List[int],
        n: int = 20,
        m: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> np.ndarray:
        """
        Compute distances from target to all candidates (more efficient than full matrix).

        Args:
            target_person_id: Target person ID
            candidate_person_ids: List of candidate person IDs
            n: Number of trajectories per prompt
            m: Maximum trajectory length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Array of distances from target to each candidate
        """
        distances = []

        print(f"\nComputing distances from person {target_person_id} to {len(candidate_person_ids)} candidates...")

        for candidate_id in tqdm(candidate_person_ids, desc="Computing"):
            distance = self.compute_distance(
                target_person_id, candidate_id, n, m, temperature, top_p, use_cache=True
            )
            distances.append(distance)

        return np.array(distances)

    def cache_trajectories(
        self,
        person_ids: List[int],
        n: int = 20,
        m: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9
    ):
        """
        Pre-generate and cache trajectories for efficiency.

        Args:
            person_ids: List of person IDs to cache trajectories for
            n: Number of trajectories per person
            m: Maximum trajectory length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        print(f"\nCaching trajectories for {len(person_ids)} persons...")

        for person_id in tqdm(person_ids, desc="Caching"):
            cache_key = (person_id, n, m, temperature, top_p)
            if cache_key not in self._trajectory_cache:
                seq = self.get_person_sequence(person_id)
                trajectories = self.sample_trajectories(seq, n, m, temperature, top_p)
                self._trajectory_cache[cache_key] = trajectories

        print(f"Cached {len(self._trajectory_cache)} trajectory sets")

    def decode_trajectory(self, trajectory_tokens: List[int]) -> List[str]:
        """
        Convert trajectory tokens to human-readable format.

        Args:
            trajectory_tokens: List of token IDs

        Returns:
            List of token strings
        """
        return [self.vocab.get(token_id, f"<UNK:{token_id}>") for token_id in trajectory_tokens]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return statistics about cached trajectories, scores, etc.

        Returns:
            Dictionary with statistics
        """
        return {
            'num_persons_loaded': len(self.person_sequences),
            'num_cached_trajectory_sets': len(self._trajectory_cache),
            'num_cached_scores': len(self._score_cache),
            'person_ids': list(self.person_sequences.keys()),
            'model_name': self.model_name,
            'experiment_name': self.experiment_name,
            'vocab_size': len(self.vocab)
        }

    def clear_caches(self):
        """Clear all caches to free memory"""
        self._trajectory_cache.clear()
        self._score_cache.clear()
        print("Caches cleared")
