import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import polars as pl
from torch.utils.data import DataLoader

from src.encoder_nano_risk import PretrainNanoEncoder
from src.datamodule2 import BaseLightningDataModule


class EmbeddingNanoEncoder(PretrainNanoEncoder):
    """
    PretrainNanoEncoder extended with integrated embedding extraction capabilities.
    
    Provides methods to extract and save high-dimensional embeddings from sequences,
    with built-in support for different pooling strategies and person ID tracking.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Embedding extraction settings
        self.pooling_strategy = "mean"
        self.embedding_max_seq_len = self.hparams.max_seq_len
        
    def set_pooling_strategy(self, strategy: str = "mean"):
        """Set the pooling strategy for embedding extraction"""
        valid_strategies = ["mean", "last", "max", "cls"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid pooling strategy. Choose from: {valid_strategies}")
        self.pooling_strategy = strategy
        
    def mean_pooling_unpacked(self, embeddings: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Mean pooling over unpacked Flash Attention format"""
        batch_size = len(cu_seqlens) - 1
        pooled = []
        
        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_embeddings = embeddings[start_idx:end_idx]  # [seq_len, hidden_dim]
            mean_emb = seq_embeddings.mean(dim=0)  # [hidden_dim]
            pooled.append(mean_emb)
        
        return torch.stack(pooled)  # [batch_size, hidden_dim]
    
    def last_token_pooling_unpacked(self, embeddings: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Last token pooling over unpacked Flash Attention format"""
        batch_size = len(cu_seqlens) - 1
        pooled = []
        
        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            if end_idx > start_idx:
                last_emb = embeddings[end_idx - 1]  # Last token
            else:
                # Empty sequence fallback
                last_emb = torch.zeros(embeddings.size(-1), device=embeddings.device)
            pooled.append(last_emb)
        
        return torch.stack(pooled)
    
    def max_pooling_unpacked(self, embeddings: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Max pooling over unpacked Flash Attention format"""
        batch_size = len(cu_seqlens) - 1
        pooled = []
        
        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            if end_idx > start_idx:
                seq_embeddings = embeddings[start_idx:end_idx]
                max_emb, _ = seq_embeddings.max(dim=0)
            else:
                # Empty sequence fallback
                max_emb = torch.zeros(embeddings.size(-1), device=embeddings.device)
            pooled.append(max_emb)
        
        return torch.stack(pooled)
    
    def cls_token_pooling_unpacked(self, embeddings: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """First token (CLS) pooling over unpacked Flash Attention format"""
        batch_size = len(cu_seqlens) - 1
        pooled = []
        
        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            if end_idx > start_idx:
                first_emb = embeddings[start_idx]  # First token
            else:
                # Empty sequence fallback
                first_emb = torch.zeros(embeddings.size(-1), device=embeddings.device)
            pooled.append(first_emb)
        
        return torch.stack(pooled)
    
    def get_pooling_function_unpacked(self, strategy: str) -> Callable:
        """Get the unpacked pooling function based on strategy name"""
        pooling_functions = {
            'mean': self.mean_pooling_unpacked,
            'last': self.last_token_pooling_unpacked,
            'max': self.max_pooling_unpacked,
            'cls': self.cls_token_pooling_unpacked
        }
        return pooling_functions[strategy]
    @torch.no_grad()
    def extract_batch_embeddings(
        self, 
        batch: Dict[str, Any], 
        person_ids: Optional[List] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract embeddings from a single batch.
        Expects batch to already be in Flash Attention unpacked format.
        """
        device = next(self.parameters()).device

        # Move batch to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device)

        # Check if batch is in unpacked format (has Flash Attention keys)
        if 'cu_seqlens' not in batch or 'indices' not in batch:
            raise ValueError("Batch should be in unpacked Flash Attention format when reaching extract_batch_embeddings")

        # Determine if we should use autocast (same logic as generation)
        should_autocast = device.type == 'cuda'
        if should_autocast:
            # Check if model was trained with mixed precision
            model_dtype = next(self.parameters()).dtype
            should_autocast = model_dtype == torch.float16 or hasattr(self, '_mixed_precision')

        # Forward pass to get final layer embeddings (in unpacked format)
        if should_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                embeddings = self.forward(batch, repad=False)
        else:
            embeddings = self.forward(batch, repad=False)

        # Get original batch dimensions
        original_batch_size = len(batch['cu_seqlens']) - 1

        # Apply pooling directly on unpacked embeddings using cu_seqlens
        # This is more efficient and avoids the repadding step
        pooling_fn = self.get_pooling_function_unpacked(self.pooling_strategy)
        pooled_embeddings = pooling_fn(embeddings, batch['cu_seqlens'])

        # Create metadata - we need sequence lengths for each sequence
        sequence_lengths = []
        for i in range(original_batch_size):
            seq_len = batch['cu_seqlens'][i + 1] - batch['cu_seqlens'][i]
            sequence_lengths.append(seq_len.item())

        # Prepare metadata
        metadata = {
            'sequence_lengths': sequence_lengths,
            'batch_size': pooled_embeddings.size(0)
        }

        if person_ids is not None:
            metadata['person_ids'] = person_ids

        # Return CPU tensors to save memory
        return pooled_embeddings.cpu(), metadata


    def mean_pooling_packed(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over packed (standard batch) format"""
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        
        # Apply mask and sum
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, dim=1)
        
        # Count non-masked tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # Compute mean
        return sum_embeddings / sum_mask


    def last_token_pooling_packed(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last token pooling over packed format"""
        batch_size = embeddings.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1  # Get last valid position
        sequence_lengths = sequence_lengths.clamp(min=0)  # Ensure non-negative
        
        batch_indices = torch.arange(batch_size, device=embeddings.device)
        return embeddings[batch_indices, sequence_lengths]


    def max_pooling_packed(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Max pooling over packed format"""
        # Set masked positions to very negative values
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        embeddings_masked = embeddings.clone()
        embeddings_masked[input_mask_expanded == 0] = -1e9
        
        # Max pool
        max_embeddings, _ = torch.max(embeddings_masked, dim=1)
        return max_embeddings


    def cls_token_pooling_packed(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """First token (CLS) pooling over packed format"""
        return embeddings[:, 0, :]  # Just take first token


    def get_pooling_function_packed(self, strategy: str) -> Callable:
        """Get the packed pooling function based on strategy name"""
        pooling_functions = {
            'mean': self.mean_pooling_packed,
            'last': self.last_token_pooling_packed,
            'max': self.max_pooling_packed,
            'cls': self.cls_token_pooling_packed
        }
        return pooling_functions[strategy]


    # Alternative version that uses packed format pooling
    @torch.no_grad()
    def extract_batch_embeddings_alternative(
        self, 
        batch: Dict[str, Any], 
        person_ids: Optional[List] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Alternative version that repads and uses packed pooling functions.
        Use this if you prefer to work with standard batch format.
        """
        from flash_attn.bert_padding import pad_input
        
        device = next(self.parameters()).device
        
        # Move batch to device
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        
        # Check if batch is in unpacked format
        if 'cu_seqlens' not in batch or 'indices' not in batch:
            raise ValueError("Batch should be in unpacked Flash Attention format when reaching extract_batch_embeddings")
        
        # Determine autocast
        should_autocast = device.type == 'cuda'
        if should_autocast:
            model_dtype = next(self.parameters()).dtype
            should_autocast = model_dtype == torch.float16 or hasattr(self, '_mixed_precision')
        
        # Forward pass to get final layer embeddings (in unpacked format)
        if should_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                embeddings = self.forward(batch, repad=False)
        else:
            embeddings = self.forward(batch, repad=False)
        
        # Get original batch dimensions for repacking
        original_batch_size = len(batch['cu_seqlens']) - 1
        max_seq_len = batch['max_seqlen_in_batch']
        
        # Repad embeddings back to batch format for pooling
        embeddings_padded = pad_input(
            embeddings, 
            batch['indices'], 
            original_batch_size, 
            max_seq_len
        )
        
        # Create attention mask for pooling
        attention_mask = torch.zeros(original_batch_size, max_seq_len, device=device)
        for i in range(original_batch_size):
            seq_len = batch['cu_seqlens'][i + 1] - batch['cu_seqlens'][i]
            attention_mask[i, :seq_len] = 1.0
        
        # Apply pooling on the padded embeddings
        if embeddings_padded.dtype != torch.float32:
            embeddings_padded = embeddings_padded.float()
        if attention_mask.dtype != torch.float32:
            attention_mask = attention_mask.float()
            
        pooling_fn = self.get_pooling_function_packed(self.pooling_strategy)
        pooled_embeddings = pooling_fn(embeddings_padded, attention_mask)
        
        # Prepare metadata
        sequence_lengths = attention_mask.sum(dim=1).cpu().tolist()
        metadata = {
            'attention_mask': attention_mask.cpu(),
            'sequence_lengths': sequence_lengths,
            'batch_size': pooled_embeddings.size(0)
        }
        
        if person_ids is not None:
            metadata['person_ids'] = person_ids
        
        # Return CPU tensors to save memory
        return pooled_embeddings.cpu(), metadata

    
    def extract_datamodule_embeddings(
        self, 
        datamodule: BaseLightningDataModule,
        pooling_strategy: str = "mean",
        max_seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        include_datasets: List[str] = ["train", "val", "predict"],
        max_sequences: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract embeddings from entire datamodule
        
        Args:
            datamodule: Lightning datamodule with data
            pooling_strategy: Pooling strategy to use
            max_seq_len: Override max sequence length
            batch_size: Override batch size for extraction
            include_datasets: Which datasets to process ["train", "val", "predict"]
            max_sequences: Maximum number of sequences to process (None for all)
        
        Returns:
            embeddings: numpy array of all embeddings
            metadata: list of metadata dictionaries with person_ids
        """
        self.eval()
        
        # Set pooling strategy
        self.set_pooling_strategy(pooling_strategy)
        
        # Override max seq len if provided
        if max_seq_len is not None:
            self.embedding_max_seq_len = max_seq_len
        
        # Ensure datamodule is prepared
        if not hasattr(datamodule, 'train_dataset') or datamodule.train_dataset is None:
            datamodule.setup()
        
        # Collect datasets to process
        datasets_to_process = []
        
        for dataset_name in include_datasets:
            dataset_attr = f"{dataset_name}_dataset"
            if hasattr(datamodule, dataset_attr):
                dataset = getattr(datamodule, dataset_attr)
                if dataset is not None:
                    datasets_to_process.append((dataset_name, dataset))
        
        if not datasets_to_process:
            raise ValueError("No datasets found to process!")
        
        all_embeddings = []
        all_metadata = []
        total_sequences_processed = 0
        

        print(f"Extracting embeddings with {pooling_strategy} pooling...")
        print(f"Max sequence length: {self.embedding_max_seq_len}")
        if max_sequences is not None:
            print(f"Max sequences to process: {max_sequences:,}")
        else:
            print("Processing all available sequences")
        
        for dataset_name, dataset in datasets_to_process:
            if max_sequences is not None and total_sequences_processed >= max_sequences:
                print(f"Reached maximum sequences limit ({max_sequences}), stopping")
                break
                
            sequences_remaining = None
            if max_sequences is not None:
                sequences_remaining = max_sequences - total_sequences_processed
                
            print(f"\nProcessing {dataset_name} dataset ({len(dataset)} samples)...")
            if sequences_remaining is not None:
                print(f"Will process up to {sequences_remaining:,} sequences from this dataset")
            
            # Create dataloader - handle sequence length modifications if needed
            if max_seq_len is not None and max_seq_len != datamodule.max_seq_len:
                # Need custom dataloader to handle different sequence length
                dataloader = self._create_custom_dataloader(dataset, datamodule, batch_size)
            else:
                # Use datamodule's default dataloader
                if batch_size is not None:
                    # Create custom dataloader with specified batch size
                    from torch.utils.data import DataLoader
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=datamodule.collate_fn(),
                        pin_memory=True
                    )
                else:
                    # Use datamodule's default dataloader
                    if dataset_name == "train":
                        dataloader = datamodule.train_dataloader()
                    elif dataset_name == "val":
                        dataloader = datamodule.val_dataloader()  
                    else:  # predict
                        dataloader = datamodule.predict_dataloader()
            
            # NOTE: All batches need unpacking since on_after_batch_transfer is not called
            # when manually iterating over dataloaders
            
            dataset_embeddings = []
            dataset_metadata = []
            
            # Track current position in dataset for person_id extraction
            current_idx = 0
            if max_sequences is not None:
                total_seqs = min(max_sequences, len(dataset))
            else:
                total_seqs = len(dataset)
            pbar = tqdm(dataloader,desc=f"Extracting {dataset_name}", total=total_seqs,
                unit="seq",
                leave=True,
            )

            for batch_idx, batch in enumerate(pbar):
                # Check if we've reached the maximum sequences limit
                if max_sequences is not None and total_sequences_processed >= max_sequences:
                    break

                try:
                    # Extract person IDs for this batch if available (before any batch modifications)
                    person_ids = None
                    batch_size_actual = batch['event'].size(0)
                    
                    if hasattr(dataset, 'observations') and 'person_id' in dataset.observations:
                        end_idx = min(current_idx + batch_size_actual, len(dataset.observations['person_id']))
                        person_ids = dataset.observations['person_id'][current_idx:end_idx]
                        current_idx += batch_size_actual
                    
                    # If we have a sequence limit, check if this batch would exceed it
                    if max_sequences is not None:
                        sequences_left = max_sequences - total_sequences_processed
                        if sequences_left <= 0:
                            break
                        elif batch_size_actual > sequences_left:
                            # Truncate the batch to fit within the limit (in packed format)
                            batch = self._truncate_batch(batch, sequences_left)
                            if person_ids is not None:
                                person_ids = person_ids[:sequences_left]
                            batch_size_actual = sequences_left
                    if hasattr(self, '_mixed_precision') and self._mixed_precision:
                        with torch.autocast(device_type='cuda', enabled=True):
                                
                            # All batches need unpacking since on_after_batch_transfer is not called
                            batch = self._prepare_batch_for_extraction(batch, datamodule)
                            
                            # Extract embeddings
                            embeddings, metadata = self.extract_batch_embeddings(batch, person_ids)
                            
                            dataset_embeddings.append(embeddings.numpy())
                            
                            # Add dataset info to metadata
                            metadata['dataset'] = dataset_name
                            metadata['batch_idx'] = batch_idx
                            dataset_metadata.append(metadata)
                            
                            total_sequences_processed += batch_size_actual
                            # Manually advance the bar by the number of sequences in this batch
                            pbar.update(batch_size_actual)
                            # And optionally show it as a postfix
                            pbar.set_postfix(seqs=total_sequences_processed)

                except Exception as e:
                    print(f"Error processing batch {batch_idx} in {dataset_name}: {e}")
                    continue
            
            if dataset_embeddings:
                all_embeddings.extend(dataset_embeddings)
                all_metadata.extend(dataset_metadata)
                print(f"Completed {dataset_name}: {len(dataset_embeddings)} batches processed")
        
        if not all_embeddings:
            raise ValueError("No embeddings were extracted!")
        
        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings)
        print(f"Total embeddings extracted: {embeddings_array.shape[0]:,} x {embeddings_array.shape[1]}D")
        
        return embeddings_array, all_metadata

    def _create_custom_dataloader(self, dataset, datamodule, batch_size):
        """Create custom dataloader for sequence length modifications"""
        from torch.utils.data import DataLoader
        
        # Use a smaller batch size or the specified one
        effective_batch_size = batch_size if batch_size is not None else 32
        
        return DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
            collate_fn=datamodule.collate_fn(),
            pin_memory=True
        )
    
    def _truncate_batch(self, batch, max_sequences):
        """Truncate batch to contain at most max_sequences (assumes packed format)"""
        # Batch is in packed format when this is called
        for key in ['event', 'abspos', 'age', 'segment']:
            if key in batch and batch[key].size(0) > max_sequences:
                batch[key] = batch[key][:max_sequences]
        
        return batch
    
    def _prepare_batch_for_extraction(self, batch, datamodule):
        """Prepare batch for extraction by handling sequence length and unpacking"""
        device = next(self.parameters()).device
        
        # Move to device first
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        
        # Handle sequence length modifications if needed
        if hasattr(self, 'embedding_max_seq_len') and self.embedding_max_seq_len != datamodule.max_seq_len:
            for key in ['event', 'abspos', 'age', 'segment']:
                if key in batch:
                    tensor = batch[key]
                    if tensor.size(1) > self.embedding_max_seq_len:
                        # Truncate
                        batch[key] = tensor[:, :self.embedding_max_seq_len]
                    elif tensor.size(1) < self.embedding_max_seq_len:
                        # Pad
                        pad_size = self.embedding_max_seq_len - tensor.size(1)
                        if key == 'event':
                            padding = torch.zeros(tensor.size(0), pad_size, dtype=tensor.dtype, device=device)
                        else:
                            padding = torch.zeros(tensor.size(0), pad_size, dtype=tensor.dtype, device=device)
                        batch[key] = torch.cat([tensor, padding], dim=1)
        
        # Apply unpacking (same as datamodule.on_after_batch_transfer)
        if 'attn_mask' not in batch:
            batch["attn_mask"] = batch["event"] != 0
        
        # Use datamodule's unpad method
        batch = datamodule.unpad(batch)
        
        return batch

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        config_override: Optional[Dict] = None
    ):
        """Save embeddings and metadata to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving embeddings to {output_dir}")
        
        # Save embeddings
        np.save(output_dir / "embeddings.npy", embeddings)
        
        # Save metadata
        with open(output_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        # Create person_id mapping CSV for easy access
        self._save_person_id_mapping(metadata, output_dir)
        
        # Save configuration
        config = {
            'model_class': self.__class__.__name__,
            'pooling_strategy': self.pooling_strategy,
            'max_seq_len': self.embedding_max_seq_len,
            'embedding_shape': embeddings.shape,
            'model_d_model': self.hparams.d_model,
            'extraction_timestamp': str(torch.cuda.get_device_name()) if torch.cuda.is_available() else 'CPU'
        }
        
        # Override config if provided
        if config_override:
            config.update(config_override)
        
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("All files saved:")
        print(f"   Embeddings: embeddings.npy ({embeddings.shape[0]:,} x {embeddings.shape[1]})")
        print(f"   Metadata: metadata.pkl")
        print(f"   Person ID mapping: person_id_mapping.csv")
        print(f"   Configuration: config.json")

    def _save_person_id_mapping(self, metadata: List[Dict[str, Any]], output_dir: Path):
        """Create a CSV mapping embedding indices to person IDs"""
        person_id_data = []
        embedding_idx = 0
        
        for batch_meta in metadata:
            batch_size = batch_meta['batch_size']
            dataset = batch_meta['dataset']
            
            if 'person_ids' in batch_meta:
                person_ids = batch_meta['person_ids']
                for i, person_id in enumerate(person_ids):
                    # Handle sequence_lengths as tensor or list
                    seq_len = None
                    if 'sequence_lengths' in batch_meta:
                        seq_lengths = batch_meta['sequence_lengths']
                        if torch.is_tensor(seq_lengths):
                            seq_len = seq_lengths[i].item()
                        else:
                            seq_len = seq_lengths[i]
                    
                    person_id_data.append({
                        'embedding_idx': embedding_idx + i,
                        'person_id': person_id,
                        'dataset': dataset,
                        'sequence_length': seq_len
                    })
            else:
                # If no person IDs available, create placeholder entries
                for i in range(batch_size):
                    seq_len = None
                    if 'sequence_lengths' in batch_meta:
                        seq_lengths = batch_meta['sequence_lengths']
                        if torch.is_tensor(seq_lengths):
                            seq_len = seq_lengths[i].item()
                        else:
                            seq_len = seq_lengths[i]
                    
                    person_id_data.append({
                        'embedding_idx': embedding_idx + i,
                        'person_id': None,
                        'dataset': dataset,
                        'sequence_length': seq_len
                    })
            
            embedding_idx += batch_size
        
        # Save as CSV
        df = pl.DataFrame(person_id_data)
        df.write_csv(output_dir / "person_id_mapping.csv")

    def extract_and_save_embeddings(
        self,
        datamodule: BaseLightningDataModule,
        output_dir: Union[str, Path],
        pooling_strategy: str = "mean",
        max_seq_len: Optional[int] = None,
        include_datasets: List[str] = ["train", "val", "predict"],
        max_sequences: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Complete pipeline: extract high-dimensional embeddings and save everything
        
        Args:
            datamodule: Lightning datamodule
            output_dir: Directory to save results
            pooling_strategy: How to pool sequence embeddings
            max_seq_len: Override max sequence length
            include_datasets: Which datasets to process
            max_sequences: Maximum number of sequences to process (None for all)
        
        Returns:
            Dictionary with results and paths
        """
        print("EMBEDDING EXTRACTION PIPELINE")
        print("=" * 50)
        
        # Extract embeddings
        embeddings, metadata = self.extract_datamodule_embeddings(
            datamodule=datamodule,
            pooling_strategy=pooling_strategy,
            max_seq_len=max_seq_len,
            include_datasets=include_datasets,
            max_sequences=max_sequences
        )
        
        # Save everything
        config_override = {
            'include_datasets': include_datasets,
            'max_sequences': max_sequences
        }
        
        self.save_embeddings(
            embeddings=embeddings,
            metadata=metadata,
            output_dir=output_dir,
            config_override=config_override
        )
        
        results = {
            'embeddings': embeddings,
            'metadata': metadata,
            'output_dir': Path(output_dir),
            'pooling_strategy': pooling_strategy
        }
        
        print(f"\nPipeline complete! Results saved to: {output_dir}")
        return results

    @classmethod
    def from_checkpoint(
        cls, 
        checkpoint_path: Union[str, Path], 
        map_location: str = 'auto'
    ) -> 'EmbeddingNanoEncoder':
        """
        Load EmbeddingNanoEncoder from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            map_location: Device to load model on
        
        Returns:
            EmbeddingNanoEncoder instance
        """
        if map_location == 'auto':
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Extract hyperparameters
        hparams = checkpoint['hyper_parameters']
        
        # Initialize model
        model = cls(**hparams)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Handle mixed precision
        precision = hparams.get('precision', '32-true')
        if '16' in str(precision):
            print(f"Model trained with mixed precision (precision: {precision})")
            # Mark model for autocast usage
            model._mixed_precision = True
            # Convert model to float16 for consistency with training
            model = model.half()
        else:
            model._mixed_precision = False
        

        model.eval()

        
        print(f"Model loaded successfully")
        print(f"   - Hidden dimensions: {hparams['d_model']}")
        print(f"   - Max sequence length: {hparams['max_seq_len']}")
        print(f"   - Device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Mixed precision: {getattr(model, '_mixed_precision', False)}")
        
        return model