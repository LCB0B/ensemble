"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from typing import Dict, Any, Tuple
import pandas as pd
from torch.optim.lr_scheduler import LinearLR
from src.lr_schedulers import CosineWarmupScheduler
from src.time2vec import Time2Vec
from src.utils import print_main
from src.encoder_blocks import (
    Block,
    CumulativeProbabilityLayer,
    SinusoidalPositionalEmbedding,
    RotaryPositionalEmbeddings,
)
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import AUROC, AveragePrecision, Accuracy, Recall, Precision

import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


import pdb

class BaseNanoEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.t2v_abspos = Time2Vec(
            output_dim=self.hparams.d_model,
            clip_min=-100,
            clip_max=100,
            init_scale=1e-4,
        )
        self.t2v_age = Time2Vec(
            output_dim=self.hparams.d_model,
            clip_min=-100,
            clip_max=100,
            init_scale=1e-4,
        )
        self.embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.d_model, padding_idx=0
        )
        # self.pos_embeddings = SinusoidalPositionalEmbedding(
        #     self.hparams.max_seq_len, self.hparams.d_model // self.hparams.num_heads
        # )
        self.pos_embeddings = RotaryPositionalEmbeddings(
            max_seq_len=self.hparams.max_seq_len,
            dim=self.hparams.d_model // self.hparams.num_heads,
        )
        self.segment_embeddings = nn.Embedding(
            self.hparams.max_seq_len, self.hparams.d_model, padding_idx=0
        )
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(self.hparams.dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            d_model=self.hparams.d_model,
                            num_heads=self.hparams.num_heads,
                            dropout=self.hparams.dropout,
                            bias=self.hparams.bias,
                            dim_feedforward=self.hparams.dim_feedforward,
                            compiled=self.hparams.compile,
                            swiglu=self.hparams.swiglu,
                        )
                        for _ in range(self.hparams.num_layers)
                    ]
                ),
                ln_f=torch.nn.LayerNorm(self.hparams.d_model, bias=self.hparams.bias),
            )
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.hparams.num_layers)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_information(self, batch):
        """Embeds information using available keys"""
        x = self.embedding(batch["event"])

        x += self.t2v_abspos(batch["abspos"])
        x += self.t2v_age(batch["age"])

        if "segment" in batch:
            x += self.segment_embeddings(batch["segment"])

        return x

    @torch.compile(dynamic=False)
    def forward(self, batch: Dict[str, Any]):
        x = self.embed_information(batch)


        # (bs, seq_len) -> (bs, num_heads, seq_len, embed_size_per_head)
        sinusoidal_pos = (
            self.pos_embeddings
        )  # self.pos_embeddings(x.shape)[None, None, :, :]

        # forward the GPT model itself
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, attn_mask=batch["attn_mask"], sinusoidal_pos=sinusoidal_pos)
        x = self.transformer.ln_f(x)
        return x

    @torch.compile(dynamic=False)
    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, output = self.standard_step(batch, batch_idx)
        self.log("train/loss", loss, batch_size=output.size(0))

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, output = self.standard_step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True, batch_size=output.size(0))

        return loss

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print_main(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print_main(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=torch.tensor(self.hparams.learning_rate),
            betas=(self.hparams.beta1, self.hparams.beta2),
            fused=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LinearLR(
                    optimizer,
                    start_factor=1e-4,
                    end_factor=1,
                    total_iters=int(
                        self.hparams["steps_per_epoch"]
                        * self.hparams["optimizer_warmup_epochs"]
                    ),
                ),
                # "scheduler": CosineWarmupScheduler(
                #     optimizer,
                #     warmup=int(
                #         self.hparams["steps_per_epoch"]
                #         * self.hparams["optimizer_warmup_epochs"]
                #     ),
                #     max_iters=self.hparams.optimizer_max_iters,
                # ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

class CausalEncoder(BaseNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Loss function for next-token prediction
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Decoder layer to map the output embeddings to vocabulary logits
        self.decoder = nn.Linear(
            self.hparams.d_model, self.hparams.vocab_size, bias=False
        )

        # Tie weights
        if hasattr(self, 'embedding') and self.embedding.weight.shape == self.decoder.weight.shape:
             self.decoder.weight = self.embedding.weight
        else:
             print("Warning: Could not tie decoder and embedding weights.")

    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the standard step for causal language modeling.

        Args:
            batch (Dict[str, Any]): A dictionary containing input data and target labels.
            batch_idx (int): The index of the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The loss value and the logits.
        """
        # Forward pass
        x = self.forward(batch)

        # Decode outputs into vocabulary logits
        logits = self.decoder(x)  # Shape: (batch_size, seq_len, vocab_size)

        # Flatten logits and targets for loss computation
        logits = logits.view(-1, self.hparams.vocab_size)  # Flatten for CrossEntropyLoss
        targets = batch["target"].view(-1)

        # Calculate loss
        loss = self.criterion(logits, targets)
        return loss, logits
    
    
    def predict_step(self, batch, batch_idx: int, predict_index=-1, strategy="most_likely", k=5, p=0.9, temperature=1.0):
        """
        Predict step supporting multiple prediction strategies with temperature scaling for a specified position.

        Args:
            batch (dict): Input batch containing 'input_ids' and possibly 'attention_mask'.
            batch_idx (int): Index of the current batch.
            predict_index (int): Index of the hidden state to use for prediction (default: -1, last position).
            strategy (str): Prediction strategy ("most_likely", "top_k", "top_p").
            k (int): Number of tokens to consider for top-k sampling (only for "top_k").
            p (float): Cumulative probability threshold for nucleus sampling (only for "top_p").
            temperature (float): Temperature parameter for scaling logits.
        
        Returns:
            Tensor: Predicted token indices for the batch.
        """
        # Forward pass through the model
        output = self(batch)
        # Get logits for the specified position
        logits = self.decoder(output[:, predict_index, :])  # Use predict_index instead of -1

        # Apply temperature scaling
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)  # Convert to probabilities

        # Most likely outcome (argmax)
        if strategy == "most_likely":
            predictions = torch.argmax(probs, dim=-1)

        # Top-k sampling
        elif strategy == "top_k":
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
            sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
            predictions = top_k_indices[range(top_k_indices.size(0)), sampled_indices]

        # Nucleus sampling (Top-p sampling)
        elif strategy == "top_p":
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Mask tokens beyond the top-p cumulative probability
            mask = cumulative_probs <= p
            mask[:, 0] = True  # Ensure at least one token is kept
            sorted_probs[~mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)  # Re-normalize
            sampled_indices = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
            predictions = sorted_indices[range(sorted_indices.size(0)), sampled_indices]

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        return predictions
    
    def generate_sequence(self, batch, prompt_length, max_new_tokens, strategy="most_likely", k=5, p=0.95, temperature=1.0):
        """
        Generate a sequence of event tokens starting from a prompt using the predict_step method.

        Args:
            batch (dict): Input batch with keys 'event', 'age', 'abspos', 'segment', 'attn_mask', etc.
                        The 'event' key contains the initial prompt sequence, possibly padded.
            prompt_length (int): Length of the prompt sequence in the 'event' tensor.
            max_new_tokens (int): Number of new tokens to generate.
            strategy (str): Sampling strategy for predict_step ("most_likely", "top_k", "top_p").
            k (int): Number of tokens to consider for top-k sampling.
            p (float): Cumulative probability txwhreshold for nucleus sampling.
            temperature (float): Temperature for scaling logits in predict_step.

        Returns:
            torch.Tensor: Generated event tokens of shape (batch_size, max_new_tokens).
        """
        # Make a deep copy of the batch to avoid modifying the original
        current_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_size = current_batch['event'].size(0)
        max_seq_len = current_batch['event'].size(1)
        generated_tokens = []

        # Ensure the batch has enough space for new tokens
        if prompt_length + max_new_tokens > max_seq_len:
            raise ValueError(f"Prompt length ({prompt_length}) + max_new_tokens ({max_new_tokens}) exceeds maximum sequence length ({max_seq_len})")

        for _ in range(max_new_tokens):
            # Current length is the prompt plus the number of generated tokens
            current_length = prompt_length + len(generated_tokens)
            # Predict the next token using predict_step at the last position of the current sequence
            next_token = self.predict_step(
                current_batch,
                batch_idx=0,
                predict_index=current_length - 1,
                strategy=strategy,
                k=k,
                p=p,
                temperature=temperature
            )
            # Update the 'event' sequence with the predicted token
            current_batch['event'][:, current_length] = next_token
            # Collect the predicted token
            generated_tokens.append(next_token.unsqueeze(-1))

        # Combine all generated tokens into a single tensor
        generated_sequence = torch.cat(generated_tokens, dim=1)  # Shape: (batch_size, max_new_tokens)
        return generated_sequence
    
    def get_logits_and_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass and returns aligned logits and targets.

        Args:
            batch: A batch dictionary containing 'event' and other necessary keys.

        Returns:
            A dictionary containing:
                - 'logits': Logits predicting the next token (shape: B, S-1, V).
                - 'targets': Target token IDs (shape: B, S-1).
                - 'inputs': Input token IDs used for prediction (shape: B, S-1).
        """
        # Ensure necessary keys are present
        if 'event' not in batch:
            raise ValueError("Batch dictionary must contain 'event' key.")

        inputs = batch['event'][:, :-1].detach()  # Input: t0 to t(n-1)
        targets = batch['event'][:, 1:].detach()   # Target: t1 to tn

        # Check for valid sequence length
        if inputs.shape[1] == 0 or targets.shape[1] == 0:
             # Return empty tensors or raise error, depending on desired handling
             # Returning empty tensors might be better for evaluation loop robustness
             print(f"Warning: Sequence length <= 1 after slicing. Returning empty logits/targets.")
             return {'logits': torch.empty(inputs.shape[0], 0, self.hparams.vocab_size, device=inputs.device),
                     'targets': torch.empty_like(inputs),
                     'inputs': torch.empty_like(inputs)}

        # Forward pass using the base class method
        # Ensure the forward method handles the full batch dictionary correctly
        hidden_states = self.forward(batch) # Assumes forward uses the whole batch

        # Get logits from decoder
        logits = self.decoder(hidden_states) # Shape: (B, S, V)

        # Align logits with targets
        aligned_logits = logits[:, :targets.shape[1], :] # Shape: (B, S-1, V)

        return {'logits': aligned_logits, 'targets': targets, 'inputs': inputs}

    def evaluate_batch_perplexity_accuracy(
        self,
        batch: Dict[str, Any],
        pad_token_id: int = 0,
        return_token_loss: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculates loss, accuracy metrics for next-token prediction on a batch.
        
        Args:
            batch: A batch dictionary.
            pad_token_id: The ID of the padding token.
            return_token_loss: Whether to return per-token losses
            
        Returns:
            A dictionary containing metrics and optionally per-token losses
        """
        outputs = self.get_logits_and_targets(batch)
        logits = outputs['logits']
        targets = outputs['targets']

        # Handle empty sequences returned by get_logits_and_targets
        if targets.shape[1] == 0:
            return {'batch_loss': 0.0, 'correct_preds': 0, 'total_tokens': 0,
                    'predictions_flat': torch.empty(0, dtype=torch.long, device=targets.device),
                    'targets_flat': torch.empty(0, dtype=torch.long, device=targets.device),
                    'valid_mask_flat': torch.empty(0, dtype=torch.bool, device=targets.device)}

        # Flatten for loss and accuracy calculation
        logits_flat = logits.reshape(-1, logits.size(-1))  # (B * S', V)
        targets_flat = targets.reshape(-1)                 # (B * S')

        # Calculate per-token loss using cross entropy with reduction='none'
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        per_token_loss = criterion(logits_flat, targets_flat)  # Shape: (B * S')

        # Create mask for valid (non-padding) tokens
        valid_mask_flat = (targets_flat != pad_token_id)

        # Calculate total loss for the batch (sum over valid tokens)
        batch_loss = per_token_loss[valid_mask_flat].sum().item()

        # Calculate accuracy
        predictions_flat = torch.argmax(logits_flat, dim=-1)
        correct_preds_mask = (predictions_flat == targets_flat) & valid_mask_flat
        correct_preds = correct_preds_mask.sum().item()
        total_tokens = valid_mask_flat.sum().item()
        
        # Calculate per-sequence losses
        batch_size = targets.size(0)
        seq_length = targets.size(1)
        
        # Reshape per_token_loss to [batch_size, seq_length]
        per_token_loss_reshaped = per_token_loss.reshape(batch_size, seq_length)
        valid_mask_reshaped = valid_mask_flat.reshape(batch_size, seq_length)
        
        # Calculate per-sequence loss and token counts
        sequence_losses = []
        sequence_lengths = []
        for i in range(batch_size):
            seq_valid_mask = valid_mask_reshaped[i]
            if seq_valid_mask.sum() > 0:
                seq_loss = per_token_loss_reshaped[i][seq_valid_mask].sum().item()
                seq_len = seq_valid_mask.sum().item()
                sequence_losses.append(seq_loss)
                sequence_lengths.append(seq_len)
            else:
                sequence_losses.append(0.0)
                sequence_lengths.append(0)

        if return_token_loss:
            # Return per-token loss for further analysis if needed
            return {
                'batch_loss': batch_loss,
                'correct_preds': correct_preds,
                'total_tokens': total_tokens,
                'predictions_flat': predictions_flat,
                'targets_flat': targets_flat,
                'valid_mask_flat': valid_mask_flat,
                'per_token_loss': per_token_loss,
                'token_losses': sequence_losses,
                'sequence_lengths': sequence_lengths
            }
        else:
            return {
                'batch_loss': batch_loss,
                'correct_preds': correct_preds,
                'total_tokens': total_tokens,
                'predictions_flat': predictions_flat,
                'targets_flat': targets_flat,
                'valid_mask_flat': valid_mask_flat
            }

    # generate_sequence and predict_step remain as they were

    # (Optional) Add a method specifically for evaluating generated sequences
    def evaluate_generated_sequence_perplexity(
        self,
        full_sequence_batch: Dict[str, Any], # Batch dict containing the FULL sequence (prompt+generated)
        prompt_length: int,
        pad_token_id: int = 0,
        return_logits: bool = False,
        ) -> Dict[str, float]:
        """
        Calculates the perplexity of the GENERATED part of sequences,
        conditioned on the prompt.

        Args:
            full_sequence_batch: Batch dict w/ 'event' key shape (B, full_len)
            prompt_length: The length of the initial prompt.
            pad_token_id: The ID of the padding token.

        Returns:
            Dict containing 'generated_loss_sum' and 'generated_token_count'.
        """
        # Get logits for the full sequence (predicting token 1 to full_len)
        # We need hidden states from position prompt_length-1 onwards
        outputs = self.get_logits_and_targets(full_sequence_batch)
        logits = outputs['logits'] # Shape (B, full_len - 1, V)

        # Targets are the generated tokens (from index prompt_length to end)
        targets = full_sequence_batch['event'][:, prompt_length:].detach() # Shape (B, generated_len)

        # Logits corresponding to the generated targets start from index prompt_length-1
        logits_for_generated = logits[:, prompt_length-1:, :] # Shape (B, generated_len, V)

        # Check if shapes match
        if logits_for_generated.shape[1] != targets.shape[1]:
             print(f"Warning: Shape mismatch in evaluate_generated_sequence_perplexity. Logits: {logits_for_generated.shape[1]}, Targets: {targets.shape[1]}. Skipping calculation.")
             return {'generated_loss_sum': 0.0, 'generated_token_count': 0}
        if targets.shape[1] == 0: # No tokens were generated
            return {'generated_loss_sum': 0.0, 'generated_token_count': 0}

        # Flatten for loss calculation
        logits_flat = logits_for_generated.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        # Calculate per-token loss
        per_token_loss = self.criterion(logits_flat, targets_flat)

        # Mask out padding ONLY WITHIN THE GENERATED PART
        valid_mask_flat = (targets_flat != pad_token_id)
        batch_loss_sum = (per_token_loss * valid_mask_flat).sum().item()
        valid_token_count = valid_mask_flat.sum().item()

        if return_logits:
            # Return logits for the generated part as well
            return {
                'generated_loss_sum': batch_loss_sum,
                'generated_token_count': valid_token_count,
                #'logits': logits_for_generated,  # Shape (B, generated_len, V)
                #'targets': targets,               # Shape (B, generated_len)
                'per_token_loss': per_token_loss, # Shape (B * generated_len)
            }
        else:
            return {
                'generated_loss_sum': batch_loss_sum,
                'generated_token_count': valid_token_count
                
            }
        
    def compute_sequence_log_prob(self, batch):
        """
        Compute the log probability of each sequence in the batch.

        Args:
            batch (dict): Batch containing 'input_ids' (tensor of shape [batch_size, seq_len])
                        and 'attention_mask' (tensor of shape [batch_size, seq_len]).

        Returns:
            Tensor: Log probabilities for each sequence, shape [batch_size].
        """
        if __debug__:
            pdb.set_trace()
        # Forward pass to get hidden states
        hidden_states = self(batch)  # [batch_size, seq_len, hidden_size]
        
        # Get logits for all positions
        logits = self.decoder(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        # Compute log probabilities over the vocabulary
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # Extract input_ids and shift for targets
        targets = batch['target']      # [batch_size, seq_len-1], tokens to predict
        
        # Gather log probabilities for the actual tokens (shifted by 1)
        log_probs_per_token = log_probs[:, :, :].gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
   
        #Handle padding
        dense_mask = (targets != 0).float()  # [64, 1023]
    
        # Sum log probabilities over valid positions
        sequence_log_probs = (log_probs_per_token * dense_mask).sum(dim=-1)  # [64
        # Sum log probabilities for valid positions
        
        return sequence_log_probs
    


#########
#########


#########
#########

class PretrainNanoEncoder(BaseNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.decoder = nn.Linear(
            self.hparams.d_model, self.hparams.vocab_size, bias=False
        )
        # TODO: Does this overwrite the embedding when loading a finetuned model?
        # Tie weights (https://paperswithcode.com/method/weight-tying)
        if self.embedding.weight.shape == self.decoder.weight.shape:
            self.embedding.weight = self.decoder.weight

    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard MLM step

        Args:
            batch (Dict[str, Any]): Batch of data and targets.
            batch_idx (int): Batch index.

        Returns: 
            torch.Tensor: Loss value
        """
        # Forward pass
        x = self.forward(batch)

        # Decodes and reshapes
        decoded_output = self.decoder(x)
        decoded_output = decoded_output.view(-1, self.hparams["vocab_size"])

        # Calculates MLM loss
        loss = self.criterion(decoded_output, batch["target"].view(-1))

        return loss, decoded_output


class FinetuneNanoEncoder(BaseNanoEncoder):
    """NanoEncoder adapted for binary classification (finetuning)"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.metrics = {
            "AUROC": AUROC("binary", ignore_index=-100),
            "PRAUC": AveragePrecision("binary", ignore_index=-100),
        }
        self.validation_step_outputs = {"target": [], "preds": []}
        self.decoder_finetune = nn.Linear(
            self.hparams["d_model"], 1
        )  # DON'T REUSE IDENTICAL DECODER NAME AS BASE `NANOENCODER`, AS THIS WILL BREAK LOADING OF CHECKPOINTS FROM PRETRAINED MODELS
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard Binary Classification step.

        Args:
            batch (Dict[str, Any]): Batch of data and targets.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value
        """
        # Forward pass
        x = self.forward(batch)

        # Decodes and reshapes
        cls = x[:, 0]
        decoded_output = self.decoder_finetune(cls).view(-1)
        loss = self.criterion(decoded_output, batch["target"].view(-1))

        return loss, decoded_output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True, batch_size=decoded_output.size(0))

        # Compute metrics
        preds = torch.sigmoid(decoded_output).detach()
        targets = batch["target"].view(-1).long().detach()
        self.validation_step_outputs["target"].append(targets)
        self.validation_step_outputs["preds"].append(preds)

        return loss

    def on_validation_epoch_end(self):
        targets = torch.cat(self.validation_step_outputs["target"])
        preds = torch.cat(self.validation_step_outputs["preds"])

        for name, metric in self.metrics.items():
            self.log(
                name,
                metric(preds, targets),
                sync_dist=True,
            )

        self.validation_step_outputs["target"].clear()
        self.validation_step_outputs["preds"].clear()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        # Forward pass
        x = self.forward(batch)

        # Decodes and reshapes
        cls = x[:, 0]
        decoded_output = self.decoder_finetune(cls).view(-1)

        # Return the prediction (outputs)
        return decoded_output


class RiskNanoEncoder(FinetuneNanoEncoder):
    """Extends Finetuning to work with Cumulative Risk Predictions"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        num_follow_up = len(self.hparams.prediction_windows)
        self.decoder_finetune = torch.nn.Linear(self.hparams.d_model, num_follow_up)
        self.metrics = {
            "AUROC": AUROC(
                "multilabel",
                ignore_index=-100,
                num_labels=num_follow_up,
                compute_on_cpu=True,
                average="none",
            ),
            "PRAUC": AveragePrecision(
                "multilabel",
                ignore_index=-100,
                num_labels=num_follow_up,
                compute_on_cpu=True,
                average="none",
            ),
        }
        self.decoder_finetune = CumulativeProbabilityLayer(
            self.hparams["d_model"], num_follow_up
        )
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard Risk Trajectories step.

        Args:
            batch (Dict[str, Any]): Batch of data and targets.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value
        """

        # Forward pass
        x = self.forward(batch)
        decoded_output = self.decoder_finetune(x)

        event_mask = batch["event_mask"]
        num_selected = event_mask.sum(dim=-1, keepdim=True)
        decoded_output = torch.bmm(event_mask, decoded_output) / (num_selected + 1e-8)

        loss = self.criterion(decoded_output, batch["target"])
        mask = batch["target"] != -100
        loss = (loss * mask).sum() / mask.sum()  # loss = loss[mask].mean() (equivalent)

        return loss, decoded_output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True, batch_size=decoded_output.size(0))

        # Compute metrics
        preds = torch.sigmoid(decoded_output).detach().cpu()
        targets = batch["target"].long().detach().cpu()
        self.validation_step_outputs["target"].append(targets)
        self.validation_step_outputs["preds"].append(preds)

        return loss

    def on_validation_epoch_end(self):
        targets = torch.cat(
            [t.view(-1, t.size(-1)) for t in self.validation_step_outputs["target"]]
        )
        preds = torch.cat(
            [p.view(-1, p.size(-1)) for p in self.validation_step_outputs["preds"]]
        )
        windows = self.hparams.prediction_windows

        for name, metric in self.metrics.items():
            result = metric(preds, targets)
            self.log(f"{name}_mean", result.mean(), sync_dist=True)
            for i, res in enumerate(result):
                self.log(f"{name}_interval {windows[i]}y", res, sync_dist=True)

        self.validation_step_outputs["target"].clear()
        self.validation_step_outputs["preds"].clear()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        _, decoded_output = self.standard_step(batch, batch_idx)
        # Compute metrics
        preds = torch.sigmoid(decoded_output).detach().cpu()
        targets = batch["target"].long().detach().cpu()
        self.validation_step_outputs["target"].append(targets)
        self.validation_step_outputs["preds"].append(preds)
        return preds

    def on_predict_epoch_end(self):
        self.age_bracket_metrics()

    def age_bracket_metrics(self):
        tbuckets = []
        pbuckets = []
        for tbatch, pbatch in zip(
            self.validation_step_outputs["target"],
            self.validation_step_outputs["preds"],
        ):
            for i in range(0, len(tbatch[0]), 40):
                b_idx = i // 40
                if len(tbuckets) <= b_idx:
                    tbuckets.append([])
                    pbuckets.append([])
                tbuckets[b_idx].extend(tbatch[:, i : i + 40])
                pbuckets[b_idx].extend(pbatch[:, i : i + 40])

        windows = self.hparams.prediction_windows
        num_follow_up = len(self.hparams.prediction_windows)
        results = []
        valid_i = []
        for i, (targets, preds) in enumerate(zip(tbuckets, pbuckets)):
            if (torch.cat(targets) == -100).all():
                continue
            result = self.metrics["AUROC"](
                torch.cat(preds).view(-1, num_follow_up),
                torch.cat(targets).view(-1, num_follow_up),
            )
            if result.sum() == 0:
                continue
            # foo = torch.cat(targets)
            # print(
            #     i * 20,
            #     i * 20 + 20,
            #     (foo != -100).sum() / num_follow_up,
            #     (foo != -100).sum() / (foo.size(0) * foo.size(1)),
            # )
            results.append(result.tolist())
            valid_i.append(i)
        brackets = list(range(i))
        valid_brackets = [brackets[i] * 20 for i in valid_i]

        df = pd.DataFrame(
            results,
            columns=[f"{w}y" for w in windows],
            index=[f"{b}-{b+20}" for b in valid_brackets],
        ).T.round(3)
        # print(df)
        self.logger.experiment.add_text(
            "Age brackets", df.to_markdown(), self.global_step
        )


class ParentRiskNanoEncoder(RiskNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

        self.pos_embeddings = RotaryPositionalEmbeddings(
            max_seq_len=self.hparams.max_seq_len * 3,
            dim=self.hparams.d_model // self.hparams.num_heads,
        )
        self.segment_embeddings = nn.Embedding(
            self.hparams.max_seq_len * 3, self.hparams.d_model, padding_idx=0
        )

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x
