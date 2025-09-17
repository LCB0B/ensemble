"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from torch.nn.attention.flex_attention import create_block_mask
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torchmetrics import AUROC, AveragePrecision

from src.encoder_blocks import Block
from src.lr_schedulers import CosineWarmupScheduler
from src.time2vec import Time2Vec
from src.utils import print_main


class BaseNanoEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.t2v_age = Time2Vec(
            output_dim=self.hparams.d_model,
            clip_min=-100,
            clip_max=100,
            init_scale=1e-4,
        )
        self.embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.d_model, padding_idx=0
        )

        self.segment_embeddings = nn.Embedding(
            self.hparams.max_seq_len // 2, self.hparams.d_model, padding_idx=0
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
                            max_seq_len=self.hparams.max_seq_len,
                            causal=self.hparams.causal,
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
        x += self.t2v_age(batch["age"])
        x += self.segment_embeddings(batch["segment"])

        return x

    @torch.compile(dynamic=True)
    def forward(self, batch: Dict[str, Any]):
        B, T = batch["attn_mask"].shape

        x = self.embed_information(batch)

        # forward the GPT model itself
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, batch)
        x = self.transformer.ln_f(x)

        # Repad
        if x.ndim == 2:
            x = pad_input(x, batch["indices"], B, T)
        return x

    @staticmethod
    def pad_to_multiple(x, n_tokens):
        total = x.size(0)
        padding_len = n_tokens - total
        if padding_len == 0:
            return x
        return torch.cat((x, torch.zeros((padding_len, x.size(1)), device=x.device)))

    @torch.compile(dynamic=False)
    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
                    total_iters=int(self.hparams["warmup_steps"]),
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    def log(self, *args, **kwargs):
        return super().log(*args, **kwargs)


class PretrainNanoEncoder(BaseNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Use triton cross_entropy_loss
        # self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion = CrossEntropyLoss(ignore_index=-100)

        self.decoder = nn.Linear(
            self.hparams.d_model, self.hparams.vocab_size, bias=False
        )
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

        # Sparse token prediction
        labels = batch["target"].view(-1)
        last_hidden_state = x.view(labels.shape[0], -1)
        # filter out the non-masked tokens
        mask_tokens = labels != -100
        last_hidden_state = last_hidden_state[mask_tokens]
        labels = labels[mask_tokens]
        batch["labels"] = labels

        # Decodes and reshapes
        decoded_output = self.decoder(last_hidden_state)
        decoded_output = decoded_output.view(-1, self.hparams["vocab_size"])

        # Calculates MLM loss
        loss = self.criterion(decoded_output, labels)

        return loss, decoded_output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.standard_step(batch, batch_idx)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.standard_step(batch, batch_idx)
        self.log(
            "val/loss", loss, sync_dist=True, batch_size=batch["attn_mask"].shape[0]
        )

        return loss


class FinetuneNanoEncoder(BaseNanoEncoder):
    """NanoEncoder adapted for binary classification (finetuning)"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.metrics = {
            "AUROC": AUROC("binary"),
            "PRAUC": AveragePrecision("binary"),
        }
        self.validation_step_outputs = {"target": [], "preds": []}

        # DON'T REUSE IDENTICAL DECODER NAME AS BASE `NANOENCODER`, AS THIS WILL BREAK LOADING OF CHECKPOINTS FROM PRETRAINED MODELS
        # Define the BIGRU decoder with an output layer for binary classification
        self.decoder_finetune = nn.GRU(
            input_size=self.hparams["d_model"],
            hidden_size=self.hparams["d_model"],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_layer = nn.Linear(
            self.hparams["d_model"] * 2, 1
        )  # *2 for bidirectional
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

        # Pass through the BIGRU decoder
        gru_output, _ = self.decoder_finetune(x)
        final_hidden_states = gru_output[:, 0, :].squeeze(1)

        # Get dimensions and define forward hidden size
        hidden_size = gru_output.size(-1)
        forward_hidden_size = hidden_size // 2  # for each direction

        # Use mask to select the final hidden state for each sequence in both directions
        last_indices = (
            batch["sequence_lens"].view(-1, 1, 1).expand(-1, 1, forward_hidden_size)
        ).to(torch.int64)

        # Forward final state (from sequence_lens)
        forward_final_states = torch.gather(
            gru_output[:, :, :forward_hidden_size], 1, last_indices - 1
        ).squeeze(1)

        # Backward final state (from 0th index)
        backward_final_states = gru_output[:, 0, forward_hidden_size:].squeeze(1)

        # Concatenate forward and backward final states
        final_hidden_states = torch.cat(
            (forward_final_states, backward_final_states), dim=-1
        )

        # Pass through the output layer for binary classification
        decoded_output = self.output_layer(final_hidden_states).view(-1)

        # Calculate binary classification loss
        loss = self.criterion(decoded_output, batch["target"].view(-1))

        return loss, decoded_output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.standard_step(batch, batch_idx)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step(batch, batch_idx)
        self.log(
            "val/loss", loss, sync_dist=True, batch_size=batch["attn_mask"].shape[0]
        )

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
            try:
                self.log(
                    name,
                    metric(preds, targets),
                    sync_dist=True,
                )
            except ValueError as e:  # If targets only contain 1 class
                print_main("Metric check error:", e)
                metric.reset()
        self.validation_step_outputs["target"].clear()
        self.validation_step_outputs["preds"].clear()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        # Forward pass
        x = self.forward(batch)

        # Pass through the BIGRU decoder
        gru_output, _ = self.decoder_finetune(x)

        # Get dimensions and define forward hidden size
        hidden_size = gru_output.size(-1)
        forward_hidden_size = hidden_size // 2  # for each direction

        # Use mask to select the final hidden state for each sequence in both directions
        last_indices = (
            batch["sequence_lens"].view(-1, 1, 1).expand(-1, 1, forward_hidden_size)
        ).to(torch.int64)

        # Forward final state (from sequence_lens)
        forward_final_states = torch.gather(
            gru_output[:, :, :forward_hidden_size], 1, last_indices - 1
        ).squeeze(1)

        # Backward final state (from 0th index)
        backward_final_states = gru_output[:, 0, forward_hidden_size:].squeeze(1)

        # Concatenate forward and backward final states
        final_hidden_states = torch.cat(
            (forward_final_states, backward_final_states), dim=-1
        )

        # Pass through the output layer for binary classification
        decoded_output = self.output_layer(final_hidden_states).view(-1)

        preds = torch.sigmoid(decoded_output).detach()
        return preds
