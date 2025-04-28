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
from torchmetrics import AUROC, AveragePrecision

from torch.nn.attention.flex_attention import create_block_mask
from src.utils import flex_attn_padding

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
                # "scheduler": LinearLR(
                #     optimizer,
                #     start_factor=1e-4,
                #     end_factor=1,
                #     total_iters=int(
                #         self.hparams["steps_per_epoch"]
                #         * self.hparams["optimizer_warmup_epochs"]
                #     ),
                # ),
                "scheduler": CosineWarmupScheduler(
                    optimizer,
                    warmup=int(
                        self.hparams["steps_per_epoch"]
                        * self.hparams["optimizer_warmup_epochs"]
                    ),
                    max_iters=self.hparams.optimizer_max_iters,
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

class CausalNanoEncoder(BaseNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.decoder = nn.Linear(
            self.hparams.d_model, self.hparams.vocab_size, bias=False
        )
        # Tie weights for efficient parameter sharing
        if self.embedding.weight.shape == self.decoder.weight.shape:
            self.embedding.weight = self.decoder.weight

    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard causal prediction step (next-block prediction).

        Args:
            batch (Dict[str, Any]): Batch of data and targets.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value
        """
        # Forward pass
        x = self.forward(batch)
        pdb.set_trace()
        # Handle flex attention block prediction
        if self.hparams["collate_method"] == "flatten_and_expand":
            # Create block mask for causal prediction
            attn_mask = create_block_mask(
                flex_attn_padding((batch["sequence_lens"])), 
                x.size(0),
                None,
                x.size(1),
                x.size(1),
                _compile=True,
            )
            # Apply causal masking
            for block in self.transformer.h:
                x = block(x, attn_mask)

        # Decodes and reshapes
        decoded_output = self.decoder(x)
        decoded_output = decoded_output.view(-1, self.hparams["vocab_size"])

        # Calculates next-block prediction loss
        pdb.set_trace()
        loss = self.criterion(decoded_output, batch["target"].view(-1))

        return loss, decoded_output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.standard_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.standard_step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True)
        return loss

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

        num_follow_up = 3
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
