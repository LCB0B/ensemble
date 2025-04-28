"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from typing import Callable, Dict, Any, Tuple

from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ChainedScheduler,
    SequentialLR,
)
from src.lr_schedulers import CosineWarmupScheduler, TensorCosineWarmupScheduler
from src.time2vec import Time2Vec
from src.utils import print_main
from src.encoder_blocks import Block, SinusoidalPositionalEmbedding
from src.utils import flex_attn_padding
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import AUROC, AveragePrecision
from torch.nn.attention.flex_attention import create_block_mask


class BaseNanoEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # self.t2v_abspos = Time2Vec(
        #     output_dim=self.hparams.d_model,
        #     clip_min=-100,
        #     clip_max=100,
        #     init_scale=1e-4,
        # )
        self.t2v_age = Time2Vec(
            output_dim=self.hparams.d_model,
            clip_min=-100,
            clip_max=100,
            init_scale=1e-4,
        )
        self.embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.d_model, padding_idx=0
        )
        self.pos_embeddings = SinusoidalPositionalEmbedding(
            self.hparams.max_seq_len, self.hparams.d_model // self.hparams.num_heads
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
                            self.hparams.d_model,
                            self.hparams.num_heads,
                            self.hparams.dropout,
                            self.hparams.bias,
                            self.hparams.dim_feedforward,
                            self.hparams.compile,
                            self.hparams.swiglu,
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

    @torch.compile(dynamic=False)
    def forward(self, batch: Dict[str, Any]):

        x = self.embed_information(batch)
        sinusoidal_pos = self.pos_embeddings(x.shape)[None, None, :, :]

        # forward the GPT model itself
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, batch["attn_mask"], sinusoidal_pos=sinusoidal_pos)
        x = self.transformer.ln_f(x)

        return x

    @torch.compile(dynamic=False)
    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer_grouped_parameters = list()
        names_of_params_in_optimizer = list()

        # Split names into decay and no decay lists
        skip_list = ["embedding"]
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or any(
                skip_names in name for skip_names in skip_list
            ):
                no_decay.append(name)
            else:
                decay.append(name)

        # Give lowest learning rate to pre-first-layer parameters
        lowest_level_lr = self.hparams.learning_rate * (
            self.hparams.layer_lr_decay**self.hparams.num_layers
        )

        # Embedding
        embedding_no_decay = [
            [n, p]
            for n, p in self.named_parameters()
            if ("embedding" in n) and p.requires_grad
        ]
        names_of_params_in_optimizer.extend([n for n, _ in embedding_no_decay])
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in embedding_no_decay],
                "weight_decay": 0.0,
                "lr": lowest_level_lr,
            }
        )
        # t2v
        t2v_decay = [
            [n, p]
            for n, p in self.named_parameters()
            if ("t2v" in n) and any(n in name for name in decay)
        ]
        names_of_params_in_optimizer.extend([n for n, _ in t2v_decay])
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in t2v_decay],
                "weight_decay": self.hparams.weight_decay,
                "lr": lowest_level_lr,
            }
        )

        t2v_nodecay = [
            [n, p]
            for n, p in self.named_parameters()
            if ("t2v" in n) and any(n in name for name in no_decay)
        ]
        names_of_params_in_optimizer.extend([n for n, _ in t2v_nodecay])
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in t2v_nodecay],
                "weight_decay": 0,
                "lr": lowest_level_lr,
            }
        )

        # Learning rate for layers, increasing from lowest (0) to highest (num_layers - 1)
        for i in range(0, self.hparams.num_layers):

            # Layer-specific learning rate
            layer_lr = self.hparams.learning_rate * (
                (self.hparams.layer_lr_decay) ** (self.hparams.num_layers - i)
            )

            # Decay params
            layer_decay = [
                [n, p]
                for n, p in self.named_parameters()
                if (f"transformer.h.{i}" in n) and any(n in name for name in decay)
            ]
            names_of_params_in_optimizer.extend([n for n, _ in layer_decay])

            optimizer_grouped_parameters.append(
                {
                    "params": [p for n, p in layer_decay],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": layer_lr,
                }
            )

            # No decay params
            layer_no_decay = [
                [n, p]
                for n, p in self.named_parameters()
                if (f"transformer.h.{i}" in n) and any(n in name for name in no_decay)
            ]

            # There's a final layernorm which doesn't match the naming conventions of the block (which are pre-norm)
            if i == self.hparams.num_layers - 1:
                final_layer_norm = [
                    [n, p]
                    for n, p in self.named_parameters()
                    if ("transformer.ln_f" in n) and any(n in name for name in no_decay)
                ]
                layer_no_decay.extend(final_layer_norm)

            names_of_params_in_optimizer.extend([n for n, _ in layer_no_decay])

            optimizer_grouped_parameters.append(
                {
                    "params": [p for n, p in layer_no_decay],
                    "weight_decay": 0.0,
                    "lr": layer_lr,
                }
            )

        # Take the remainder (decoder-related) and add them to respective decay and no decay groups

        remaining_params = [
            [n, p]
            for n, p in self.named_parameters()
            if n not in names_of_params_in_optimizer and p.requires_grad
        ]

        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in remaining_params
                    if any(n in name for name in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate,
            }
        )

        optimizer_grouped_parameters.append(
            {
                "params": [
                    p for n, p in remaining_params if any(n in name for name in decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            }
        )

        # All the groups into the optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=torch.tensor(self.hparams.learning_rate),
            betas=(self.hparams.beta1, self.hparams.beta2),
            # fused=True,  # True
            foreach=True,
            capturable=True,
        )

        # n_warmup_steps = int(
        #     self.hparams["steps_per_epoch"] * self.hparams["optimizer_warmup_epochs"]
        # )
        # scheduler1 = LinearLR(
        #     optimizer, start_factor=1e-4, end_factor=1, total_iters=n_warmup_steps
        # )
        # scheduler2 = CosineAnnealingLR(
        #     optimizer, T_max=self.hparams.optimizer_max_iters
        # )

        # # Chain
        # # scheduler = ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)
        # # Sequential
        # scheduler = SequentialLR(
        #     optimizer, schedulers=[scheduler1, scheduler2], milestones=[n_warmup_steps]
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step",
            #     "frequency": 1,
            #     "name": "learning_rate",
            # },
        }

    @torch.compiler.disable
    def log(self, *args, **kwargs):
        return super().log(*args, **kwargs)

    # @torch.compile(dynamic=False)
    # def optimizer_step(
    #     self,
    #     epoch: int,
    #     batch_idx: int,
    #     optimizer: Optimizer | LightningOptimizer,
    #     optimizer_closure: Callable[[], Any] | None = None,
    # ) -> None:
    #     return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)


class PretrainNanoEncoder(BaseNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        if self.hparams.collate_method == "flatten_and_expand":
            self.decoder = nn.Linear(
                self.hparams.d_model, self.hparams.vocab_size, bias=False
            )
            # TODO: Does this overwrite the embedding when loading a finetuned model?
            # Tie weights (https://paperswithcode.com/method/weight-tying)
            if self.embedding.weight.shape == self.decoder.weight.shape:
                self.embedding.weight = self.decoder.weight
        elif self.hparams.collate_method == "channel":
            self.decoder = nn.Linear(
                self.hparams.d_model, self.hparams.vocab_size * 9, bias=False
            )
        else:
            raise ValueError(
                f"Unknown collate_method given: {self.hparams.collate_method}"
            )

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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.standard_step(batch, batch_idx)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.standard_step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True)

        return loss


class FinetuneNanoEncoder(BaseNanoEncoder):
    """NanoEncoder adapted for binary classification (finetuning)"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
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
            batch["last_data_idx"].view(-1, 1, 1).expand(-1, 1, forward_hidden_size)
        ).to(torch.int64)

        # Forward final state (from last_data_idx)
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

    # def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
    #     loss, _ = self.standard_step(batch, batch_idx)
    #     self.log("train/loss", loss)

    #     return loss
    @torch.compile(dynamic=False)
    def compiled_opt_step(self, opt):
        opt.step()

        
    @torch.compile(dynamic=False)
    def compiled_opt_sch_step(self, opt, sch):
        opt.step()
        sch.step()


    @torch.compile(dynamic=False)
    def compiled_manual_backward(self, loss):
        self.manual_backward(loss)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # opt = self.optimizers()
        # sch = self.lr_schedulers()
        # opt.zero_grad()
        # loss, _ = self.standard_step(batch, batch_idx)
        # self.manual_backward(loss)
        # opt.step()
        # sch.step()
        opt = self.optimizers()
        #sch = self.lr_schedulers()
        opt.zero_grad()
        loss, _ = self.standard_step(batch, batch_idx)
        self.compiled_manual_backward(loss)
        self.compiled_opt_step(opt)
        #self.compiled_opt_sch_step(opt, sch)
        #sch.step()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step(batch, batch_idx)
        self.log(
            "val/loss", loss, sync_dist=True, batch_size=self.hparams["batch_size"]
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
                    batch_size=self.hparams["batch_size"],
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
            batch["last_data_idx"].view(-1, 1, 1).expand(-1, 1, forward_hidden_size)
        ).to(torch.int64)

        # Forward final state (from last_data_idx)
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
