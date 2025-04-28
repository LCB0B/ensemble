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
from src.lr_schedulers import CosineWarmupScheduler
from src.time2vec import Time2Vec
from src.utils import print_main
from src.encoder_blocks import Block
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
                            self.hparams.compile,
                            self.hparams.swiglu,
                        )
                        for _ in range(self.hparams.num_layers)
                    ]
                ),
                ln_f=torch.nn.LayerNorm(self.hparams.d_model, bias=self.hparams.bias),
            )
        )
        print("hello Magnus")
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

    def forward(self, batch: Dict[str, Any]):
        x, mask, abspos, age = (
            batch["event"],
            batch["last_data_idx"],  # torch.tensor([1])
            batch["abspos"],
            batch["age"],
        )
        x = self.embedding(x)

        if self.hparams["collate_method"] == "flatten_and_expand":
            # Create block mask
            attn_mask = create_block_mask(
                flex_attn_padding(batch["last_data_idx"]),
                x.size(0),
                None,
                x.size(1),
                x.size(1),
                _compile=True,
            )

        elif self.hparams["collate_method"] == "channel":
            x *= mask.unsqueeze(-1)  # (B, L, C) -> (B, L, C, 1)
            x = x.sum(dim=2)  # (B, L, C, D) -> (B, L, D)

            denom = mask.sum(
                dim=2, keepdim=True
            )  # Amount of relevant channels which are present
            x /= denom + 0.0001  # Divide with amount of non-zero channels

            attn_mask = mask[:, :, 0]

        # x += self.t2v_abspos(abspos)  # Add positional encoding
        x += self.t2v_age(age)  # Add positional encoding

        # forward the GPT model itself
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, attn_mask)
        x = self.transformer.ln_f(x)

        return x

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
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            fused=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
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

    @torch.compiler.disable()
    def log(self, *args, **kwargs):
        return super().log(*args, **kwargs)


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
        loss = self.criterion(
            decoded_output, batch["target"].view(-1).to(torch.float32)
        )

        return loss, decoded_output

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.standard_step(batch, batch_idx)
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
        # print(batch['event'].shape)
        # print(batch.keys())
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
