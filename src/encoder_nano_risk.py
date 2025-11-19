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
from flash_attn.bert_padding import pad_input
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR
from src.encoder_blocks import Block, CumulativeProbabilityLayer
from src.metrics import CustomROCS, FusedAccuracyAtK, BestAtThreshold
from src.time2vec import Time2Vec
from src.utils import print_main


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
                            max_seq_len=self.hparams.max_seq_len,
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

        x += self.segment_embeddings(batch["segment"])

        return x

    @torch.compile(dynamic=True)
    def forward(self, batch: Dict[str, Any], repad=True):
        B, T = batch["attn_mask"].shape
        x = self.embed_information(batch)

        # forward the GPT model itself
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, batch)
        x = self.transformer.ln_f(x)

        # Repad
        if (x.ndim == 2) and repad:
            x = pad_input(x, batch["indices"], B, T)

        return x

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
                    total_iters=self.hparams["warmup_steps"],
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    def update_warmup_steps(self, warmup_steps):
        self.hparams["warmup_steps"] = warmup_steps


class PretrainNanoEncoder(BaseNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.metrics = FusedAccuracyAtK([1, 10, 100], reduce="sum")
        self.metric_vals = {f"MLM top{key}": [] for key in self.metrics.top_k}
        self.n_metric_vals = 0
        self.criterion = FACrossEntropyLoss()

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
        x = self.forward(batch, repad=False)

        # Sparse token prediction
        labels = batch["target"]
        if -100 in labels:
            mask_tokens = labels != -100
            x = x[mask_tokens]
            labels = labels[mask_tokens]
        batch["labels"] = labels

        # Decodes and reshapes
        decoded_output = self.decoder(x)

        # Calculates CE loss
        loss = self.criterion(decoded_output, labels)

        return loss, decoded_output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True, batch_size=batch["event"].size(0))

        # Compute metrics
        logits = decoded_output.detach()
        targets = batch["labels"].view(-1).detach()
        res = self.metrics(logits, targets)
        for k, val in res:
            if isinstance(val, list):
                self.metric_vals[f"MLM top{k}"].extend(val)
            else:
                self.metric_vals[f"MLM top{k}"].append(val)
        self.n_metric_vals += len(logits)

        return loss

    def on_validation_epoch_end(self):
        for name, metric in self.metric_vals.items():
            self.log(
                name,
                sum(metric) / self.n_metric_vals,
                sync_dist=True,
            )
            self.metric_vals[name].clear()
        self.n_metric_vals = 0


class PredictionFinetuneNanoEncoder(BaseNanoEncoder):
    """Extends NanoEncoder to work with new FT scheme"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.decoder_finetune = torch.nn.Linear(
            self.hparams.d_model, len(self.hparams.prediction_windows)
        )
        self.metrics = [
            CustomROCS(
                pred_times=self.hparams.pred_times,
                prediction_windows=self.hparams.prediction_windows,
            ),
            BestAtThreshold(
                pred_times=self.hparams.pred_times,
                prediction_windows=self.hparams.prediction_windows,
            ),
        ]
        self.validation_step_outputs = {"target": [], "preds": []}
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int, repad=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass
        x = self.forward(batch, repad=repad)
        decoded_output = self.decoder_finetune(x)

        predict_tokens = decoded_output[batch["predict_tokens"]].view(-1)
        predict_targets = batch["target"][batch["target"] != -100]
        loss = self.criterion(predict_tokens, predict_targets)

        if repad:
            decoded_output = torch.nn.utils.rnn.pad_sequence(
                [
                    row[mask]
                    for row, mask in zip(decoded_output, batch["predict_tokens"])
                ],
                batch_first=True,
                padding_side=(
                    self.hparams.padding_side
                    if self.hparams.get("padding_side")
                    else "left"
                ),
                padding_value=torch.nan,
            )
        else:
            decoded_output = predict_tokens
        return loss, decoded_output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step(batch, batch_idx, repad=True)
        self.log("val/loss", loss, sync_dist=True, batch_size=decoded_output.size(0))

        # Compute metrics
        preds = torch.sigmoid(decoded_output).detach().cpu()
        targets = batch["target"].long().detach().cpu()
        if not self.hparams.pred_times:
            targets = targets.view(-1, 1, targets.size(-1))
            preds = preds.view(-1, 1, preds.size(-1))
        self.validation_step_outputs["target"].append(targets)
        self.validation_step_outputs["preds"].append(preds)

        return loss

    def on_validation_epoch_end(self):
        targets = torch.cat(self.validation_step_outputs["target"], dim=0)
        preds = torch.cat(self.validation_step_outputs["preds"], dim=0)

        for metric in self.metrics:
            res = metric(preds, targets)
            for name, metric in res.items():
                self.log(name, metric, sync_dist=True)

        self.validation_step_outputs["target"].clear()
        self.validation_step_outputs["preds"].clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        _, decoded_output = self.standard_step(batch, batch_idx)
        # Compute metrics
        preds = torch.sigmoid(decoded_output).detach().cpu()
        targets = batch["target"].long().detach().cpu()
        self.validation_step_outputs["target"].append(targets)
        self.validation_step_outputs["preds"].append(preds)
        return preds


class RiskPredictionFinetuneNanoEncoder(PredictionFinetuneNanoEncoder):
    """Extends Finetuning to work with Cumulative Risk Predictions"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.metrics = [
            CustomROCS(
                pred_times=self.hparams.pred_times,
                prediction_windows=self.hparams.prediction_windows,
            )
        ]

        self.decoder_finetune = CumulativeProbabilityLayer(
            self.hparams["d_model"], len(self.hparams.prediction_windows)
        )


class FamilyPredictionFinetuneNanoEncoder(PredictionFinetuneNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x


class FamilyRiskPredictionFinetuneNanoEncoder(RiskPredictionFinetuneNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x


class FamilyPretrainNanoEncoder(PretrainNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x
