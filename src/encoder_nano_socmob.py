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
import pandas as pd
import torch
import torch.nn as nn
from flash_attn.bert_padding import pad_input
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import (
    AUROC,
    AveragePrecision,
    MeanAbsoluteError,
    MeanSquaredError,
)

from src.encoder_blocks import Block, CumulativeProbabilityLayer
from src.metrics import CustomROCS, FusedAccuracyAtK
from src.time2vec import Time2Vec
from src.utils import print_main


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

        # x += self.t2v_abspos(batch["abspos"])
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
        self.metric = CustomROCS(
            pred_times=self.hparams.pred_times,
            prediction_windows=self.hparams.prediction_windows,
        )
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

        res = self.metric(preds, targets)
        for name, metric in res.items():
            self.log(name, metric, sync_dist=True)

        self.validation_step_outputs["target"].clear()
        self.validation_step_outputs["preds"].clear()

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        _, decoded_output = self.standard_step(batch, batch_idx)
        # Compute metrics
        preds = torch.sigmoid(decoded_output).detach().cpu()
        targets = batch["target"].long().detach().cpu()
        self.validation_step_outputs["target"].append(targets)
        self.validation_step_outputs["preds"].append(preds)
        return preds


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
        x = self.forward(batch)

        # Decodes and reshapes
        cls = x[:, 0]
        decoded_output = self.decoder_finetune(cls).view(-1)

        return decoded_output


class RiskPredictionFinetuneNanoEncoder(PredictionFinetuneNanoEncoder):
    """Extends Finetuning to work with Cumulative Risk Predictions"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.decoder_finetune = CumulativeProbabilityLayer(
            self.hparams["d_model"], len(self.hparams.prediction_windows)
        )

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        _, decoded_output = self.standard_step(batch, batch_idx)
        # Compute metrics
        preds = torch.sigmoid(decoded_output).detach().cpu()
        targets = batch["target"].long().detach().cpu()
        self.validation_step_outputs["target"].append(targets)
        self.validation_step_outputs["preds"].append(preds)
        return preds

    def risk_step(self, batch, batch_idx):
        x = self.forward(batch)
        decoded_output = self.decoder_finetune(x)

        event_mask = batch["event_mask"]
        num_selected = event_mask.sum(dim=-1, keepdim=True)
        decoded_output = torch.bmm(event_mask, decoded_output) / (
            num_selected + 1e-8
        )  # mean aggrigation

        return torch.sigmoid(decoded_output)

    def on_predict_epoch_end(self):
        self.age_bracket_metrics()

    def age_bracket_metrics(self):
        tbuckets, pbuckets = [], []
        prediction_intervals = 20

        for tbatch, pbatch in zip(
            self.validation_step_outputs["target"],
            self.validation_step_outputs["preds"],
        ):
            for i in range(0, len(tbatch[0]), prediction_intervals):
                b_idx = i // prediction_intervals
                if len(tbuckets) <= b_idx:
                    tbuckets.append([])
                    pbuckets.append([])
                tbuckets[b_idx].extend(tbatch[:, i : i + prediction_intervals])
                pbuckets[b_idx].extend(pbatch[:, i : i + prediction_intervals])
        windows = self.hparams.prediction_windows
        results = []
        valid_i = []
        for i, (targets, preds) in enumerate(zip(tbuckets, pbuckets)):
            if len(targets) == 0 or (torch.cat(targets) == -100).all():
                continue
            result = self.metrics["AUROC"](
                torch.cat(preds).view(-1, len(windows)),
                torch.cat(targets).view(-1, len(windows)),
            )
            if result.sum() == 0:
                continue
            results.append(result.tolist())
            valid_i.append(i)
        brackets = list(range(i))
        valid_brackets = [brackets[i] * 20 for i in valid_i]

        df = pd.DataFrame(
            results,
            columns=[f"{w}y" for w in windows],
            index=[f"{b}-{b+20}" for b in valid_brackets],
        ).T.round(3)
        self.logger.experiment.add_text(
            "Age brackets", df.to_markdown(), self.global_step
        )


class ParentRiskPredictionFinetuneNanoEncoder(RiskPredictionFinetuneNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x


class ParentPretrainNanoEncoder(PretrainNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x


# from typing import Literal


# class FamilyRiskPredictionFinetuneNanoEncoder(RiskPredictionFinetuneNanoEncoder):
#     def __init__(self, *args, feature_set: Literal["own", "parents", "both"], **kwargs):
#         super().__init__()

#         # N_FAMILY_EMBS_MAP = {"own": 1, "parents": 2, "both": 3}
#         # self.typ_embeddings = nn.Embedding(
#         #     N_FAMILY_EMBS_MAP[feature_set], self.hparams.d_model, padding_idx=0
#         # )
#         self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

#     def embed_information(self, batch):
#         x = super().embed_information(batch)

#         x += self.typ_embeddings(batch["family_type"])

#         return x


class FamilyPretrainNanoEncoder(PretrainNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x


class MultiHeadFamilyRegressionFinetuneNanoEncoder(BaseNanoEncoder):
    def __init__(self, n_predict_tokens: int, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)
        self.n_predict_tokens = n_predict_tokens
        d_model = self.hparams["d_model"]

        # One weight per decoder: (n_tokens, d_model, 1)
        self.decoder_weights = nn.Parameter(torch.empty(n_predict_tokens, d_model, 1))
        self.decoder_biases = nn.Parameter(torch.zeros(n_predict_tokens))

        nn.init.xavier_uniform_(self.decoder_weights)

        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.validation_step_outputs = {"target_regression": [], "preds": []}
        self.criterion = torch.nn.MSELoss()

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x

    # def standard_step(
    #     self, batch: Dict[str, Any], batch_idx: int
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # Forward pass
    #     x = self.forward(batch, repad=False)

    #     # Select hidden states (1 = CLS/PREDICT token)
    #     hidden_states = x[batch["event"] == 1]

    #     # Decode
    #     decoded_output = self.decoder_finetune(hidden_states).view(-1)

    #     # if self.n_predict_tokens is None:
    #     #     # One attention mask per person
    #     #     B, _ = batch["attn_mask"].shape
    #     #     # Remainder once viewed with 0th dimension as n people must be amount of predict tokens
    #     #     # Requires an assumption of equal amount of predict tokens, which is the case for the regression setup
    #     #     self.n_predict_tokens = decoded_output.view(B, -1).shape[1]

    #     # Calculate loss
    #     targets = batch["target_regression"].view(-1)
    #     loss = self.criterion(decoded_output, targets)

    #     return loss, decoded_output.detach()

    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.forward(batch, repad=False)  # shape (B, S, D)
        B = batch["attn_mask"].shape[0]

        # Mask prediction tokens
        pred_mask = batch["event"] == 1
        hidden_states = x[pred_mask]  # (B*T, D)
        hidden_states = hidden_states.view(B, self.n_predict_tokens, -1)  # (B, T, D)

        # Prepare for batched matmul
        H = hidden_states.unsqueeze(2)  # (B, T, 1, D)
        W = self.decoder_weights.unsqueeze(0)  # (1, T, D, 1)
        logits = (H @ W).squeeze(-1).squeeze(-1)  # (B, T)
        logits += self.decoder_biases  # (B, T)

        preds = logits.view(-1)  # (B*T,)
        targets = batch["target_regression"].view(-1)  # (B*T,)
        loss = self.criterion(preds, targets)

        return loss, preds.detach()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True, batch_size=decoded_output.size(0))

        # Compute metrics
        targets = batch["target_regression"].view(-1)
        self.validation_step_outputs["target_regression"].append(targets)
        self.validation_step_outputs["preds"].append(decoded_output)

        return loss

    def on_validation_epoch_end(self):

        targets = torch.cat(self.validation_step_outputs["target_regression"])
        preds = torch.cat(self.validation_step_outputs["preds"])

        self.log(
            "RMSE",
            self.rmse(preds * 100, targets * 100),
            sync_dist=True,
        )
        self.log(
            "MAE",
            self.mae(preds * 100, targets * 100),
            sync_dist=True,
        )

        targets = targets.view(-1, self.n_predict_tokens).T
        preds = preds.view(-1, self.n_predict_tokens).T

        print(
            f"Validation end, targets shape {targets.shape}, preds shape {preds.shape}"
        )
        # Log for each prediction token, e.g. different time points
        for i in range(self.n_predict_tokens):
            self.log(
                f"RMSE_predict_token_{i + 1}",
                self.rmse(preds[i] * 100, targets[i] * 100),
                sync_dist=True,
            )
            self.log(
                f"MAE_predict_token_{i + 1}",
                self.mae(preds[i] * 100, targets[i] * 100),
                sync_dist=True,
            )

        self.validation_step_outputs["target_regression"].clear()
        self.validation_step_outputs["preds"].clear()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = self.forward(batch, repad=False)
        B = batch["attn_mask"].shape[0]

        pred_mask = batch["event"] == 1
        hidden_states = x[pred_mask].view(B, self.n_predict_tokens, -1)  # (B, T, D)

        H = hidden_states.unsqueeze(2)  # (B, T, 1, D)
        W = self.decoder_weights.unsqueeze(0)  # (1, T, D, 1)
        logits = torch.matmul(H, W).squeeze(-1).squeeze(-1)  # (B, T)
        logits += self.decoder_biases  # (B, T)

        return logits.view(-1)


class FamilyRegressionFinetuneNanoEncoder(BaseNanoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.typ_embeddings = nn.Embedding(4, self.hparams.d_model, padding_idx=0)

        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()

        self.validation_step_outputs = {"target_regression": [], "preds": []}
        self.decoder_finetune = nn.Linear(
            self.hparams["d_model"], 1
        )  # DON'T REUSE IDENTICAL DECODER NAME AS BASE `NANOENCODER`, AS THIS WILL BREAK LOADING OF CHECKPOINTS FROM PRETRAINED MODELS
        self.criterion = torch.nn.MSELoss()
        self.n_predict_tokens = None

    def embed_information(self, batch):
        x = super().embed_information(batch)

        x += self.typ_embeddings(batch["family_type"])

        return x

    def standard_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Forward pass
        x = self.forward(batch, repad=False)

        # Select hidden states (1 = CLS/PREDICT token)
        hidden_states = x[batch["event"] == 1]

        # Decode
        decoded_output = self.decoder_finetune(hidden_states).view(-1)

        if self.n_predict_tokens is None:
            # One attention mask per person
            B, _ = batch["attn_mask"].shape
            # Remainder once viewed with 0th dimension as n people must be amount of predict tokens
            # Requires an assumption of equal amount of predict tokens, which is the case for the regression setup
            self.n_predict_tokens = decoded_output.view(B, -1).shape[1]

        # Calculate loss
        targets = batch["target_regression"].view(-1)
        loss = self.criterion(decoded_output, targets)

        return loss, decoded_output.detach()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step(batch, batch_idx)
        self.log("val/loss", loss, sync_dist=True, batch_size=decoded_output.size(0))

        # Compute metrics
        targets = batch["target_regression"].view(-1)
        self.validation_step_outputs["target_regression"].append(targets)
        self.validation_step_outputs["preds"].append(decoded_output)

        return loss

    def on_validation_epoch_end(self):

        targets = torch.cat(self.validation_step_outputs["target_regression"])
        preds = torch.cat(self.validation_step_outputs["preds"])

        self.log(
            "RMSE",
            self.rmse(preds, targets),
            sync_dist=True,
        )
        self.log(
            "MAE",
            self.mae(preds, targets),
            sync_dist=True,
        )

        targets = targets.view(-1, self.n_predict_tokens).T
        preds = preds.view(-1, self.n_predict_tokens).T

        print(
            f"Validation end, targets shape {targets.shape}, preds shape {preds.shape}"
        )
        # Log for each prediction token, e.g. different time points
        for i in range(self.n_predict_tokens):
            self.log(
                f"RMSE_predict_token_{i + 1}",
                self.rmse(preds[i], targets[i]),
                sync_dist=True,
            )
            self.log(
                f"MAE_predict_token_{i + 1}",
                self.mae(preds[i], targets[i]),
                sync_dist=True,
            )

        self.validation_step_outputs["target_regression"].clear()
        self.validation_step_outputs["preds"].clear()

    def predict_step(self, batch: Dict[str, Any], batch_idx: int):
        x = self.forward(batch, repad=False)

        hidden_states = x[batch["event"] == 1]
        decoded_output = self.decoder_finetune(hidden_states)

        return decoded_output
