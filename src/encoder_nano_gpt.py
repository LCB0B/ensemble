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
from flash_attn.bert_padding import pad_input,unpad_input
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FACrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR
from src.encoder_blocks import Block, CumulativeProbabilityLayer
from src.metrics import CustomROCS, FusedAccuracyAtK, BestAtThreshold
from src.time2vec import Time2Vec
from src.utils import print_main

import torch.nn.functional as F

class BaseNanoEncoder(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(
            self.hparams.vocab_size, self.hparams.d_model, padding_idx=0
        )
        # self.t2v_abspos = Time2Vec(
        #     output_dim=self.hparams.d_model,
        #     clip_min=-100,
        #     clip_max=100,
        #     init_scale=1e-4,
        # )
        # self.t2v_age = Time2Vec(
        #     output_dim=self.hparams.d_model,
        #     clip_min=-100,
        #     clip_max=100,
        #     init_scale=1e-4,
        # )
        # self.segment_embeddings = nn.Embedding(
        #     self.hparams.max_seq_len, self.hparams.d_model, padding_idx=0
        # )
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
        # x += self.t2v_age(batch["age"])

        # x += self.segment_embeddings(batch["segment"])

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

        self.ignore_tokens = torch.tensor([-100] + self.hparams.ignore_tokens)

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

    def standard_step2(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.forward(batch, repad=False)

        labels = batch["target"]

        if self.ignore_tokens is not None:
            if self.ignore_tokens.device != labels.device:
                self.ignore_tokens = self.ignore_tokens.to(labels.device)
            mask = ~torch.isin(labels, self.ignore_tokens)
            x = x[mask]
            labels = labels[mask]

        batch["labels"] = labels

        # Decodes and reshapes
        decoded_output = self.decoder(x)

        # Calculates CE loss
        loss = self.criterion(decoded_output, labels)

        return loss, decoded_output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, decoded_output = self.standard_step2(batch, batch_idx)
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

        return

class GenerativeNanoEncoder(PretrainNanoEncoder):
    """
    Generation wrapper for the GPT backbone:
      - FlashAttention-backed forward pass
      - Optional KV-cache fast path if Block provides it
      - Top-p / temperature sampling
    """
    def __init__(self, *args, max_generation_length=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_generation_length = max_generation_length
        
    def prepare_for_inference(self, device=None, dtype="auto", warmup=True, pad_token_id=0):
        """
        cast params and buffers to a single dtype and device, fix rotary caches,
        optional tiny warmup so caches materialize in the right dtype
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # pick dtype
        if isinstance(dtype, torch.dtype):
            target_dtype = dtype
        elif isinstance(dtype, str):
            dmap = {
                "fp16": torch.float16, "float16": torch.float16,
                "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                "fp32": torch.float32, "float32": torch.float32,
                "auto": None,
            }
            target_dtype = dmap.get(dtype, None)
        else:
            target_dtype = None

        if target_dtype is None:
            if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16

        # move params and buffers
        self.to(device=device, dtype=target_dtype)

        # fix rotary buffers if they already exist
        for m in self.modules():
            rot = getattr(getattr(m, "attn", None), "rotary", None) or getattr(m, "rotary", None)
            if rot is not None:
                for name in ("cos_cached", "sin_cached", "freqs_cached"):
                    buf = getattr(rot, name, None)
                    if isinstance(buf, torch.Tensor):
                        setattr(rot, name, buf.to(device=device, dtype=target_dtype))

        # optional 1-token warmup to (re)materialize caches in the right dtype
        if warmup:
            with torch.inference_mode():
                evt = torch.zeros(1, 1, dtype=torch.long, device=device)
                msk = evt != pad_token_id
                _ = self.forward_generation({"event": evt, "attn_mask": msk}, use_autocast=False)

        # we unified dtype, no need for runtime autocast
        self._mixed_precision = False

        return self


    def embed_information(self, batch):
        return self.embedding(batch["event"])

    def forward_generation(self, batch, use_autocast=False):
    # returns (B, T, D) without any autocast
        x_unp, indices, cu, max_seqlen = self._forward_generation_unpadded_impl(batch)
        B = (len(cu) - 1)
        x = pad_input(x_unp, indices, B, max_seqlen)
        return x
        
    def _forward_generation_unpadded_impl(self, batch):
        # returns unpadded hidden states and varlen metadata
        if "cu_seqlens" in batch:
            # caller already provided varlen layout
            x = self.embed_information(batch)
            x = self.transformer.drop(x)
            for block in self.transformer.h:
                x = block(x, batch)
            x = self.transformer.ln_f(x)
            return x, batch["indices"], batch["cu_seqlens"], batch["max_seqlen_in_batch"]

        if "attn_mask" not in batch:
            batch["attn_mask"] = (batch["event"] != 0)

        batch["attn_mask"] = batch["attn_mask"].to(torch.bool)

        ret = unpad_input(batch["event"].unsqueeze(-1), batch["attn_mask"])
        _, indices, cu_seqlens, max_seqlen_in_batch = ret[:4]

        unpacked = {
            "event": batch["event"].flatten()[indices],
            "attn_mask": batch["attn_mask"],
            "indices": indices,
            "max_seqlen_in_batch": max_seqlen_in_batch,
            "cu_seqlens": cu_seqlens,
            "total": int(cu_seqlens[-1].item()),
        }

        x = self.embed_information(unpacked)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, unpacked)
        x = self.transformer.ln_f(x)
        return x, indices, cu_seqlens, max_seqlen_in_batch
    

    def _forward_generation_impl(self, batch):
        if "cu_seqlens" in batch:
            return super().forward(batch, repad=True)
        if "attn_mask" not in batch:
            batch["attn_mask"] = (batch["event"] != 0)
        batch["attn_mask"] = batch["attn_mask"].to(torch.bool)
        _, indices, cu_seqlens, max_seqlen_in_batch,total= unpad_input(
            batch["event"].unsqueeze(-1), batch["attn_mask"]
        )
        unpacked = {
            "event": batch["event"].flatten()[indices],
            "attn_mask": batch["attn_mask"],
            "indices": indices,
            "max_seqlen_in_batch": max_seqlen_in_batch,
            "cu_seqlens": cu_seqlens,
            "total": total.sum().item(),
        }

        x = self.embed_information(unpacked)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, unpacked)  # FlashAttention path inside Block
        x = self.transformer.ln_f(x)

        # Repad back to (B,T,D)
        B = len(cu_seqlens) - 1
        T = max_seqlen_in_batch
        x = pad_input(x, indices, B, T)
        return x

    @torch.no_grad()
    def generate(
    self,
    batch,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    top_p=0.9,
    pad_token_id=0,
    eos_token_id=None,
    use_cache=False,                 # ignored here; no kv-cache path in this function
    return_attention_mask=True,
    use_autocast=False,              # kept for api compatibility; not used
    preallocate_cache=True,
    profile: bool = False,
    n_samples_per_person: int = 1,
    start_after_n_tokens: int | None = None,
    return_logits: bool = False,
):
        device = batch["event"].device

        if "attn_mask" not in batch or batch["attn_mask"].shape != batch["event"].shape:
            batch["attn_mask"] = (batch["event"] != pad_token_id)
        else:
            batch["attn_mask"] = batch["attn_mask"].to(torch.bool)

        if batch["event"].ndim == 1:
            batch["event"] = batch["event"].unsqueeze(0)
            batch["attn_mask"] = batch["attn_mask"].unsqueeze(0)

        B, T = batch["event"].shape

        if n_samples_per_person and n_samples_per_person > 1:
            def _tile(x):
                return x.repeat_interleave(n_samples_per_person, dim=0) if torch.is_tensor(x) and x.dim() > 0 and x.size(0) == B else x
            batch = {k: _tile(v) for k, v in batch.items()}
            B = B * n_samples_per_person

        actual_len = batch["attn_mask"].sum(1).long()
        if start_after_n_tokens is None:
            prompt_len_vec = actual_len
        else:
            limit = min(int(start_after_n_tokens), T)
            prompt_len_vec = torch.minimum(actual_len, torch.full_like(actual_len, limit))

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        keep = pos < prompt_len_vec.unsqueeze(1)
        batch["attn_mask"] = keep
        batch["event"][~keep] = pad_token_id

        max_prompt_len = int(prompt_len_vec.max().item())
        write_pos = prompt_len_vec.clone()
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        gen_len_vec = torch.zeros(B, dtype=torch.long, device=device)
        logits_steps = []

        for _ in range(max_new_tokens):
            can_write_any = torch.any((~finished) & (write_pos < T))
            if not can_write_any:
                break

            max_cur = int(write_pos.max().item())
            cur = {
                "event": batch["event"][:, :max_cur],
                "attn_mask": batch["attn_mask"][:, :max_cur],
            }

            x_unp, indices, cu, _ = self._forward_generation_unpadded_impl(cur)
            last_idx = cu[1:] - 1
            last_hidden = x_unp[last_idx]

            logits = self.decoder(last_hidden).float()  # keep sampling in fp32
            if return_logits:
                logits_steps.append(logits.detach().cpu())
            next_tokens = self._sample_tokens(logits, temperature, top_k, top_p, finished, pad_token_id)

            can_write = (~finished) & (write_pos < T)
            rows = torch.nonzero(can_write).squeeze(-1)
            if rows.numel() == 0:
                break
            wp = write_pos[rows]
            batch["event"][rows, wp] = next_tokens[rows]
            batch["attn_mask"][rows, wp] = True

            if eos_token_id is not None:
                finished[rows] |= (next_tokens[rows] == eos_token_id)
            gen_len_vec[rows] += 1
            write_pos[rows] += 1
            if finished.all():
                break

        gen_max = int(gen_len_vec.max().item())
        final_max = int(write_pos.max().item())

        generated = torch.full((B, gen_max), pad_token_id, dtype=batch["event"].dtype, device=device)
        for i in range(B):
            li = int(prompt_len_vec[i].item())
            gi = int(gen_len_vec[i].item())
            if gi > 0:
                generated[i, :gi] = batch["event"][i, li:li+gi]

        out = {
            "prompt": batch["event"][:, :max_prompt_len],
            "generated": generated,
            "full": batch["event"][:, :final_max],
            "generation_lengths": gen_len_vec,
            "finished": finished,
            "prompt_length": max_prompt_len,
            "prompt_length_per_row": prompt_len_vec,
        }
        if return_attention_mask:
            out["generated_attn_mask"] = batch["attn_mask"][:, :final_max]
        out["generated_events"] = out["generated"]
        out["full_sequences"] = out["full"]
        if return_logits:
            out["logits"] = torch.stack(logits_steps, dim=1) if len(logits_steps) > 0 else torch.empty(0)
        return out

    # helpers
    def _sample_tokens(self, logits, temperature, top_k, top_p, finished_mask, pad_token_id):
        dtype = logits.dtype
        min_val = torch.finfo(dtype).min  # dtype-safe "minus infinity"

        if temperature is None or temperature <= 0:
            next_tokens = torch.argmax(logits, dim=-1)
        else:
            if temperature != 1.0:
                logits = logits / torch.as_tensor(temperature, device=logits.device, dtype=dtype)

            if top_k is not None and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, min_val)

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                # keep at least one token (same behavior as previous impl)
                cutoff = cumsum - probs > top_p
                sorted_logits = sorted_logits.masked_fill(cutoff, min_val)
                inv = torch.argsort(sorted_idx, dim=-1)
                logits = torch.gather(sorted_logits, -1, inv)

            probs = F.softmax(logits, dim=-1).clamp_min_(0)
            rowsum = probs.sum(dim=-1, keepdim=True)

            # if a row became all-zero (e.g., extreme masking), fall back to argmax for that row
            need_fallback = (rowsum.squeeze(-1) == 0)
            if need_fallback.any():
                next_tokens = torch.empty(logits.size(0), device=logits.device, dtype=torch.long)
                if (~need_fallback).any():
                    toks_ok = torch.multinomial(probs[~need_fallback], num_samples=1).squeeze(-1)
                    next_tokens[~need_fallback] = toks_ok
                # argmax fallback on the already-masked logits
                next_tokens[need_fallback] = torch.argmax(logits[need_fallback], dim=-1)
            else:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        if finished_mask is not None and finished_mask.any():
            next_tokens = torch.where(
                finished_mask.to(torch.bool),
                torch.full_like(next_tokens, 0 if pad_token_id is None else int(pad_token_id)),
                next_tokens,
            )
        return next_tokens

    

    @torch.no_grad()
    @torch.no_grad()
    def generate_kv(
        self,
        batch,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=0.9,
        pad_token_id=0,
        eos_token_id=None,
        n_samples_per_person: int = 1,
        start_after_n_tokens: int | None = None,
        return_attention_mask=True,
        return_logits: bool = False,
    ):
        # try FA kvcache import; if unavailable, fallback
        try:
            from flash_attn.flash_attn_interface import flash_attn_with_kvcache
        except Exception:
            return self.generate(
                batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                n_samples_per_person=n_samples_per_person,
                start_after_n_tokens=start_after_n_tokens,
                return_attention_mask=return_attention_mask,
                return_logits=return_logits,
            )

        device = batch["event"].device
        if "attn_mask" not in batch or batch["attn_mask"].shape != batch["event"].shape:
            batch["attn_mask"] = (batch["event"] != pad_token_id)
        else:
            batch["attn_mask"] = batch["attn_mask"].to(torch.bool)

        if batch["event"].ndim == 1:
            batch["event"] = batch["event"].unsqueeze(0)
            batch["attn_mask"] = batch["attn_mask"].unsqueeze(0)

        B, T = batch["event"].shape

        if n_samples_per_person and n_samples_per_person > 1:
            def _tile(x):
                return x.repeat_interleave(n_samples_per_person, dim=0) if torch.is_tensor(x) and x.dim() > 0 and x.size(0) == B else x
            batch = {k: _tile(v) for k, v in batch.items()}
            B = B * n_samples_per_person

        actual_len = batch["attn_mask"].sum(1).long()
        if start_after_n_tokens is None:
            prompt_len_vec = actual_len
        else:
            limit = min(int(start_after_n_tokens), T)
            prompt_len_vec = torch.minimum(actual_len, torch.full_like(actual_len, limit))

        # truncate tails so decoding begins exactly after min(limit, actual_len)
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        keep = pos < prompt_len_vec.unsqueeze(1)
        batch["attn_mask"] = keep
        batch["event"][~keep] = pad_token_id

        L = int(prompt_len_vec.max().item())
        max_total_len = min(T, L + max_new_tokens)

        event_trim = batch["event"][:, :L]
        mask_trim  = batch["attn_mask"][:, :L]

        ret = unpad_input(event_trim.unsqueeze(-1), mask_trim)
        _, indices_L, cu_seqlens, max_seqlen_in_batch = ret[:4]
        indices_L = indices_L.to(torch.long)

        unpacked = {
            "event": event_trim.flatten()[indices_L],
            "attn_mask": mask_trim,
            "indices": indices_L,
            "max_seqlen_in_batch": max_seqlen_in_batch,
            "cu_seqlens": cu_seqlens,
            "total": int(cu_seqlens[-1].item()),
        }

        x = self.embed_information(unpacked)
        x = self.transformer.drop(x)

        head_dim = self.transformer.h[0].attn.head_dim
        n_heads  = self.transformer.h[0].attn.num_heads
        d_model  = self.hparams.d_model

        def _pad_kv(k_or_v_unp):  # (total, H, Hd) -> (B, L, H, Hd)
            return pad_input(k_or_v_unp, indices_L, B, L)

        # precompute rotary cos/sin up to max_total_len from inv_freq (one tensor per layer)
        def _rope_cos_sin_from_invfreq(rotary, seqlen, rotary_dim, device, dtype):
            half = rotary_dim // 2
            inv = getattr(rotary, "inv_freq", None)
            if inv is None:
                base = 10000.0
                inv = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
            else:
                inv = inv.to(device=device, dtype=dtype)
            pos = torch.arange(seqlen, device=device, dtype=dtype).unsqueeze(1)  # (S,1)
            freqs = pos * inv.unsqueeze(0)                                       # (S,half)
            return torch.cos(freqs), torch.sin(freqs)                            # (S,half)

        caches = []
        for block in self.transformer.h:
            x_norm = block.ln_1(x)
            q, k, v = block.attn.Wqkv(x_norm).chunk(3, dim=-1)
            q = q.view(-1, n_heads, head_dim)
            k = k.view(-1, n_heads, head_dim)
            v = v.view(-1, n_heads, head_dim)

            q, k = block.attn.rotary(q, k, cu_seqlens, max_seqlen_in_batch)

            qkv = torch.stack((q, k, v), dim=1)  # (total, 3, H, Hd)
            y = block.attn.self_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch,causal=True)
            y = y.view(-1, d_model)
            y = block.attn.resid_dropout(block.attn.out_proj(y))
            x = x + y
            x = x + block.mlp(block.ln_2(x))

            k_cache = _pad_kv(k)
            v_cache = _pad_kv(v)
            if max_total_len > L:
                pad_len = max_total_len - L
                k_cache = torch.cat([k_cache,
                                    torch.zeros(B, pad_len, n_heads, head_dim, dtype=k_cache.dtype, device=k_cache.device)], 1)
                v_cache = torch.cat([v_cache,
                                    torch.zeros(B, pad_len, n_heads, head_dim, dtype=v_cache.dtype, device=v_cache.device)], 1)

            # precompute rope tables for this layer/dtype
            rot = block.attn.rotary
            rotary_dim = int(getattr(rot, "rotary_dim", self.transformer.h[0].attn.head_dim))
            interleaved = bool(getattr(rot, "interleaved", True))
            cos, sin = _rope_cos_sin_from_invfreq(rot, max_total_len, rotary_dim, x.device, x.dtype)

            caches.append({
                "k": k_cache, "v": v_cache,
                "seqlens": prompt_len_vec.clone(),
                "max_len": max_total_len,
                "ln_1": block.ln_1, "ln_2": block.ln_2, "mlp": block.mlp,
                "Wqkv": block.attn.Wqkv, "out_proj": block.attn.out_proj, "drop": block.attn.resid_dropout,
                "rotary_cos": cos, "rotary_sin": sin,
                "rotary_interleaved": interleaved,                # <<< pass through
                "softmax_scale": getattr(block.attn, "softmax_scale", None),
                "n_heads": block.attn.num_heads,
                "head_dim": block.attn.head_dim,
            })

        x = self.transformer.ln_f(x)
        last_idx_prompt = cu_seqlens[1:] - 1
        last_hidden_current = x[last_idx_prompt]  # (B, d_model, now LN-normalized)

        write_pos = prompt_len_vec.clone()
        finished = torch.zeros(B, dtype=torch.bool, device=x.device)
        gen_len_vec = torch.zeros(B, dtype=torch.long, device=x.device)
        logits_steps = []

        for _ in range(max_new_tokens):
            can_write = (~finished) & (write_pos < T)
            rows = torch.nonzero(can_write).squeeze(-1)
            if rows.numel() == 0:
                break

            logits = self.decoder(last_hidden_current[rows]).float()
            if return_logits:
                logits_steps.append(logits.detach().cpu())
            next_tokens = self._sample_tokens(logits, temperature, top_k, top_p, finished[rows], pad_token_id)

            wp = write_pos[rows]
            batch["event"][rows, wp] = next_tokens
            batch["attn_mask"][rows, wp] = True
            if eos_token_id is not None:
                finished[rows] |= (next_tokens == eos_token_id)
            gen_len_vec[rows] += 1
            write_pos[rows] += 1

            x_step = self.embedding(next_tokens)  # (B_act, d)

            for li, cache in enumerate(caches):
                x_norm = cache["ln_1"](x_step)
                q_, k_, v_ = cache["Wqkv"](x_norm).chunk(3, dim=-1)
                q_ = q_.view(-1, n_heads, head_dim)
                k_ = k_.view(-1, n_heads, head_dim)
                v_ = v_.view(-1, n_heads, head_dim)

                # per-step shapes for fa kernel
                q_in   = q_.unsqueeze(1).contiguous()               # (B_act, 1, H, Hd)
                k_new  = k_.unsqueeze(1).contiguous()               # (B_act, 1, H, Hd)
                v_new  = v_.unsqueeze(1).contiguous()
                k_rows = cache["k"].index_select(0, rows).contiguous()
                v_rows = cache["v"].index_select(0, rows).contiguous()

                # current cache lens before appending the new token
                cache_lens = torch.clamp(cache["seqlens"][rows].to(torch.int32), min=0)

                # precomputed rope tables (crop to max_len)
                cos = cache["rotary_cos"][:cache["max_len"]] # (S, rotary_dim//2)
                sin = cache["rotary_sin"][:cache["max_len"]]

                # call fa kernel: updates k_rows/v_rows in-place and attends
                #print('call fa kernel: updates k_rows/v_rows in-place and attends')
                #print(q_in.shape, k_rows.shape, v_rows.shape, k_new.shape, v_new.shape,cos.shape, sin.shape, cache_lens.min().item(), cache_lens.max().item(), cache["max_len"])
                
                o = flash_attn_with_kvcache(
                    q_in, k_rows, v_rows,
                    k=k_new, v=v_new,
                    rotary_cos=cos, rotary_sin=sin,
                    cache_seqlens=cache_lens,
                    causal=True,
                    rotary_interleaved=cache["rotary_interleaved"],      # <<< match training
                    softmax_scale=cache["softmax_scale"],                # <<< match training scale if set
                    num_splits=2, #heuristic-dependent; measure 1/2/0 (auto)
                )  # (B_act, 1, H, Hd)

                # write the updated rows back to the full cache tensors
                cache["k"].index_copy_(0, rows, k_rows)
                cache["v"].index_copy_(0, rows, v_rows)
                cache["seqlens"][rows] = write_pos[rows]  # len advanced by 1

                o = o.view(o.size(0), -1)
                y = cache["drop"](cache["out_proj"](o))
                x_step = x_step + y
                x_step = x_step + cache["mlp"](cache["ln_2"](x_step))
                # bring the 1-token hidden through the final LN before it becomes "current state"
            
            x_step = self.transformer.ln_f(x_step)
            if rows.numel() < B:
                last_hidden_current[rows] = x_step
            else:
                last_hidden_current = x_step

            if finished.all():
                break

        gen_max = int(gen_len_vec.max().item())
        final_max = int(write_pos.max().item())
        generated = torch.full((B, gen_max), pad_token_id, dtype=batch["event"].dtype, device=device)
        for i in range(B):
            li = int(prompt_len_vec[i].item())
            gi = int(gen_len_vec[i].item())
            if gi > 0:
                generated[i, :gi] = batch["event"][i, li:li+gi]

        out = {
            "prompt": batch["event"][:, :L],
            "generated": generated,
            "full": batch["event"][:, :final_max],
            "generation_lengths": gen_len_vec,
            "finished": finished,
            "prompt_length": L,
            "prompt_length_per_row": prompt_len_vec,
        }
        if return_attention_mask:
            out["generated_attn_mask"] = batch["attn_mask"][:, :final_max]
        if return_logits and len(logits_steps) > 0:
            out["logits"] = torch.stack(logits_steps, dim=1)
        return out
