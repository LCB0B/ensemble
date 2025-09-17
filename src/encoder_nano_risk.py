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
import torch.nn.functional as F

from flash_attn.bert_padding import pad_input, unpad_input
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



class GenerativeNanoEncoder(PretrainNanoEncoder):
    """Extends PretrainNanoEncoder with efficient generation capabilities"""
    
    def __init__(self, *args, max_generation_length=1024, extension_chunk_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_generation_length = max_generation_length
        self.extension_chunk_size = extension_chunk_size
        
        # TODO: Add KV cache initialization here
        # self.kv_cache = None  # Will store past key-value pairs
        # self.cache_enabled = False
    
    def forward_generation(self, batch, use_autocast=True):
        """
        Forward pass specifically for generation with autocast support
        
        Args:
            batch: Input batch in standard format
            use_autocast: Whether to use CUDA autocast for mixed precision
        """
        device = batch['event'].device
        should_autocast = (use_autocast and 
                          device.type == 'cuda' and 
                          hasattr(self, '_mixed_precision') and 
                          self._mixed_precision)
        
        if should_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return self._forward_generation_impl(batch)
        else:
            return self._forward_generation_impl(batch)
    
    def _forward_generation_impl(self, batch):
        """Implementation of generation forward pass"""
        if 'cu_seqlens' in batch:
            # Already unpacked (training format) - use parent's forward
            return super().forward(batch, repad=True)
        else:
            # Standard format (generation) - need to unpack for Flash Attention
            return self._unpack_and_forward_generation(batch)
    
    def _unpack_and_forward_generation(self, batch):
        """Convert standard format to Flash Attention unpacked format and forward"""
        from flash_attn.bert_padding import unpad_input, pad_input
        
        # Create attention mask if not present
        if 'attn_mask' not in batch:
            batch['attn_mask'] = (batch['event'] != 0)
        
        # Unpack the batch like the datamodule does
        _, indices, cu_seqlens, max_seqlen_in_batch, total = unpad_input(
            batch["event"].unsqueeze(-1), batch["attn_mask"]
        )
        
        # Create unpacked batch
        unpacked_batch = {
            'event': batch["event"].flatten()[indices],
            'abspos': batch["abspos"].flatten()[indices], 
            'age': batch["age"].flatten()[indices],
            'segment': batch["segment"].flatten()[indices],
            'attn_mask': batch["attn_mask"],
            'indices': indices,
            'max_seqlen_in_batch': max_seqlen_in_batch,
            'cu_seqlens': cu_seqlens,
            'total': total.sum().item()
        }
        
        # Forward through Flash Attention layers - generation-specific implementation
        x = self.embed_information(unpacked_batch)
        
        # TODO: KV Cache - If we have past_key_values, only process new tokens
        # if past_key_values is not None:
        #     x = x[-total.sum().item():]  # Only new tokens
        
        x = self.transformer.drop(x)
        
        # TODO: KV Cache - Collect new key-values during forward pass
        # new_key_values = []
        
        for i, block in enumerate(self.transformer.h):
            # TODO: KV Cache - Pass and collect layer-specific cache
            # layer_past = past_key_values[i] if past_key_values else None
            # x, new_kv = block.forward_generation(x, unpacked_batch, past_key_values=layer_past, return_cache=True)
            # new_key_values.append(new_kv)
            
            # Use the original block forward method for generation
            x = block(x, unpacked_batch)
        
        x = self.transformer.ln_f(x)
        
        # Repad if requested
        B = len(cu_seqlens) - 1
        T = max_seqlen_in_batch
        x = pad_input(x, indices, B, T)
        
        # TODO: KV Cache - Return both hidden states and new cache
        # return x, new_key_values
        return x
    
    def _ensure_batch_2d(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Ensure tensors are (B, T); if (T,) -> (1, T)."""
        for key in ["event", "abspos", "age", "segment", "attn_mask"]:
            if key in batch and torch.is_tensor(batch[key]) and batch[key].ndim == 1:
                batch[key] = batch[key].unsqueeze(0)
        return batch

    
    def embed_information(self, batch):
        """
        Embeds token + temporal/segment features with safe dtype handling for mixed precision.
        Ensures all Time2Vec inputs match module parameter dtype (fp16/bf16) after model.half().
        """
        tok_dtype = self.embedding.weight.dtype
        x = self.embedding(batch["event"])  # (B,T,D) in tok_dtype

        abspos = batch["abspos"]
        age = batch["age"]
        segment = batch["segment"]

        if abspos.dtype != tok_dtype:
            abspos = abspos.to(tok_dtype)
        if age.dtype != tok_dtype:
            age = age.to(tok_dtype)

        # Time2Vec modules were also cast by model.half(), so feed matching dtype
        x = x + self.t2v_abspos(abspos)
        x = x + self.t2v_age(age)
        x = x + self.segment_embeddings(segment)  # segment indices ok (long) -> returns tok_dtype
        return x

    # Replace ONLY the use_cache branch inside generate() with this updated version (full generate function shown for clarity)

    @torch.no_grad()
    def generate(
        self,
        batch,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        top_p=None,
        pad_token_id=0,
        eos_token_id=None,
        use_cache=False,
        return_attention_mask=True,
        use_autocast=True,
        profile: bool = False,
        debug: bool = False,
        debug_steps: int = 3,
        preallocate_cache: bool = True,   # NEW FLAG
    ):
        import os, time
        debug = debug or os.environ.get("DEBUG_GEN") == "1"
        def dprint(m): 
            if debug: print(f"[GEN] {m}")

        if 'cu_seqlens' in batch and 'indices' in batch:
            batch = self._repack_batch(batch)
        batch = self._ensure_batch_2d(batch)

        if "attn_mask" not in batch or batch["attn_mask"].shape != batch["event"].shape:
            batch["attn_mask"] = (batch["event"] != 0).to(batch["event"].device, torch.float32)

        B, T0 = batch["event"].shape
        device = batch["event"].device
        model_dtype = self.embedding.weight.dtype

        # Temporal tensors
        def ensure(name, make_fn):
            if name not in batch or batch[name].shape != (B, T0):
                batch[name] = make_fn()
        ensure("abspos", lambda: torch.arange(T0, device=device, dtype=model_dtype).unsqueeze(0).expand(B, -1))
        ensure("age", lambda: torch.zeros(B, T0, dtype=model_dtype, device=device))
        ensure("segment", lambda: torch.zeros(B, T0, dtype=torch.long, device=device))

        # Cast continuous to model dtype
        for k in ("abspos","age"):
            if batch[k].dtype != model_dtype:
                batch[k] = batch[k].to(model_dtype)

        prompt_len = T0
        if max_new_tokens <= 0:
            out = {
                "prompt": batch["event"],
                "generated": batch["event"][:, 0:0],
                "full": batch["event"],
                "generation_lengths": torch.zeros(B, dtype=torch.long, device=device),
                "finished": torch.zeros(B, dtype=torch.bool, device=device),
                "prompt_length": prompt_len,
            }
            out["generated_events"] = out["generated"]
            out["full_sequences"] = out["full"]
            return out

        # Preallocate sequence tensors
        max_len = prompt_len + max_new_tokens
        for k in ["event","abspos","age","segment","attn_mask"]:
            need = max_len - batch[k].size(1)
            if need > 0:
                if k == "abspos":
                    last = batch[k][:, -1:].to(model_dtype)
                    incr = torch.arange(1, need+1, device=device, dtype=model_dtype)
                    pad_block = (last + incr).expand(B, -1)
                elif k == "age":
                    pad_block = torch.zeros(B, need, dtype=model_dtype, device=device)
                elif k == "segment":
                    pad_block = batch[k][:, -1:].expand(B, need)
                else:
                    pad_block = torch.zeros(B, need, dtype=batch[k].dtype, device=device)
                batch[k] = torch.cat([batch[k], pad_block], dim=1)

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generation_lengths = torch.zeros(B, dtype=torch.long, device=device)

        if profile:
            wall_start = time.perf_counter()
            first_step_ms = None
            decode_ms_acc = 0.0
        else:
            wall_start = first_step_ms = decode_ms_acc = None

        # -------- NO CACHE PATH (unchanged) --------
        if not use_cache:
            current_length = prompt_len
            for step in range(max_new_tokens):
                cur = {k: (v[:, :current_length] if torch.is_tensor(v) and v.ndim==2 else v)
                       for k,v in batch.items()}
                if profile and device.type=="cuda":
                    torch.cuda.synchronize(); t0=time.perf_counter()
                if use_autocast and device.type=="cuda":
                    with torch.autocast(device_type="cuda",
                                        dtype=torch.float16 if getattr(self,"_mixed_precision",False) else torch.float32):
                        hidden = self.forward_generation(cur, use_autocast=False)
                else:
                    hidden = self.forward_generation(cur, use_autocast=False)
                if profile and device.type=="cuda":
                    torch.cuda.synchronize()
                    dt=(time.perf_counter()-t0)*1000
                    if step==0: first_step_ms=dt
                    else: decode_ms_acc += dt
                last_pos = cur["attn_mask"].sum(1).long() - 1
                last_hidden = hidden[torch.arange(B, device=device), last_pos]
                if last_hidden.dtype != self.decoder.weight.dtype:
                    last_hidden = last_hidden.to(self.decoder.weight.dtype)
                logits = self.decoder(last_hidden).float()
                next_tokens = self._sample_tokens(logits, temperature, top_k, top_p, finished)
                if eos_token_id is not None:
                    finished |= (next_tokens == eos_token_id)
                generation_lengths[~finished] += 1
                wp = current_length
                batch["event"][:, wp] = torch.where(finished,
                                                    torch.full_like(next_tokens, pad_token_id),
                                                    next_tokens)
                batch["attn_mask"][:, wp] = (~finished).to(batch["attn_mask"].dtype)
                batch["abspos"][:, wp] = batch["abspos"][:, wp-1] + 1
                batch["age"][:, wp] = batch["age"][:, wp-1]
                batch["segment"][:, wp] = batch["segment"][:, wp-1]
                current_length += 1
                if finished.all(): break

        # -------- CACHE PATH WITH PREALLOCATION --------
        else:
            prompt_batch = {
                "event": batch["event"][:, :prompt_len],
                "abspos": batch["abspos"][:, :prompt_len],
                "age": batch["age"][:, :prompt_len],
                "segment": batch["segment"][:, :prompt_len],
                "attn_mask": batch["attn_mask"][:, :prompt_len],
            }
            x = self.embed_information(prompt_batch)  # (B,T,D)
            caches = []
            max_total_len = prompt_len + max_new_tokens
            # Build prompt & preallocate per layer
            for block in self.transformer.h:
                if preallocate_cache:
                    x, layer_cache = block.build_kv_cache_prealloc(x, prompt_batch["attn_mask"], max_total_len)
                else:
                    x, layer_cache = block.build_kv_cache(x, prompt_batch["attn_mask"])
                caches.append(layer_cache)
            x = self.transformer.ln_f(x)
            last_hidden = x[:, -1:, :]
            current_length = prompt_len

            for step in range(max_new_tokens):
                if profile and device.type=="cuda":
                    torch.cuda.synchronize(); t0=time.perf_counter()

                lh = last_hidden.squeeze(1)
                if lh.dtype != self.decoder.weight.dtype:
                    lh = lh.to(self.decoder.weight.dtype)
                logits = self.decoder(lh).float()
                next_tokens = self._sample_tokens(logits, temperature, top_k, top_p, finished)
                if eos_token_id is not None:
                    finished |= (next_tokens == eos_token_id)
                generation_lengths[~finished] += 1

                wp = current_length
                if wp >= batch["event"].shape[1]: break
                batch["event"][:, wp] = torch.where(finished,
                                                    torch.full_like(next_tokens, pad_token_id),
                                                    next_tokens)
                batch["attn_mask"][:, wp] = (~finished).to(batch["attn_mask"].dtype)
                batch["abspos"][:, wp] = batch["abspos"][:, wp-1] + 1
                batch["age"][:, wp] = batch["age"][:, wp-1]
                batch["segment"][:, wp] = batch["segment"][:, wp-1]

                # New token embedding
                token_embed = ( self.embedding(batch["event"][:, wp:wp+1])
                                + self.t2v_abspos(batch["abspos"][:, wp:wp+1])
                                + self.t2v_age(batch["age"][:, wp:wp+1])
                                + self.segment_embeddings(batch["segment"][:, wp:wp+1]) )

                new_token_hidden = token_embed
                for li, block in enumerate(self.transformer.h):
                    cache_li = caches[li]
                    if preallocate_cache and len(cache_li) == 3:
                        new_token_hidden, caches[li] = block.incremental_forward_prealloc(
                            new_token_hidden, cache_li, position_index=wp
                        )
                    else:
                        new_token_hidden, caches[li] = block.incremental_forward(
                            new_token_hidden, cache_li, position_index=wp
                        )
                new_token_hidden = self.transformer.ln_f(new_token_hidden)
                last_hidden = new_token_hidden

                if profile and device.type=="cuda":
                    torch.cuda.synchronize()
                    dt=(time.perf_counter()-t0)*1000
                    if step==0: first_step_ms=dt
                    else: decode_ms_acc += dt

                current_length += 1
                if finished.all(): break

        # ----- Final packaging -----
        gen_len = int(generation_lengths.max().item())
        final_len = prompt_len + gen_len
        prompt_tokens = batch["event"][:, :prompt_len]
        generated = batch["event"][:, prompt_len:final_len]
        full_seq = batch["event"][:, :final_len]
        out = {
            "prompt": prompt_tokens,
            "generated": generated,
            "full": full_seq,
            "generation_lengths": generation_lengths,
            "finished": finished,
            "prompt_length": prompt_len,
        }
        if return_attention_mask:
            out["generated_attn_mask"] = batch["attn_mask"][:, :final_len]
        out["generated_events"] = out["generated"]
        out["full_sequences"] = out["full"]

        if profile:
            if device.type=="cuda": torch.cuda.synchronize()
            total_ms = (time.perf_counter()-wall_start)*1000.0
            gen_tokens_total = int(generation_lengths.sum().item())
            prompt_tokens_total = int(batch["attn_mask"][:, :prompt_len].sum().item())
            total_tokens = prompt_tokens_total + gen_tokens_total
            out["profile"] = {
                "batch_size": B,
                "prompt_len": prompt_len,
                "gen_tokens": gen_tokens_total,
                "prompt_tokens": prompt_tokens_total,
                "total_tokens": total_tokens,
                "first_step_ms": first_step_ms,
                "decode_ms": decode_ms_acc,
                "total_ms": total_ms,
                "tokens_per_sec_total": total_tokens / (total_ms/1000.0) if total_ms else 0.0,
                "gen_tokens_per_sec": gen_tokens_total / (total_ms/1000.0) if total_ms else 0.0,
                "decode_gen_tokens_per_sec": gen_tokens_total / (decode_ms_acc/1000.0) if decode_ms_acc else 0.0,
                "used_cache": use_cache,
                "preallocate_cache": preallocate_cache,
            }
        return out 
    
    def _repack_batch(self, batch):
        """Convert unpacked Flash Attention format to standard format"""
        from flash_attn.bert_padding import pad_input
        
        B = len(batch['cu_seqlens']) - 1
        T = batch['max_seqlen_in_batch']
        
        repacked = {}
        for key in ['event', 'abspos', 'age', 'segment']:
            if key in batch:
                if key == 'event':
                    padded = pad_input(batch[key].unsqueeze(-1), batch['indices'], B, T)
                    repacked[key] = padded.squeeze(-1)
                else:
                    padded = pad_input(batch[key].unsqueeze(-1), batch['indices'], B, T)
                    repacked[key] = padded.squeeze(-1)
        
        repacked['attn_mask'] = (repacked['event'] != 0).float()
        return repacked
    
    def _prepare_generation_tensors(self, batch, max_length):
        """Pre-allocate tensors for efficient generation"""
        batch_size, current_length = batch['event'].shape
        device = batch['event'].device
        
        # Pre-allocate larger tensors
        new_batch = {}
        for key in ['event', 'segment']:
            new_tensor = torch.zeros(batch_size, max_length, dtype=batch[key].dtype, device=device)
            new_tensor[:, :current_length] = batch[key]
            new_batch[key] = new_tensor
        
        for key in ['abspos', 'age', 'attn_mask']:
            if key in batch:
                new_tensor = torch.zeros(batch_size, max_length, dtype=batch[key].dtype, device=device)
                new_tensor[:, :current_length] = batch[key]
                new_batch[key] = new_tensor
        
        return new_batch
    
    def _get_current_batch(self, batch, current_length):
        """Extract current sequence up to current_length"""
        return {key: value[:, :current_length] for key, value in batch.items()}
    

    def _sample_tokens(self, logits, temperature, top_k, top_p, finished):
        """Apply sampling to logits"""
        batch_size = logits.shape[0]
        device = logits.device
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k = top_k_logits[:, -1:].expand_as(logits)
            logits = torch.where(logits < min_top_k, 
                               torch.full_like(logits, float('-inf')), 
                               logits)
        
        # Apply top-p filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 0] = False  # Keep at least one token
            
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample tokens
        probs = F.softmax(logits, dim=-1)
        
        # Check for NaN or invalid probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            # Use greedy sampling as fallback
            next_tokens = torch.argmax(logits, dim=-1)
        else:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        # Use pad tokens for finished sequences
        pad_token_id = 0  # Default pad token
        next_tokens = torch.where(finished, 
                                torch.full_like(next_tokens, pad_token_id), 
                                next_tokens)
        
        return next_tokens
    
    def _extend_sequences(self, batch, next_tokens, current_length):
        """Efficiently extend sequences with new tokens"""
        batch_size = next_tokens.shape[0]
        device = next_tokens.device
        
        # Add new events
        batch['event'][:, current_length] = next_tokens
        
        # TODO: Improve temporal feature prediction
        # For now, use simple heuristics for temporal features
        if current_length > 0:
            # Use last values + small increment for temporal continuity
            batch['abspos'][:, current_length] = batch['abspos'][:, current_length-1] + 1.0
            batch['age'][:, current_length] = batch['age'][:, current_length-1] + 0.1
            batch['segment'][:, current_length] = batch['segment'][:, current_length-1]
        else:
            # Fallback values
            batch['abspos'][:, current_length] = torch.zeros(batch_size, dtype=torch.float32, device=device)
            batch['age'][:, current_length] = torch.zeros(batch_size, dtype=torch.float32, device=device) 
            batch['segment'][:, current_length] = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Update attention mask
        batch['attn_mask'][:, current_length] = 1.0
        
        return batch
    
    # TODO: Implement KV caching methods
    # def _init_cache(self, batch_size, device):
    #     """Initialize KV cache for all layers"""
    #     cache = []
    #     for _ in range(len(self.transformer.h)):
    #         layer_cache = {
    #             'key': None,
    #             'value': None
    #         }
    #         cache.append(layer_cache)
    #     return cache
    #
    # def _update_cache(self, past_cache, new_cache):
    #     """Update KV cache with new key-value pairs"""
    #     if past_cache is None:
    #         return new_cache
    #     
    #     updated_cache = []
    #     for past_layer, new_layer in zip(past_cache, new_cache):
    #         updated_layer = {
    #             'key': torch.cat([past_layer['key'], new_layer['key']], dim=-2),
    #             'value': torch.cat([past_layer['value'], new_layer['value']], dim=-2)
    #         }
    #         updated_cache.append(updated_layer)
    #     return updated_cache





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
