""" The backbone Blocks for the nano encoder"""

import os
import torch
import torch.nn as nn
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb
from flash_attn.modules.mha import FlashSelfAttention
from flash_attn.modules.mlp import Mlp

# See https://github.com/pytorch/torchtune/issues/1185
# 'Reduce VRAM usage by reducing fragmentation'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class FlashRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim, max_seqlen, device=torch.device("cuda")):
        super().__init__(dim, device=device)
        self._update_cos_sin_cache(max_seqlen, device=device, dtype=torch.float16)

    def forward(self, q, k, cu_seqlens, max_seqlen):
        """Aplies 'apply_rotary_emb` on q and k"""
        q = apply_rotary_emb(
            q,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        k = apply_rotary_emb(
            k,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return q, k


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention using FlashAttention"""

    def __init__(self, num_heads, d_model, dropout, bias, max_seq_len, causal=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads
        # key, query, value projections for all heads, but in a batch
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        # output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = FlashRotaryEmbedding(self.head_dim, max_seq_len)
        self.self_attn = FlashSelfAttention(causal=causal)

    def forward(self, x, cu_seqlens, max_seqlen):
        total, C = x.shape
        # Standard
        q, k, v = self.Wqkv(x).chunk(3, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)

        q, k = self.rotary(q, k, cu_seqlens, max_seqlen)
        qkv = torch.stack((q, k, v), dim=1)

        y = self.self_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        y = y.view(total, C)

        # output projection
        y = self.resid_dropout(self.out_proj(y))

        return y


class SwiGLU(nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=bias)
        self.l2 = nn.Linear(dim, dim, bias=bias)
        self.l3 = nn.Linear(dim, dim, bias=bias)
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.l3(self.swish(self.l2(x)) * self.l1(x))


class MLP(nn.Module):
    """Standard MLP with FlashAttention MLP(Linear, GELU, Linear) and dropout"""

    def __init__(self, d_model, bias, dropout):
        super().__init__()
        self.mlp = Mlp(
            d_model,
            bias1=bias,
            bias2=bias,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Standard Block with LayerNorm, MultiHeadAttention and MLP"""

    def __init__(
        self,
        d_model,
        num_heads,
        dropout,
        bias,
        max_seq_len,
        causal=True,
    ):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(d_model, bias=bias)
        self.attn = MultiHeadAttention(
            num_heads, d_model, dropout, bias, max_seq_len, causal
        )
        self.ln_2 = torch.nn.LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model, bias, dropout)

    def forward(self, x, batch):
        x = x + self.attn(
            self.ln_1(x),
            cu_seqlens=batch["cu_seqlens"],
            max_seqlen=batch["max_seqlen_in_batch"],
        ).to(
            x.dtype
        )  # Manual upcasting due to issues (unpad + compile)
        x = x + self.mlp(self.ln_2(x)).to(
            x.dtype
        )  # Manual upcasting due to issues (unpad + compile)
        return x


class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, dim_model, num_follow_up=3):
        super().__init__()
        self.base_hazard_fc = nn.Linear(dim_model, 1)
        self.hazard_fc = nn.Linear(dim_model, num_follow_up)
        self.relu = nn.ReLU()

    def hazards(self, x: torch.Tensor) -> torch.Tensor:
        raw_hazard = self.hazard_fc(x)
        return self.relu(raw_hazard)  # Risk cannot be negative

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_hazards = self.base_hazard_fc(x)
        hazards = self.hazards(x)
        cum_prob = hazards.cumsum(dim=-1) + base_hazards
        return cum_prob
