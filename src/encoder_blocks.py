""" The backbone Blocks for the nano encoder"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb
from flash_attn.modules.mha import FlashSelfAttention
from flash_attn.modules.mlp import Mlp

# See https://github.com/pytorch/torchtune/issues/1185
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class FlashRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim, max_seqlen, device=torch.device("cuda")):
        super().__init__(dim, device=device)
        self._update_cos_sin_cache(max_seqlen, device=device, dtype=torch.float16)

    def forward(self, q, k, cu_seqlens, max_seqlen):
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
    """FlashAttention path for training / full prompt pass.
       A lightweight SDPA path is used during incremental KV generation in Block.incremental_forward."""
    def __init__(self, num_heads, d_model, dropout, bias, max_seq_len, causal=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // self.num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = FlashRotaryEmbedding(self.head_dim, max_seq_len)
        self.self_attn = FlashSelfAttention(causal=causal)

    def forward(self, x, cu_seqlens, max_seqlen):
        total, C = x.shape
        q, k, v = self.Wqkv(x).chunk(3, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)
        q, k = self.rotary(q, k, cu_seqlens, max_seqlen)
        qkv = torch.stack((q, k, v), dim=1)
        y = self.self_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        y = y.view(total, C)
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
    def __init__(self, d_model, bias, dropout):
        super().__init__()
        self.mlp = Mlp(d_model, bias1=bias, bias2=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer Block with:
      - FlashAttention path for packed training
      - Exact rotary
      - Two cache modes (cat & preallocated)
    """
    def __init__(self, d_model, num_heads, dropout, bias, max_seq_len, causal=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, bias=bias)
        self.attn = MultiHeadAttention(num_heads, d_model, dropout, bias, max_seq_len, causal)
        self.ln_2 = nn.LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model, bias, dropout)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)

    def forward(self, x, batch):
        x = x + self.attn(
            self.ln_1(x),
            cu_seqlens=batch["cu_seqlens"],
            max_seqlen=batch["max_seqlen_in_batch"],
        ).to(x.dtype)
        x = x + self.mlp(self.ln_2(x)).to(x.dtype)
        return x

    # --- Rotary helpers ---
    def _rope_cos_sin(self, positions: torch.Tensor, dtype, device):
        inv_freq = self.rotary_inv_freq.to(device=device)
        freqs = torch.einsum("t,f->tf", positions.float(), inv_freq)
        emb = torch.stack((freqs, freqs), dim=-1).reshape(freqs.size(0), -1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def _apply_rope(self, q, k, positions):
        # q,k: (B,H,T,hd)
        cos, sin = self._rope_cos_sin(positions, q.dtype, q.device)
        cos = cos.view(1, 1, -1, self.head_dim)
        sin = sin.view(1, 1, -1, self.head_dim)
        def rotate_half(t):
            t1 = t[..., : self.head_dim // 2]
            t2 = t[..., self.head_dim // 2 :]
            return torch.cat((-t2, t1), dim=-1)
        return (q * cos + rotate_half(q) * sin,
                k * cos + rotate_half(k) * sin)

    # --- Original (cat) cache ---
    @torch.no_grad()
    def build_kv_cache(self, x, attn_mask):
        B, T, D = x.shape
        h = self.ln_1(x)
        qkv = self.attn.Wqkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        pos = torch.arange(T, device=x.device)
        q, k = self._apply_rope(q, k, pos)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn_out = attn_out.permute(0,2,1,3).contiguous().view(B,T,D)
        y = self.attn.out_proj(attn_out)
        x = x + self.attn.resid_dropout(y)
        x = x + self.mlp(self.ln_2(x))
        return x, (k, v)

    @torch.no_grad()
    def incremental_forward(self, x_token, cache, position_index):
        k_cache, v_cache = cache
        B, H, L, hd = k_cache.shape
        h = self.ln_1(x_token)
        qkv = self.attn.Wqkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B,1,H,hd).permute(0,2,1,3)
        k = k.view(B,1,H,hd).permute(0,2,1,3)
        v = v.view(B,1,H,hd).permute(0,2,1,3)
        pos = torch.tensor([position_index], device=x_token.device)
        q, k = self._apply_rope(q, k, pos)
        k_new = torch.cat([k_cache, k], dim=2)
        v_new = torch.cat([v_cache, v], dim=2)
        attn_out = F.scaled_dot_product_attention(q, k_new, v_new, is_causal=True)
        D = H*hd
        attn_out = attn_out.permute(0,2,1,3).contiguous().view(B,1,D)
        y = self.attn.resid_dropout(self.attn.out_proj(attn_out))
        x_token = x_token + y
        x_token = x_token + self.mlp(self.ln_2(x_token))
        return x_token, (k_new, v_new)

    # --- Preallocated cache ---
    @torch.no_grad()
    def build_kv_cache_prealloc(self, x, attn_mask, max_total_len):
        B, T, D = x.shape
        h = self.ln_1(x)
        qkv = self.attn.Wqkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B,T,self.num_heads,self.head_dim).permute(0,2,1,3)
        k = k.view(B,T,self.num_heads,self.head_dim).permute(0,2,1,3)
        v = v.view(B,T,self.num_heads,self.head_dim).permute(0,2,1,3)
        pos = torch.arange(T, device=x.device)
        q, k = self._apply_rope(q, k, pos)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn_out = attn_out.permute(0,2,1,3).contiguous().view(B,T,D)
        y = self.attn.resid_dropout(self.attn.out_proj(attn_out))
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        # Preallocate
        K_alloc = torch.empty(B, self.num_heads, max_total_len, self.head_dim,
                              dtype=k.dtype, device=k.device)
        V_alloc = torch.empty_like(K_alloc)
        K_alloc[:,:, :T] = k
        V_alloc[:,:, :T] = v
        return x, (K_alloc, V_alloc, T)

    @torch.no_grad()
    def incremental_forward_prealloc(self, x_token, cache_tuple, position_index):
        K_alloc, V_alloc, cur_len = cache_tuple
        B,H,cap,hd = K_alloc.shape
        assert position_index == cur_len and cur_len < cap
        h = self.ln_1(x_token)
        qkv = self.attn.Wqkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B,1,H,hd).permute(0,2,1,3)
        k = k.view(B,1,H,hd).permute(0,2,1,3)
        v = v.view(B,1,H,hd).permute(0,2,1,3)
        pos = torch.tensor([position_index], device=x_token.device)
        q, k = self._apply_rope(q, k, pos)
        # write
        K_alloc[:,:,cur_len] = k.squeeze(2)
        V_alloc[:,:,cur_len] = v.squeeze(2)
        k_slice = K_alloc[:,:,:cur_len+1]
        v_slice = V_alloc[:,:,:cur_len+1]
        attn_out = F.scaled_dot_product_attention(q, k_slice, v_slice, is_causal=True)
        D = H*hd
        attn_out = attn_out.permute(0,2,1,3).contiguous().view(B,1,D)
        y = self.attn.resid_dropout(self.attn.out_proj(attn_out))
        x_token = x_token + y
        x_token = x_token + self.mlp(self.ln_2(x_token))
        return x_token, (K_alloc, V_alloc, cur_len+1)

class CumulativeProbabilityLayer(nn.Module):
    def __init__(self, dim_model, num_follow_up=3):
        super().__init__()
        self.base_hazard_fc = nn.Linear(dim_model, 1)
        self.hazard_fc = nn.Linear(dim_model, num_follow_up)
        self.relu = nn.ReLU()

    def hazards(self, x: torch.Tensor) -> torch.Tensor:
        raw_hazard = self.hazard_fc(x)
        return self.relu(raw_hazard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_hazards = self.base_hazard_fc(x)
        hazards = self.hazards(x)
        cum_prob = hazards.cumsum(dim=-1) + base_hazards
        return cum_prob