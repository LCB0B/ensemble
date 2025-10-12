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

        # Ensure q and k have the same dtype as the rotary embedding cached tensors
        if hasattr(self.rotary, '_cos_cached') and self.rotary._cos_cached is not None:
            cos_dtype = self.rotary._cos_cached.dtype
            if q.dtype != cos_dtype:
                q = q.to(cos_dtype)
            if k.dtype != cos_dtype:
                k = k.to(cos_dtype)

        q, k = self.rotary(q, k, cu_seqlens, max_seqlen)

        # Ensure v also has the same dtype for FlashAttention
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        qkv = torch.stack((q, k, v), dim=1)
        y = self.self_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        y = y.view(total, C)

        # Ensure y has the same dtype as the output projection weights
        if y.dtype != self.out_proj.weight.dtype:
            y = y.to(self.out_proj.weight.dtype)

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

    # --- Flash Attention KV cache (NEW) ---
    @torch.no_grad()
    def build_flash_kv_cache(self, x, cu_seqlens, max_seqlen, max_total_len):
        """
        Build KV cache using Flash Attention for prefill stage.

        Args:
            x: Unpacked hidden states (total_tokens, D)
            cu_seqlens: Cumulative sequence lengths tensor
            max_seqlen: Maximum sequence length in batch
            max_total_len: Maximum total length (prompt + generation)

        Returns:
            x: Updated hidden states (total_tokens, D)
            cache: Tuple of (K_alloc, V_alloc, current_len, batch_info)
        """
        from flash_attn.bert_padding import pad_input

        total, D = x.shape
        B = len(cu_seqlens) - 1

        # Layer norm
        h = self.ln_1(x)

        # Compute Q, K, V using Flash Attention's Wqkv
        qkv = self.attn.Wqkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for Flash Attention (total_tokens, num_heads, head_dim)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)

        # Ensure dtype consistency for rotary
        if hasattr(self.attn.rotary, '_cos_cached') and self.attn.rotary._cos_cached is not None:
            cos_dtype = self.attn.rotary._cos_cached.dtype
            if q.dtype != cos_dtype:
                q = q.to(cos_dtype)
            if k.dtype != cos_dtype:
                k = k.to(cos_dtype)

        # Apply rotary embeddings using Flash Attention's rotary
        q, k = self.attn.rotary(q, k, cu_seqlens, max_seqlen)

        # Ensure v has same dtype
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # Run Flash Attention
        qkv_stack = torch.stack((q, k, v), dim=1)
        attn_out = self.attn.self_attn(qkv_stack, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        attn_out = attn_out.view(total, D)

        # Ensure dtype consistency for output projection
        if attn_out.dtype != self.attn.out_proj.weight.dtype:
            attn_out = attn_out.to(self.attn.out_proj.weight.dtype)

        # Output projection and residual
        y = self.attn.resid_dropout(self.attn.out_proj(attn_out))
        x = x + y

        # MLP
        x = x + self.mlp(self.ln_2(x))

        # Convert K, V from unpacked to padded format for caching
        # Pad k, v to (B, T, H, hd)
        k_padded = pad_input(k.unsqueeze(-1),
                            torch.arange(total, device=x.device),
                            B, max_seqlen).squeeze(-1)  # (B, T, H, hd)
        v_padded = pad_input(v.unsqueeze(-1),
                            torch.arange(total, device=x.device),
                            B, max_seqlen).squeeze(-1)  # (B, T, H, hd)

        # Transpose to (B, H, T, hd) for SDPA format
        k_padded = k_padded.permute(0, 2, 1, 3)
        v_padded = v_padded.permute(0, 2, 1, 3)

        # Preallocate cache
        K_alloc = torch.zeros(B, self.num_heads, max_total_len, self.head_dim,
                             dtype=k_padded.dtype, device=k_padded.device)
        V_alloc = torch.zeros(B, self.num_heads, max_total_len, self.head_dim,
                             dtype=v_padded.dtype, device=v_padded.device)

        # Copy prompt K, V into cache
        K_alloc[:, :, :max_seqlen, :] = k_padded
        V_alloc[:, :, :max_seqlen, :] = v_padded

        # Store batch info for later use
        batch_info = {
            'B': B,
            'prompt_len': max_seqlen
        }

        return x, (K_alloc, V_alloc, max_seqlen, batch_info)

    @torch.no_grad()
    def incremental_forward_flash(self, x_token, cache_tuple, position_index):
        """
        Incremental forward with Flash Attention-compatible KV cache.
        Uses SDPA but with exact RoPE/dtype matching to Flash Attention training.

        Args:
            x_token: New token embedding (B, 1, D)
            cache_tuple: (K_alloc, V_alloc, current_len, batch_info)
            position_index: Absolute position of new token

        Returns:
            x_token: Updated hidden state (B, 1, D)
            updated_cache: Updated cache tuple
        """
        K_alloc, V_alloc, cur_len, batch_info = cache_tuple
        B, H, cap, hd = K_alloc.shape
        D = H * hd

        assert position_index == cur_len and cur_len < cap, \
            f"Position mismatch: position_index={position_index}, cur_len={cur_len}, cap={cap}"

        # Layer norm
        h = self.ln_1(x_token)

        # Compute Q, K, V for new token
        qkv = self.attn.Wqkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, 1, H, hd) then transpose to (B, H, 1, hd)
        q = q.view(B, 1, H, hd).permute(0, 2, 1, 3)
        k = k.view(B, 1, H, hd).permute(0, 2, 1, 3)
        v = v.view(B, 1, H, hd).permute(0, 2, 1, 3)

        # Ensure dtype consistency BEFORE RoPE - match cached K, V dtype
        cache_dtype = K_alloc.dtype
        if q.dtype != cache_dtype:
            q = q.to(cache_dtype)
        if k.dtype != cache_dtype:
            k = k.to(cache_dtype)
        if v.dtype != cache_dtype:
            v = v.to(cache_dtype)

        # Apply rotary using the SAME method as Flash Attention prefill
        # Use the simpler _apply_rope for single position
        pos = torch.tensor([position_index], device=x_token.device)
        q, k = self._apply_rope(q, k, pos)

        # Write new K, V to cache
        K_alloc[:, :, cur_len, :] = k.squeeze(2)
        V_alloc[:, :, cur_len, :] = v.squeeze(2)

        # Get full K, V for attention
        k_full = K_alloc[:, :, :cur_len+1, :]
        v_full = V_alloc[:, :, :cur_len+1, :]

        # Ensure all inputs to SDPA have same dtype
        if q.dtype != k_full.dtype:
            q = q.to(k_full.dtype)
        if v.dtype != k_full.dtype:
            v = v.to(k_full.dtype)

        # Run SDPA with same settings as training
        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)

        # Reshape back to (B, 1, D)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, 1, D)

        # Ensure dtype consistency for output projection
        if attn_out.dtype != self.attn.out_proj.weight.dtype:
            attn_out = attn_out.to(self.attn.out_proj.weight.dtype)

        # Output projection and residual
        y = self.attn.resid_dropout(self.attn.out_proj(attn_out))
        x_token = x_token + y

        # MLP
        x_token = x_token + self.mlp(self.ln_2(x_token))

        return x_token, (K_alloc, V_alloc, cur_len + 1, batch_info)

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