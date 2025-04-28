""" The backbone Blocks for the nano encoder"""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

# torch._dynamo.config.optimize_ddp = False


class SelfAttention(nn.Module):
    """SelfAttention using FlashAttention"""

    def __init__(self, num_heads, d_model, dropout, bias, compiled: bool):
        super().__init__()
        assert d_model % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.flex_attention = (
            flex_attention if compiled else torch.compile(flex_attention, dynamic=False)
        )

    def forward(self, x, attn_mask, sinusoidal_pos=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (d_model)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Apply rope
        if sinusoidal_pos is not None:
            # q, k = self.apply_rope(sinusoidal_pos, q, k)
            q = sinusoidal_pos(q)
            k = sinusoidal_pos(k)
        v = v.to(q.dtype)

        y = self.flex_attention(q, k, v, block_mask=attn_mask)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    @staticmethod
    def apply_rope(sinusoidal_pos, q, k):
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)

        # sin/cos [θ0, θ1, θ2, ..., θd/2-1] -> sin/cos_pos [θ0, θ0, θ1, θ1, ..., θd/2-1, θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

        # rotate_half_query [-q1, q0, -q3, q2, ...., -qd-1, qd-2]
        rotate_half_query = torch.stack(
            [-q[..., 1::2], q[..., ::2]], dim=-1
        ).reshape_as(q)
        query = q * cos_pos + rotate_half_query * sin_pos

        # rotate_half_key [-k1, k0, -k3, k2, ..., -kd-1, kd-2]
        rotate_half_key = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(
            k
        )
        key = k * cos_pos + rotate_half_key * sin_pos

        return query, key


# from https://pytorch.org/torchtune/0.2/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        max_seq_len: int,
        dim: int,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class SinusoidalPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        n_pos, dim = out.shape
        positions = torch.arange(n_pos).unsqueeze(1) / torch.float_power(
            10_000, 2 * (torch.arange(dim) // 2) / dim
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, :sentinel] = torch.sin(positions[:, 0::2])
        out[:, sentinel:] = torch.cos(positions[:, 1::2])
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size) -> torch.Tensor:
        bs, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            0,
            seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


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
    """Standard MLP with Linear, GELU, Linear and dropout"""

    def __init__(self, d_model, bias, dropout, dim_feedforward, swiglu):
        super().__init__()
        self.c_fc = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.act_fn = SwiGLU(dim_feedforward, bias=bias) if swiglu else nn.GELU()
        self.c_proj = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act_fn(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Standard Block with LayerNorm, SelfAttention and MLP"""

    def __init__(
        self,
        d_model,
        num_heads,
        dropout,
        bias,
        dim_feedforward,
        compiled,
        swiglu,
    ):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(d_model, bias=bias)
        self.attn = SelfAttention(num_heads, d_model, dropout, bias, compiled)
        self.ln_2 = torch.nn.LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model, bias, dropout, dim_feedforward, swiglu=swiglu)

    def forward(self, x, attn_mask, sinusoidal_pos=None):
        x = x + self.attn(
            self.ln_1(x), attn_mask=attn_mask, sinusoidal_pos=sinusoidal_pos
        )
        x = x + self.mlp(self.ln_2(x))
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
