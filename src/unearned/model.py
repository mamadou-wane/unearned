"""Reference causal language model with owned Transformer modules."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Dimensions and numerical constants for a bias-free, untied decoder.

    All matrices start from N(0, init_std**2), and RMSNorm scales start at one.
    Blocks use pre-norm residual branches with no dropout. Every query head
    rotates its full, even width; adjacent query heads share one KV head.
    Constants use built-in int/float values for primitive checkpoint metadata.
    They must lie in the positive normal fp32 range; rope_theta >= 1
    keeps rotation frequencies bounded by one.
    """

    vocab_size: int
    max_seq_len: int
    d_model: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    d_ff: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_std: float = 0.02

    def __post_init__(self):
        for name in (
            'vocab_size', 'max_seq_len', 'd_model', 'num_layers', 'num_heads',
            'num_kv_heads', 'd_ff',
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f'{name} must be an integer')
            if value <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('rms_norm_eps', 'rope_theta', 'init_std'):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise TypeError(f'{name} must be a built-in int or float')
            if not torch.finfo(torch.float32).tiny <= value <= torch.finfo(torch.float32).max:
                raise ValueError(f'{name} must be in the positive normal fp32 range')
        if self.rope_theta < 1:
            raise ValueError('rope_theta must be at least one')
        if self.d_model % self.num_heads:
            raise ValueError('d_model must be divisible by num_heads')
        if self.num_heads % self.num_kv_heads:
            raise ValueError('num_heads must be divisible by num_kv_heads')
        if self.head_dim % 2:
            raise ValueError('head_dim must be even for RoPE')

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


class Linear(nn.Module):
    """Bias-free projection from [..., in_features] to [..., out_features]."""

    def __init__(self, in_features: int, out_features: int, init_std: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=init_std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """Gather int32/int64 token ids into vectors, retaining leading dimensions."""

    def __init__(self, vocab_size: int, d_model: int, init_std: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, std=init_std)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight.index_select(0, token_ids.reshape(-1)).reshape(
            *token_ids.shape, self.weight.shape[1]
        )


class RMSNorm(nn.Module):
    """Scale the last axis by its root mean square with epsilon inside the root."""

    def __init__(self, d_model: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)) * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotate adjacent coordinates of [batch, heads, sequence, head_dim].

    Position p rotates pair i by p * theta**(-2*i/head_dim). Positions start
    at zero on each forward; values are not rotated.
    """

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x: Tensor) -> Tensor:
        frequency = self.theta ** (
            -torch.arange(0, self.head_dim, 2, device=x.device, dtype=x.dtype)
            / self.head_dim
        )
        positions = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)
        angles = positions[:, None] * frequency[None, :]
        cosine, sine = angles.cos(), angles.sin()
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)


class CausalSelfAttention(nn.Module):
    """Inclusive causal GQA from [batch, sequence, d_model] to the same shape."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        kv_width = config.num_kv_heads * config.head_dim
        self.q_proj = Linear(config.d_model, config.d_model, config.init_std)
        self.k_proj = Linear(config.d_model, kv_width, config.init_std)
        self.v_proj = Linear(config.d_model, kv_width, config.init_std)
        self.out_proj = Linear(config.d_model, config.d_model, config.init_std)
        self.rope = RotaryPositionEmbedding(config.head_dim, config.rope_theta)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, d_model = x.shape
        q = self.q_proj(x).reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        # [B, KV heads, queries per KV head, T, D]; broadcast each KV head
        # across its contiguous query group, including shared-KV gradients.
        q = q.reshape(
            batch, self.num_kv_heads, self.num_heads // self.num_kv_heads,
            length, self.head_dim,
        )
        scores = (q @ k.unsqueeze(2).transpose(-2, -1)) / math.sqrt(self.head_dim)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        probabilities = scores.masked_fill(future, -torch.inf).softmax(dim=-1)
        attended = (probabilities @ v.unsqueeze(2)).reshape(
            batch, self.num_heads, length, self.head_dim
        )
        return self.out_proj(attended.transpose(1, 2).reshape(batch, length, d_model))


class SwiGLU(nn.Module):
    """Down-project SiLU(gate(x)) * up(x), with hidden width d_ff."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = Linear(config.d_model, config.d_ff, config.init_std)
        self.up_proj = Linear(config.d_model, config.d_ff, config.init_std)
        self.down_proj = Linear(config.d_ff, config.d_model, config.init_std)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        return self.down_proj((gate * torch.sigmoid(gate)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TransformerLM(nn.Module):
    """Map token ids [B, T] to raw logits [B, T, vocab_size], without mutation.

    Inputs must be nonempty int32/int64 tensors on the model's device, with
    ids in [0, vocab_size) and T <= max_seq_len. There is no padding mask:
    each supplied token participates in its row's inclusive causal context.
    The checked numerical reference is CPU fp32; positions restart at zero.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size, config.d_model, config.init_std)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.output = Linear(config.d_model, config.vocab_size, config.init_std)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError('token_ids must have int32 or int64 dtype')
        if token_ids.ndim != 2:
            raise ValueError('token_ids must have shape [batch, sequence]')
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError('token_ids must have nonempty batch and sequence axes')
        if token_ids.shape[1] > self.config.max_seq_len:
            raise ValueError('sequence exceeds max_seq_len')
        if (token_ids < 0).any() or (token_ids >= self.config.vocab_size).any():
            raise ValueError('token_ids must be in [0, vocab_size)')
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.output(self.final_norm(x))
