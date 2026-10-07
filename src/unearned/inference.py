"""CPU fp32 inference with explicit, caller-owned per-layer KV state."""

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from unearned.model import ModelConfig, TransformerLM


@dataclass(frozen=True, eq=False)
class LayerKV:
    """Post-RoPE keys and unrotated values, both [B, Hkv, cached length, Dhead]."""

    key: Tensor
    value: Tensor


@dataclass(frozen=True, eq=False)
class KVCache:
    """Read-only state for equal-length rows on one unchanged model instance.

    Append returns independent storage, so an old cache can branch into separate
    continuations. The frozen structure does not make tensors immutable: ordinary
    in-place tensor or parameter changes invalidate the cache through version
    counters and retained storage identities. Parameter replacement, loading
    weights, dtype round trips and config changes invalidate it. Start a fresh
    prefill after changing the model. Retained storage references keep replaced
    allocations alive until this cache is released.

    Use the owned TransformerLM structure without changing modules, their numeric
    attributes or forward behavior while a cache exists. Concurrent mutation,
    .data/external-storage writes, forged metadata and cache serialization are
    unsupported. This is an ownership contract, not a security boundary.
    """

    layers: tuple[LayerKV, ...]
    length: int
    batch_size: int
    _model: TransformerLM = field(repr=False)
    _config: ModelConfig = field(repr=False)
    _weights: tuple = field(repr=False)
    _tensors: tuple = field(repr=False)


def _tensor_state(tensors) -> tuple:
    # Module dtype conversions may replace storage without advancing _version.
    # Retain each storage so its address cannot be recycled while the cache lives.
    return tuple((tensor, tensor._version, tensor.untyped_storage()) for tensor in tensors)


def _same_state(tensors, saved) -> bool:
    return len(tensors) == len(saved) and all(
        tensor is original and tensor._version == version
        and tensor.untyped_storage().data_ptr() == storage.data_ptr()
        for tensor, (original, version, storage) in zip(tensors, saved)
    )


def _validate_model(model):
    if type(model) is not TransformerLM:
        raise TypeError('cached inference requires the owned TransformerLM')
    if any(module.training for module in model.modules()):
        raise ValueError('cached inference requires eval mode')
    if torch.is_autocast_enabled('cpu'):
        raise ValueError('CPU autocast is unsupported by fp32 cached inference')
    parameters = tuple(model.parameters())
    if any(p.device.type != 'cpu' or p.dtype != torch.float32
           or p.layout != torch.strided or p.is_inference() for p in parameters):
        raise ValueError('model parameters must be ordinary CPU fp32 tensors')
    if len(model.blocks) != model.config.num_layers:
        raise ValueError('model layer count does not match its configuration')
    return parameters


def _validate_cache(model, cache, parameters):
    if type(cache) is not KVCache:
        raise TypeError('cache must be a KVCache returned by this API')
    if cache._model is not model or cache._config != model.config:
        raise ValueError('cache model or configuration does not match')
    if not _same_state(parameters, cache._weights):
        raise ValueError('model weights changed after cache creation')
    if type(cache.length) is not int or cache.length <= 0:
        raise ValueError('cache length must be a positive integer')
    if type(cache.batch_size) is not int or cache.batch_size <= 0:
        raise ValueError('cache batch size must be a positive integer')
    if type(cache.layers) is not tuple or len(cache.layers) != model.config.num_layers:
        raise ValueError('cache must contain one layer entry per model layer')
    shape = (cache.batch_size, model.config.num_kv_heads, cache.length, model.config.head_dim)
    tensors = []
    for layer in cache.layers:
        if type(layer) is not LayerKV:
            raise TypeError('cache layers must contain LayerKV entries')
        for tensor in (layer.key, layer.value):
            if not isinstance(tensor, Tensor):
                raise TypeError('cached keys and values must be tensors')
            if (tensor.device.type != 'cpu' or tensor.dtype != torch.float32
                    or tensor.layout != torch.strided or tensor.shape != shape
                    or tensor.requires_grad or tensor.is_inference()):
                raise ValueError('cache tensors must match CPU fp32 layer shape without autograd')
            tensors.append(tensor)
    if not _same_state(tensors, cache._tensors):
        raise ValueError('cache tensors changed after creation')
    if cache.length > model.config.max_seq_len:
        raise ValueError('cached context exceeds max_seq_len')


def validate_cache(model: TransformerLM, cache: KVCache) -> None:
    """Check cache ownership and model state without appending or executing it.

    A sampler must call this before using saved logits, even when its next
    token ends the sequence and therefore never reaches decode.
    """
    _validate_cache(model, cache, _validate_model(model))


def _validate(model, token_ids, cache) -> tuple:
    parameters = _validate_model(model)
    if not isinstance(token_ids, Tensor):
        raise TypeError('token_ids must be a tensor')
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError('token_ids must have int32 or int64 dtype')
    if token_ids.device.type != 'cpu' or token_ids.layout != torch.strided:
        raise ValueError('token_ids must be a dense CPU tensor')
    if token_ids.ndim != 2 or any(size == 0 for size in token_ids.shape):
        raise ValueError('token_ids must have nonempty shape [batch, sequence]')
    if (token_ids < 0).any() or (token_ids >= model.config.vocab_size).any():
        raise ValueError('token_ids must be in [0, vocab_size)')
    past = 0
    if cache is not None:
        _validate_cache(model, cache, parameters)
        if cache.batch_size != token_ids.shape[0]:
            raise ValueError('cache batch size must match token_ids')
        past = cache.length
    if past + token_ids.shape[1] > model.config.max_seq_len:
        raise ValueError('cached context exceeds max_seq_len')
    return _tensor_state(parameters)


def _attention(attention, x: Tensor, previous: LayerKV | None, past: int):
    batch, length, width = x.shape
    q = attention.q_proj(x).reshape(
        batch, length, attention.num_heads, attention.head_dim,
    ).transpose(1, 2)
    k = attention.k_proj(x).reshape(
        batch, length, attention.num_kv_heads, attention.head_dim,
    ).transpose(1, 2)
    v = attention.v_proj(x).reshape(
        batch, length, attention.num_kv_heads, attention.head_dim,
    ).transpose(1, 2)
    q = attention.rope(q, position_offset=past)
    k = attention.rope(k, position_offset=past)
    if previous is None:
        k, v = k.clone(memory_format=torch.contiguous_format), v.clone(memory_format=torch.contiguous_format)
    else:
        k, v = torch.cat((previous.key, k), dim=2), torch.cat((previous.value, v), dim=2)
    q = q.reshape(batch, attention.num_kv_heads, attention.num_heads // attention.num_kv_heads,
                  length, attention.head_dim)
    scores = (q @ k.unsqueeze(2).transpose(-2, -1)) / math.sqrt(attention.head_dim)
    # A suffix query at local row i sees keys through absolute position past+i.
    key_positions = torch.arange(past + length, device=x.device)
    query_positions = past + torch.arange(length, device=x.device)
    future = key_positions[None, :] > query_positions[:, None]
    probabilities = scores.masked_fill(future, -torch.inf).softmax(dim=-1)
    attended = (probabilities @ v.unsqueeze(2)).reshape(batch, attention.num_heads, length, attention.head_dim)
    result = attention.out_proj(attended.transpose(1, 2).reshape(batch, length, width))
    return result, LayerKV(k, v)


@torch.inference_mode(False)
@torch.no_grad()
def prefill(model: TransformerLM, token_ids: Tensor, cache: KVCache | None = None) -> tuple[Tensor, KVCache]:
    """Append a nonempty prompt chunk; return logits [B,T,V] and a new cache.

    None starts at position zero. Rows share a length; no padding or BOS is
    inserted. Requires an eval-mode CPU fp32 model with autocast disabled.
    All incoming state is checked before model work. Failure leaves the input
    cache untouched, and success retains no autograd graph or old KV storage.
    Existing parameter gradients and RNG state are not changed.
    """
    weights = _validate(model, token_ids, cache)
    past = 0 if cache is None else cache.length
    x = model.embedding(token_ids)
    layers = []
    for index, block in enumerate(model.blocks):
        previous = None if cache is None else cache.layers[index]
        attended, layer = _attention(block.attention, block.attention_norm(x), previous, past)
        x = x + attended
        x = x + block.ffn(block.ffn_norm(x))
        layers.append(layer)
    logits = model.output(model.final_norm(x))
    if not torch.isfinite(logits).all():
        raise ValueError('cached inference produced nonfinite logits')
    state = KVCache(tuple(layers), past + token_ids.shape[1], token_ids.shape[0],
                    model, model.config, weights,
                    _tensor_state(t for layer in layers for t in (layer.key, layer.value)))
    return logits, state


def decode(model: TransformerLM, token_ids: Tensor, cache: KVCache) -> tuple[Tensor, KVCache]:
    """Append exactly one token per existing row; return [B,1,V] logits and new state."""
    if cache is None:
        raise ValueError('decode requires an existing cache')
    if not isinstance(token_ids, Tensor) or token_ids.ndim != 2 or token_ids.shape[1] != 1:
        raise ValueError('decode requires token_ids with shape [batch, 1]')
    return prefill(model, token_ids, cache)
