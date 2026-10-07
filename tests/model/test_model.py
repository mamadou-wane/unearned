"""Independent equations and CPU references for the causal Transformer."""

import json
import math
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
import torch.nn.functional as F

from unearned.model import (
    CausalSelfAttention,
    Embedding,
    Linear,
    ModelConfig,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU,
    TransformerLM,
)


# Different reduction orders in the fp32 reference may differ by a few ulps.
FP32 = dict(rtol=1e-5, atol=1e-6)
FP64 = dict(rtol=1e-12, atol=1e-12)


@pytest.fixture(autouse=True)
def reproducible_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3407)
        yield


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=19,
        max_seq_len=7,
        d_model=16,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=23,
    )


def reference_rope(x, theta):
    """Rotate with explicit 2x2 matrices, one position and pair at a time."""
    positions = []
    width = x.shape[-1]
    for position in range(x.shape[-2]):
        pairs = []
        for coordinate in range(0, width, 2):
            angle = position * theta ** (-coordinate / width)
            rotation = x.new_tensor(
                [[math.cos(angle), -math.sin(angle)],
                 [math.sin(angle), math.cos(angle)]]
            )
            pairs.append(x[..., position, coordinate:coordinate + 2] @ rotation.T)
        positions.append(torch.cat(pairs, dim=-1))
    return torch.stack(positions, dim=-2)


def reference_attention(x, attention, config):
    batch, length, _ = x.shape
    width = config.d_model // config.num_heads
    q = F.linear(x, attention.q_proj.weight).reshape(
        batch, length, config.num_heads, width
    ).transpose(1, 2)
    k = F.linear(x, attention.k_proj.weight).reshape(
        batch, length, config.num_kv_heads, width
    ).transpose(1, 2)
    v = F.linear(x, attention.v_proj.weight).reshape(
        batch, length, config.num_kv_heads, width
    ).transpose(1, 2)
    q = reference_rope(q, config.rope_theta)
    k = reference_rope(k, config.rope_theta)
    repeats = config.num_heads // config.num_kv_heads
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return F.linear(
        attended.transpose(1, 2).reshape(batch, length, config.d_model),
        attention.out_proj.weight,
    )


def reference_norm(x, norm):
    return x / torch.sqrt(x.square().sum(dim=-1, keepdim=True) / x.shape[-1]
                          + norm.eps) * norm.weight


def reference_swiglu(x, layer):
    gate = F.linear(x, layer.gate_proj.weight)
    value = F.linear(x, layer.up_proj.weight)
    return F.linear(F.silu(gate) * value, layer.down_proj.weight)


def reference_model(token_ids, model):
    x = F.embedding(token_ids, model.embedding.weight)
    for block in model.blocks:
        x = x + reference_attention(
            reference_norm(x, block.attention_norm), block.attention, model.config
        )
        x = x + reference_swiglu(reference_norm(x, block.ffn_norm), block.ffn)
    return F.linear(reference_norm(x, model.final_norm), model.output.weight)


def test_config_is_immutable_and_derives_even_head_width(config):
    assert config.head_dim == 4
    with pytest.raises(FrozenInstanceError):
        config.d_model = 32


@pytest.mark.parametrize(
    'field',
    ['vocab_size', 'max_seq_len', 'd_model', 'num_layers', 'num_heads',
     'num_kv_heads', 'd_ff'],
)
@pytest.mark.parametrize('value', [0, -1, 1.5, True, '4'])
def test_config_rejects_nonpositive_or_noninteger_dimensions(config, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        replace(config, **{field: value})


@pytest.mark.parametrize('changes', [
    {'d_model': 18},
    {'d_model': 12},
    {'num_kv_heads': 3},
    {'num_kv_heads': 8},
])
def test_config_rejects_incompatible_head_dimensions(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


@pytest.mark.parametrize('field', ['rms_norm_eps', 'rope_theta', 'init_std'])
@pytest.mark.parametrize('value', [0, -1, math.inf, math.nan, True, 'small'])
def test_config_rejects_invalid_numerical_constants(config, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        replace(config, **{field: value})


def test_embedding_gathers_rows_and_accumulates_repeated_id_gradients():
    embedding = Embedding(5, 3, init_std=0.02)
    weights = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    with torch.no_grad():
        embedding.weight.copy_(weights)
    ids = torch.tensor([[4, 1, 4], [0, 1, 2]])
    torch.testing.assert_close(embedding(ids), F.embedding(ids, weights), rtol=0, atol=0)
    embedding(ids).sum().backward()
    expected_gradient = torch.tensor([1, 2, 1, 0, 2])[:, None].expand(5, 3)
    torch.testing.assert_close(embedding.weight.grad, expected_gradient.float(), rtol=0, atol=0)


def test_linear_matches_reference_and_input_weight_gradients():
    layer = Linear(3, 5, init_std=0.02).double()
    x = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    actual = layer(x)
    expected = F.linear(x, layer.weight)
    torch.testing.assert_close(actual, expected, **FP64)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (x, layer.weight), cotangent)
    expected_gradients = torch.autograd.grad(expected, (x, layer.weight), cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, **FP64)


def test_rmsnorm_equation_zero_input_and_gradients():
    norm = RMSNorm(4, eps=1e-5).double()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
    x = torch.tensor([[0., 0., 0., 0.], [1., -2., 3., -4.]],
                     dtype=torch.float64, requires_grad=True)
    actual = norm(x)
    torch.testing.assert_close(actual, reference_norm(x, norm), **FP64)
    torch.testing.assert_close(actual[0], torch.zeros(4, dtype=torch.float64), rtol=0, atol=0)
    assert torch.autograd.gradcheck(
        lambda value, weight: torch.func.functional_call(norm, {'weight': weight}, (value,)),
        (x, norm.weight), eps=1e-6, atol=1e-5, rtol=1e-3,
    )


def test_rmsnorm_preserves_dtype_and_normalizes_last_axis():
    x = torch.randn(2, 3, 4)
    norm = RMSNorm(4, eps=1e-5)
    assert norm(x).dtype == torch.float32
    torch.testing.assert_close(norm(x), reference_norm(x, norm), **FP32)


@pytest.mark.parametrize('head_dim', [2, 4, 8])
def test_rope_matches_rotation_matrices_and_preserves_norm(head_dim):
    rope = RotaryPositionEmbedding(head_dim, theta=10000.0)
    x = torch.randn(2, 3, 5, head_dim, dtype=torch.float64, requires_grad=True)
    actual = rope(x)
    torch.testing.assert_close(actual, reference_rope(x, 10000.0), **FP64)
    torch.testing.assert_close(actual[..., 0, :], x[..., 0, :], rtol=0, atol=0)
    torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1), **FP64)
    assert torch.autograd.gradcheck(rope, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_rope_dot_product_depends_on_relative_position():
    rope = RotaryPositionEmbedding(8, theta=10000.0)
    q = torch.randn(8, dtype=torch.float64).expand(1, 1, 9, 8)
    k = torch.randn(8, dtype=torch.float64).expand(1, 1, 9, 8)
    rotated_q, rotated_k = rope(q), rope(k)
    for left, right, shift in [(0, 3, 2), (4, 1, 3), (2, 2, 5)]:
        before = (rotated_q[..., left, :] * rotated_k[..., right, :]).sum()
        after = (rotated_q[..., left + shift, :] * rotated_k[..., right + shift, :]).sum()
        torch.testing.assert_close(before, after, **FP64)


def test_swiglu_equation_and_input_weight_gradients(config):
    layer = SwiGLU(config).double()
    x = torch.randn(2, 3, config.d_model, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.3)
    actual = layer(x)
    expected = reference_swiglu(x, layer)
    torch.testing.assert_close(actual, expected, **FP64)
    cotangent = torch.randn_like(actual)
    inputs = (x, *layer.parameters())
    actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, **FP64)


@pytest.mark.parametrize('kv_heads', [1, 2, 4])
@pytest.mark.parametrize('batch,length', [(1, 1), (2, 5), (1, 7)])
def test_attention_matches_repeated_kv_reference_and_gradients(config, kv_heads, batch, length):
    config = replace(config, num_kv_heads=kv_heads)
    attention = CausalSelfAttention(config)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.normal_(std=0.3)
    x = torch.randn(batch, length, config.d_model, requires_grad=True)
    actual = attention(x)
    expected = reference_attention(x, attention, config)
    torch.testing.assert_close(actual, expected, **FP32)
    cotangent = torch.randn_like(actual)
    inputs = (x, *attention.parameters())
    actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, **FP32)


def test_equal_kv_and_query_heads_match_torch_multihead_attention(config):
    config = replace(config, num_kv_heads=config.num_heads)
    attention = CausalSelfAttention(config)
    reference = torch.nn.MultiheadAttention(config.d_model, config.num_heads, bias=False,
                                            batch_first=True)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.normal_(std=0.3)
        # Native MHA receives independently projected/rotated Q, K and V.
        reference.in_proj_weight.copy_(torch.eye(config.d_model).repeat(3, 1))
        reference.out_proj.weight.copy_(attention.out_proj.weight)
    x = torch.randn(2, 5, config.d_model)
    projected = []
    for projection in [attention.q_proj, attention.k_proj]:
        heads = F.linear(x, projection.weight).reshape(2, 5, config.num_heads, config.head_dim)
        rotated = reference_rope(heads.transpose(1, 2), config.rope_theta)
        projected.append(rotated.transpose(1, 2).reshape(2, 5, config.d_model))
    q, k = projected
    v = F.linear(x, attention.v_proj.weight)
    expected, _ = reference(q, k, v, need_weights=False,
                            attn_mask=torch.ones(5, 5, dtype=torch.bool).triu(1))
    torch.testing.assert_close(attention(x), expected, **FP32)


def test_zero_queries_and_keys_give_inclusive_prefix_mean(config):
    config = replace(config, d_model=8, num_heads=2, num_kv_heads=2)
    attention = CausalSelfAttention(config)
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.k_proj.weight.zero_()
        attention.v_proj.weight.copy_(torch.eye(8))
        attention.out_proj.weight.copy_(torch.eye(8))
    x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    expected = x.cumsum(dim=1) / torch.arange(1, 5)[None, :, None]
    torch.testing.assert_close(attention(x), expected, **FP32)


def test_attention_has_zero_gradient_from_future_positions(config):
    attention = CausalSelfAttention(config)
    x = torch.randn(2, 5, config.d_model, requires_grad=True)
    gradient, = torch.autograd.grad(attention(x)[:, 2].square().sum(), (x,))
    torch.testing.assert_close(gradient[:, 3:], torch.zeros_like(gradient[:, 3:]), rtol=0, atol=0)
    assert gradient[:, :3].abs().sum() > 0


@pytest.mark.parametrize('batch,length', [(1, 1), (2, 4), (3, 7)])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_model_logits_match_independent_composition_and_have_finite_gradients(config, batch, length, dtype):
    model = TransformerLM(config)
    ids = torch.randint(config.vocab_size, (batch, length), dtype=dtype)
    actual = model(ids)
    assert actual.shape == (batch, length, config.vocab_size)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference_model(ids, model), **FP32)
    F.cross_entropy(actual.flatten(0, 1), ids.long().flatten()).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_model_noncontiguous_ids_and_batch_independence(config):
    model = TransformerLM(config)
    ids = torch.randint(config.vocab_size, (3, 10))[:, ::2]
    assert not ids.is_contiguous()
    combined = model(ids)
    for row in range(ids.shape[0]):
        torch.testing.assert_close(combined[row:row + 1], model(ids[row:row + 1]), **FP32)
    torch.testing.assert_close(combined, model(ids.contiguous()), rtol=0, atol=0)


def test_model_does_not_read_future_tokens_or_mutate_inputs_and_state(config):
    model = TransformerLM(config)
    ids = torch.randint(config.vocab_size, (2, 7))
    original_ids = ids.clone()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    original_logits = model(ids)
    for prefix_length in [1, 3, 6]:
        changed = ids.clone()
        changed[:, prefix_length:] = (changed[:, prefix_length:] + 1) % config.vocab_size
        torch.testing.assert_close(model(changed)[:, :prefix_length],
                                   original_logits[:, :prefix_length], rtol=0, atol=0)
        torch.testing.assert_close(model(ids[:, :prefix_length]),
                                   original_logits[:, :prefix_length], **FP32)
    torch.testing.assert_close(ids, original_ids, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_model_owns_untied_weights_and_normal_initialization(config):
    model = TransformerLM(config)
    assert model.embedding.weight.data_ptr() != model.output.weight.data_ptr()
    assert all('bias' not in name for name, _ in model.named_parameters())
    for module in model.modules():
        assert not isinstance(module, (torch.nn.Embedding, torch.nn.Linear,
                                       torch.nn.RMSNorm, torch.nn.MultiheadAttention,
                                       torch.nn.Transformer, torch.nn.Dropout))
        if isinstance(module, RMSNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight), rtol=0, atol=0)
    # A same-seed comparison establishes init_std scaling without a statistical threshold.
    torch.manual_seed(19)
    first = TransformerLM(config)
    torch.manual_seed(19)
    second = TransformerLM(replace(config, init_std=2 * config.init_std))
    for (name, value), (_, scaled) in zip(first.named_parameters(), second.named_parameters()):
        torch.testing.assert_close(scaled, value if value.ndim == 1 else 2 * value, rtol=0, atol=0)


@pytest.mark.parametrize('ids,error', [
    (torch.ones(3, dtype=torch.int64), ValueError),
    (torch.ones(1, 2, 3, dtype=torch.int64), ValueError),
    (torch.empty(0, 3, dtype=torch.int64), ValueError),
    (torch.empty(2, 0, dtype=torch.int64), ValueError),
    (torch.zeros(1, 8, dtype=torch.int64), ValueError),
    (torch.tensor([[-1]]), ValueError),
    (torch.tensor([[19]]), ValueError),
    (torch.tensor([[1.5]]), TypeError),
    (torch.tensor([[True]]), TypeError),
    (torch.tensor([[1]], dtype=torch.int16), TypeError),
])
def test_model_rejects_invalid_token_inputs(config, ids, error):
    with pytest.raises(error):
        TransformerLM(config)(ids)


def test_model_accepts_minimal_dimensions():
    config = ModelConfig(vocab_size=1, max_seq_len=1, d_model=2, num_layers=1,
                         num_heads=1, num_kv_heads=1, d_ff=1)
    logits = TransformerLM(config)(torch.zeros(1, 1, dtype=torch.int64))
    assert logits.shape == (1, 1, 1)
    assert torch.isfinite(logits).all()


def test_model_parameter_gradients_match_independent_composition_in_float64(config):
    model = TransformerLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.normal_(std=0.3)
    ids = torch.tensor([[0, 18, 0, 2], [5, 5, 3, 1]])
    # Promote the same fp32 stress weights/cotangent to isolate the algorithm
    # from accumulated roundoff across the two complete computation graphs.
    cotangent = torch.randn(2, 4, config.vocab_size).double()
    model = model.double()
    actual = model(ids)
    expected = reference_model(ids, model)
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()), cotangent)
    expected_gradients = torch.autograd.grad(expected, tuple(model.parameters()), cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, **FP64)


@pytest.mark.parametrize('field', ['rms_norm_eps', 'rope_theta', 'init_std'])
@pytest.mark.parametrize('value', [1e-50, 1e50])
def test_config_rejects_constants_outside_normal_fp32_range(config, field, value):
    with pytest.raises(ValueError, match=field):
        replace(config, **{field: value})


def test_config_rejects_rope_base_below_one(config):
    with pytest.raises(ValueError, match='rope_theta'):
        replace(config, rope_theta=0.5)


def comparison_errors(actual, expected):
    difference = (actual.detach().double() - expected.detach().double()).abs()
    magnitude = expected.detach().double().abs()
    relative = torch.where(magnitude == 0,
                           torch.where(difference == 0, 0.0, torch.inf),
                           difference / magnitude)
    allowance = FP32['atol'] + FP32['rtol'] * magnitude
    return {
        'max_absolute': difference.max().item(),
        'max_relative': relative.max().item(),
        'max_allowance_fraction': (difference / allowance).max().item(),
    }


@pytest.mark.parametrize('seed', [0, 17, 3407])
@pytest.mark.parametrize('length', [1, 4, 7])
def test_default_fp32_model_outputs_and_all_parameter_gradients(config, seed, length):
    torch.manual_seed(seed)
    model = TransformerLM(config)
    ids = torch.randint(config.vocab_size, (2, length))
    actual = model(ids)
    expected = reference_model(ids, model)
    cotangent = torch.randn_like(actual)
    parameters = tuple(model.named_parameters())
    actual_gradients = torch.autograd.grad(actual, tuple(p for _, p in parameters), cotangent)
    expected_gradients = torch.autograd.grad(expected, tuple(p for _, p in parameters), cotangent)
    errors = {
        'seed': seed,
        'length': length,
        'init_std': config.init_std,
        'dtype': str(actual.dtype),
        'outputs': comparison_errors(actual, expected),
        'gradients': {
            name: comparison_errors(actual_gradient, expected_gradient)
            for (name, _), actual_gradient, expected_gradient
            in zip(parameters, actual_gradients, expected_gradients)
        },
    }
    print('FP32_CASE ' + json.dumps(errors, sort_keys=True))
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, **FP32)
    for (name, _), actual_gradient, expected_gradient in zip(
        parameters, actual_gradients, expected_gradients
    ):
        assert actual_gradient.dtype == expected_gradient.dtype == torch.float32
        torch.testing.assert_close(actual_gradient, expected_gradient, **FP32, msg=name)
