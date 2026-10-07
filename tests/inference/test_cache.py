"""Inference cache contracts checked against the uncached Transformer."""

import importlib
import math
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
import torch.nn.functional as F

from unearned.model import ModelConfig, RotaryPositionEmbedding, TransformerLM


LOGITS = dict(rtol=0, atol=1e-6)


@pytest.fixture(autouse=True)
def reproducible_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3407)
        yield


@pytest.fixture
def api():
    return importlib.import_module('unearned.inference')


def make_model(*, context=16, kv_heads=2, seed=3407):
    torch.manual_seed(seed)
    return TransformerLM(ModelConfig(
        vocab_size=31, max_seq_len=context, d_model=16, num_layers=2,
        num_heads=4, num_kv_heads=kv_heads, d_ff=24,
    )).eval()


def tokens(batch=2, length=9):
    return (torch.arange(batch * length).reshape(batch, length) * 7 + 3) % 31


def cache_snapshot(cache):
    return [
        (layer.key.clone(), layer.value.clone(), layer.key._version, layer.value._version)
        for layer in cache.layers
    ]


def assert_cache_unchanged(cache, snapshot):
    for layer, (key, value, key_version, value_version) in zip(cache.layers, snapshot):
        assert torch.equal(layer.key, key)
        assert torch.equal(layer.value, value)
        assert layer.key._version == key_version
        assert layer.value._version == value_version


def independent_rope(x, theta, start=0):
    positions = []
    for index in range(x.shape[-2]):
        pairs = []
        for coordinate in range(0, x.shape[-1], 2):
            angle = (start + index) * theta ** (-coordinate / x.shape[-1])
            rotation = x.new_tensor([
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ])
            pairs.append(x[..., index, coordinate:coordinate + 2] @ rotation.T)
        positions.append(torch.cat(pairs, dim=-1))
    return torch.stack(positions, dim=-2)


def independent_kv(model, token_ids):
    inputs = []
    handles = [
        block.attention.register_forward_pre_hook(
            lambda _module, args: inputs.append(args[0].detach().clone())
        )
        for block in model.blocks
    ]
    try:
        with torch.no_grad():
            model(token_ids)
    finally:
        for handle in handles:
            handle.remove()
    result = []
    for block, x in zip(model.blocks, inputs):
        shape = (*token_ids.shape, model.config.num_kv_heads, model.config.head_dim)
        key = F.linear(x, block.attention.k_proj.weight).reshape(shape).transpose(1, 2)
        value = F.linear(x, block.attention.v_proj.weight).reshape(shape).transpose(1, 2)
        result.append((independent_rope(key, model.config.rope_theta), value))
    return result


@pytest.mark.parametrize('seed,batch,kv_heads,parts', [
    (3407, 1, 1, (5, 3, 1)),
    (19, 2, 2, (1, 2, 5, 1)),
    (97, 3, 4, (2, 7, 1, 8, 13, 1)),
])
def test_chunked_prefill_and_decode_match_uncached_prefixes(api, seed, batch, kv_heads, parts):
    model = make_model(context=sum(parts), kv_heads=kv_heads, seed=seed)
    ids = tokens(batch, sum(parts))
    cache = None
    length = 0
    for index, count in enumerate(parts):
        old_cache = cache
        old_snapshot = cache_snapshot(cache) if cache is not None else None
        chunk = ids[:, length:length + count]
        if index == len(parts) - 1:
            actual, cache = api.decode(model, chunk, cache)
        else:
            actual, cache = api.prefill(model, chunk, cache)
        with torch.no_grad():
            expected = model(ids[:, :length + count])[:, length:]
        assert actual.shape == (batch, count, model.config.vocab_size)
        torch.testing.assert_close(actual, expected, **LOGITS)
        assert cache.length == length + count
        assert cache.batch_size == batch
        assert len(cache.layers) == model.config.num_layers
        for layer in cache.layers:
            assert layer.key.shape == (batch, kv_heads, length + count, 4)
            assert layer.value.shape == layer.key.shape
            assert layer.key.dtype == layer.value.dtype == torch.float32
            assert layer.key.device == layer.value.device == torch.device('cpu')
        if old_cache is not None:
            assert_cache_unchanged(old_cache, old_snapshot)
            for before, after in zip(old_cache.layers, cache.layers):
                for name in ('key', 'value'):
                    old_tensor, new_tensor = getattr(before, name), getattr(after, name)
                    assert torch.equal(new_tensor[:, :, :length], old_tensor)
                    assert new_tensor.untyped_storage().data_ptr() != old_tensor.untyped_storage().data_ptr()
        length += count
    for layer, (key, value) in zip(cache.layers, independent_kv(model, ids)):
        torch.testing.assert_close(layer.key, key, **LOGITS)
        torch.testing.assert_close(layer.value, value, **LOGITS)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_full_prefill_logits_and_kv_contents_match_independent_reference(dtype):
    api = importlib.import_module('unearned.inference')
    model = make_model()
    ids = tokens().to(dtype)
    logits, cache = api.prefill(model, ids)
    with torch.no_grad():
        torch.testing.assert_close(logits, model(ids), **LOGITS)
    for layer, (key, value) in zip(cache.layers, independent_kv(model, ids)):
        torch.testing.assert_close(layer.key, key, **LOGITS)
        torch.testing.assert_close(layer.value, value, **LOGITS)
    assert isinstance(cache, api.KVCache)
    assert isinstance(cache.layers, tuple)
    assert all(isinstance(layer, api.LayerKV) for layer in cache.layers)
    with pytest.raises(FrozenInstanceError):
        cache.length = 0
    with pytest.raises(FrozenInstanceError):
        cache.layers[0].key = torch.empty(0)


def test_appended_prompt_is_causal_at_nonzero_past_length(api):
    model = make_model()
    ids = tokens()
    _, original = api.prefill(model, ids[:, :3])
    suffix = ids[:, 3:8]
    changed = suffix.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % 31
    first, _ = api.prefill(model, suffix, original)
    second, _ = api.prefill(model, changed, original)
    assert torch.equal(first[:, :3], second[:, :3])
    with torch.no_grad():
        torch.testing.assert_close(first, model(ids[:, :8])[:, 3:], **LOGITS)


def test_batch_rows_and_independent_sequences_do_not_share_state(api):
    model = make_model()
    ids = tokens(2, 8)
    _, together = api.prefill(model, ids[:, :5])
    combined, combined_cache = api.decode(model, ids[:, 5:6], together)
    rows = []
    for row in range(2):
        _, cache = api.prefill(model, ids[row:row + 1, :5])
        result, _ = api.decode(model, ids[row:row + 1, 5:6], cache)
        rows.append(result)
    torch.testing.assert_close(combined, torch.cat(rows), **LOGITS)
    _, unrelated = api.prefill(model, ids.flip(0)[:, :2])
    api.decode(model, ids[:, 2:3], unrelated)
    repeated, repeated_cache = api.decode(model, ids[:, 5:6], together)
    assert torch.equal(repeated, combined)
    for left, right in zip(combined_cache.layers, repeated_cache.layers):
        assert torch.equal(left.key, right.key)
        assert left.key.untyped_storage().data_ptr() != right.key.untyped_storage().data_ptr()
        assert left.value.untyped_storage().data_ptr() != right.value.untyped_storage().data_ptr()


@pytest.mark.parametrize('site', ['second_ffn', 'output'])
def test_late_failure_leaves_input_cache_unchanged_and_reusable(api, monkeypatch, site):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    snapshot = cache_snapshot(cache)
    error = RuntimeError('injected after attention has computed new keys and values')
    module = model.blocks[1].ffn if site == 'second_ffn' else model.output

    def fail(_x):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(module, 'forward', fail)
        with pytest.raises(RuntimeError) as raised:
            api.prefill(model, ids[:, 4:7], cache)
        assert raised.value is error
    assert_cache_unchanged(cache, snapshot)
    actual, _ = api.prefill(model, ids[:, 4:7], cache)
    with torch.no_grad():
        torch.testing.assert_close(actual, model(ids[:, :7])[:, 4:], **LOGITS)


@pytest.mark.parametrize('offset', [1, 7, 31])
def test_rope_offset_matches_absolute_rotation_matrices(offset):
    x = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    rope = RotaryPositionEmbedding(4, 10000.0)
    torch.testing.assert_close(
        rope(x, position_offset=offset), independent_rope(x, 10000.0, offset),
        rtol=0, atol=1e-12,
    )
    assert torch.equal(rope(x), rope(x, position_offset=0))


@pytest.mark.parametrize('offset', [-1, True, 1.5, '1'])
def test_rope_rejects_invalid_position_offset(offset):
    with pytest.raises((TypeError, ValueError)):
        RotaryPositionEmbedding(4, 10000.0)(torch.ones(1, 1, 2, 4), position_offset=offset)


@pytest.mark.parametrize('field,value', [
    ('length', -1), ('length', 0), ('length', True), ('length', 4.0), ('length', 3), ('length', 5),
    ('batch_size', -1), ('batch_size', True), ('batch_size', 2.0), ('batch_size', 1),
])
def test_cache_rejects_inconsistent_or_invalid_scalar_metadata(api, field, value):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    snapshot = cache_snapshot(cache)
    malformed = replace(cache, **{field: value})
    with pytest.raises((TypeError, ValueError)):
        api.decode(model, ids[:, 4:5], malformed)
    assert_cache_unchanged(cache, snapshot)


@pytest.mark.parametrize('kind', [
    'missing_layer', 'extra_layer', 'list_layers', 'wrong_layer', 'non_tensor',
    'rank', 'batch', 'heads', 'length', 'width', 'dtype', 'device', 'sparse',
])
@pytest.mark.parametrize('field', ['key', 'value'])
def test_cache_rejects_malformed_layer_storage(api, kind, field):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    first = cache.layers[0]
    layers = cache.layers
    if kind == 'missing_layer':
        layers = layers[:-1]
    elif kind == 'extra_layer':
        layers = layers + layers[:1]
    elif kind == 'list_layers':
        layers = list(layers)
    elif kind == 'wrong_layer':
        layers = (None,) + layers[1:]
    else:
        tensor = getattr(first, field)
        invalid = {
            'non_tensor': None,
            'rank': tensor[0],
            'batch': tensor[:1],
            'heads': tensor[:, :1],
            'length': tensor[:, :, :3],
            'width': tensor[..., :2],
            'dtype': tensor.double(),
            'device': torch.empty_like(tensor, device='meta'),
            'sparse': tensor.to_sparse(),
        }[kind]
        layers = (replace(first, **{field: invalid}),) + layers[1:]
    malformed = replace(cache, layers=layers)
    with pytest.raises((TypeError, ValueError)):
        api.decode(model, ids[:, 4:5], malformed)


@pytest.mark.parametrize('field', ['key', 'value'])
def test_caller_mutation_invalidates_cache(api, field):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    getattr(cache.layers[1], field).add_(1)
    with pytest.raises((TypeError, ValueError)):
        api.decode(model, ids[:, 4:5], cache)


@pytest.mark.parametrize('change', ['different_model', 'inplace', 'replacement', 'load', 'config'])
def test_cache_is_bound_to_model_and_unchanged_weights(api, change):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    if change == 'different_model':
        model = make_model()
    elif change == 'inplace':
        with torch.no_grad():
            model.output.weight.add_(0.01)
    elif change == 'replacement':
        model.output.weight = torch.nn.Parameter(model.output.weight.detach().clone())
    elif change == 'load':
        model.load_state_dict(model.state_dict())
    else:
        model.config = replace(model.config, max_seq_len=32)
    with pytest.raises((TypeError, ValueError)):
        api.decode(model, ids[:, 4:5], cache)
    fresh, _ = api.prefill(model, ids[:, :5])
    with torch.no_grad():
        torch.testing.assert_close(fresh, model(ids[:, :5]), **LOGITS)


@pytest.mark.parametrize('kind', [
    'non_tensor', 'float', 'bool', 'rank_one', 'rank_three', 'empty_batch',
    'empty_sequence', 'negative_id', 'large_id', 'device',
])
def test_prefill_rejects_invalid_inputs(api, kind):
    model = make_model()
    invalid = {
        'non_tensor': [[1]],
        'float': torch.ones(1, 1),
        'bool': torch.ones(1, 1, dtype=torch.bool),
        'rank_one': torch.tensor([1]),
        'rank_three': torch.ones(1, 1, 1, dtype=torch.long),
        'empty_batch': torch.empty(0, 1, dtype=torch.long),
        'empty_sequence': torch.empty(1, 0, dtype=torch.long),
        'negative_id': torch.tensor([[-1]]),
        'large_id': torch.tensor([[31]]),
        'device': torch.ones(1, 1, dtype=torch.long, device='meta'),
    }[kind]
    with pytest.raises((TypeError, ValueError)):
        api.prefill(model, invalid)


@pytest.mark.parametrize('operation', ['prefill', 'decode'])
@pytest.mark.parametrize('mode', ['training', 'double'])
def test_inference_rejects_unsupported_model_mode_and_dtype(api, operation, mode):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    model.train() if mode == 'training' else model.double()
    with pytest.raises((TypeError, ValueError)):
        getattr(api, operation)(model, ids[:, 4:5], cache)


def test_decode_requires_one_token_per_existing_batch_row_and_cache(api):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    for invalid in (ids[:, 4:6], ids[:, 4:4], ids[:1, 4:5]):
        with pytest.raises((TypeError, ValueError)):
            api.decode(model, invalid, cache)
    with pytest.raises((TypeError, ValueError)):
        api.decode(model, ids[:, 4:5], None)
    with pytest.raises((TypeError, ValueError)):
        api.prefill(model, ids[:, 4:5], object())


@pytest.mark.parametrize('operation', ['prefill', 'decode'])
def test_cpu_autocast_is_rejected_before_cached_work(api, monkeypatch, operation):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    snapshot = cache_snapshot(cache)

    def unexpected_embedding(_ids):
        pytest.fail('autocast must be rejected before model work')

    monkeypatch.setattr(model.embedding, 'forward', unexpected_embedding)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises((TypeError, ValueError)):
            getattr(api, operation)(model, ids[:, 4:5], cache)
    assert_cache_unchanged(cache, snapshot)


def test_model_with_inference_tensors_is_rejected_explicitly(api):
    with torch.inference_mode():
        model = make_model()
    assert all(parameter.is_inference() for parameter in model.parameters())
    with pytest.raises((TypeError, ValueError)):
        api.prefill(model, tokens())


def test_exact_context_boundary_singleton_and_overflow(api):
    model = make_model(context=1)
    ids = tokens(1, 2)
    actual, cache = api.prefill(model, ids[:, :1])
    with torch.no_grad():
        torch.testing.assert_close(actual, model(ids[:, :1]), **LOGITS)
    snapshot = cache_snapshot(cache)
    with pytest.raises((TypeError, ValueError)):
        api.prefill(model, ids)
    for function in (api.prefill, api.decode):
        with pytest.raises((TypeError, ValueError)):
            function(model, ids[:, 1:2], cache)
    assert_cache_unchanged(cache, snapshot)


@pytest.mark.parametrize('outer_inference_mode', [False, True])
def test_cache_has_no_autograd_and_preserves_existing_training_gradients(api, outer_inference_mode):
    model = make_model()
    ids = tokens()
    model(ids[:, :4]).sum().backward()
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    parameters = [parameter.detach().clone() for parameter in model.parameters()]
    with torch.inference_mode(outer_inference_mode):
        logits, cache = api.prefill(model, ids[:, :4])
        continuation, appended = api.decode(model, ids[:, 4:5], cache)
    for tensor in (logits, continuation):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
    for state in (cache, appended):
        for layer in state.layers:
            for tensor in (layer.key, layer.value):
                assert not tensor.requires_grad
                assert tensor.grad_fn is None
                assert not tensor.is_inference()
                assert isinstance(tensor._version, int)
    for parameter, value, gradient in zip(model.parameters(), parameters, gradients):
        assert torch.equal(parameter, value)
        assert torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize('operation', ['prefill', 'decode'])
def test_cpu_cache_ignores_ambient_default_device(api, operation):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    chunk = ids[:, 4:5] if operation == 'decode' else ids[:, 4:7]
    with torch.no_grad():
        expected = model(ids[:, :4 + chunk.shape[1]])[:, 4:]
    with torch.device('meta'):
        actual, new_cache = getattr(api, operation)(model, chunk, cache)
    torch.testing.assert_close(actual, expected, **LOGITS)
    assert all(t.device.type == 'cpu' for layer in new_cache.layers for t in (layer.key, layer.value))


@pytest.mark.parametrize('intermediate', [torch.float16, torch.float64])
def test_dtype_round_trip_invalidates_weight_binding(api, intermediate):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    parameter = model.embedding.weight
    before = parameter.detach().clone()
    version = parameter._version
    model.to(dtype=intermediate).float()
    print('DTYPE_ROUND_TRIP', intermediate, 'same_parameter', parameter is model.embedding.weight,
          'same_version', version == parameter._version,
          'changed_values', torch.count_nonzero(before != parameter).item())
    if intermediate == torch.float16:
        assert not torch.equal(before, parameter)
    with pytest.raises(ValueError):
        api.decode(model, ids[:, 4:5], cache)
    fresh, _ = api.prefill(model, ids[:, :5])
    with torch.no_grad():
        torch.testing.assert_close(fresh, model(ids[:, :5]), **LOGITS)


@pytest.mark.parametrize('operation', ['prefill', 'decode'])
@pytest.mark.parametrize('field', ['key', 'value'])
def test_final_layer_validation_precedes_any_model_work(api, monkeypatch, operation, field):
    model = make_model()
    ids = tokens()
    _, cache = api.prefill(model, ids[:, :4])
    snapshot = cache_snapshot(cache)
    last = cache.layers[-1]
    broken = replace(last, **{field: getattr(last, field)[:, :, :-1]})
    malformed = replace(cache, layers=cache.layers[:-1] + (broken,))

    def unexpected_embedding(_ids):
        pytest.fail('a malformed final layer must be rejected before embedding')

    monkeypatch.setattr(model.embedding, 'forward', unexpected_embedding)
    with pytest.raises(ValueError):
        getattr(api, operation)(model, ids[:, 4:5], malformed)
    assert_cache_unchanged(cache, snapshot)


def test_cache_success_and_late_failure_preserve_rng(api, monkeypatch):
    import random
    import numpy as np

    model = make_model()
    ids = tokens()
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()

    def check_rng():
        assert random.getstate() == python_state
        current = np.random.get_state()
        assert current[0] == numpy_state[0] and current[2:] == numpy_state[2:]
        np.testing.assert_array_equal(current[1], numpy_state[1])
        assert torch.equal(torch.get_rng_state(), torch_state)

    _, cache = api.prefill(model, ids[:, :4])
    api.decode(model, ids[:, 4:5], cache)
    check_rng()

    def fail(_x):
        raise RuntimeError('late failure')

    monkeypatch.setattr(model.blocks[-1].ffn, 'forward', fail)
    with pytest.raises(RuntimeError, match='late failure'):
        api.decode(model, ids[:, 4:5], cache)
    check_rng()
