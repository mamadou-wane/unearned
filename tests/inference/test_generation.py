"""Sampling, ownership, stopping and independent-slot behavior."""

import importlib
from dataclasses import FrozenInstanceError

import pytest
import torch

from unearned.model import ModelConfig, TransformerLM


@pytest.fixture(autouse=True)
def preserve_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture
def api():
    return importlib.import_module('unearned.generation')


def make_model(context=32, vocab_size=7):
    torch.manual_seed(812)
    return TransformerLM(ModelConfig(
        vocab_size=vocab_size, max_seq_len=context, d_model=16,
        num_layers=2, num_heads=4, num_kv_heads=2, d_ff=24,
    )).eval()


VOCAB = (b'a', b'b', b'c', b'{', b'}trailing', b'"}"', b'<eos>')


def start(api, model, prompt=(0, 1), **kwargs):
    kwargs.setdefault('slot_id', 19)
    kwargs.setdefault('vocab', VOCAB)
    return api.start_slot(model, prompt, **kwargs)


def snapshot(slot):
    return (slot.sampled_ids, slot.output, slot.decoded_bytes, slot.rng_state,
            slot.parser, slot.stop_reason, slot.logits.clone(), slot.logits._version,
            tuple((t.clone(), t._version) for layer in slot.cache.layers
                  for t in (layer.key, layer.value)))


def unchanged(slot, saved):
    assert (slot.sampled_ids, slot.output, slot.decoded_bytes, slot.rng_state,
            slot.parser, slot.stop_reason) == saved[:6]
    assert torch.equal(slot.logits, saved[6])
    assert slot.logits._version == saved[7]
    for tensor, (value, version) in zip(
        (t for layer in slot.cache.layers for t in (layer.key, layer.value)), saved[8],
    ):
        assert torch.equal(tensor, value)
        assert tensor._version == version


def test_greedy_ties_pending_token_and_caps(api):
    model = make_model()
    with torch.no_grad():
        model.output.weight.zero_()
    slot = start(api, model, config=api.GenerationConfig(max_new_tokens=3))
    original, initial = slot, snapshot(slot)
    for count in range(1, 4):
        slot = api.advance(model, slot)
        assert slot.sampled_ids == (0,) * count
        assert slot.cache.length == 2 + count - 1
        assert slot.log_probs == (0.0,) * count
        assert slot.rng_state is None
    assert slot.stop_reason == 'max_new_tokens'
    assert slot.output == b'aaa'
    assert api.advance(model, slot) is slot
    unchanged(original, initial)
    with pytest.raises(FrozenInstanceError):
        slot.stop_reason = None


@pytest.mark.parametrize('context,prompt,budget,reason,count', [
    (2, (0, 1), 10, 'context', 0),
    (3, (0, 1), 10, 'context', 1),
    (4, (0, 1), 2, 'max_new_tokens', 2),
    (4, (0, 1), 0, 'max_new_tokens', 0),
])
def test_context_and_token_budget(api, context, prompt, budget, reason, count):
    model = make_model(context)
    slot = start(api, model, prompt, config=api.GenerationConfig(max_new_tokens=budget))
    result, = api.generate(model, (slot,))
    assert len(result.sampled_ids) == count
    assert result.stop_reason == reason
    assert result.cache.length == len(prompt) + max(0, count - 1)


def test_eos_by_id_is_retained_and_precedes_parser_and_cap(api, monkeypatch):
    model = make_model()
    config = api.GenerationConfig(mode='sample', max_new_tokens=1, eos_token_id=4)
    slot = start(api, model, config=config, function_prefix=b'int f() {')
    monkeypatch.setattr(torch, 'multinomial', lambda *a, **k: torch.tensor([4]))
    result = api.advance(model, slot)
    assert result.sampled_ids == (4,)
    assert result.decoded_bytes == b'}trailing'
    assert result.output == b''
    assert result.parser == slot.parser
    assert result.stop_reason == 'eos'
    assert result.cache.length == 2


def test_function_stop_preserves_whole_token_and_extracts_byte_boundary(api, monkeypatch):
    model = make_model(context=3)
    slot = start(api, model, config=api.GenerationConfig(mode='sample', max_new_tokens=1),
                 function_prefix=b'int f() {')
    monkeypatch.setattr(torch, 'multinomial', lambda *a, **k: torch.tensor([4]))
    result = api.advance(model, slot)
    assert result.sampled_ids == (4,)
    assert result.decoded_bytes == b'}trailing'
    assert result.output == b'}'
    assert result.stop_reason == 'function'
    assert result.parser.end == len(b'int f() {}')


def test_temperature_draws_and_log_probs_match_explicit_generator(api):
    model = make_model()
    config = api.GenerationConfig(mode='sample', temperature=0.7, max_new_tokens=8)
    slot = start(api, model, config=config, base_seed=901)
    generator = torch.Generator(device='cpu').manual_seed(slot.seed)
    history = torch.tensor([slot.prompt_ids], device='cpu')
    assert slot.config.top_p == 1.0
    for _ in range(8):
        with torch.no_grad():
            logits = model(history)[0, -1]
        probabilities = (logits / 0.7).softmax(-1)
        expected = torch.multinomial(probabilities, 1, generator=generator).item()
        slot = api.advance(model, slot)
        assert slot.sampled_ids[-1] == expected
        assert slot.log_probs[-1] == pytest.approx(probabilities[expected].log().item(), abs=1e-6)
        history = torch.cat((history, torch.tensor([[expected]], device='cpu')), dim=1)
    assert slot.rng_state == bytes(generator.get_state().tolist())


def test_slot_independence_reordering_stops_reuse_and_global_rng(api):
    model = make_model(context=32)
    config = api.GenerationConfig(mode='sample', max_new_tokens=9)
    slots = tuple(start(api, model, tuple(range(length)), slot_id=index,
                        config=config, base_seed=76) for index, length in enumerate((1, 3, 7)))
    before = torch.get_rng_state().clone()
    baseline = api.generate(model, slots)
    assert torch.equal(torch.get_rng_state(), before)
    reordered = api.generate(model, slots[::-1])[::-1]
    for original, a, b in zip(slots, baseline, reordered):
        alone, = api.generate(model, (original,))
        assert a.sampled_ids == b.sampled_ids == alone.sampled_ids
        assert a.rng_state == b.rng_state == alone.rng_state
    active = slots[0]
    stopped = start(api, model, (2,), slot_id=1,
                    config=api.GenerationConfig(mode='sample', max_new_tokens=0))
    for step in range(9):
        torch.rand(13, device='cpu')
        active, returned = api.advance_slots(model, (active, stopped))
        assert returned is stopped
        if step == 3:
            reused = start(api, model, (2,), slot_id=1, config=config)
            reused = api.advance(model, reused)
            assert len(reused.sampled_ids) == 1
    assert active.sampled_ids == baseline[0].sampled_ids
    assert active.rng_state == baseline[0].rng_state
    assert api.advance_slots(model, ()) == ()


@pytest.mark.parametrize('failure_site', ['draw', 'decode', 'parser'])
@pytest.mark.parametrize('exception_type', [RuntimeError, KeyboardInterrupt])
def test_failure_preserves_all_slot_state_and_retry(api, monkeypatch, failure_site, exception_type):
    model = make_model()
    slot = start(api, model, config=api.GenerationConfig(mode='sample'),
                 vocab=(b'a',) * 7, function_prefix=b'int f() {')
    slot = api.advance(model, slot)
    expected = api.advance(model, slot)
    saved = snapshot(slot)
    global_rng = torch.get_rng_state().clone()
    error = exception_type('injected after work')
    owner, name = {'draw': (api, '_sample'), 'decode': (api, 'decode'),
                   'parser': (api.FunctionStop, 'feed')}[failure_site]
    original = getattr(owner, name)
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise error
    with monkeypatch.context() as patch:
        patch.setattr(owner, name, fail)
        with pytest.raises(exception_type) as raised:
            api.advance(model, slot)
        assert raised.value is error
    unchanged(slot, saved)
    assert torch.equal(torch.get_rng_state(), global_rng)
    retry = api.advance(model, slot)
    assert retry.sampled_ids == expected.sampled_ids
    assert retry.rng_state == expected.rng_state
    assert retry.parser == expected.parser


@pytest.mark.parametrize('change', ['weights', 'logits', 'cache', 'roundtrip'])
def test_stale_state_rejected_before_terminal_draw(api, monkeypatch, change):
    model = make_model()
    slot = start(api, model, config=api.GenerationConfig(max_new_tokens=1, eos_token_id=0))
    with torch.no_grad():
        if change == 'weights': model.output.weight.add_(1)
        if change == 'logits': slot.logits.add_(1)
        if change == 'cache': slot.cache.layers[-1].value.add_(1)
        if change == 'roundtrip': model.half().float()
    monkeypatch.setattr(api, '_sample', lambda *a: pytest.fail('sampling occurred before validation'))
    with pytest.raises(ValueError):
        api.advance(model, slot)


def test_partial_batch_failure_is_atomic(api, monkeypatch):
    model = make_model()
    slots = tuple(start(api, model, slot_id=i,
                        config=api.GenerationConfig(mode='sample')) for i in range(2))
    saved = tuple(snapshot(s) for s in slots)
    original = api._sample
    calls = 0
    def fail_second(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        if calls == 2: raise KeyboardInterrupt('second slot')
        return result
    monkeypatch.setattr(api, '_sample', fail_second)
    with pytest.raises(KeyboardInterrupt):
        api.advance_slots(model, slots)
    for slot, before in zip(slots, saved): unchanged(slot, before)


@pytest.mark.parametrize('kwargs', [
    {'mode': 'other'}, {'temperature': 0}, {'temperature': float('nan')},
    {'temperature': float('inf')}, {'temperature': True}, {'max_new_tokens': -1},
    {'max_new_tokens': True}, {'eos_token_id': -1}, {'eos_token_id': True},
])
def test_invalid_config(api, kwargs):
    with pytest.raises((TypeError, ValueError)):
        api.GenerationConfig(**kwargs)


@pytest.mark.parametrize('prompt,kwargs', [
    ((), {}), ((7,), {}), ((True,), {}), ((1.2,), {}), ((-1,), {}),
    ((0,), {'slot_id': -1}), ((0,), {'base_seed': 2**64}),
    ((0,), {'vocab': (b'a',)}), ((0,), {'vocab': (b'a',) * 6 + ('x',)}),
    ((0,), {'function_prefix': b'int f() {}'}),
])
def test_invalid_start(api, prompt, kwargs):
    with pytest.raises((TypeError, ValueError)):
        start(api, make_model(), prompt, **kwargs)


def test_vocabulary_is_snapshotted_and_duplicate_slot_ids_rejected(api):
    model = make_model()
    vocab = list(VOCAB)
    slot = start(api, model, vocab=vocab)
    vocab[0] = b'changed'
    assert slot.vocab == VOCAB
    with pytest.raises(ValueError, match='slot_id'):
        api.advance_slots(model, (slot, slot))


def test_ambient_inference_mode_and_default_device(api):
    model = make_model()
    with torch.inference_mode(), torch.device('meta'):
        slot = start(api, model)
        result = api.advance(model, slot)
    assert result.logits.device.type == 'cpu'
    assert not result.logits.is_inference()
    assert result.cache.length == 2


def test_sampler_parses_comment_string_and_raw_literal_across_tokens(api, monkeypatch):
    model = make_model(context=16)
    vocabulary = (b'/', b'* } */ const char* s="', b'}',
                  b'"; auto raw=R"tag(', b'} )ta', b'g"; return 0;', b'}trailing')
    slot = start(api, model, (6,), vocab=vocabulary, function_prefix=b'int f(){ ',
                 config=api.GenerationConfig(mode='sample', max_new_tokens=7))
    scripted = iter(range(7))
    monkeypatch.setattr(torch, 'multinomial',
                        lambda *a, **k: torch.tensor([next(scripted)], device='cpu'))
    for count in range(1, 8):
        previous = slot
        slot = api.advance(model, slot)
        assert len(previous.sampled_ids) == count - 1
        assert slot.sampled_ids == tuple(range(count))
        assert slot.stop_reason == ('function' if count == 7 else None)
        assert slot.parser.source == b'int f(){ ' + b''.join(vocabulary[:count])
    assert slot.decoded_bytes == b''.join(vocabulary)
    assert slot.output == b''.join(vocabulary[:-1]) + b'}'
    assert slot.cache.length == 7


def test_reused_slot_starts_fresh_and_replays_same_identity(api):
    model = make_model()
    config = api.GenerationConfig(mode='sample', max_new_tokens=6)
    original = start(api, model, config=config, base_seed=99)
    completed, = api.generate(model, (original,))
    reused = start(api, model, config=config, base_seed=99)
    assert reused.cache is not original.cache
    assert reused.sampled_ids == ()
    assert reused.rng_state == original.rng_state
    replay, = api.generate(model, (reused,))
    assert replay.sampled_ids == completed.sampled_ids
    assert replay.rng_state == completed.rng_state


def test_default_384_token_cap(api):
    model = make_model(context=416)
    result, = api.generate(model, (start(api, model, (0,)),))
    assert len(result.sampled_ids) == 384
    assert result.stop_reason == 'max_new_tokens'
    assert result.cache.length == 384
