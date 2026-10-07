"""Long greedy decoding against independent uncached histories on pinned fp32 weights."""

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import unearned.inference as inference_source
import unearned.model as model_source
from unearned.inference import decode, prefill
from unearned.model import ModelConfig, TransformerLM


GENERATED_TOKENS = 256
MODEL_SEED = 3407
LOGITS_ATOL = 1e-6


def pinned_model():
    assert torch.get_default_dtype() == torch.float32
    config = ModelConfig(
        vocab_size=31, max_seq_len=288, d_model=16, num_layers=2,
        num_heads=4, num_kv_heads=2, d_ff=24, init_std=0.02,
    )
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.manual_seed(MODEL_SEED)
        model = TransformerLM(config).eval()
    assert all(parameter.device.type == 'cpu' and parameter.dtype == torch.float32
               for parameter in model.parameters())
    return model


def weights_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        metadata = json.dumps(
            {'name': name, 'dtype': str(tensor.dtype), 'shape': list(tensor.shape)},
            sort_keys=True, separators=(',', ':'),
        ).encode()
        digest.update(len(metadata).to_bytes(8, 'big'))
        digest.update(metadata)
        digest.update(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    return {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in (
            ('src/unearned/model.py', model_source.__file__),
            ('src/unearned/inference.py', inference_source.__file__),
            ('test_long_decode.py', __file__),
        )
    }


def write_optional_evidence(record, cached_logits, reference_logits):
    directory = os.environ.get('UNEARNED_M04_EVIDENCE_DIR')
    if directory is None:
        return
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    stem = record['case']
    paired_path = destination / f'{stem}-paired-logits.pt'
    json_path = destination / f'{stem}.json'
    # Evidence is append-only: a repeat needs a fresh destination.
    if paired_path.exists() or json_path.exists():
        raise FileExistsError(f'evidence already exists for {stem}')
    paired = {
        'metadata': {key: value for key, value in record.items() if key != 'steps'},
        'cached_logits': torch.stack(cached_logits),
        'reference_logits': torch.stack(reference_logits),
    }
    with paired_path.open('xb') as output:
        torch.save(paired, output)
    record['paired_logits'] = {
        'file': paired_path.name,
        'shape': list(paired['cached_logits'].shape),
        'dtype': str(paired['cached_logits'].dtype),
        'sha256': hashlib.sha256(paired_path.read_bytes()).hexdigest(),
    }
    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    with json_path.open('x') as output:
        json.dump(json_safe(record), output, indent=2, allow_nan=False)
        output.write('\n')


@pytest.mark.parametrize('batch,prompt_length', [(1, 1), (2, 32)])
def test_256_generated_tokens_match_independent_uncached_greedy(batch, prompt_length):
    initial_rng = torch.get_rng_state().clone()
    cached_model, reference_model = pinned_model(), pinned_model()
    model_hash = weights_sha256(cached_model)
    assert weights_sha256(reference_model) == model_hash
    prompt = (
        torch.arange(batch * prompt_length, dtype=torch.long, device='cpu')
        .reshape(batch, prompt_length) * 7 + 3
    ) % cached_model.config.vocab_size
    cached_history, reference_history = prompt.clone(), prompt.clone()
    cached_steps, reference_steps, rows = [], [], []
    record = {
        'case': f'greedy-b{batch}-p{prompt_length}-g{GENERATED_TOKENS}',
        'provenance': 'fresh initialized pinned weights; no trained checkpoint',
        'model_seed': MODEL_SEED,
        'config': asdict(cached_model.config),
        'weights_sha256': model_hash,
        'weights_hash_format': 'ordered state_dict; 8-byte big-endian JSON metadata length, '
                               'canonical name/dtype/shape JSON, contiguous raw tensor bytes',
        'source_sha256': source_hashes(),
        'torch_version': str(torch.__version__),
        'device': 'cpu',
        'dtype': 'torch.float32',
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'prompt_ids': prompt.tolist(),
        'generated_tokens_per_row': GENERATED_TOKENS,
        'rtol': 0,
        'atol': LOGITS_ATOL,
        'status': 'running',
        'steps': rows,
    }
    with torch.no_grad():
        cached_output, cache = prefill(cached_model, prompt)
        for step in range(GENERATED_TOKENS):
            assert cache.length == prompt_length + step
            assert cached_history.shape == reference_history.shape == (batch, prompt_length + step)
            actual = cached_output[:, -1, :]
            expected = reference_model(reference_history)[:, -1, :]
            cached_steps.append(actual.clone())
            reference_steps.append(expected.clone())
            # Promote before subtraction so the record captures the exact difference
            # between represented fp32 values, without fp32 subtraction rounding.
            errors = (actual.double() - expected.double()).abs()
            maxima, coordinates = errors.max(dim=-1)
            cached_next, reference_next = actual.argmax(dim=-1), expected.argmax(dim=-1)
            indices = torch.arange(batch, device='cpu')
            row = {
                'prediction_step': step,
                'generated_token_number': step + 1,
                'input_length': prompt_length + step,
                'cache_length': cache.length,
                'max_absolute_error_by_row': maxima.tolist(),
                'worst_vocab_id_by_row': coordinates.tolist(),
                'cached_logit_at_worst_by_row': actual[indices, coordinates].tolist(),
                'reference_logit_at_worst_by_row': expected[indices, coordinates].tolist(),
                'cached_next_ids': cached_next.tolist(),
                'reference_next_ids': reference_next.tolist(),
                'all_finite': bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                'identical_tokens': bool(torch.equal(cached_next, reference_next)),
            }
            rows.append(row)
            try:
                assert row['all_finite'], f'nonfinite logits at step {step}'
                torch.testing.assert_close(actual, expected, rtol=0, atol=LOGITS_ATOL)
                assert torch.equal(cached_next, reference_next), f'greedy divergence at step {step}'
            except AssertionError:
                record.update(status='failed', failed_prediction_step=step,
                              cached_history_ids=cached_history.tolist(),
                              reference_history_ids=reference_history.tolist())
                write_optional_evidence(record, cached_steps, reference_steps)
                raise
            cached_history = torch.cat((cached_history, cached_next[:, None]), dim=1)
            reference_history = torch.cat((reference_history, reference_next[:, None]), dim=1)
            if step + 1 < GENERATED_TOKENS:
                cached_output, cache = decode(cached_model, cached_next[:, None], cache)
    assert len(rows) == GENERATED_TOKENS
    assert torch.equal(cached_history, reference_history)
    assert cached_history.shape == (batch, prompt_length + GENERATED_TOKENS)
    # The final sampled token has not been consumed by the model.
    assert cache.length == cached_history.shape[1] - 1
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert weights_sha256(cached_model) == weights_sha256(reference_model) == model_hash
    maximum = max(error for row in rows for error in row['max_absolute_error_by_row'])
    record.update(
        status='passed', observed_steps=len(rows), max_absolute_error=maximum,
        cached_history_ids=cached_history.tolist(), reference_history_ids=reference_history.tolist(),
        final_cache_length=cache.length, final_sample_cached=False,
        identical_generated_tokens=True, rng_unchanged=True, weights_unchanged=True,
    )
    write_optional_evidence(record, cached_steps, reference_steps)
    print(f"LONG_DECODE {record['case']} steps={len(rows)} "
          f"max_abs_error={maximum:.17g} tokens_equal=True weights_sha256={model_hash}")


def test_256_sampler_steps_match_uncached_histories_for_unequal_reordered_slots():
    import unearned.generation as generation_source
    import unearned.stopping as stopping_source
    from unearned.generation import GenerationConfig, advance_slots, start_slot

    initial_rng = torch.get_rng_state().clone()
    cached_model, reference_model = pinned_model(), pinned_model()
    model_hash = weights_sha256(cached_model)
    assert weights_sha256(reference_model) == model_hash
    slot_ids, prompt_lengths = (101, 202, 303), (1, 7, 32)
    prompts = {
        slot_id: tuple((position * 7 + 3 + index) % 31 for position in range(length))
        for index, (slot_id, length) in enumerate(zip(slot_ids, prompt_lengths))
    }
    config = GenerationConfig(max_new_tokens=GENERATED_TOKENS)
    vocab = tuple(bytes([token]) for token in range(31))
    states = {
        slot_id: start_slot(cached_model, prompts[slot_id], vocab=vocab,
                            slot_id=slot_id, config=config, base_seed=0)
        for slot_id in slot_ids
    }
    reference_histories = {
        slot_id: torch.tensor([prompts[slot_id]], dtype=torch.long, device='cpu')
        for slot_id in slot_ids
    }
    sources = source_hashes()
    for name, module in (('generation', generation_source), ('stopping', stopping_source)):
        sources[f'src/unearned/{name}.py'] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    cached_steps, reference_steps, rows = [], [], []
    record = {
        'case': 'sampler-unequal-p1-7-32-g256',
        'provenance': 'fresh initialized pinned weights; no trained checkpoint',
        'model_seed': MODEL_SEED,
        'config': asdict(cached_model.config),
        'generation_config': asdict(config),
        'weights_sha256': model_hash,
        'weights_hash_format': 'ordered state_dict; 8-byte big-endian JSON metadata length, '
                               'canonical name/dtype/shape JSON, contiguous raw tensor bytes',
        'source_sha256': sources,
        'torch_version': str(torch.__version__),
        'device': 'cpu',
        'dtype': 'torch.float32',
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'slot_row_order': list(slot_ids),
        'prompt_ids': {str(slot_id): list(prompts[slot_id]) for slot_id in slot_ids},
        'base_seed': 0,
        'derived_seeds': {str(slot_id): states[slot_id].seed for slot_id in slot_ids},
        'vocab_hex': [piece.hex() for piece in vocab],
        'function_stopping': False,
        'generated_tokens_per_row': GENERATED_TOKENS,
        'rtol': 0,
        'atol': LOGITS_ATOL,
        'status': 'running',
        'steps': rows,
    }
    with torch.no_grad():
        for step in range(GENERATED_TOKENS):
            rotation = step % len(slot_ids)
            order = slot_ids[rotation:] + slot_ids[:rotation]
            if step % 2:
                order = order[::-1]
            advanced = advance_slots(cached_model, tuple(states[slot_id] for slot_id in order))
            assert tuple(slot.slot_id for slot in advanced) == order
            updated = {slot.slot_id: slot for slot in advanced}
            assert set(updated) == set(slot_ids)
            actual = torch.stack([updated[slot_id].logits for slot_id in slot_ids])
            expected = torch.stack([
                reference_model(reference_histories[slot_id])[0, -1, :]
                for slot_id in slot_ids
            ])
            cached_steps.append(actual.clone())
            reference_steps.append(expected.clone())
            errors = (actual.double() - expected.double()).abs()
            maxima, coordinates = errors.max(dim=-1)
            sampled = torch.tensor([updated[slot_id].sampled_ids[-1] for slot_id in slot_ids],
                                   dtype=torch.long, device='cpu')
            reference_next = expected.argmax(dim=-1)
            cached_argmax = actual.argmax(dim=-1)
            indices = torch.arange(len(slot_ids), device='cpu')
            row = {
                'prediction_step': step,
                'generated_token_number': step + 1,
                'processing_order': list(order),
                'input_length_by_row': [reference_histories[slot_id].shape[1] for slot_id in slot_ids],
                'cache_length_by_row': [updated[slot_id].cache.length for slot_id in slot_ids],
                'max_absolute_error_by_row': maxima.tolist(),
                'worst_vocab_id_by_row': coordinates.tolist(),
                'cached_logit_at_worst_by_row': actual[indices, coordinates].tolist(),
                'reference_logit_at_worst_by_row': expected[indices, coordinates].tolist(),
                'sampled_ids': sampled.tolist(),
                'cached_argmax_ids': cached_argmax.tolist(),
                'reference_next_ids': reference_next.tolist(),
                'all_finite': bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                'identical_tokens': bool(torch.equal(sampled, reference_next)),
                'stop_reason_by_row': [updated[slot_id].stop_reason for slot_id in slot_ids],
            }
            rows.append(row)
            try:
                assert row['all_finite'], f'nonfinite sampler logits at step {step}'
                torch.testing.assert_close(actual, expected, rtol=0, atol=LOGITS_ATOL)
                assert torch.equal(sampled, cached_argmax), f'sampler is not greedy at step {step}'
                assert torch.equal(sampled, reference_next), f'sampler divergence at step {step}'
                for slot_id, length in zip(slot_ids, prompt_lengths):
                    slot = updated[slot_id]
                    assert slot.prompt_ids == prompts[slot_id]
                    assert slot.sampled_ids[:-1] == states[slot_id].sampled_ids
                    assert len(slot.sampled_ids) == step + 1
                    assert slot.cache.length == length + step
                    assert reference_histories[slot_id].shape == (1, length + step)
                    assert slot.rng_state is None
                    assert slot.log_probs == (0.0,) * (step + 1)
                    assert slot.stop_reason == ('max_new_tokens' if step == GENERATED_TOKENS - 1 else None)
            except AssertionError:
                record.update(
                    status='failed', failed_prediction_step=step,
                    sampled_ids={str(slot_id): list(updated[slot_id].sampled_ids) for slot_id in slot_ids},
                    reference_history_ids={str(slot_id): reference_histories[slot_id].tolist()[0]
                                           for slot_id in slot_ids},
                )
                write_optional_evidence(record, cached_steps, reference_steps)
                raise
            for index, slot_id in enumerate(slot_ids):
                reference_histories[slot_id] = torch.cat(
                    (reference_histories[slot_id], reference_next[index].reshape(1, 1)), dim=1,
                )
            states = updated
    assert len(rows) == GENERATED_TOKENS
    for slot_id, length in zip(slot_ids, prompt_lengths):
        slot = states[slot_id]
        assert reference_histories[slot_id].tolist()[0] == list(slot.prompt_ids + slot.sampled_ids)
        assert len(slot.sampled_ids) == GENERATED_TOKENS
        assert slot.cache.length == length + GENERATED_TOKENS - 1
        assert slot.decoded_bytes == slot.output == bytes(slot.sampled_ids)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert weights_sha256(cached_model) == weights_sha256(reference_model) == model_hash
    maximum = max(error for row in rows for error in row['max_absolute_error_by_row'])
    record.update(
        status='passed', observed_steps=len(rows), max_absolute_error=maximum,
        sampled_ids={str(slot_id): list(states[slot_id].sampled_ids) for slot_id in slot_ids},
        reference_history_ids={str(slot_id): reference_histories[slot_id].tolist()[0] for slot_id in slot_ids},
        final_cache_lengths={str(slot_id): states[slot_id].cache.length for slot_id in slot_ids},
        final_sample_cached=False, identical_generated_tokens=True,
        rng_unchanged=True, weights_unchanged=True,
    )
    write_optional_evidence(record, cached_steps, reference_steps)
    print(f"LONG_DECODE {record['case']} steps={len(rows)} "
          f"max_abs_error={maximum:.17g} tokens_equal=True weights_sha256={model_hash}")
