"""Hand-checked next-token updates and independent interrupted-run comparisons."""

import copy
import json
import math
import pickle
import random
from dataclasses import replace
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from unearned.model import ModelConfig, TransformerLM
from unearned.tokenizer import BPETokenizer
from unearned.training import Trainer, TrainingConfig


CORPUS = b'int x = 1;\n' * 4


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def independent_rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def independent_live_state(trainer):
    parameters = dict(trainer.model.named_parameters())
    return {
        'parameters': {name: value.detach().clone() for name, value in parameters.items()},
        'moments': {name: (state.step, state.exp_avg.detach().clone(), state.exp_avg_sq.detach().clone())
                    for name, parameter in parameters.items()
                    if (state := trainer.optimizer.state.get(parameter)) is not None},
        'completed': trainer.completed_steps, 'cursor': trainer.data_cursor, 'lr': trainer.optimizer.lr,
    }


def prime_rng_streams():
    random.gauss(0, 1)
    np.random.standard_normal(3)
    torch.randn(5)


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.device == expected.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.fixture(autouse=True)
def deterministic_cpu_environment():
    rng = independent_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    seed_all(101)
    yield
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    torch.set_rng_state(rng[2])
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.set_num_threads(threads)


def make_trainer(corpus=CORPUS, tokenizer=None, config=None, model_config=None):
    tokenizer = BPETokenizer([]) if tokenizer is None else tokenizer
    config = config or TrainingConfig(batch_size=2, sequence_length=4, peak_lr=0.003,
                                     min_lr=0.0003, warmup_steps=5, total_steps=50)
    model_config = model_config or ModelConfig(
        vocab_size=len(tokenizer.vocab), max_seq_len=8, d_model=8, num_layers=1,
        num_heads=2, num_kv_heads=1, d_ff=12,
    )
    return Trainer(TransformerLM(model_config), corpus, tokenizer, config)


def test_batches_shift_targets_overlap_and_drop_tail_without_end_to_start_transition():
    config = TrainingConfig(batch_size=2, sequence_length=3, peak_lr=0.001,
                            min_lr=0.0001, warmup_steps=0, total_steps=4)
    trainer = make_trainer(corpus=bytes(range(16)), config=config)
    x, y = trainer.next_batch()
    assert x.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert y.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert x.dtype == y.dtype == torch.int64
    x.fill_(99)
    assert trainer.next_batch()[0].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert trainer.data_cursor == 0
    trainer.step()
    assert trainer.completed_steps == 1 and trainer.data_cursor == 6
    x, y = trainer.next_batch()
    assert x.tolist() == [[6, 7, 8], [9, 10, 11]]
    assert y.tolist() == [[7, 8, 9], [10, 11, 12]]
    trainer.step()
    assert trainer.data_cursor == 0
    assert trainer.next_batch()[0].tolist() == [[0, 1, 2], [3, 4, 5]]


def test_step_matches_explicit_preupdate_loss_and_reference_optimizer():
    config = TrainingConfig(batch_size=2, sequence_length=3, peak_lr=0.002,
                            min_lr=0.0002, warmup_steps=0, total_steps=4)
    trainer = make_trainer(corpus=b'abcdefghijklm', config=config)
    reference = copy.deepcopy(trainer.model)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.002, betas=config.betas,
                                  eps=config.eps, weight_decay=config.weight_decay,
                                  foreach=False, fused=False)
    x = torch.tensor([[97, 98, 99], [100, 101, 102]])
    y = torch.tensor([[98, 99, 100], [101, 102, 103]])
    logits = reference(x)
    loss = (torch.logsumexp(logits, dim=-1) - logits.gather(-1, y[..., None]).squeeze(-1)).mean()
    loss.backward()
    optimizer.step()
    actual = trainer.step()
    assert actual == pytest.approx(loss.item(), rel=1e-6, abs=0)
    for parameter, expected in zip(trainer.model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-6)
        assert parameter.grad is None
        assert trainer.optimizer.state[parameter].step == 1
    assert trainer.completed_steps == 1 and trainer.data_cursor == 6


def test_zero_warmup_rate_still_advances_moments_then_uses_completed_count():
    trainer = make_trainer()
    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    trainer.step()
    assert trainer.optimizer.lr == 0
    assert trainer.completed_steps == 1
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, before[name])
        assert trainer.optimizer.state[parameter].step == 1
    assert any(state.exp_avg.abs().sum() > 0 for state in trainer.optimizer.state.values())
    trainer.step()
    assert trainer.optimizer.lr == pytest.approx(0.003 / 5, rel=0, abs=0)
    assert any(not torch.equal(parameter, before[name]) for name, parameter in trainer.model.named_parameters())
    assert trainer.completed_steps == 2


def test_failed_gradient_does_not_advance_count_cursor_or_rate_and_requires_restore(tmp_path):
    trainer = make_trainer()
    checkpoint = tmp_path / 'before.pt'
    trainer.save_checkpoint(checkpoint)
    before = trainer.state_dict()
    parameter = next(trainer.model.parameters())
    hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, math.nan))
    with pytest.raises(ValueError):
        trainer.step()
    hook.remove()
    assert trainer.completed_steps == 0 and trainer.data_cursor == 0
    assert trainer.optimizer.lr == 0 and trainer.optimizer.state == {}
    assert_tree_equal(trainer.model.state_dict(), before['model'])
    with pytest.raises(RuntimeError):
        trainer.step()
    with pytest.raises(RuntimeError):
        trainer.save_checkpoint(tmp_path / 'failed.pt')
    assert not (tmp_path / 'failed.pt').exists()
    trainer.load_checkpoint(checkpoint)
    assert math.isfinite(trainer.step())
    assert trainer.completed_steps == 1


@pytest.mark.parametrize('split', [0, 17])
def test_fifty_steps_resume_matches_independent_uninterrupted_run(tmp_path, split):
    seed_all(101)
    prime_rng_streams()
    baseline = make_trainer()
    initial_weights = {name: p.detach().clone() for name, p in baseline.model.named_parameters()}
    baseline_losses = baseline.run(50)
    expected_live = independent_live_state(baseline)
    expected = baseline.state_dict()
    expected_rng = independent_rng_state()
    assert expected_rng[0][2] is not None and expected_rng[1][3] == 1
    expected_draws = (random.random(), random.gauss(0, 1), np.random.rand(4),
                      np.random.standard_normal(3), torch.randn(4))

    seed_all(101)
    prime_rng_streams()
    interrupted = make_trainer()
    assert_tree_equal(interrupted.model.state_dict(), initial_weights)
    prefix = interrupted.run(split)
    before_save_rng = independent_rng_state()
    path = tmp_path / f'step-{split}.pt'
    interrupted.save_checkpoint(path)
    assert_tree_equal(independent_rng_state(), before_save_rng)
    assert interrupted.optimizer.lr == interrupted.schedule(max(0, split - 1))
    if split == 0:
        assert interrupted.optimizer.state == {}

    seed_all(907)
    random.random()
    np.random.rand(19)
    torch.rand(23)
    resumed = make_trainer()
    assert all(left is not right for left, right in zip(resumed.model.parameters(), interrupted.model.parameters()))
    resumed.load_checkpoint(path)
    assert_tree_equal(independent_rng_state(), before_save_rng)
    losses = prefix + resumed.run(50 - split)
    relative_errors = [abs(actual - reference) / abs(reference) if reference != 0
                       else (0 if actual == 0 else math.inf)
                       for actual, reference in zip(losses, baseline_losses)]
    assert len(losses) == len(baseline_losses) == 50
    assert max(relative_errors) <= 1e-6
    actual_live = independent_live_state(resumed)
    assert_tree_equal(actual_live, expected_live)
    actual = resumed.state_dict()
    assert_tree_equal(actual, expected)
    assert_tree_equal(independent_rng_state(), expected_rng)
    assert_tree_equal((random.random(), random.gauss(0, 1), np.random.rand(4),
                       np.random.standard_normal(3), torch.randn(4)), expected_draws)
    assert resumed.completed_steps == 50 and resumed.data_cursor == 0
    assert all(state.step == 50 for state in resumed.optimizer.state.values())
    print('RESUME_CASE ' + json.dumps({
        'split': split, 'seed': 101, 'steps': 50, 'losses': losses,
        'baseline_losses': baseline_losses, 'max_relative_loss_error': max(relative_errors),
        'max_absolute_loss_error': max(abs(a - b) for a, b in zip(losses, baseline_losses)),
        'final_model_optimizer_counters_data_rng_exact': True,
        'final_completed_steps': resumed.completed_steps, 'final_data_cursor': resumed.data_cursor,
        'parameter_tensors': len(tuple(resumed.model.parameters())),
        'total_parameters': sum(parameter.numel() for parameter in resumed.model.parameters()),
        'cuda_rng_states': len(actual['rng']['cuda']),
        'rng_priming': {'python_gauss': 1, 'numpy_normals': 3, 'torch_normals': 5},
        'max_absolute_parameter_error': max((actual_live['parameters'][name].double() - value.double()).abs().max().item()
                                            for name, value in expected_live['parameters'].items()),
        'max_absolute_moment_errors': [max((actual_live['moments'][name][index].double() - values[index].double()).abs().max().item()
                                           for name, values in expected_live['moments'].items()) for index in (1, 2)],
        'max_parameter_step_error': max(abs(actual_live['moments'][name][0] - values[0])
                                         for name, values in expected_live['moments'].items()),
        'identity': actual['identity'], 'provenance': actual['provenance'],
    }, sort_keys=True))


def test_captured_and_loaded_state_have_no_live_tensor_aliases():
    trainer = make_trainer()
    trainer.run(3)
    state = trainer.state_dict()
    frozen = copy.deepcopy(state)
    other = make_trainer()
    other.load_state_dict(state)
    state['model']['embedding.weight'].fill_(999)
    state['optimizer']['state'][0]['exp_avg'].fill_(999)
    assert_tree_equal(trainer.state_dict(), frozen)
    assert_tree_equal(other.state_dict(), frozen)
    snapshot = other.state_dict()
    other.step()
    assert_tree_equal(snapshot, frozen)
    assert_tree_equal(trainer.state_dict(), frozen)


@pytest.mark.parametrize('mismatch', ['corpus', 'tokenizer', 'model', 'training'])
def test_resume_rejects_changed_inputs_and_configuration_without_mutation(mismatch):
    tokenizer = BPETokenizer([(120, 121)])
    original = make_trainer(tokenizer=tokenizer)
    original.run(2)
    state = original.state_dict()
    kwargs = {'tokenizer': BPETokenizer([(120, 121)])}
    if mismatch == 'corpus':
        kwargs['corpus'] = CORPUS + b'!'
    elif mismatch == 'tokenizer':
        kwargs['tokenizer'] = BPETokenizer([(122, 122)])
    elif mismatch == 'model':
        kwargs['model_config'] = replace(original.model.config, d_ff=13)
    else:
        kwargs['config'] = replace(original.config, peak_lr=0.004)
    target = make_trainer(**kwargs)
    before = target.state_dict()
    rng = independent_rng_state()
    with pytest.raises(ValueError):
        target.load_state_dict(state)
    assert_tree_equal(target.state_dict(), before)
    assert_tree_equal(independent_rng_state(), rng)


@pytest.mark.parametrize('corruption', [
    'format', 'model_dtype', 'model_nonfinite', 'extra_model_key', 'optimizer_count',
    'step', 'cursor', 'lr', 'python_rng', 'python_rng_cache', 'python_rng_nan', 'python_rng_bool',
    'numpy_rng', 'torch_rng', 'cuda_rng',
    'source', 'lock', 'runtime',
])
def test_invalid_checkpoint_is_rejected_before_live_state_or_rng_mutation(corruption):
    original = make_trainer()
    original.run(2)
    state = original.state_dict()
    if corruption == 'format':
        state['format_version'] = 9
    elif corruption == 'model_dtype':
        state['model']['embedding.weight'] = state['model']['embedding.weight'].double()
    elif corruption == 'model_nonfinite':
        state['model']['embedding.weight'][0, 0] = math.inf
    elif corruption == 'extra_model_key':
        state['model']['extra'] = torch.ones(1)
    elif corruption == 'optimizer_count':
        state['optimizer']['state'][-1]['step'] = 1
    elif corruption == 'step':
        state['completed_steps'] = -1
    elif corruption == 'cursor':
        state['data_cursor'] = 1
    elif corruption == 'lr':
        state['optimizer']['lr'] = 0.99
    elif corruption == 'python_rng':
        state['rng']['python'] = (9, (), None)
    elif corruption in ('python_rng_cache', 'python_rng_nan', 'python_rng_bool'):
        version, internal, _ = state['rng']['python']
        invalid_cache = {'python_rng_cache': 'bad', 'python_rng_nan': math.nan, 'python_rng_bool': True}[corruption]
        state['rng']['python'] = (version, internal, invalid_cache)
    elif corruption == 'numpy_rng':
        state['rng']['numpy'][1] = [0]
    elif corruption == 'torch_rng':
        state['rng']['torch_cpu'] = torch.ones(2, dtype=torch.uint8)
    elif corruption == 'cuda_rng':
        state['rng']['cuda'] = [torch.zeros(2, dtype=torch.uint8)]
    elif corruption == 'source':
        state['provenance']['source_sha256']['training.py'] = 'changed'
    elif corruption == 'lock':
        state['provenance']['lock_sha256'] = 'changed'
    else:
        state['provenance']['runtime']['torch_threads'] += 1
    target = make_trainer()
    target.step()
    before, rng = target.state_dict(), independent_rng_state()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        target.load_state_dict(state)
    assert_tree_equal(target.state_dict(), before)
    assert_tree_equal(independent_rng_state(), rng)


class UnsupportedCheckpointValue:
    pass


def test_checkpoint_refuses_overwrite_and_unsupported_pickle_without_changing_rng(tmp_path):
    trainer = make_trainer()
    path = tmp_path / 'existing.pt'
    path.write_bytes(b'existing record')
    rng = independent_rng_state()
    with pytest.raises(FileExistsError):
        trainer.save_checkpoint(path)
    assert path.read_bytes() == b'existing record'
    assert list(tmp_path.iterdir()) == [path]
    unsupported = tmp_path / 'unsupported.pt'
    torch.save(UnsupportedCheckpointValue(), unsupported)
    before = trainer.state_dict()
    with pytest.raises(pickle.UnpicklingError):
        trainer.load_checkpoint(unsupported)
    assert_tree_equal(trainer.state_dict(), before)
    assert_tree_equal(independent_rng_state(), rng)


@pytest.mark.parametrize('changes', [
    {'batch_size': 0}, {'batch_size': True}, {'sequence_length': -1},
    {'sequence_length': 1.5}, {'warmup_steps': 50}, {'peak_lr': math.nan},
    {'betas': [0.9, 0.999]},
])
def test_invalid_training_configuration(changes):
    arguments = dict(batch_size=2, sequence_length=4, peak_lr=0.003,
                     min_lr=0.0003, warmup_steps=5, total_steps=50)
    arguments.update(changes)
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(**arguments)


def test_training_rejects_short_data_incompatible_model_and_invalid_run_lengths():
    with pytest.raises(ValueError):
        make_trainer(corpus=b'12345678')
    trainer = make_trainer()
    with pytest.raises(ValueError):
        Trainer(trainer.model.double(), CORPUS, BPETokenizer([]), trainer.config)
    with pytest.raises(ValueError):
        make_trainer(model_config=replace(trainer.model.config, vocab_size=257))
    with pytest.raises(ValueError):
        make_trainer(config=replace(trainer.config, sequence_length=9))
    trainer = make_trainer()
    for count in [-1, True, 1.5, 51]:
        with pytest.raises((ValueError, TypeError)):
            trainer.run(count)
    assert trainer.run(0) == []
    assert trainer.completed_steps == 0


def test_finite_gradient_overflow_blocks_commit_and_checkpoint():
    trainer = make_trainer()
    parameter = next(trainer.model.parameters())
    hook = parameter.register_hook(lambda gradient: torch.full_like(gradient, 1e30))
    with pytest.raises(ValueError, match='nonfinite state'):
        trainer.step()
    hook.remove()
    assert trainer.completed_steps == 0 and trainer.data_cursor == 0
    assert trainer.optimizer.lr == 0
    with pytest.raises(RuntimeError):
        trainer.state_dict()


@pytest.mark.parametrize('initializer', ['__init__.py', 'tokenizer/__init__.py'])
def test_resume_detects_changed_import_initializer(tmp_path, monkeypatch, initializer):
    import unearned.training as training

    package = tmp_path / 'src' / 'unearned'
    (package / 'tokenizer').mkdir(parents=True)
    for name in ['__init__.py', 'model.py', 'optim.py', 'training.py',
                 'tokenizer/__init__.py', 'tokenizer/bpe.py']:
        (package / name).write_text('initial source identity')
    monkeypatch.setattr(training, '__file__', str(package / 'training.py'))
    trainer = make_trainer()
    saved = trainer.state_dict()
    before, rng = independent_live_state(trainer), independent_rng_state()
    (package / initializer).write_text('changed import behavior')
    with pytest.raises(ValueError, match='source, lock or runtime'):
        trainer.load_state_dict(saved)
    assert_tree_equal(independent_live_state(trainer), before)
    assert_tree_equal(independent_rng_state(), rng)


@pytest.mark.parametrize('kind,field,value', [
    ('model', 'rms_norm_eps', np.float64(1e-5)),
    ('model', 'rope_theta', np.float64(10000)),
    ('model', 'init_std', np.float64(0.02)),
    ('training', 'peak_lr', np.float64(0.003)),
    ('training', 'min_lr', np.float64(0.0003)),
    ('training', 'eps', np.float64(1e-8)),
    ('training', 'weight_decay', np.float64(0.01)),
    ('training', 'betas', (np.float64(0.9), 0.999)),
    ('training', 'betas', (0.9, np.float64(0.999))),
])
def test_checkpoint_configuration_rejects_nonprimitive_numeric_scalars(kind, field, value):
    # NumPy float64 passes isinstance(value, float), but is not safe primitive
    # checkpoint metadata for the required weights_only loader.
    config = (ModelConfig(vocab_size=256, max_seq_len=8, d_model=8, num_layers=1,
                          num_heads=2, num_kv_heads=1, d_ff=12) if kind == 'model' else
              TrainingConfig(batch_size=2, sequence_length=4, peak_lr=0.003,
                             min_lr=0.0003, warmup_steps=5, total_steps=50))
    with pytest.raises(TypeError, match='built-in'):
        replace(config, **{field: value})


def _consume_rng():
    # Recovery must restore state after the failed attempt consumed randomness.
    random.random()
    np.random.standard_normal(3)
    torch.randn(5)


@contextmanager
def inject_step_failure(monkeypatch, trainer, point, primary, cleanup_error=None):
    """Inject once at a real update/commit boundary; restore all patched methods."""
    first = next(trainer.model.parameters())
    real_addcdiv = torch.Tensor.addcdiv_
    real_zero_grad = trainer.optimizer.zero_grad
    real_setattr = Trainer.__setattr__
    real_item = torch.Tensor.item
    witness = {'point': point, 'primary_raised': False, 'cleanup_raised': False,
               'zero_grad_calls': 0}

    def raise_primary():
        witness['primary_raised'] = True
        witness['completed_at_fault'] = trainer.completed_steps
        witness['cursor_at_fault'] = trainer.data_cursor
        witness['parameter_steps_at_fault'] = [
            trainer.optimizer.state[p].step for p in trainer.optimizer.parameters
        ]
        _consume_rng()
        raise primary

    def addcdiv_then_interrupt(parameter, *args, **kwargs):
        result = real_addcdiv(parameter, *args, **kwargs)
        if parameter is first and not witness['primary_raised']:
            raise_primary()
        return result

    def clear_then_interrupt():
        witness['zero_grad_calls'] += 1
        real_zero_grad()
        if cleanup_error is not None and witness['primary_raised']:
            witness['cleanup_raised'] = True
            raise cleanup_error
        call = 1 if point == 'initial_zero_grad' else 2
        if point in ('initial_zero_grad', 'final_zero_grad'):
            if witness['zero_grad_calls'] == call and not witness['primary_raised']:
                raise_primary()

    def interrupt_cursor_assignment(instance, name, value):
        if instance is trainer and name == 'data_cursor' and not witness['primary_raised']:
            # The count has advanced, but the cursor still names the prior batch.
            assert trainer.completed_steps == 18
            assert trainer.data_cursor == 16
            assert value == 24
            witness['attempted_cursor'] = value
            raise_primary()
        return real_setattr(instance, name, value)

    def interrupt_loss_item(tensor, *args, **kwargs):
        if (tensor.ndim == 0 and tensor.dtype == torch.float32
                and trainer.completed_steps == 18 and trainer.data_cursor == 24
                and not witness['primary_raised']):
            raise_primary()
        return real_item(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        if point == 'partial_optimizer':
            patch.setattr(torch.Tensor, 'addcdiv_', addcdiv_then_interrupt)
        elif point == 'between_commits':
            patch.setattr(Trainer, '__setattr__', interrupt_cursor_assignment)
        elif point == 'loss_item':
            patch.setattr(torch.Tensor, 'item', interrupt_loss_item)
        if point in ('initial_zero_grad', 'final_zero_grad') or cleanup_error is not None:
            patch.setattr(trainer.optimizer, 'zero_grad', clear_then_interrupt)
        yield witness


def _reference_and_interrupted_trainer(tmp_path):
    seed_all(101)
    prime_rng_streams()
    reference = make_trainer()
    reference.run(17)
    assert reference.completed_steps == 17 and reference.data_cursor == 16
    assert all(state.step == 17 for state in reference.optimizer.state.values())
    checkpoint = tmp_path / 'step-17.pt'
    reference.save_checkpoint(checkpoint)
    saved = reference.state_dict()
    saved_live, saved_rng = independent_live_state(reference), independent_rng_state()
    expected_losses = reference.run(33)
    expected_live, expected_rng = independent_live_state(reference), independent_rng_state()

    interrupted = make_trainer()
    interrupted.load_checkpoint(checkpoint)
    assert_tree_equal(independent_live_state(interrupted), saved_live)
    assert_tree_equal(independent_rng_state(), saved_rng)
    return interrupted, checkpoint, saved, saved_live, saved_rng, expected_losses, expected_live, expected_rng


def _assert_partial_update_witness(trainer, before, witness):
    names = list(before['parameters'])
    actual = independent_live_state(trainer)
    first = names[0]
    assert witness['parameter_steps_at_fault'] == [18] + [17] * (len(names) - 1)
    assert not torch.equal(actual['parameters'][first], before['parameters'][first])
    assert actual['moments'][first][0] == 18
    assert not torch.equal(actual['moments'][first][1], before['moments'][first][1])
    assert not torch.equal(actual['moments'][first][2], before['moments'][first][2])
    for name in names[1:]:
        assert_tree_equal(actual['parameters'][name], before['parameters'][name])
        assert_tree_equal(actual['moments'][name], before['moments'][name])


def _assert_blocked_until_restore(trainer, tmp_path, saved):
    live, rng = independent_live_state(trainer), independent_rng_state()
    failed_path = tmp_path / 'must-not-publish.pt'
    operations = [trainer.step, lambda: trainer.run(1), lambda: trainer.run(0),
                  trainer.state_dict, lambda: trainer.save_checkpoint(failed_path)]
    for operation in operations:
        with pytest.raises(RuntimeError):
            operation()
        assert_tree_equal(independent_live_state(trainer), live)
        assert_tree_equal(independent_rng_state(), rng)
    assert not failed_path.exists()
    invalid = copy.deepcopy(saved)
    invalid['data_cursor'] = 0
    with pytest.raises(ValueError):
        trainer.load_state_dict(invalid)
    assert_tree_equal(independent_live_state(trainer), live)
    assert_tree_equal(independent_rng_state(), rng)
    with pytest.raises(RuntimeError):
        trainer.run(0)


def _assert_same_trainer_recovers(trainer, checkpoint, saved_live, saved_rng,
                                expected_losses, expected_live, expected_rng):
    trainer.load_checkpoint(checkpoint)
    assert_tree_equal(independent_live_state(trainer), saved_live)
    assert_tree_equal(independent_rng_state(), saved_rng)
    assert all(p.grad is None for p in trainer.model.parameters())
    actual_losses = trainer.run(33)
    assert actual_losses == expected_losses
    assert_tree_equal(independent_live_state(trainer), expected_live)
    assert_tree_equal(independent_rng_state(), expected_rng)
    assert trainer.completed_steps == 50 and trainer.data_cursor == 0
    assert all(state.step == 50 for state in trainer.optimizer.state.values())
    trainer.state_dict()
    print('INTERRUPTION_RECOVERY ' + json.dumps({
        'restored_checkpoint_step': 17, 'continuation_steps': 33,
        'final_completed_steps': 50, 'final_cursor': 0,
        'maximum_loss_error': max(abs(actual - expected)
                                  for actual, expected in zip(actual_losses, expected_losses)),
        'parameters_moments_counters_cursor_rng_exact': True,
    }))


@pytest.mark.parametrize('point,error_type', [
    ('partial_optimizer', KeyboardInterrupt),
    ('partial_optimizer', RuntimeError),
    ('between_commits', KeyboardInterrupt),
    ('initial_zero_grad', KeyboardInterrupt),
    ('final_zero_grad', KeyboardInterrupt),
    ('loss_item', KeyboardInterrupt),
])
def test_step_failures_preserve_error_block_reuse_and_restore_exactly(tmp_path, monkeypatch,
                                                                   point, error_type):
    (trainer, checkpoint, saved, before, before_rng,
     expected_losses, expected_live, expected_rng) = _reference_and_interrupted_trainer(tmp_path)
    error = error_type(f'injected {point}')
    with inject_step_failure(monkeypatch, trainer, point, error) as witness:
        with pytest.raises(BaseException) as caught:
            trainer.step()
    print('INTERRUPTION_CASE ' + json.dumps({
        **witness, 'error_type': error_type.__name__,
        'propagated_type': type(caught.value).__name__, 'same_error_object': caught.value is error,
    }))
    assert caught.value is error
    assert witness['primary_raised']
    assert not torch.equal(independent_rng_state()[2], before_rng[2])
    if point == 'partial_optimizer':
        _assert_partial_update_witness(trainer, before, witness)
    if point == 'between_commits':
        assert (witness['completed_at_fault'], witness['cursor_at_fault'],
                witness['attempted_cursor']) == (18, 16, 24)
    if point in ('initial_zero_grad', 'final_zero_grad', 'between_commits'):
        assert all(p.grad is None for p in trainer.model.parameters())
    _assert_blocked_until_restore(trainer, tmp_path, saved)
    _assert_same_trainer_recovers(trainer, checkpoint, before, before_rng,
                                 expected_losses, expected_live, expected_rng)


@pytest.mark.parametrize('primary_type,cleanup_type', [
    (KeyboardInterrupt, RuntimeError),
    (RuntimeError, KeyboardInterrupt),
])
def test_cleanup_failure_cannot_mask_original_update_error(tmp_path, monkeypatch,
                                                         primary_type, cleanup_type):
    (trainer, checkpoint, saved, before, before_rng,
     expected_losses, expected_live, expected_rng) = _reference_and_interrupted_trainer(tmp_path)
    primary, cleanup = primary_type('primary failure'), cleanup_type('cleanup failure')
    with inject_step_failure(monkeypatch, trainer, 'partial_optimizer', primary, cleanup) as witness:
        with pytest.raises(BaseException) as caught:
            trainer.step()
    print('INTERRUPTION_CASE ' + json.dumps({
        **witness, 'primary_type': primary_type.__name__, 'cleanup_type': cleanup_type.__name__,
        'propagated_type': type(caught.value).__name__, 'same_error_object': caught.value is primary,
    }))
    assert caught.value is primary and caught.value is not cleanup
    assert witness['primary_raised'] and witness['cleanup_raised']
    _assert_partial_update_witness(trainer, before, witness)
    assert all(p.grad is None for p in trainer.model.parameters())
    _assert_blocked_until_restore(trainer, tmp_path, saved)
    _assert_same_trainer_recovers(trainer, checkpoint, before, before_rng,
                                 expected_losses, expected_live, expected_rng)


def test_normal_entry_rejections_do_not_poison_a_successful_boundary(tmp_path):
    trainer = make_trainer()
    trainer.run(17)
    before, rng = independent_live_state(trainer), independent_rng_state()
    with pytest.raises(ValueError):
        trainer.run(-1)
    assert trainer.run(0) == []
    assert_tree_equal(independent_live_state(trainer), before)
    assert_tree_equal(independent_rng_state(), rng)
    trainer.state_dict()
    trainer.run(33)
    completed, rng = independent_live_state(trainer), independent_rng_state()
    with pytest.raises(ValueError):
        trainer.step()
    assert trainer.run(0) == []
    assert_tree_equal(independent_live_state(trainer), completed)
    assert_tree_equal(independent_rng_state(), rng)
    trainer.save_checkpoint(tmp_path / 'completed.pt')
