"""Scalar mathematics, optimizer ownership, and CPU AdamW oracle comparisons."""

import json
import math
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from unearned.optim import AdamW, WarmupCosineSchedule


FP32 = dict(rtol=1e-6, atol=1e-6)


def test_scalar_steps_expose_bias_correction_epsilon_and_decoupled_decay():
    parameter = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    optimizer = AdamW([parameter], lr=0.1, betas=(0.5, 0.75), eps=0.5, weight_decay=0.2)
    parameter.grad = torch.tensor(-2.0, dtype=torch.float64)
    optimizer.step()
    state = optimizer.state[parameter]
    # m=-1, v=1, m_hat=-2, v_hat=4: 2*.98 - .1*(-2)/(2+.5) = 2.04.
    assert parameter.item() == pytest.approx(2.04, abs=1e-14, rel=0)
    assert state.exp_avg.item() == -1
    assert state.exp_avg_sq.item() == 1
    assert state.step == 1
    parameter.grad.fill_(1)
    optimizer.step()
    # The new first moment is zero, so only decoupled decay changes the weight.
    assert parameter.item() == pytest.approx(1.9992, abs=1e-14, rel=0)
    assert state.exp_avg.item() == 0
    assert state.exp_avg_sq.item() == 1
    assert state.step == 2


def test_constant_gradient_has_closed_form_moments_and_updates():
    parameter = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    optimizer = AdamW([parameter], lr=0.03, betas=(0.8, 0.9), eps=0.1, weight_decay=0)
    parameter.grad = torch.tensor(0.4, dtype=torch.float64)
    for step in range(1, 11):
        optimizer.step()
        state = optimizer.state[parameter]
        assert state.exp_avg.item() == pytest.approx(0.4 * (1 - 0.8**step), abs=1e-14, rel=0)
        assert state.exp_avg_sq.item() == pytest.approx(0.16 * (1 - 0.9**step), abs=1e-14, rel=0)
        assert parameter.item() == pytest.approx(3 - step * 0.03 * 0.4 / 0.5, abs=1e-14, rel=0)


def test_none_skips_everything_but_zero_gradient_creates_state_and_decays():
    missing = nn.Parameter(torch.tensor(2.0))
    zero = nn.Parameter(torch.tensor(2.0))
    optimizer = AdamW([missing, zero], lr=0.2, betas=(0.5, 0.75), weight_decay=0.1)
    assert optimizer.state == {}
    optimizer.step()
    assert optimizer.state == {}
    zero.grad = torch.zeros_like(zero)
    optimizer.step()
    assert missing.item() == 2
    assert missing not in optimizer.state
    assert zero.item() == pytest.approx(1.96, abs=1e-7, rel=0)
    assert optimizer.state[zero].step == 1
    assert optimizer.state[zero].exp_avg.item() == 0
    assert optimizer.state[zero].exp_avg_sq.item() == 0


def test_none_preserves_existing_state_while_zero_decays_moments():
    parameter = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    optimizer = AdamW([parameter], lr=0.1, betas=(0.5, 0.75), eps=0.5, weight_decay=0.2)
    parameter.grad = torch.tensor(2.0, dtype=torch.float64)
    optimizer.step()
    before = parameter.detach().clone()
    parameter.grad = None
    optimizer.step()
    state = optimizer.state[parameter]
    assert torch.equal(parameter, before)
    assert state.step == 1
    assert state.exp_avg.item() == 1
    assert state.exp_avg_sq.item() == 1
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert state.step == 2
    assert state.exp_avg.item() == 0.5
    assert state.exp_avg_sq.item() == 0.75
    expected = before.item() * 0.98 - 0.1 * (0.5 / 0.75) / (math.sqrt(0.75 / 0.4375) + 0.5)
    assert parameter.item() == pytest.approx(expected, abs=1e-14, rel=0)


def test_zero_learning_rate_advances_moments_without_changing_parameter():
    parameter = nn.Parameter(torch.tensor([2.0, -1.0]))
    optimizer = AdamW([parameter], lr=0, betas=(0.5, 0.75), weight_decay=0.5)
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([2.0, -2.0])
    optimizer.step()
    assert torch.equal(parameter, before)
    assert optimizer.state[parameter].step == 1
    torch.testing.assert_close(optimizer.state[parameter].exp_avg, torch.tensor([1., -1.]), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[parameter].exp_avg_sq, torch.ones(2), rtol=0, atol=0)


def test_moment_storage_is_owned_and_zero_grad_sets_none():
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(first.detach().clone())
    supplied = [first, second]
    optimizer = AdamW(iter(supplied), lr=0.01)
    supplied.clear()
    gradient = torch.tensor([0.5, -0.5])
    first.grad = gradient
    second.grad = gradient.clone()
    optimizer.step()
    tensors = [first, second, first.grad, second.grad]
    for parameter in (first, second):
        state = optimizer.state[parameter]
        tensors.extend((state.exp_avg, state.exp_avg_sq))
        assert not state.exp_avg.requires_grad
        assert not state.exp_avg_sq.requires_grad
    assert len({t.data_ptr() for t in tensors}) == len(tensors)
    assert torch.equal(gradient, torch.tensor([0.5, -0.5]))
    saved_moment = optimizer.state[first].exp_avg.clone()
    gradient.fill_(99)
    assert torch.equal(optimizer.state[first].exp_avg, saved_moment)
    optimizer.zero_grad()
    assert first.grad is second.grad is None
    assert torch.equal(optimizer.state[first].exp_avg, saved_moment)


@pytest.mark.parametrize('bad_gradient', ['nan', 'inf', 'sparse'])
@pytest.mark.parametrize('initialized', [False, True])
def test_invalid_later_gradient_is_rejected_before_any_update(bad_gradient, initialized):
    parameters = [nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2))]
    optimizer = AdamW(parameters, lr=0.1)
    if initialized:
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    weights = [parameter.detach().clone() for parameter in parameters]
    states = {parameter: (state.step, state.exp_avg.clone(), state.exp_avg_sq.clone())
              for parameter, state in optimizer.state.items()}
    parameters[0].grad = torch.ones(2)
    if bad_gradient == 'sparse':
        parameters[1].grad = torch.sparse_coo_tensor([[0]], [1.0], (2,), check_invariants=True)
    else:
        parameters[1].grad = torch.tensor([float(bad_gradient), 1.0])
    with pytest.raises(ValueError):
        optimizer.step()
    for parameter, weight in zip(parameters, weights):
        assert torch.equal(parameter, weight)
    assert optimizer.state.keys() == states.keys()
    for parameter, (step, moment, variance) in states.items():
        state = optimizer.state[parameter]
        assert state.step == step
        assert torch.equal(state.exp_avg, moment)
        assert torch.equal(state.exp_avg_sq, variance)


@pytest.mark.parametrize('settings', [
    {'lr': -1}, {'lr': math.nan}, {'lr': True},
    {'betas': (1, 0.9)}, {'betas': (0.9, -0.1)}, {'betas': (0.9,)},
    {'betas': (math.inf, 0.9)}, {'eps': 0}, {'eps': 1e-50},
    {'eps': math.inf}, {'weight_decay': -1}, {'weight_decay': math.nan},
])
def test_optimizer_rejects_invalid_hyperparameters(settings):
    with pytest.raises((ValueError, TypeError)):
        AdamW([nn.Parameter(torch.ones(2))], **settings)


def test_optimizer_rejects_empty_duplicate_grouped_or_unsupported_parameters():
    parameter = nn.Parameter(torch.ones(2))
    for parameters in [[], [parameter, parameter], [{'params': [parameter]}],
                       [nn.Parameter(torch.ones(2, dtype=torch.complex64))],
                       [nn.Parameter(torch.ones(2, dtype=torch.float16))]]:
        with pytest.raises((ValueError, TypeError)):
            AdamW(parameters)


def test_learning_rate_assignment_is_validated_without_losing_old_value():
    optimizer = AdamW([nn.Parameter(torch.ones(1))], lr=0.01)
    optimizer.lr = 0.02
    assert optimizer.lr == 0.02
    with pytest.raises(ValueError):
        optimizer.lr = -1
    assert optimizer.lr == 0.02


def tensor_discrepancy(actual, expected):
    difference = (actual.detach().double() - expected.detach().double()).abs()
    allowance = FP32['atol'] + FP32['rtol'] * expected.detach().double().abs()
    return {'max_absolute': difference.max().item(),
            'max_allowance_fraction': (difference / allowance).max().item()}


@pytest.mark.parametrize('weight_decay', [0.0, 0.1])
@pytest.mark.parametrize('betas,eps', [((0.9, 0.999), 1e-8), ((0.0, 0.5), 1e-3)])
@pytest.mark.parametrize('intermittent', [False, True])
def test_adamw_matches_torch_over_100_fp32_steps(weight_decay, betas, eps, intermittent):
    generator = torch.Generator().manual_seed(73)
    owned = [nn.Parameter(torch.randn(shape, generator=generator)) for shape in [(), (2, 3), (5,)]]
    reference = [nn.Parameter(parameter.detach().clone()) for parameter in owned]
    settings = dict(lr=0.003, betas=betas, eps=eps, weight_decay=weight_decay)
    optimizer = AdamW(owned, **settings)
    oracle = torch.optim.AdamW(reference, **settings, foreach=False, fused=False)
    history = []
    for update in range(100):
        lr = 0.0 if update % 13 == 0 else 0.003 * (1 + update % 4) / 4
        optimizer.lr = lr
        oracle.param_groups[0]['lr'] = lr
        for index, (parameter, expected_parameter) in enumerate(zip(owned, reference)):
            absent = intermittent and index > 0 and (update < index or update % 7 == index)
            gradient = None if absent else torch.randn(parameter.shape, generator=generator)
            if gradient is not None and update % 11 == 0:
                gradient.zero_()
            parameter.grad = gradient
            expected_parameter.grad = None if gradient is None else gradient.clone()
        optimizer.step()
        oracle.step()
        differences = []
        for index, (parameter, expected_parameter) in enumerate(zip(owned, reference)):
            assert parameter.dtype == expected_parameter.dtype == torch.float32
            difference = {'parameter_index': index, 'parameter': tensor_discrepancy(parameter, expected_parameter)}
            torch.testing.assert_close(parameter, expected_parameter, **FP32)
            if parameter not in optimizer.state:
                assert expected_parameter not in oracle.state
                difference['step'] = 0
            else:
                state, expected_state = optimizer.state[parameter], oracle.state[expected_parameter]
                difference['step'] = state.step
                difference['step_error'] = state.step - int(expected_state['step'].item())
                assert difference['step_error'] == 0
                for name in ('exp_avg', 'exp_avg_sq'):
                    actual, expected = getattr(state, name), expected_state[name]
                    assert actual.dtype == expected.dtype == torch.float32
                    difference[name] = tensor_discrepancy(actual, expected)
                    torch.testing.assert_close(actual, expected, **FP32)
            differences.append(difference)
        history.append({'update': update + 1, 'lr': lr, 'parameters': differences})
    print('ADAMW_ORACLE ' + json.dumps({'settings': settings, 'seed': 73, 'intermittent': intermittent,
                                      'foreach': False, 'fused': False, 'history': history}, sort_keys=True))


@pytest.mark.parametrize('step,expected', [
    (0, 0.0), (1, 0.5), (2, 1.0),
    (3, 0.55 + 0.45 / math.sqrt(2)), (4, 0.55),
    (5, 0.55 - 0.45 / math.sqrt(2)), (6, 0.1), (7, 0.1),
])
def test_warmup_cosine_boundaries(step, expected):
    schedule = WarmupCosineSchedule(peak_lr=1.0, min_lr=0.1, warmup_steps=2, total_steps=6)
    assert schedule(step) == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize('warmup,total,expected', [
    (0, 4, [1.0, 0.55 + 0.45 / math.sqrt(2), 0.55, 0.55 - 0.45 / math.sqrt(2), 0.1]),
    (3, 4, [0.0, 1 / 3, 2 / 3, 1.0, 0.1]),
    (0, 1, [1.0, 0.1]),
])
def test_schedule_without_warmup_or_with_one_decay_interval(warmup, total, expected):
    schedule = WarmupCosineSchedule(peak_lr=1.0, min_lr=0.1, warmup_steps=warmup, total_steps=total)
    assert [schedule(step) for step in range(total + 1)] == pytest.approx(expected, rel=0, abs=1e-15)


def test_schedule_is_stateless_and_immutable():
    schedule = WarmupCosineSchedule(peak_lr=1.0, min_lr=0.1, warmup_steps=2, total_steps=6)
    assert [schedule(step) for step in [6, 2, 0, 4, 2]] == pytest.approx([0.1, 1.0, 0.0, 0.55, 1.0])
    with pytest.raises(FrozenInstanceError):
        schedule.total_steps = 10


@pytest.mark.parametrize('settings', [
    {'peak_lr': -1}, {'min_lr': -1}, {'min_lr': 2}, {'peak_lr': math.inf},
    {'min_lr': math.nan}, {'warmup_steps': -1}, {'warmup_steps': True},
    {'warmup_steps': 1.5}, {'total_steps': 0}, {'total_steps': 2},
])
def test_schedule_rejects_invalid_configuration(settings):
    arguments = dict(peak_lr=1.0, min_lr=0.1, warmup_steps=2, total_steps=6)
    arguments.update(settings)
    with pytest.raises((ValueError, TypeError)):
        WarmupCosineSchedule(**arguments)


@pytest.mark.parametrize('step', [-1, 0.5, True, '1'])
def test_schedule_rejects_invalid_step(step):
    schedule = WarmupCosineSchedule(peak_lr=1, min_lr=0, warmup_steps=0, total_steps=1)
    with pytest.raises((ValueError, TypeError)):
        schedule(step)


def populated_optimizer():
    parameters = [nn.Parameter(torch.tensor([1.0, -2.0])),
                  nn.Parameter(torch.tensor([3.0, 4.0])),
                  nn.Parameter(torch.tensor(5.0))]
    optimizer = AdamW(parameters, lr=0.03)
    parameters[0].grad = torch.tensor([0.5, -0.5])
    parameters[2].grad = torch.tensor(2.0)
    optimizer.step()
    parameters[2].grad = None
    optimizer.step()
    return optimizer


def assert_optimizer_snapshot_equal(actual, expected):
    for key in ('lr', 'betas', 'eps', 'weight_decay'):
        assert actual[key] == expected[key]
    assert len(actual['state']) == len(expected['state'])
    for entry, reference in zip(actual['state'], expected['state']):
        if reference is None:
            assert entry is None
        else:
            assert entry['step'] == reference['step']
            for name in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(entry[name], reference[name], rtol=0, atol=0)


def test_optimizer_snapshot_preserves_initial_lazy_state():
    parameters = [nn.Parameter(torch.ones(2)), nn.Parameter(torch.tensor(3.0))]
    optimizer = AdamW(parameters, lr=0.2)
    assert optimizer.state_dict() == {
        'lr': 0.2, 'betas': (0.9, 0.999), 'eps': 1e-8,
        'weight_decay': 0.01, 'state': [None, None],
    }
    optimizer.load_state_dict(optimizer.state_dict())
    assert optimizer.state == {}


def test_optimizer_loads_manual_state_by_parameter_order_without_changing_parameters():
    parameters = [nn.Parameter(torch.ones(2)), nn.Parameter(torch.tensor(3.0))]
    parameters[0].grad = torch.tensor([4.0, 5.0])
    gradient = parameters[0].grad
    optimizer = AdamW(parameters)
    optimizer.load_state_dict({
        'lr': 0.25, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01,
        'state': [{'step': 7, 'exp_avg': torch.tensor([-0.5, 0.25]),
                   'exp_avg_sq': torch.tensor([0.125, 0.0])}, None],
    })
    assert optimizer.lr == 0.25
    assert optimizer.state[parameters[0]].step == 7
    assert parameters[1] not in optimizer.state
    assert torch.equal(optimizer.state[parameters[0]].exp_avg, torch.tensor([-0.5, 0.25]))
    assert torch.equal(optimizer.state[parameters[0]].exp_avg_sq, torch.tensor([0.125, 0.0]))
    assert torch.equal(parameters[0], torch.ones(2))
    assert parameters[1].item() == 3.0
    assert parameters[0].grad is gradient
    assert torch.equal(gradient, torch.tensor([4.0, 5.0]))
    assert parameters[1].grad is None


def test_optimizer_snapshot_round_trip_preserves_counts_and_replaces_old_state():
    source = populated_optimizer()
    payload = source.state_dict()
    target = AdamW([nn.Parameter(parameter.detach().clone()) for parameter in source.parameters], lr=0)
    for parameter in target.parameters:
        parameter.grad = torch.zeros_like(parameter)
    target.step()
    target.load_state_dict(payload)
    assert_optimizer_snapshot_equal(target.state_dict(), payload)
    assert [entry['step'] if entry else None for entry in payload['state']] == [2, None, 1]
    assert target.parameters[1] not in target.state
    for original, restored in zip(source.parameters, target.parameters):
        gradient = torch.full_like(original, 0.75)
        original.grad, restored.grad = gradient, gradient.clone()
    source.step()
    target.step()
    for original, restored in zip(source.parameters, target.parameters):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)
    assert_optimizer_snapshot_equal(source.state_dict(), target.state_dict())


def test_optimizer_capture_and_load_own_detached_moment_storage():
    optimizer = populated_optimizer()
    snapshot = optimizer.state_dict()
    moment = optimizer.state[optimizer.parameters[0]].exp_avg
    saved = snapshot['state'][0]['exp_avg'].clone()
    moment.add_(1)
    assert torch.equal(snapshot['state'][0]['exp_avg'], saved)
    snapshot['state'][0]['exp_avg'].requires_grad_(True)
    restored = AdamW([nn.Parameter(parameter.detach().clone()) for parameter in optimizer.parameters])
    restored.load_state_dict(snapshot)
    tensors = []
    for index in (0, 2):
        for name in ('exp_avg', 'exp_avg_sq'):
            original = getattr(optimizer.state[optimizer.parameters[index]], name)
            captured = snapshot['state'][index][name]
            loaded = getattr(restored.state[restored.parameters[index]], name)
            tensors.extend((original, captured, loaded))
            assert not loaded.requires_grad
            assert loaded.grad_fn is None
    assert len({tensor.data_ptr() for tensor in tensors}) == len(tensors)
    assert snapshot['state'][2]['exp_avg'].grad_fn is None
    with torch.no_grad():
        snapshot['state'][0]['exp_avg'].fill_(99)
        snapshot['state'][2]['exp_avg_sq'].fill_(99)
    snapshot['state'][2]['step'] = 99
    snapshot['state'][1] = snapshot['state'][0]
    assert torch.equal(restored.state[restored.parameters[0]].exp_avg, saved)
    assert restored.state[restored.parameters[2]].exp_avg_sq.item() != 99
    assert restored.state[restored.parameters[2]].step == 1
    assert restored.parameters[1] not in restored.state


@pytest.mark.parametrize('defect', [
    'not_dict', 'missing_key', 'extra_key', 'betas_list', 'betas_mismatch',
    'eps_mismatch', 'decay_mismatch', 'invalid_lr', 'nonfinite_lr', 'boolean_lr',
    'state_tuple', 'state_length', 'entry_type', 'entry_missing_key', 'entry_extra_key',
    'step_zero', 'step_boolean', 'step_float', 'moment_type', 'moment_shape',
    'moment_dtype', 'moment_device', 'moment_sparse', 'moment_nan', 'moment_inf',
    'negative_variance',
])
def test_optimizer_rejects_malformed_state_without_partial_mutation(defect):
    optimizer = populated_optimizer()
    before = optimizer.state_dict()
    old_state = optimizer.state
    parameters = [parameter.detach().clone() for parameter in optimizer.parameters]
    gradients = [parameter.grad for parameter in optimizer.parameters]
    payload = optimizer.state_dict()
    payload['lr'] = 0.07
    payload['state'][0]['step'] = 8
    payload['state'][0]['exp_avg'].fill_(3)
    payload['state'][0]['exp_avg_sq'].fill_(4)
    last = payload['state'][-1]
    if defect == 'not_dict':
        payload = []
    elif defect == 'missing_key':
        del payload['eps']
    elif defect == 'extra_key':
        payload['schema'] = 1
    elif defect == 'betas_list':
        payload['betas'] = list(payload['betas'])
    elif defect == 'betas_mismatch':
        payload['betas'] = (0.8, 0.999)
    elif defect == 'eps_mismatch':
        payload['eps'] = 1e-5
    elif defect == 'decay_mismatch':
        payload['weight_decay'] = 0.1
    elif defect == 'invalid_lr':
        payload['lr'] = -1
    elif defect == 'nonfinite_lr':
        payload['lr'] = math.nan
    elif defect == 'boolean_lr':
        payload['lr'] = True
    elif defect == 'state_tuple':
        payload['state'] = tuple(payload['state'])
    elif defect == 'state_length':
        payload['state'].pop()
    elif defect == 'entry_type':
        payload['state'][-1] = 2
    elif defect == 'entry_missing_key':
        del last['exp_avg_sq']
    elif defect == 'entry_extra_key':
        last['extra'] = 2
    elif defect == 'step_zero':
        last['step'] = 0
    elif defect == 'step_boolean':
        last['step'] = True
    elif defect == 'step_float':
        last['step'] = 1.5
    elif defect == 'moment_type':
        last['exp_avg'] = 0.0
    elif defect == 'moment_shape':
        last['exp_avg'] = torch.ones(1)
    elif defect == 'moment_dtype':
        last['exp_avg'] = last['exp_avg'].double()
    elif defect == 'moment_device':
        last['exp_avg'] = torch.empty((), device='meta')
    elif defect == 'moment_sparse':
        last['exp_avg'] = last['exp_avg'].to_sparse()
    elif defect == 'moment_nan':
        last['exp_avg'].fill_(math.nan)
    elif defect == 'moment_inf':
        last['exp_avg_sq'].fill_(math.inf)
    elif defect == 'negative_variance':
        last['exp_avg_sq'].fill_(-1)
    with pytest.raises((ValueError, TypeError)):
        optimizer.load_state_dict(payload)
    assert optimizer.state is old_state
    assert_optimizer_snapshot_equal(optimizer.state_dict(), before)
    for parameter, value, gradient in zip(optimizer.parameters, parameters, gradients):
        assert torch.equal(parameter, value)
        assert parameter.grad is gradient


def test_optimizer_capture_and_load_do_not_consume_rng():
    import random
    import numpy as np

    optimizer = populated_optimizer()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    optimizer.load_state_dict(optimizer.state_dict())
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
