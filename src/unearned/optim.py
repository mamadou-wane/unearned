"""Owned AdamW updates and a stateless warmup/cosine learning-rate schedule."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be a real number')
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f'{name} must be finite') from None
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


@dataclass
class AdamWState:
    """Moments have the parameter's shape/dtype/device; step counts present gradients."""

    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor


class AdamW:
    """Update a fixed, flat list of dense real fp32/fp64 parameters in place.

    State is allocated only for parameters with gradients. None skips decay,
    moments and step count; a zero gradient does not. Moments own their storage.
    Only lr is mutable through this API. Construct after choosing the model's
    dtype/device, and keep parameter shapes fixed. No groups, sparse updates,
    closures, differentiable updates or mixed-precision state are supported.
    """

    def __init__(
        self, parameters: Iterable[nn.Parameter], lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError('parameters must not be empty')
        if any(not isinstance(parameter, nn.Parameter) for parameter in self.parameters):
            raise TypeError('parameters must be a flat iterable of nn.Parameter objects')
        if len({id(parameter) for parameter in self.parameters}) != len(self.parameters):
            raise ValueError('parameters must be distinct')
        if any(parameter.layout != torch.strided or parameter.dtype not in (torch.float32, torch.float64)
               for parameter in self.parameters):
            raise ValueError('parameters must be dense real fp32/fp64 tensors')
        self.lr = lr
        if len(betas) != 2:
            raise ValueError('betas must contain two coefficients')
        self._betas = tuple(_finite_real(value, 'beta') for value in betas)
        if any(not 0 <= beta < 1 for beta in self._betas):
            raise ValueError('betas must be in [0, 1)')
        self._eps = _finite_real(eps, 'eps')
        if any(not torch.finfo(parameter.dtype).tiny <= self._eps <= torch.finfo(parameter.dtype).max
               for parameter in self.parameters):
            raise ValueError('eps must be positive and normal in every parameter dtype')
        self._weight_decay = _finite_real(weight_decay, 'weight_decay')
        if self._weight_decay < 0:
            raise ValueError('weight_decay must be nonnegative')
        self.state: dict[nn.Parameter, AdamWState] = {}

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float):
        value = _finite_real(value, 'lr')
        if value < 0:
            raise ValueError('lr must be nonnegative')
        self._lr = value

    def zero_grad(self):
        """Discard gradients without changing weights or optimizer state."""
        for parameter in self.parameters:
            parameter.grad = None

    def state_dict(self) -> dict:
        """Return an independent snapshot indexed by the fixed parameter order."""
        states = []
        for parameter in self.parameters:
            state = self.state.get(parameter)
            states.append(None if state is None else {
                'step': state.step,
                'exp_avg': state.exp_avg.detach().clone(),
                'exp_avg_sq': state.exp_avg_sq.detach().clone(),
            })
        return {'lr': self.lr, 'betas': self._betas, 'eps': self._eps,
                'weight_decay': self._weight_decay, 'state': states}

    def load_state_dict(self, payload: dict):
        """Validate and copy a snapshot before replacing state; leave weights/gradients alone.

        Parameter identity is positional. The caller must separately verify model
        configuration and parameter names/order before loading a checkpoint.
        """
        if not isinstance(payload, dict):
            raise TypeError('optimizer snapshot must be a dictionary')
        if payload.keys() != {'lr', 'betas', 'eps', 'weight_decay', 'state'}:
            raise ValueError('optimizer snapshot keys do not match')
        betas = payload['betas']
        if not isinstance(betas, tuple) or len(betas) != 2:
            raise TypeError('snapshot betas must be a pair in a tuple')
        betas = tuple(_finite_real(beta, 'beta') for beta in betas)
        eps = _finite_real(payload['eps'], 'eps')
        decay = _finite_real(payload['weight_decay'], 'weight_decay')
        if (betas, eps, decay) != (self._betas, self._eps, self._weight_decay):
            raise ValueError('optimizer fixed hyperparameters do not match')
        lr = _finite_real(payload['lr'], 'lr')
        if lr < 0:
            raise ValueError('lr must be nonnegative')
        entries = payload['state']
        if not isinstance(entries, list):
            raise TypeError('optimizer state must be a list in parameter order')
        if len(entries) != len(self.parameters):
            raise ValueError('optimizer state length does not match parameters')
        restored = {}
        for parameter, entry in zip(self.parameters, entries):
            if entry is None:
                continue
            if not isinstance(entry, dict):
                raise TypeError('initialized optimizer state must be a dictionary')
            if entry.keys() != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('optimizer state entry keys do not match')
            if type(entry['step']) is not int or entry['step'] <= 0:
                raise ValueError('initialized optimizer step must be a positive integer')
            moments = []
            for name in ('exp_avg', 'exp_avg_sq'):
                moment = entry[name]
                if not isinstance(moment, Tensor):
                    raise TypeError('optimizer moments must be tensors')
                if (moment.layout != torch.strided or moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype or moment.device != parameter.device):
                    raise ValueError('optimizer moments must match parameter shape, dtype and device')
                if not torch.isfinite(moment).all():
                    raise ValueError('optimizer moments must be finite')
                if name == 'exp_avg_sq' and (moment < 0).any():
                    raise ValueError('second moments must be nonnegative')
                moments.append(moment.detach().clone())
            restored[parameter] = AdamWState(entry['step'], *moments)
        self.state = restored
        self._lr = lr

    @torch.no_grad()
    def step(self):
        """Apply one update; reject sparse/nonfinite gradients before any mutation."""
        decay_factor = 1 - self.lr * self._weight_decay
        if not math.isfinite(decay_factor):
            raise ValueError('lr * weight_decay must be finite')
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.layout != torch.strided or not torch.isfinite(gradient).all():
                raise ValueError('gradients must be dense and finite')
            if parameter in self.state:
                state = self.state[parameter]
                for moment in (state.exp_avg, state.exp_avg_sq):
                    if (moment.shape != parameter.shape or moment.dtype != parameter.dtype
                            or moment.device != parameter.device):
                        raise ValueError('parameter shape, dtype and device must match its state')
        beta1, beta2 = self._betas
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if parameter not in self.state:
                self.state[parameter] = AdamWState(
                    step=0, exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter)
                )
            state = self.state[parameter]
            state.step += 1
            state.exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
            state.exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            corrected_mean = state.exp_avg / (1 - beta1**state.step)
            corrected_variance = state.exp_avg_sq / (1 - beta2**state.step)
            denominator = corrected_variance.sqrt().add_(self._eps)
            # Decay uses the pre-update weight and never enters either moment.
            parameter.mul_(decay_factor)
            parameter.addcdiv_(corrected_mean, denominator, value=-self.lr)


@dataclass(frozen=True)
class WarmupCosineSchedule:
    """Learning rate at a completed-update count, with no internal counter.

    For W warmup steps: rate(0)=0 and rate(W)=peak. Without warmup, rate(0)
    is peak. Cosine decay reaches min_lr at total_steps and stays there.
    Before each update use optimizer.lr = schedule(completed_steps), then
    advance that global count after optimizer.step(). Thus the update at
    count total_steps-1 is above the floor when peak_lr > min_lr.
    """

    peak_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int

    def __post_init__(self):
        peak = _finite_real(self.peak_lr, 'peak_lr')
        floor = _finite_real(self.min_lr, 'min_lr')
        if not 0 <= floor <= peak:
            raise ValueError('rates must satisfy 0 <= min_lr <= peak_lr')
        for name in ('warmup_steps', 'total_steps'):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f'{name} must be an integer')
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError('steps must satisfy 0 <= warmup_steps < total_steps')

    def __call__(self, completed_steps: int) -> float:
        if type(completed_steps) is not int:
            raise TypeError('completed_steps must be an integer')
        if completed_steps < 0:
            raise ValueError('completed_steps must be nonnegative')
        if completed_steps < self.warmup_steps:
            return self.peak_lr * completed_steps / self.warmup_steps
        if completed_steps >= self.total_steps:
            return float(self.min_lr)
        progress = (completed_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
