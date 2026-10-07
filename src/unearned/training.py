"""CPU fp32 next-token training and trusted local checkpoint/resume."""

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from unearned.model import TransformerLM
from unearned.optim import AdamW, WarmupCosineSchedule
from unearned.tokenizer import BPETokenizer


@dataclass(frozen=True)
class TrainingConfig:
    """Training scalars use built-in numbers to keep checkpoint metadata primitive."""

    batch_size: int
    sequence_length: int
    peak_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    def __post_init__(self):
        for name in ('batch_size', 'sequence_length'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.betas) is not tuple:
            raise TypeError('betas must be an immutable tuple')
        for name in ('peak_lr', 'min_lr', 'eps', 'weight_decay'):
            if type(getattr(self, name)) not in (int, float):
                raise TypeError(f'{name} must be a built-in int or float')
        if any(type(beta) not in (int, float) for beta in self.betas):
            raise TypeError('betas must contain built-in int or float values')
        WarmupCosineSchedule(self.peak_lr, self.min_lr, self.warmup_steps, self.total_steps)


def _json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _provenance() -> dict:
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    source_names = ('__init__.py', 'model.py', 'optim.py', 'training.py',
                    'tokenizer/__init__.py', 'tokenizer/bpe.py')
    try:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=False)
        revision = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        revision = None
    lock = root / 'pylock.toml'
    return {
        'source_revision': revision,
        'source_sha256': {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                          for name in source_names},
        'lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
        'runtime': {
            'python': platform.python_version(), 'platform': platform.platform(),
            'machine': platform.machine(), 'torch': str(torch.__version__), 'numpy': np.__version__,
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'default_dtype': str(torch.get_default_dtype()), 'device': 'cpu', 'dtype': 'torch.float32',
            'cuda_initialized': torch.cuda.is_initialized(),
        },
    }


def _capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
                  numpy_state[3], numpy_state[4]],
        'torch_cpu': torch.get_rng_state().clone(),
        'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [],
    }


def _numpy_state(values):
    return (values[0], np.array(values[1], dtype=np.uint32), values[2], values[3], values[4])


def _checked_rng(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.keys() != {'python', 'numpy', 'torch_cpu', 'cuda'}:
        raise ValueError('checkpoint RNG keys do not match')
    state = copy.deepcopy(payload)
    python_state = state['python']
    if not isinstance(python_state, tuple) or len(python_state) != 3:
        raise ValueError('invalid Python RNG state')
    cached = python_state[2]
    if cached is not None:
        if type(cached) not in (int, float):
            raise ValueError('Python cached Gaussian must be finite or None')
        try:
            finite = math.isfinite(cached)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError('Python cached Gaussian must be finite or None')
    random.Random(0).setstate(python_state)
    numpy_state = state['numpy']
    if (not isinstance(numpy_state, list) or len(numpy_state) != 5
            or numpy_state[0] != 'MT19937' or not isinstance(numpy_state[1], list)
            or len(numpy_state[1]) != 624
            or any(type(value) is not int or not 0 <= value < 2**32 for value in numpy_state[1])
            or type(numpy_state[2]) is not int or not 0 <= numpy_state[2] <= 624
            or type(numpy_state[3]) is not int or numpy_state[3] not in (0, 1)
            or type(numpy_state[4]) not in (int, float) or not math.isfinite(numpy_state[4])):
        raise ValueError('invalid NumPy RNG state')
    np.random.RandomState(0).set_state(_numpy_state(numpy_state))
    cpu = state['torch_cpu']
    if (not isinstance(cpu, torch.Tensor) or cpu.device.type != 'cpu'
            or cpu.dtype != torch.uint8 or cpu.layout != torch.strided or cpu.ndim != 1):
        raise ValueError('invalid torch CPU RNG state')
    torch.Generator(device='cpu').set_state(cpu)
    cuda = state['cuda']
    current_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    if not isinstance(cuda, list) or len(cuda) != len(current_cuda):
        raise ValueError('CUDA RNG device inventory does not match')
    for saved, current in zip(cuda, current_cuda):
        if (not isinstance(saved, torch.Tensor) or saved.device.type != 'cpu'
                or saved.dtype != torch.uint8 or saved.layout != torch.strided
                or saved.shape != current.shape):
            raise ValueError('invalid CUDA RNG state')
    return state


def _restore_rng(state: dict):
    random.setstate(state['python'])
    np.random.set_state(_numpy_state(state['numpy']))
    torch.set_rng_state(state['torch_cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class Trainer:
    """Own one CPU fp32 model's updates over a single in-memory byte corpus.

    The caller supplies any document separators; no BOS/EOS is added. Each
    batch consumes B*T consecutive transitions, with one overlapping target
    token. Only full batches are used; an incomplete tail is dropped, and
    wrapping never creates a corpus-end-to-start target. Positions reset in
    each row. There is no shuffle, padding, dropout, or gradient accumulation.

    This object exclusively owns model/optimizer mutation and global RNG
    capture/restore on one thread. Checkpoints exist only between successful
    updates, with gradients cleared. An update error blocks further training
    and capture until a valid checkpoint is restored; partial updates are not
    rolled back. Only trusted, locally-created checkpoints may be loaded.
    """

    def __init__(self, model: TransformerLM, corpus: bytes, tokenizer: BPETokenizer,
                 config: TrainingConfig):
        if type(model) is not TransformerLM or not isinstance(corpus, bytes):
            raise TypeError('training requires an owned TransformerLM and a bytes corpus')
        if config.sequence_length > model.config.max_seq_len:
            raise ValueError('training sequence length exceeds model context')
        if model.config.vocab_size != len(tokenizer.vocab):
            raise ValueError('model vocabulary must match tokenizer')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 or not p.requires_grad
               for p in model.parameters()):
            raise ValueError('training requires CPU fp32 trainable parameters')
        self._tokens = tuple(tokenizer.encode(corpus))
        self._batch_tokens = config.batch_size * config.sequence_length
        self._batches = (len(self._tokens) - 1) // self._batch_tokens
        if self._batches < 1:
            raise ValueError('corpus needs at least batch_size * sequence_length + 1 tokens')
        self.config, self.model = config, model
        self.schedule = WarmupCosineSchedule(config.peak_lr, config.min_lr,
                                             config.warmup_steps, config.total_steps)
        self.optimizer = AdamW(model.parameters(), lr=self.schedule(0), betas=config.betas,
                               eps=config.eps, weight_decay=config.weight_decay)
        self.completed_steps = 0
        self.data_cursor = 0
        self._failed = False
        self._identity = {
            'model_config': asdict(model.config), 'training_config': asdict(config),
            'parameter_names': [name for name, _ in model.named_parameters()],
            'corpus_sha256': hashlib.sha256(corpus).hexdigest(),
            'tokenizer_sha256': _json_hash({'merges': tokenizer.merges,
                                           'special_tokens': [value.hex() for value in tokenizer.special_tokens],
                                           'vocab': [value.hex() for value in tokenizer.vocab]}),
            'tokens_sha256': _json_hash(self._tokens), 'token_count': len(self._tokens),
            'batching': 'contiguous-full-batches-v1', 'full_batches': self._batches,
        }
        self._provenance = _provenance()
        self.model.train()
        self.optimizer.zero_grad()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return overlapping [B,T] views of a fresh int64 CPU window; do not advance."""
        window = torch.tensor(self._tokens[self.data_cursor:self.data_cursor + self._batch_tokens + 1],
                              dtype=torch.int64, device='cpu')
        shape = (self.config.batch_size, self.config.sequence_length)
        return window[:-1].reshape(shape), window[1:].reshape(shape)

    def step(self) -> float:
        """Return mean pre-update next-token cross entropy, then commit one update."""
        if self._failed:
            raise RuntimeError('restore a valid checkpoint after a failed update')
        if self.completed_steps >= self.config.total_steps:
            raise ValueError('configured training updates are complete')
        previous_lr = self.optimizer.lr
        # Stay unusable until every part of the update has completed.
        self._failed = True
        try:
            self.optimizer.zero_grad()
            inputs, targets = self.next_batch()
            loss = F.cross_entropy(self.model(inputs).flatten(0, 1), targets.flatten())
            if not torch.isfinite(loss):
                raise ValueError('training loss must be finite')
            loss.backward()
            self.optimizer.lr = self.schedule(self.completed_steps)
            self.optimizer.step()
            tensors = list(self.model.parameters())
            tensors += [moment for state in self.optimizer.state.values()
                        for moment in (state.exp_avg, state.exp_avg_sq)]
            if any(not torch.isfinite(value).all() for value in tensors):
                raise ValueError('optimizer update produced nonfinite state')
            self.optimizer.zero_grad()
            self.completed_steps += 1
            self.data_cursor = (self.completed_steps % self._batches) * self._batch_tokens
            result = loss.item()
            self._failed = False
            return result
        except BaseException:
            self._failed = True
            try:
                self.optimizer.lr = previous_lr
                self.optimizer.zero_grad()
            except BaseException:
                # Cleanup is best-effort; preserve the original failure and require restore.
                pass
            raise

    def run(self, updates: int) -> list[float]:
        if self._failed:
            raise RuntimeError('restore a valid checkpoint after a failed update')
        if type(updates) is not int or not 0 <= updates <= self.config.total_steps - self.completed_steps:
            raise ValueError('updates must be an integer within the remaining training budget')
        return [self.step() for _ in range(updates)]

    def state_dict(self) -> dict:
        """Capture independent tensors and primitive metadata without consuming RNG."""
        if self._failed or any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('checkpoint requires a successful update boundary and cleared gradients')
        if _provenance() != self._provenance:
            raise ValueError('source, lock or runtime changed since trainer construction')
        return {
            'format_version': 1, 'identity': copy.deepcopy(self._identity),
            'provenance': copy.deepcopy(self._provenance),
            'model': {name: value.detach().clone() for name, value in self.model.state_dict().items()},
            'optimizer': self.optimizer.state_dict(), 'completed_steps': self.completed_steps,
            'data_cursor': self.data_cursor, 'rng': _capture_rng(),
        }

    def load_state_dict(self, payload: dict):
        """Preflight identities and all state before changing live objects or RNG."""
        required = {'format_version', 'identity', 'provenance', 'model', 'optimizer',
                    'completed_steps', 'data_cursor', 'rng'}
        if not isinstance(payload, dict) or payload.keys() != required:
            raise ValueError('checkpoint keys do not match')
        if type(payload['format_version']) is not int or payload['format_version'] != 1:
            raise ValueError('unsupported checkpoint version')
        if payload['identity'] != self._identity:
            raise ValueError('checkpoint corpus, tokenizer, configuration or parameter identity mismatch')
        if payload['provenance'] != self._provenance or _provenance() != self._provenance:
            raise ValueError('checkpoint source, lock or runtime mismatch')
        completed = payload['completed_steps']
        if type(completed) is not int or not 0 <= completed <= self.config.total_steps:
            raise ValueError('invalid completed-update count')
        cursor = payload['data_cursor']
        if type(cursor) is not int or cursor != (completed % self._batches) * self._batch_tokens:
            raise ValueError('data cursor does not match completed-update count')
        live = self.model.state_dict()
        saved = payload['model']
        if not isinstance(saved, dict) or saved.keys() != live.keys():
            raise ValueError('model state keys do not match')
        weights = {}
        for name, parameter in live.items():
            value = saved[name]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or value.device != parameter.device or not torch.isfinite(value).all()):
                raise ValueError('model tensors must match shape, dtype, device and be finite')
            weights[name] = value.detach().clone()
        optimizer = AdamW(self.model.parameters(), lr=self.schedule(0), betas=self.config.betas,
                          eps=self.config.eps, weight_decay=self.config.weight_decay)
        optimizer.load_state_dict(payload['optimizer'])
        if optimizer.lr != self.schedule(max(0, completed - 1)):
            raise ValueError('saved learning rate must be the last-used rate, or rate(0) before training')
        for parameter in optimizer.parameters:
            state = optimizer.state.get(parameter)
            if (completed == 0 and state is not None) or (completed > 0 and (state is None or state.step != completed)):
                raise ValueError('optimizer counts do not match completed updates')
        rng = _checked_rng(payload['rng'])
        self.model.load_state_dict(weights, strict=True)
        self.optimizer = optimizer
        self.completed_steps, self.data_cursor = completed, cursor
        self.optimizer.zero_grad()
        _restore_rng(rng)
        self._failed = False

    def save_checkpoint(self, path: str | Path):
        """Durably publish to a new path; never replace an existing checkpoint."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        snapshot = self.state_dict()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(snapshot, stream)
                stream.flush()
                os.fsync(stream.fileno())
            # A same-directory hard link publishes a complete file without overwriting a racing writer.
            os.link(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink()

    def load_checkpoint(self, path: str | Path):
        """Load only trusted local files; never fall back to unrestricted pickle."""
        self.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
