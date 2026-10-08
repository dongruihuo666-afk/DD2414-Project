#!/usr/bin/env python3
"""Atomic training checkpoints and a resumable deterministic sampler."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


CHECKPOINT_SCHEMA_VERSION = 1


def _canonical_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


class StatefulShuffleSampler(Sampler):
    """Deterministic epoch shuffling with an explicit next-sample position.

    This sampler is intended for ``num_workers=0``. Multiprocess prefetch can
    advance a sampler beyond the last optimizer update and would need a separate
    acknowledgement protocol before exact resume could be claimed.
    """

    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.position = 0

    def _order(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.data_source), generator=generator).tolist()

    def __iter__(self):
        order = self._order()
        while self.position < len(order):
            index = order[self.position]
            self.position += 1
            yield index
        self.epoch += 1
        self.position = 0

    def __len__(self):
        return len(self.data_source) - self.position

    def state_dict(self):
        return {
            'seed': self.seed,
            'dataset_size': len(self.data_source),
            'epoch': self.epoch,
            'position': self.position,
        }

    def load_state_dict(self, state):
        if int(state['seed']) != self.seed:
            raise ValueError('checkpoint sampler seed does not match current run')
        if int(state['dataset_size']) != len(self.data_source):
            raise ValueError('checkpoint sampler dataset size does not match current run')
        epoch = int(state['epoch'])
        position = int(state['position'])
        if epoch < 0 or not 0 <= position <= len(self.data_source):
            raise ValueError('checkpoint sampler position is invalid')
        self.epoch = epoch
        self.position = position


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    # ``torch.load(map_location='cuda')`` also moves serialized RNG byte tensors;
    # both CPU and CUDA generator setters require their state tensors on CPU.
    torch.set_rng_state(state['torch_cpu'].cpu())
    cuda_state = state.get('torch_cuda')
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('checkpoint contains CUDA RNG state but CUDA is unavailable')
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA device count does not match current host')
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        with temporary.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(path, *, model, optimizer, scaler, sampler,
                             progress, run_config):
    """Atomically save every state needed to continue the next optimizer step."""
    if int(progress['update']) < 0:
        raise ValueError('progress update must be nonnegative')
    sampler_state = sampler.state_dict()
    if int(progress['epoch']) != int(sampler_state['epoch']):
        raise ValueError('progress epoch and sampler epoch disagree')
    payload = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'run_config': run_config,
        'run_config_sha256': _canonical_digest(run_config),
        'progress': dict(progress),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'sampler': sampler_state,
        'rng': capture_rng_state(),
    }
    _atomic_torch_save(payload, path)
    return payload['run_config_sha256']


def save_periodic_checkpoint(path, *, every_updates, force=False, model,
                             optimizer, scaler, sampler, progress, run_config):
    """Save ``latest`` at a fixed update interval or at an explicit boundary."""
    every_updates = int(every_updates)
    update = int(progress['update'])
    if every_updates < 1:
        raise ValueError('checkpoint interval must be positive')
    if update < 0:
        raise ValueError('progress update must be nonnegative')
    if not force and (update == 0 or update % every_updates != 0):
        return None
    return save_training_checkpoint(
        path, model=model, optimizer=optimizer, scaler=scaler, sampler=sampler,
        progress=progress, run_config=run_config,
    )


def load_training_checkpoint(path, *, model, optimizer, scaler, sampler,
                             expected_run_config, map_location):
    """Load and validate a local checkpoint, then restore RNG last."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError('unsupported training checkpoint schema')
    expected_digest = _canonical_digest(expected_run_config)
    if payload.get('run_config_sha256') != expected_digest:
        raise ValueError('checkpoint run configuration does not match current run')
    if _canonical_digest(payload.get('run_config')) != expected_digest:
        raise ValueError('checkpoint run configuration is internally inconsistent')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scaler.load_state_dict(payload['scaler'])
    sampler.load_state_dict(payload['sampler'])
    progress = dict(payload['progress'])
    if int(progress['epoch']) != sampler.epoch:
        raise ValueError('checkpoint progress and sampler epoch disagree')
    restore_rng_state(payload['rng'])
    return progress
