"""Explicit DDP sampling, rank-local state, and atomic collective checkpoints.

The single-GPU checkpoint format and sampler are intentionally unchanged.
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data import Dataset

from training_checkpoint import _atomic_torch_save, _canonical_digest


class SeededEpochDataset(Dataset):
    """Sample-local CPU RNG makes augmentation independent of worker prefetch.

    Does not touch CUDA RNG or change the wrapped sample/position structure.
    Construct new DataLoader workers after setting each epoch (no persistent
    workers). This is a recorded DDP recipe, not historical RNG equivalence.
    """
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, int(seed), 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        before = (random.getstate(),np.random.get_state(),torch.get_rng_state())
        seed = (self.seed+1000003*self.epoch+7919*int(index)) % 2**32
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(before[0])
            np.random.set_state(before[1])
            torch.set_rng_state(before[2])


class AcknowledgedDistributedSampler(Sampler):
    """Disjoint interleaved shards of one epoch permutation; never pad or drop.

    Equal-length ranks are mandatory. Reject non-divisible datasets instead
    of silently duplicating examples. Cursor advances only after an update.
    """
    def __init__(self, dataset, seed, rank, world_size):
        self.dataset, self.seed = dataset, int(seed)
        self.rank, self.world_size = int(rank), int(world_size)
        if not 0 <= self.rank < self.world_size or len(dataset) % self.world_size:
            raise ValueError('DDP requires valid ranks and equally divisible full data; no padding/dropping')
        self.epoch, self.position = 0, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter(order[self.rank::self.world_size][self.position:])

    def __len__(self):
        return len(self.dataset)//self.world_size-self.position

    def acknowledge(self, count):
        count = int(count)
        if count < 1 or self.position+count > len(self.dataset)//self.world_size:
            raise ValueError('invalid distributed acknowledgement')
        self.position += count
        if self.position == len(self.dataset)//self.world_size:
            self.epoch, self.position = self.epoch+1, 0

    def state_dict(self):
        return dict(seed=self.seed, dataset_size=len(self.dataset), rank=self.rank,
                    world_size=self.world_size, epoch=self.epoch, position=self.position)

    def load_state_dict(self, state):
        for key in ['seed','dataset_size','rank','world_size']:
            if state.get(key) != self.state_dict()[key]:
                raise ValueError(f'Distributed sampler {key} mismatch')
        if state['epoch'] < 0 or not 0 <= state['position'] < len(self.dataset)//self.world_size:
            raise ValueError('Invalid distributed sampler cursor')
        self.epoch, self.position = state['epoch'], state['position']


def local_rng_state(device):
    """Capture only the assigned CUDA device; never initialize the other GPU."""
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_local_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] is not None:
        if device.type != 'cuda':
            raise ValueError('CUDA RNG cannot resume on a CPU process')
        torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device)


def rank_zero_call(function):
    """Run I/O or validation once, propagate ordinary failures to every rank."""
    response = [None]
    if dist.get_rank() == 0:
        try:
            response[0] = dict(ok=True, value=function())
        except Exception as error:
            response[0] = dict(ok=False, error=f'{type(error).__name__}: {error}')
    dist.broadcast_object_list(response, src=0)
    if not response[0]['ok']:
        raise RuntimeError('Rank-zero operation failed: '+response[0]['error'])
    return response[0]['value']


def any_rank(value, device):
    flag = torch.tensor(int(bool(value)), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def mean_scalars(values, device):
    keys = sorted(values)
    packed = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(packed)
    packed /= dist.get_world_size()
    return dict(zip(keys, packed.cpu().tolist()))


def validation_shard(indices, rank, world_size):
    """Uneven validation shards are fine: use unwrapped inference, then sum."""
    if not 0 <= rank < world_size:
        raise ValueError('Invalid validation rank')
    return indices[rank::world_size]


def require_completed_ddp_pretraining(payload):
    """A final training cursor alone is NOT completion without final validation."""
    config, progress, sampler = payload['run_config'], payload['progress'], payload['sampler']
    epochs = config['epochs']
    expected_updates = epochs * config['distributed_training']['updates_per_epoch']
    validation = progress.get('validations', {}).get(str(epochs), {})
    if (config['mode'] != 'pretrain' or progress['epoch'] != epochs
            or sampler['epoch'] != epochs or sampler['position'] != 0
            or progress['update'] != expected_updates
            or progress['samples_seen'] != epochs * config['train_samples']
            or validation.get('samples') != config['val_samples']):
        raise ValueError('DDP downstream requires a completed continuation AND full final validation')


def save_ddp_checkpoint(path, *, model, optimizer, scaler, sampler, progress, config, device):
    """All ranks participate; only rank zero writes; keep every rank's RNG/BN."""
    local = dict(rank=dist.get_rank(), sampler=sampler.state_dict(), rng=local_rng_state(device),
        scaler=scaler.state_dict(),
        buffers={name:value.detach().cpu().clone() for name,value in model.named_buffers()})
    states = [None]*dist.get_world_size()
    dist.all_gather_object(states, local)
    reference = states[0]['sampler']
    for rank, state in enumerate(states):
        if (state['rank'] != rank or state['sampler']['rank'] != rank
                or any(state['sampler'][key] != reference[key]
                       for key in ['seed','dataset_size','world_size','epoch','position'])):
            raise ValueError('Ranks disagree about committed sampler progress')
    if progress['epoch'] != reference['epoch']:
        raise ValueError('DDP progress and sampler epoch disagree')
    def write():
        payload = dict(schema_version=2, checkpoint_kind='ddp_full_model',
            world_size=dist.get_world_size(), run_config=config,
            run_config_sha256=_canonical_digest(config), model=model.state_dict(),
            optimizer=optimizer.state_dict(), scaler=states[0]['scaler'],
            sampler=reference, distributed_state=states, progress=dict(progress))
        _atomic_torch_save(payload, Path(path))
    rank_zero_call(write)


def load_ddp_checkpoint(path, *, model, optimizer, scaler, sampler, config, device):
    """Same-world-size resume only; single-GPU checkpoints are initializers, not resumes."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected = _canonical_digest(config)
    if (payload.get('schema_version') != 2 or payload.get('checkpoint_kind') != 'ddp_full_model'
            or payload.get('world_size') != dist.get_world_size()
            or payload.get('run_config_sha256') != expected
            or _canonical_digest(payload.get('run_config')) != expected):
        raise ValueError('DDP checkpoint schema/world size/configuration mismatch')
    rank = dist.get_rank()
    state = payload['distributed_state'][rank]
    if state['rank'] != rank:
        raise ValueError('Checkpoint rank state mismatch')
    model.load_state_dict(payload['model'], strict=True)
    buffers = dict(model.named_buffers())
    if buffers.keys() != state['buffers'].keys():
        raise ValueError('Checkpoint rank-local buffer keys differ')
    with torch.no_grad():
        for name,value in state['buffers'].items():
            buffers[name].copy_(value.to(buffers[name].device))
    optimizer.load_state_dict(payload['optimizer'])
    scaler.load_state_dict(state['scaler'])
    sampler.load_state_dict(state['sampler'])
    progress = payload['progress']
    if progress['epoch'] != sampler.epoch:
        raise ValueError('DDP checkpoint progress epoch mismatch')
    restore_local_rng(state['rng'], device)
    return progress
