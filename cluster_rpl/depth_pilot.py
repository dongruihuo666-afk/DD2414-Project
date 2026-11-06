#!/usr/bin/env python3
"""Bounded real-model admission: stop at update two, restart and stop at three.

Run ONE treatment inside an authorized two-GPU Slurm allocation. This script
does not submit jobs, change settings after OOM, or start formal training.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

from control import digest, git_commit, read, write
from depth_study import load_config, require_allocation, run_pretrain, treatment_plan


def inspect_checkpoint(path, update):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config, progress = payload['run_config'], payload['progress']
    if (payload.get('checkpoint_kind') != 'ddp_full_model' or payload['world_size'] != 2
            or progress['update'] != update or progress['samples_seen'] != update * 2
            or progress['epoch'] != 0 or config['train_samples'] != 28130
            or config['val_samples'] != 6019 or config['epochs'] != 5):
        raise ValueError('Unexpected full-manifest pilot checkpoint progress/configuration')
    for rank, state in enumerate(payload['distributed_state']):
        if (state['rank'] != rank or state['sampler']['rank'] != rank
                or state['sampler']['position'] != update or state['sampler']['epoch'] != 0
                or state['rng']['torch_cuda'] is None):
            raise ValueError('Missing or incorrect rank-local resume state')
    for values in [payload['model'], *[state['buffers'] for state in payload['distributed_state']]]:
        for name, value in values.items():
            if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                raise ValueError('Non-finite checkpoint tensor: ' + name)
    if not payload['optimizer']['state']:
        raise ValueError('No optimizer update state')
    return payload


def verify_transition(before, after, metrics):
    import torch
    if before['run_config_sha256'] != after['run_config_sha256']:
        raise ValueError('Resume silently changed the scientific configuration')
    changed = sum(not torch.equal(value, after['model'][name])
                  for name, value in before['model'].items() if torch.is_floating_point(value))
    if changed == 0:
        raise ValueError('No model state changed at resumed update three')
    if [row['update'] for row in metrics] != [1, 2, 3]:
        raise ValueError('Resume skipped or replayed a logged optimizer update')
    for row in metrics:
        if not all(math.isfinite(row[key]) for key in ['loss', 'gradient_norm']):
            raise ValueError('Non-finite pilot loss/gradient')
    return changed


def main():
    import os
    import socket
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--treatment', choices=['none', 'radar', 'radar-dav2'], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    require_allocation(config, os.environ, socket.gethostname())
    audit = read(Path(config['run_root']) / 'depth_assets_audit.json')
    if audit['status'] != 'passed' or audit['settings'] != config:
        raise ValueError('Missing/potentially stale full-data depth asset audit')
    task = treatment_plan(config, args.treatment)['pretrain']
    task['argv'] += ['--checkpoint-every', '1', '--print-every', '1']
    directory = Path(task['run_dir'])
    if (directory / 'config.json').exists():
        raise FileExistsError('Pilot requires a fresh treatment output; never overwrite an earlier pilot/run')
    started = time.monotonic()
    code = run_pretrain(config, task, stop_after_updates=2)
    if code != 75:
        raise RuntimeError(f'First pilot worker exit {code}, expected intentional partial 75')
    checkpoint = directory / 'checkpoints/latest.pt'
    saved = directory / 'checkpoints/pilot_update002.pt'
    shutil.copyfile(checkpoint, saved)
    before = inspect_checkpoint(saved, 2)
    print('PILOT_TWO_UPDATES_FINITE; restarting in a fresh torchrun process', flush=True)
    code = run_pretrain(config, task, resume=True, stop_after_updates=3)
    if code != 75:
        raise RuntimeError(f'Resumed pilot exit {code}, expected intentional partial 75')
    after = inspect_checkpoint(checkpoint, 3)
    metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    changed = verify_transition(before, after, metrics)
    status = read(directory / 'status.json')
    report = dict(status='passed', depth_mode=args.treatment, code_commit=git_commit(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
        updates_before_restart=2, updates_after_restart=3, global_samples_seen=6,
        changed_float_model_tensors_at_resumed_update=changed,
        peak_allocated_gib_rank_zero=status['peak_allocated_gib'],
        checkpoint_sha256=digest(checkpoint), latest=status['latest'],
        elapsed_seconds=time.monotonic() - started, run_dir=str(directory),
        scope='Real two-GPU full-model finite updates and new-process state restoration; not exact CUDA numerical equivalence, full validation, throughput convergence or semantic accuracy')
    write(Path(config['run_root']) / f'admission_{args.treatment}.json', report)
    print('DEPTH_PILOT_PASSED\n' + json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
