#!/usr/bin/env python3
"""One bounded allocation: real wrappers, two updates then fresh-process resume.

Both treatments must pass before the shared admission report is published.
Never submits jobs, changes the scientific budget, or falls back after OOM.
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from control import digest, git_commit, read, write
from depth_pilot import verify_transition
from random_depth_study import ROOT, TREATMENTS, load_config, require_allocation, treatment_plan


def inspect_checkpoint(path, update, world, treatment):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config, progress = payload['run_config'], payload['progress']
    if (payload.get('checkpoint_kind') != 'ddp_full_model' or payload['world_size'] != world
            or progress['update'] != update or progress['samples_seen'] != update*world
            or progress['epoch'] != 0 or config['train_samples'] != 28130
            or config['val_samples'] != 6019 or config['epochs'] != 50
            or config.get('training_profile') != 'random-metric-bev'
            or config.get('treatment') != treatment
            or config.get('random_student_initialization', {}).get('kind') != 'random'
            or config.get('official_camera_initialization') is not None
            or config.get('pretraining_continuation') is not None):
        raise ValueError('Unexpected random-student full-manifest pilot checkpoint')
    states = payload['distributed_state']
    if len(states) != world:
        raise ValueError('Missing distributed rank state')
    for rank, state in enumerate(states):
        if (state['rank'] != rank or state['sampler']['rank'] != rank
                or state['sampler']['world_size'] != world
                or state['sampler']['position'] != update or state['sampler']['epoch'] != 0
                or state['rng']['torch_cuda'] is None):
            raise ValueError('Missing/incorrect per-rank CUDA RNG or acknowledged cursor')
    for values in [payload['model'], *[state['buffers'] for state in states]]:
        for name, value in values.items():
            if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                raise ValueError('Non-finite checkpoint tensor: ' + name)
    if not payload['optimizer']['state']:
        raise ValueError('No committed optimizer state')
    for values in payload['optimizer']['state'].values():
        for value in values.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise ValueError('Non-finite optimizer state')
    return payload


def run_treatment(config, config_path, treatment):
    task = treatment_plan(config, treatment)['pretrain']
    directory = Path(task['run_dir'])
    if (directory/'config.json').exists():
        raise FileExistsError('Admission needs a fresh treatment output; never overwrite/replay a prior pilot')
    command = ['bash', str(ROOT/'cluster_rpl/random_depth_job.sh'), str(config_path), treatment]
    started = time.monotonic()
    print('PILOT_START '+treatment, flush=True)
    result = subprocess.run([*command, '--stop-after-updates', '2'], cwd=ROOT)
    if result.returncode != 75:
        raise RuntimeError(f'{treatment}: update-two exit {result.returncode}; expected intentional partial 75')
    world = config['gpus_per_experiment']
    before = inspect_checkpoint(directory/'checkpoints/latest.pt', 2, world, treatment)
    print('PILOT_RESUME_NEW_PROCESS '+treatment, flush=True)
    result = subprocess.run([*command, '--resume', '--stop-after-updates', '3'], cwd=ROOT)
    if result.returncode != 75:
        raise RuntimeError(f'{treatment}: resume exit {result.returncode}; expected intentional partial 75')
    after = inspect_checkpoint(directory/'checkpoints/latest.pt', 3, world, treatment)
    metrics = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    changed = verify_transition(before, after, metrics)
    status = read(directory/'status.json')
    if not math.isfinite(status['peak_allocated_gib']):
        raise ValueError('Invalid peak memory telemetry')
    result = dict(status='passed', resumed_update=3, world_size=world,
        samples_seen=3*world, changed_float_tensors=changed,
        initial_student_sha256=after['run_config']['random_student_initialization']['student_state_sha256'],
        checkpoint_sha256=digest(directory/'checkpoints/latest.pt'),
        peak_allocated_gib_rank_zero=status['peak_allocated_gib'], latest=status['latest'],
        elapsed_seconds=time.monotonic()-started,
        scope='Real full-model finite updates and new-process resume, not convergence or semantic accuracy')
    write(Path(config['run_root'])/('admission_'+treatment+'.json'), result)
    print('PILOT_TREATMENT_PASSED '+json.dumps(dict(treatment=treatment, **result)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    require_allocation(config, os.environ, socket.gethostname())
    audit = read(Path(config['run_root'])/'random_depth_assets_audit.json')
    if audit.get('status') != 'passed' or audit.get('settings') != config or audit.get('code_commit') != git_commit():
        raise ValueError('Fresh same-commit/config CPU admission required')
    report_path = Path(config['run_root'])/'random_depth_pilot_passed.json'
    if report_path.exists():
        raise FileExistsError('Pilot report already exists; inspect it instead of duplicating admission')
    results = {treatment:run_treatment(config, config_path, treatment) for treatment in TREATMENTS}
    if len({value['initial_student_sha256'] for value in results.values()}) != 1:
        raise ValueError('The two treatments did not start from identical random student tensors')
    report = dict(status='passed', settings=config, code_commit=git_commit(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), treatments=results)
    write(report_path, report)
    print('BOTH_RANDOM_DEPTH_PILOTS_PASSED', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
