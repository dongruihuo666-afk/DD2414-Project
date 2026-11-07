#!/usr/bin/env python3
"""Preview or run one of TWO random-student depth studies. Never submits jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from depth_study import _run_child

ROOT = Path(__file__).resolve().parents[1]
TREATMENTS = ('dino-only', 'dino-metric-depth-bev')
METRIC_SHA256 = '9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb'


def load_config(path):
    config = json.loads(Path(path).read_text())
    for name in ['data_root', 'run_root', 'env_prefix', 'bevcar_source',
                 'metric_teacher_source', 'metric_teacher_weights']:
        if not Path(config[name]).is_absolute():
            raise ValueError(name + ' must be absolute')
    if config['node'] not in ('arwen', 'eowyn', 'galadriel') or config['gpus_per_experiment'] not in (2, 5, 10):
        raise ValueError('Select a staged Ampere node and 2/5/10 GPUs; never pad/drop full data')
    if config['workers_per_rank'] not in (0, 1, 2) or config['metric_teacher_sha256'] != METRIC_SHA256:
        raise ValueError('Changed loader or unpinned metric teacher')
    if any(key in config for key in ('camera_checkpoint', 'parent_checkpoint', 'parent_sha256')):
        raise ValueError('This random-student study must not inherit pretrained student assets')
    return config


def treatment_plan(config, treatment):
    if treatment not in TREATMENTS:
        raise ValueError('Unknown treatment')
    world = config['gpus_per_experiment']
    python = str(Path(config['env_prefix'])/'bin/python')
    directory = Path(config['run_root'])/treatment
    common = [str(ROOT/'scripts/train_multisweep_teacher.py'),
        '--training-profile', 'random-metric-bev', '--treatment', treatment,
        '--data-root', config['data_root'], '--bevcar-source', config['bevcar_source'],
        '--manifest', str(ROOT/'configs/radar_scaling_manifest_seed125.json'),
        '--image-resolution', '448', '800', '--teacher-resolution', '448', '798',
        '--depth-teacher-resolution', '448', '798', '--nsweeps', '10', '--depth-mode', 'metric-bev',
        '--velocity-mode', 'full', '--batch-size', '1', '--precision', 'bf16', '--seed', '125',
        '--num-workers', str(config['workers_per_rank']), '--camera-lr', '0.0002',
        '--learning-rate', '0.0002', '--image-weight', '0.2', '--bev-weight', '0.8',
        '--motion-weight', '0', '--depth-weight', '0', '--relative-depth-weight', '0',
        '--metric-depth-weight', '0.05', '--metric-depth-sigma', '2',
        '--train-samples', '28130', '--val-samples', '6019']
    teacher = []
    if treatment == 'dino-metric-depth-bev':
        teacher = ['--metric-teacher-source', config['metric_teacher_source'],
            '--metric-teacher-weights', config['metric_teacher_weights'],
            '--metric-teacher-sha256', config['metric_teacher_sha256']]
    pretrain = [python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        f'--nproc_per_node={world}', *common, *teacher, '--distributed', '--mode', 'pretrain',
        '--epochs', '50', '--run-dir', str(directory/'pretrain')]
    downstream = []
    for mode, location, epochs, name in [
            ('frozen-probe', 'camera_bev', 10, 'frozen_camera_bev_probe'),
            ('frozen-probe', 'fused_bev', 10, 'frozen_fused_bev_probe'),
            ('finetune', 'fused_bev', 5, 'end_to_end_segmentation')]:
        downstream.append(dict(name=name, gpus=1, epochs=epochs,
            argv=[python, *common, '--mode', mode, '--epochs', str(epochs),
                '--probe-location', location, '--checkpoint', str(directory/'pretrain/checkpoints/latest.pt'),
                '--run-dir', str(directory/name)]))
    return dict(treatment=treatment, pretrain=dict(argv=pretrain, gpus=world, epochs=50,
        global_batch_size=world, updates_per_epoch=28130//world, run_dir=str(directory/'pretrain'),
        total_training_sample_visits=1406500, validation_samples_per_epoch=6019), downstream=downstream)


def require_allocation(config, environment, hostname):
    world = config['gpus_per_experiment']
    if (not environment.get('SLURM_JOB_ID') or hostname.split('.')[0] != config['node']
            or int(environment.get('SLURM_JOB_NUM_NODES', '0')) != 1
            or int(environment.get('SLURM_CPUS_PER_TASK', '0')) < 4*world
            or not environment.get('SLURM_JOB_GPUS')):
        raise RuntimeError('Explicit single-node Slurm GPU allocation on the configured data node and four CPUs per rank required')


def require_pilot(config, commit):
    report = json.loads((Path(config['run_root'])/'random_depth_pilot_passed.json').read_text())
    if (report.get('status') != 'passed' or report.get('settings') != config
            or report.get('code_commit') != commit
            or set(report.get('treatments', {})) != set(TREATMENTS)):
        raise ValueError('Both matching real-model CUDA/resume pilots must pass before formal execution')
    for treatment in TREATMENTS:
        result = report['treatments'][treatment]
        if (result.get('status') != 'passed' or result.get('resumed_update') != 3
                or result.get('world_size') != config['gpus_per_experiment']):
            raise ValueError('Incomplete CUDA pilot: ' + treatment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--treatment', choices=TREATMENTS)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--require-pilot', action='store_true', help='Formal execution gate for both CUDA/resume pilots')
    parser.add_argument('--stop-after-updates', type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stop_after_updates < 0:
        parser.error('--stop-after-updates must be nonnegative')
    if args.execute:
        if args.treatment is None:
            parser.error('--execute requires one treatment')
        require_allocation(config, os.environ, socket.gethostname())
        audit_path = Path(config['run_root'])/'random_depth_assets_audit.json'
        audit = json.loads(audit_path.read_text())
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if audit.get('status') != 'passed' or audit.get('settings') != config or audit.get('code_commit') != commit:
            raise ValueError('Fresh allocated-CPU asset/split audit required for this exact commit/config')
        if args.require_pilot:
            require_pilot(config, commit)
        task = treatment_plan(config, args.treatment)['pretrain']
        directory = Path(task['run_dir'])
        directory.mkdir(parents=True, exist_ok=True)
        with (directory/'.launcher.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            return _run_child(task, args.resume, args.stop_after_updates)
    world = config['gpus_per_experiment']
    plans = [treatment_plan(config, t) for t in ([args.treatment] if args.treatment else TREATMENTS)]
    print(json.dumps(dict(scope='PREVIEW; no submission or allocation',
        resources=dict(node=config['node'], nodes=1, ntasks=1, gpus=world, cpus_per_task=4*world,
                       memory_gib=16*world, time_limit='47:00:00'), treatments=plans), indent=2))
    for plan in plans:
        print('\n# ' + plan['treatment'] + '\n' + shlex.join(plan['pretrain']['argv']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
