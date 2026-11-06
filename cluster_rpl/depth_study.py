#!/usr/bin/env python3
"""Preview the isolated depth matrix; explicitly run ONE allocated pretraining task.

No SSH, sbatch, cancellation, requeue, environment mutation or automatic queue.
Only stdlib is imported, so previews do not need torch, data or checkpoints.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ('none', 'radar', 'radar-dav2')
PARENT_SHA256 = 'ae9daf1124a03ee5a47f5fac68886334bcdd12dcbe5684c8a64167982c632520'


def load_config(path):
    config = json.loads(Path(path).read_text())
    for name in ['data_root', 'run_root', 'env_prefix', 'camera_checkpoint',
                 'bevcar_source', 'parent_checkpoint']:
        if not Path(config[name]).is_absolute():
            raise ValueError(f'{name} must be an explicit absolute path')
    if config['node'] != 'arwen' or config['parent_sha256'] != PARENT_SHA256:
        raise ValueError('This study pins arwen-local data and the approved velocity-present parent')
    if config['workers_per_rank'] not in [0, 1, 2]:
        raise ValueError('The eight-CPU allocation supports at most two workers per rank')
    return config


def treatment_plan(config, depth_mode):
    if depth_mode not in MODES:
        raise ValueError('Unknown depth treatment')
    directory = Path(config['run_root']) / ('depth_' + depth_mode)
    python = str(Path(config['env_prefix']) / 'bin/python')
    common = [str(ROOT / 'scripts/train_multisweep_teacher.py'),
        '--data-root', config['data_root'], '--bevcar-source', config['bevcar_source'],
        '--manifest', str(ROOT / 'configs/radar_scaling_manifest_seed125.json'),
        '--official-camera-checkpoint', config['camera_checkpoint'], '--allow-source-relocation',
        '--image-resolution', '448', '800', '--teacher-resolution', '448', '798',
        '--depth-teacher-resolution', '448', '798', '--nsweeps', '10', '--depth-mode', depth_mode,
        '--velocity-mode', 'full', '--batch-size', '1', '--precision', 'bf16',
        '--num-workers', str(config['workers_per_rank']), '--seed', '125',
        '--camera-lr', '0.00001', '--learning-rate', '0.0002', '--probe-lr', '0.001',
        '--image-weight', '0.2', '--bev-weight', '0.8', '--motion-weight', '0.5',
        '--depth-weight', '0.2', '--relative-depth-weight', '0.05',
        '--train-samples', '28130', '--val-samples', '6019']
    pretrain = [python, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc_per_node=2', *common, '--distributed', '--mode', 'pretrain', '--epochs', '5',
        '--run-dir', str(directory / 'pretrain'),
        '--pretrain-initial-checkpoint', config['parent_checkpoint'],
        '--pretrain-initial-sha256', config['parent_sha256']]
    downstream = []
    for mode, location, epochs, name in [
            ('finetune', 'fused_bev', 5, 'end_to_end_segmentation'),
            ('frozen-probe', 'camera_bev', 10, 'frozen_camera_bev_probe'),
            ('frozen-probe', 'fused_bev', 10, 'frozen_fused_bev_probe'),
            ('object-motion', 'fused_bev', 5, 'gt_box_velocity_readout')]:
        downstream.append(dict(name=name, gpus=1, epochs=epochs,
            requires_completed=str(directory / 'pretrain/summary.json'),
            argv=[python, *common, '--mode', mode, '--epochs', str(epochs),
                  '--probe-location', location, '--run-dir', str(directory / name),
                  '--checkpoint', str(directory / 'pretrain/checkpoints/latest.pt')]))
    return dict(depth_mode=depth_mode, pretrain=dict(argv=pretrain, gpus=2,
        epochs=5, total_pretraining_exposure_epochs=10, global_batch_size=2,
        updates_per_epoch=14065, run_dir=str(directory / 'pretrain')),
        downstream=downstream)


def require_allocation(config, environment, hostname):
    if (not environment.get('SLURM_JOB_ID') or hostname.split('.')[0] != config['node']
            or int(environment.get('SLURM_JOB_NUM_NODES', '0')) != 1
            or int(environment.get('SLURM_CPUS_PER_TASK', '0')) < 8):
        raise RuntimeError('Run only in an authorized single-node arwen Slurm allocation with eight CPUs')
    if not environment.get('SLURM_JOB_GPUS'):
        raise RuntimeError('Slurm GPU allocation required; the trainer additionally checks exactly two visible GPUs')


def run_pretrain(config, task, resume=False, stop_after_updates=0):
    require_allocation(config, os.environ, socket.gethostname())
    directory = Path(task['run_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_child(task, resume, stop_after_updates)


def _run_child(task, resume, stop_after_updates):
    directory = Path(task['run_dir'])
    if not resume and (directory / 'config.json').exists():
        raise FileExistsError('Output already exists; pass --resume explicitly')
    stop_file = directory / 'REQUEST_CHECKPOINT'
    # Clear only OUR previous signal flag, BEFORE starting the workers. Never
    # erase a new request inside rank zero during potentially slow startup.
    stop_file.unlink(missing_ok=True)
    command = list(task['argv'])
    if resume:
        command.append('--resume')
    if stop_after_updates:
        command += ['--stop-after-updates', str(stop_after_updates)]
    previous = {}
    def request_checkpoint(signum, frame):
        stop_file.touch()
        print(f'SIGNAL {signum}: requested a collective checkpoint at the next safe boundary', flush=True)
    for signum in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        previous[signum] = signal.signal(signum, request_checkpoint)
    environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TORCH_HOME=str(ROOT / '.cache/torch'),
        MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'))
    try:
        child = subprocess.Popen(command, cwd=ROOT, env=environment)
        # Signal the wrapper (Slurm B:USR1), NOT torchrun, whose own signal
        # handler would terminate workers before the collective checkpoint.
        result = child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if result:
        return result
    summary = directory / 'summary.json'
    if not summary.exists() or json.loads(summary.read_text()).get('status') != 'complete':
        return 75  # Partial is never a successful downstream dependency.
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--treatment', choices=MODES)
    parser.add_argument('--execute', action='store_true', help='Run ONE pretraining task inside Slurm; never submits')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stop_after_updates < 0:
        parser.error('--stop-after-updates must be nonnegative')
    if args.execute:
        if not args.treatment:
            parser.error('--execute requires exactly one --treatment')
        return run_pretrain(config, treatment_plan(config, args.treatment)['pretrain'],
                            args.resume, args.stop_after_updates)
    selected = [args.treatment] if args.treatment else MODES
    plans = [treatment_plan(config, mode) for mode in selected]
    print(json.dumps(dict(scope='PREVIEW ONLY; no allocations, subprocesses or cluster writes',
        pretraining_resources=dict(nodes=1, ntasks=1, gpus=2, cpus_per_task=8,
            memory='32G', time_limit='47:00:00', node=config['node']),
        treatments=plans), indent=2))
    for plan in plans:
        print('\n# ' + plan['depth_mode'] + ' pretraining command (requires an allocation):')
        print(shlex.join(plan['pretrain']['argv']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
