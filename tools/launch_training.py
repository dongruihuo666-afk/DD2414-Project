#!/usr/bin/env python3
"""Portable NEW-run command planner. Executes only with --execute; never submits Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def command(deployment, experiment, resume=False):
    root = Path(deployment['project_root']).expanduser().resolve()
    manifest = Path(deployment['manifest'])
    if not manifest.is_absolute():
        manifest = root / manifest
    name = experiment['run_name']
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('run_name must be one directory component')
    ranks = deployment['local_ranks']
    if type(ranks) is not int or ranks not in (2, 5, 10):
        raise ValueError('This random-metric profile supports exactly 2, 5 or 10 local ranks')
    args = experiment['arguments']
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError('arguments must be a JSON string list')
    if '--resume' in args or any(k in args for k in ('--data-root', '--run-dir', '--bevcar-source', '--manifest')):
        raise ValueError('Deployment paths/resume must not be embedded in model arguments')
    run_dir = Path(deployment['run_root']).expanduser().resolve() / name
    result = [deployment['python'], '-m', 'torch.distributed.run', '--standalone',
              '--nnodes=1', f'--nproc_per_node={ranks}', '-m', 'dd2414', 'train',
              '--data-root', deployment['data_root'], '--bevcar-source', deployment['bevcar_source'],
              '--manifest', str(manifest), '--run-dir', str(run_dir), *args]
    if '--treatment' in args and args[args.index('--treatment') + 1] == 'dino-metric-depth-bev':
        result += ['--metric-teacher-source', deployment['metric_teacher_source'],
                   '--metric-teacher-weights', deployment['metric_teacher_weights'],
                   '--metric-teacher-sha256', '9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb']
    if resume:
        result += ['--resume']
    return result, root, manifest, run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment', required=True, type=Path)
    p.add_argument('--experiment', required=True, type=Path)
    p.add_argument('--resume', action='store_true', help='Exact new-source state resume; never migrates historical runs.')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    deployment = json.loads(args.deployment.read_text())
    experiment = json.loads(args.experiment.read_text())
    argv, root, manifest, directory = command(deployment, experiment, args.resume)
    print(shlex.join(argv), flush=True)
    if not args.execute:
        return
    for path in [root / 'scripts/train_multisweep_teacher.py', Path(deployment['python']),
                 Path(deployment['data_root']), Path(deployment['bevcar_source']) / 'nets/voxelnet.py', manifest]:
        if not path.exists():
            raise FileNotFoundError(path)
    # Raw-file digest differs from the manifest's canonical unsigned JSON digest.
    expected = 'f46868da2d68fe651cddce64013981454f67c8574a1a6f46a6723f72bc84002f'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen data manifest hash differs')
    checkpoint = directory / 'checkpoints/latest.pt'
    if args.resume and not checkpoint.is_file():
        raise FileNotFoundError('Exact resume checkpoint missing: ' + str(checkpoint))
    if not args.resume and directory.exists():
        raise FileExistsError('New runs require a new output directory: ' + str(directory))
    environment = dict(os.environ, DD2414_PROJECT_ROOT=str(root))
    subprocess.run(argv, cwd=root, env=environment, check=True)


if __name__ == '__main__':
    main()
