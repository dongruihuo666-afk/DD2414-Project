#!/usr/bin/env python3
"""CPU-only arwen-to-selected-node tar stream inside a two-node Slurm job.

No dataset passes through moria/home. Source is read only; partial destinations
are never promoted until tar completes. Full sensor admission is a later job.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from random_depth_study import ROOT, load_config

SOURCE = Path('/local_storage/users/hongen/datasets/nuscenes')
ACCOUNT_ROOT = Path('/local_storage/users/hongen')
BEVCAR_COMMIT = '29cacda3bc5416d47428c1d0f017527acad34f90'


def check_owned_directory(path, expected):
    path, expected = Path(path), Path(expected)
    if path != expected or path.is_symlink() or path.resolve() != expected:
        raise ValueError('Unexpected or symlinked owner-scoped path: ' + str(path))
    if not path.is_dir() or path.stat().st_uid != os.getuid():
        raise ValueError('Expected an existing directory owned by the current account: ' + str(path))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['transfer', 'prepare', 'receive'], default='transfer')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    node = config['node']
    job_id = os.environ.get('SLURM_JOB_ID')
    host = socket.gethostname().split('.')[0]
    if not job_id or node not in ('eowyn', 'galadriel'):
        raise ValueError('Explicit Slurm allocation and a distinct supported target node required')
    destination = Path(config['data_root'])
    if destination != SOURCE:
        raise ValueError('This migration pins the same owner-scoped path on two DIFFERENT nodes')
    incoming = destination.with_name(destination.name+'.incoming-'+job_id)
    python = str(Path(config['env_prefix'])/'bin/python')
    if args.phase in ('prepare', 'receive'):
        if host != node:
            raise ValueError('Destination helper ran on the wrong node')
        if args.phase == 'prepare':
            ACCOUNT_ROOT.mkdir(mode=0o700, exist_ok=True)
            check_owned_directory(ACCOUNT_ROOT, ACCOUNT_ROOT)
            destination.parent.mkdir(mode=0o700, exist_ok=True)
            if destination.exists() or incoming.exists():
                raise FileExistsError('Refusing to overwrite existing destination or partial transfer')
            if shutil.disk_usage('/local_storage').free < 500*2**30:
                raise ValueError('Less than 500 GiB free for complete dataset staging')
            incoming.mkdir(mode=0o700)
            print('DESTINATION_READY '+str(incoming), flush=True)
            return
        check_owned_directory(incoming, incoming)
        subprocess.run(['tar', '--extract', '--no-same-owner', '-f', '-', '-C', str(incoming)], check=True)
        for name in ['sample_data.json', 'sample.json', 'scene.json']:
            if (incoming/'v1.0-trainval'/name).stat().st_size == 0:
                raise ValueError('Missing/empty transferred metadata')
        if destination.exists():
            raise FileExistsError('Destination appeared during transfer; refusing promotion')
        incoming.rename(destination)
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        write(ROOT/'.cache/random_depth_transfer.json', dict(status='copied', settings=config,
            source_node='arwen', target_node=node, source_data_root=str(SOURCE),
            target_data_root=str(destination), slurm_job_id=job_id, code_commit=commit,
            scope='Complete successful tar stream; fresh target sensor/hash/split audit still required'))
        print('TRANSFER_COMPLETE_AUDIT_REQUIRED', flush=True)
        return
    if host != 'arwen' or int(os.environ.get('SLURM_JOB_NUM_NODES', '0')) != 2:
        raise ValueError('Transfer requires a two-node CPU allocation with arwen as batch node')
    check_owned_directory(SOURCE, SOURCE)
    if not (SOURCE/'v1.0-trainval/sample_data.json').is_file():
        raise FileNotFoundError('Source full metadata missing')
    bevcar = Path(config['bevcar_source'])
    expected = Path('/Midgard/home/hongen/projects/DD2414-Project/.cache/BEVCar')
    if bevcar != expected:
        raise ValueError('BEVCar must be staged into the explicit shared project cache')
    if not bevcar.exists():
        subprocess.run(['git', 'clone', '--no-hardlinks',
            '/local_storage/users/hongen/dd2414_assets/BEVCar', str(bevcar)], check=True)
    actual = subprocess.check_output(['git', '-C', str(bevcar), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != BEVCAR_COMMIT:
        raise ValueError('Shared BEVCar source commit mismatch')
    command = ['srun', '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--nodelist='+node,
               python, str(Path(__file__).resolve()), '--config', str(config_path)]
    subprocess.run([*command, '--phase', 'prepare'], check=True)
    print('STREAMING_DATASET_ARWEN_TO_'+node.upper(), flush=True)
    sender = subprocess.Popen(['tar', '--checkpoint=500000', '--checkpoint-action=echo',
                               '-C', str(SOURCE), '-cf', '-', '.'], stdout=subprocess.PIPE)
    try:
        receiver = subprocess.run([*command, '--phase', 'receive'], stdin=sender.stdout)
    finally:
        sender.stdout.close()
    sent = sender.wait()
    if sent or receiver.returncode:
        raise RuntimeError(f'Transfer failed: sender={sent}, receiver={receiver.returncode}; source preserved')


if __name__ == '__main__':
    main()
