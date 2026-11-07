#!/usr/bin/env python3
"""Owner-authorized dataset retirement, guarded by admission or completed results.

Only the exact hongen nuScenes directory is removed. Source retirement requires
the target audit; target retirement requires ALL downstream results and a
hash-verified durable run backup. Checkpoints and workstation data are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from random_depth_study import ROOT, TREATMENTS, load_config
from stage_random_depth import SOURCE, check_owned_directory, write


def verified_target(config, audit, transfer, commit):
    if (config['node'] == 'arwen' or audit.get('status') != 'passed'
            or audit.get('settings') != config or audit.get('code_commit') != commit
            or audit.get('node', '').split('.')[0] != config['node']
            or transfer.get('status') != 'copied' or transfer.get('settings') != config
            or transfer.get('target_node') != config['node'] or transfer.get('source_node') != 'arwen'):
        raise ValueError('Matching completed target transfer and fresh admission required before deleting source')
    counts = audit.get('sensor_files', {})
    if counts.get('referenced') != 2631083 or any(counts.get(k, -1) != 0 for k in ['missing', 'empty', 'unsafe']):
        raise ValueError('Target sensor coverage incomplete')


def completed_experiments(root):
    for treatment in TREATMENTS:
        for name, epochs in [('pretrain', 50), ('frozen_camera_bev_probe', 10),
                             ('frozen_fused_bev_probe', 10), ('end_to_end_segmentation', 5)]:
            report = json.loads((root/treatment/name/'summary.json').read_text())
            validation = report.get('final_validation')
            if (report.get('status') != 'complete' or report.get('config', {}).get('epochs') != epochs
                    or report.get('config', {}).get('training_profile') != 'random-metric-bev'
                    or report.get('config', {}).get('treatment') != treatment
                    or not validation or validation.get('samples') != 6019):
                raise ValueError('All pretraining and GT readouts must finish before target data cleanup')


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def backup_results(root, destination):
    if destination.exists():
        raise FileExistsError('Backup exists; inspect it rather than overwriting durable results')
    files = list(root.rglob('*'))
    if any(path.is_symlink() for path in files):
        raise ValueError('Run backup must not traverse symlinks')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, destination)
    manifest = {}
    for path in files:
        if path.is_file():
            relative = path.relative_to(root)
            source_hash = file_hash(path)
            if file_hash(destination/relative) != source_hash:
                raise ValueError('Durable backup hash mismatch: '+str(relative))
            manifest[str(relative)] = source_hash
    write(destination/'backup_manifest.json', dict(status='verified', sha256=manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['source', 'target'], required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    host = socket.gethostname().split('.')[0]
    job = os.environ.get('SLURM_JOB_ID')
    expected_host = 'arwen' if args.phase == 'source' else config['node']
    if not job or host != expected_host or config['node'] == 'arwen':
        raise ValueError('Cleanup requires the selected owner-scoped CPU Slurm allocation')
    if Path(config['data_root']) != SOURCE:
        raise ValueError('Cleanup is limited to the exact hongen nuScenes directory')
    check_owned_directory(SOURCE, SOURCE)
    active = subprocess.check_output(['squeue', '-u', 'hongen', '-w', host, '-t', 'RUNNING', '-h', '-o', '%i'], text=True).split()
    if set(active)-{job}:
        raise RuntimeError('Other own jobs still run on this node; preserve their possible inputs')
    if args.phase == 'source':
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        audit = json.loads((ROOT/'.cache/target_random_depth_assets_audit.json').read_text())
        transfer = json.loads((ROOT/'.cache/random_depth_transfer.json').read_text())
        verified_target(config, audit, transfer, commit)
    else:
        completed_experiments(Path(config['run_root']))
        backup = Path(config['durable_backup_root'])
        expected = Path('/Midgard/home/hongen/projects/DD2414-Project/fullsize_baseline/runs')
        if not backup.is_relative_to(expected):
            raise ValueError('Durable backup must stay inside the explicit shared project results directory')
        if args.execute:
            backup_results(Path(config['run_root']), backup)
    print(json.dumps(dict(phase=args.phase, node=host, dataset=str(SOURCE), execute=args.execute)), flush=True)
    if args.execute:
        before = shutil.disk_usage('/local_storage').free
        shutil.rmtree(SOURCE)
        write(ROOT/('.cache/dataset_cleanup_'+args.phase+'.json'), dict(status='complete', phase=args.phase,
            node=host, removed=str(SOURCE), checkpoint_policy='preserved; workstation dataset untouched',
            observed_free_delta_bytes=shutil.disk_usage('/local_storage').free-before, slurm_job_id=job))


if __name__ == '__main__':
    main()
