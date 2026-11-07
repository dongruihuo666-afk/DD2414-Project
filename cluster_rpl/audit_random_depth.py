#!/usr/bin/env python3
"""Allocated-CPU admission for the random student; no official RGB/parent needed."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from check_assets import sensor_files, tree_digest
from control import BEVCAR_COMMIT, digest, read, write
from random_depth_study import load_config
from radar_scaling_data import validate_scaling_manifest


def audit_membership(root):
    from nuscenes.utils.splits import create_splits_scenes
    scenes = read(root/'v1.0-trainval/scene.json')
    samples = read(root/'v1.0-trainval/sample.json')
    manifest = read(ROOT/'configs/radar_scaling_manifest_seed125.json')
    validate_scaling_manifest(manifest)
    splits = create_splits_scenes()
    records, sets = {}, {}
    for split, expected_scenes, expected_samples in [('train', 700, 28130), ('val', 150, 6019)]:
        selected = [s for s in scenes if s['name'] in set(splits[split])]
        tokens = {s['token'] for s in selected}
        sample_tokens = {s['token'] for s in samples if s['scene_token'] in tokens}
        logs = {s['log_token'] for s in selected}
        actual = manifest['scales']['full'] if split == 'train' else manifest['val']
        if (len(selected) != expected_scenes or len(sample_tokens) != expected_samples
                or sample_tokens != set(actual['tokens'])):
            raise ValueError('Full official split/manifest mismatch: ' + split)
        records[split] = dict(scenes=len(tokens), samples=len(sample_tokens), logs=len(logs))
        sets[split] = (tokens, sample_tokens, logs)
    if any(a & b for a, b in zip(sets['train'], sets['val'])):
        raise ValueError('Train/validation scenes, samples or logs overlap')
    return dict(splits=records, overlaps=dict(scenes=0, samples=0, logs=0),
                manifest_sha256=manifest['manifest_sha256'])


def audit_assets(config):
    expected = read(ROOT/'cluster_rpl/assets.json')
    checks = {
        'dino_weight_sha256': digest(ROOT/'.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'),
        'dino_python_tree_sha256': tree_digest(ROOT/'.cache/torch/hub/facebookresearch_dinov2_main'),
        'bevcar_voxelnet_sha256': digest(Path(config['bevcar_source'])/'nets/voxelnet.py'),
        'dav2_python_tree_sha256': tree_digest(Path(config['metric_teacher_source']))}
    for key, value in checks.items():
        if value != expected[key]:
            raise ValueError('Source/weight content mismatch: ' + key)
    commit = subprocess.check_output(['git', '-C', config['bevcar_source'], 'rev-parse', 'HEAD'], text=True).strip()
    if commit != BEVCAR_COMMIT or digest(config['metric_teacher_weights']) != config['metric_teacher_sha256']:
        raise ValueError('BEVCar commit or metric Depth Anything weights differ')
    return dict(checks, metric_teacher_sha256=config['metric_teacher_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().split('.')[0] != config['node']:
        raise ValueError('Audit requires a CPU Slurm allocation on the configured data node, never login-node compute')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('A committed clean deployment is required')
    versions = {name:importlib.metadata.version(name).split('+')[0] for name in
                ['torch', 'torchvision', 'numpy', 'Pillow', 'nuscenes-devkit']}
    if sys.version_info[:2] != (3, 10) or versions != dict(torch='2.5.1', torchvision='0.20.1',
            numpy='1.23.5', Pillow='9.4.0', **{'nuscenes-devkit': '1.1.11'}):
        raise ValueError('Unexpected environment: ' + str(versions))
    started = time.monotonic()
    membership = audit_membership(Path(config['data_root']))
    assets = audit_assets(config)
    counts = sensor_files(config['data_root'], read(Path(config['data_root'])/'v1.0-trainval/sample_data.json'))
    if counts['referenced'] != 2631083 or any(counts[k] for k in ['missing', 'empty', 'unsafe']):
        raise ValueError('Incomplete sensor-file coverage: ' + str(counts))
    for record in read(Path(config['data_root'])/'v1.0-trainval/map.json'):
        if not (Path(config['data_root'])/record['filename']).is_file():
            raise ValueError('Missing referenced map')
    result = dict(status='passed', settings=config, membership=membership, assets=assets,
        node=socket.gethostname(), slurm_job_id=os.environ['SLURM_JOB_ID'],
        sensor_files=counts, versions=versions, elapsed_seconds=time.monotonic()-started,
        code_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        scope='CPU presence/hash/split audit; not bytewise dataset checksums, CUDA admission or blind test')
    write(Path(config['run_root'])/'random_depth_assets_audit.json', result)
    # Shared evidence lets a separate arwen CPU job retire only the old source
    # after this node-local destination has passed complete admission.
    write(ROOT/'.cache/target_random_depth_assets_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
