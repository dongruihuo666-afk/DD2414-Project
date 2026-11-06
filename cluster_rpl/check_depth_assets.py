#!/usr/bin/env python3
"""CPU-allocated admission audit for the three full-data depth continuations."""
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
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_full_data_protocol import audit
from check_assets import sensor_files, tree_digest
from control import BEVCAR_COMMIT, digest, git_commit, read, write
from depth_study import load_config
from source_provenance import check_continuation_source


def require_cpu_allocation(config, environment, hostname):
    if not environment.get('SLURM_JOB_ID') or hostname.split('.')[0] != config['node']:
        raise ValueError('Run the audit in a CPU Slurm allocation on the selected data node')


def check_sensor_coverage(counts):
    if counts['referenced'] != 2631083 or any(counts[key] for key in ['missing', 'empty', 'unsafe']):
        raise ValueError(f'Incomplete full-data sensor references: {counts}')


def verify_assets(config):
    import torch
    expected = read(ROOT / 'cluster_rpl/assets.json')
    protocol = read(ROOT / 'configs/full_data_official_protocol.json')
    if digest(config['camera_checkpoint']) != protocol['camera_checkpoint_sha256']:
        raise ValueError('Official RGB hash mismatch')
    if digest(config['parent_checkpoint']) != config['parent_sha256']:
        raise ValueError('Complete parent hash mismatch')
    if subprocess.check_output(['git', '-C', config['bevcar_source'], 'rev-parse', 'HEAD'], text=True).strip() != BEVCAR_COMMIT:
        raise ValueError('BEVCar source commit mismatch')
    voxel_hash = digest(Path(config['bevcar_source']) / 'nets/voxelnet.py')
    if voxel_hash != expected['bevcar_voxelnet_sha256']:
        raise ValueError('BEVCar VoxelNet hash mismatch')
    cache = ROOT / '.cache'
    checks = dict(
        dino_weight_sha256=digest(cache / 'torch/hub/checkpoints/dinov2_vits14_pretrain.pth'),
        dino_python_tree_sha256=tree_digest(cache / 'torch/hub/facebookresearch_dinov2_main'),
        dav2_weight_sha256=digest(cache / 'depth_anything_v2_vits.pth'),
        dav2_python_tree_sha256=tree_digest(cache / 'Depth-Anything-V2'))
    for key, value in checks.items():
        if value != expected[key]:
            raise ValueError('Frozen teacher content mismatch: ' + key)
    source = torch.load(config['parent_checkpoint'], map_location='cpu', weights_only=False)
    manifest = read(ROOT / 'configs/radar_scaling_manifest_seed125.json')
    initialization = dict(source['run_config']['official_camera_initialization'], path=config['camera_checkpoint'])
    continuation = check_continuation_source(source, dict(velocity_mode='full'),
        manifest['manifest_sha256'], Path(config['camera_checkpoint']), initialization, True)
    if initialization['sha256'] != protocol['camera_checkpoint_sha256'] or source['run_config']['bevcar_sha256'] != voxel_hash:
        raise ValueError('Parent encoder/BEVCar provenance mismatch')
    for name, value in source['run_config']['model_data_implementation_sha256'].items():
        if digest(ROOT / name) != value:
            raise ValueError('Parent model/data source mismatch: ' + name)
    return dict(checks, bevcar_voxelnet_sha256=voxel_hash, parent_sha256=config['parent_sha256'],
        parent=continuation, parent_model_tensors=len(source['model']),
        dav2_source_commit=expected['dav2_source_commit'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    require_cpu_allocation(config, os.environ, socket.gethostname())
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Deploy a committed clean checkout before admission')
    packages = {name: importlib.metadata.version(name) for name in
                ['torch', 'torchvision', 'numpy', 'Pillow', 'nuscenes-devkit']}
    pinned = dict(torch='2.5.1', torchvision='0.20.1', numpy='1.23.5', Pillow='9.4.0', **{'nuscenes-devkit': '1.1.11'})
    if sys.version_info[:2] != (3, 10) or any(packages[k].split('+')[0] != v for k, v in pinned.items()):
        raise ValueError(f'Unexpected environment: {packages}')
    started = time.monotonic()
    assets = verify_assets(config)
    print('DEPTH_ASSET_HASHES_PASSED', flush=True)
    result = audit(Path(config['data_root']), ROOT / '.cache/no_test_metadata_required.tgz',
                   Path(config['camera_checkpoint']), official_protocol=True)
    print('OFFICIAL_SPLIT_MEMBERSHIP_PASSED', flush=True)
    counts = sensor_files(config['data_root'], read(Path(config['data_root']) / 'v1.0-trainval/sample_data.json'))
    check_sensor_coverage(counts)
    for record in read(Path(config['data_root']) / 'v1.0-trainval/map.json'):
        if not (Path(config['data_root']) / record['filename']).is_file():
            raise ValueError('Missing referenced map')
    result.update(status='passed', assets=assets, sensor_files=counts, settings=config,
        packages=packages, python=sys.version, code_commit=git_commit(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
        elapsed_seconds=time.monotonic() - started,
        scope='CPU content/provenance/split and sensor-file presence audit; not CUDA training admission or byte-wise dataset checksum audit')
    write(Path(config['run_root']) / 'depth_assets_audit.json', result)
    print(json.dumps(dict(status='passed', sensor_files=counts, assets=assets,
        elapsed_seconds=result['elapsed_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    main()
