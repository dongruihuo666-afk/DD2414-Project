#!/usr/bin/env python3
"""Allocated-CPU audit of every metadata-referenced sensor file and fixed assets."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_full_data_protocol import audit
from control import BEVCAR_COMMIT, clean_checkout, digest, git_commit, read, settings, write


def tree_digest(root):
    import hashlib
    value = hashlib.sha256()
    for path in sorted(Path(root).rglob('*.py')):
        value.update(path.relative_to(root).as_posix().encode() + b'\0')
        value.update(path.read_bytes())
    return value.hexdigest()


def sensor_files(root, records):
    counts = dict(referenced=0, missing=0, empty=0, unsafe=0)
    examples = []
    for record in records:
        counts['referenced'] += 1
        relative = PurePosixPath(record['filename'])
        if relative.is_absolute() or '..' in relative.parts:
            counts['unsafe'] += 1
        else:
            path = Path(root) / str(relative)
            if not path.is_file():
                counts['missing'] += 1
            elif path.stat().st_size == 0:
                counts['empty'] += 1
            else:
                continue
        if len(examples) < 20:
            examples.append(str(relative))
    counts['examples'] = examples
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = settings(args.config)
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Full audit must run in a CPU allocation, never on moria')
    if (os.environ.get('SLURMD_NODENAME') or os.uname().nodename.split('.')[0]) != cfg['node']:
        raise ValueError('Audit must run on the selected data node')
    clean_checkout()
    packages = {name:importlib.metadata.version(name) for name in
                ['torch','torchvision','numpy','Pillow','nuscenes-devkit']}
    expected = {'torch':'2.5.1','torchvision':'0.20.1','numpy':'1.23.5','Pillow':'9.4.0','nuscenes-devkit':'1.1.11'}
    if any(packages[k].split('+')[0] != v for k,v in expected.items()):
        raise ValueError(f'Unexpected environment versions: {packages}')
    if sys.version_info[:2] != (3,10):
        raise ValueError('Python 3.10 is required')
    protocol = read(ROOT / 'configs/full_data_official_protocol.json')
    if digest(cfg['camera_checkpoint']) != protocol['camera_checkpoint_sha256']:
        raise ValueError('Official RGB checkpoint SHA-256 mismatch')
    if subprocess.check_output(['git','-C',cfg['bevcar_source'],'rev-parse','HEAD'],text=True).strip() != BEVCAR_COMMIT:
        raise ValueError('BEVCar checkout is not pinned to the approved source commit')
    assets = read(ROOT / 'cluster_rpl/assets.json')
    if digest(Path(cfg['bevcar_source']) / 'nets/voxelnet.py') != assets['bevcar_voxelnet_sha256']:
        raise ValueError('BEVCar source file hash mismatch')
    cache = ROOT / '.cache/torch/hub'
    if digest(cache / 'checkpoints/dinov2_vits14_pretrain.pth') != assets['dino_weight_sha256']:
        raise ValueError('DINO teacher weight hash mismatch')
    if tree_digest(cache / 'facebookresearch_dinov2_main') != assets['dino_python_tree_sha256']:
        raise ValueError('DINO source differs from the teacher used on the workstation')
    result = audit(Path(cfg['data_root']), ROOT / '.cache/no_test_metadata_required.tgz',
                   Path(cfg['camera_checkpoint']), official_protocol=True)
    import torch
    result.update(settings=cfg, code_commit=git_commit(), packages=packages,
                  python=sys.version, cuda_runtime=torch.version.cuda)
    # This full JSON table can be large: request sufficient CPU RAM (32 GiB),
    # and never run this metadata/stat-heavy operation on the login node.
    records = read(Path(cfg['data_root']) / 'v1.0-trainval/sample_data.json')
    result['sensor_files'] = sensor_files(cfg['data_root'], records)
    for record in read(Path(cfg['data_root']) / 'v1.0-trainval/map.json'):
        if not (Path(cfg['data_root']) / record['filename']).is_file():
            raise ValueError('Missing map referenced by metadata')
    counts = result['sensor_files']
    result['status'] = 'passed' if counts['referenced'] and not any(counts[k] for k in ['missing','empty','unsafe']) else 'failed'
    write(Path(cfg['run_root']) / 'assets_audit.json', result)
    print(json.dumps(dict(status=result['status'], sensor_files=counts, packages=packages), indent=2))
    if result['status'] != 'passed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
