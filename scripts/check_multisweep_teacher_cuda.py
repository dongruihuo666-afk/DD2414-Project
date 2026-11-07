#!/usr/bin/env python3
"""Bounded real-CUDA engineering checks; never scientific subset evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fullsize_training import atomic_json_write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--bevcar-source',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: this smoke cannot be marked passed')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    base = [sys.executable,'scripts/train_multisweep_teacher.py','--data-root',str(args.data_root),
        '--bevcar-source',str(args.bevcar_source),'--manifest','configs/radar_scaling_manifest_seed125.json',
        '--epochs','1','--batch-size',str(args.batch_size),'--num-workers','2',
        '--train-samples',str(args.batch_size*2),'--val-samples','4',
        '--print-every','1','--checkpoint-every','1']
    results = {}
    env = dict(os.environ,TORCH_HOME=str(ROOT/'.cache/torch'),PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR='/tmp/dd2414-mpl',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def execute(name,mode,sweeps,depth,checkpoint=None,stop=0,resume=False):
        directory = args.output_dir/name
        directory.mkdir(exist_ok=True)
        command = [*base,'--mode',mode,'--nsweeps',str(sweeps),'--depth-mode',depth,
                   '--run-dir',str(directory)]
        if checkpoint:
            command += ['--checkpoint',str(checkpoint)]
        if stop:
            command += ['--stop-after-updates',str(stop)]
        if resume:
            command.append('--resume')
        print('SMOKE '+name+(' resume' if resume else ''),flush=True)
        with (directory/'console.log').open('a') as log:
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print((directory/'console.log').read_text()[-6000:],flush=True)
            raise RuntimeError(f'{name} failed with code {result.returncode}')
        value = json.loads((directory/'summary.json').read_text())
        expected = 'partial' if stop else 'complete'
        if value['status'] != expected:
            raise RuntimeError(f'{name} has unexpected status: {value["status"]}')
        results[name] = dict(status=value['status'],update=value['progress']['update'],
            samples=value['progress']['samples_seen'],latest=value['progress']['latest'],
            peak_allocated_gib=value['telemetry']['peak_allocated_gib'],
            final_validation=value['final_validation'])
        atomic_json_write(dict(status='running',results=results),args.output_dir/'checks.json')
        return directory/'checkpoints/latest.pt'
    source = execute('double_5sweep_resume','pretrain',5,'none',stop=1)
    stopped = json.loads((source.parent.parent/'summary.json').read_text())
    if stopped['progress']['update'] != 1 or stopped['progress']['samples_seen'] != args.batch_size:
        raise RuntimeError('checkpoint stop was not at the expected acknowledged cursor')
    execute('double_5sweep_resume','pretrain',5,'none',resume=True)
    ten = execute('double_10sweep','pretrain',10,'none')
    radar = execute('double_5sweep_radar_depth','pretrain',5,'radar')
    dav2 = execute('double_5sweep_radar_and_relative_depth','pretrain',5,'radar-dav2')
    for mode in ['frozen-probe','unfrozen-probe','finetune','object-motion']:
        execute('double_5sweep_'+mode,mode,5,'none',source)
    execute('depth_finetune','finetune',5,'radar',radar)
    execute('depth_frozen_probe','frozen-probe',5,'radar-dav2',dav2)
    for name in ['double_5sweep_radar_depth','double_5sweep_radar_and_relative_depth']:
        latest = results[name]['latest']
        if latest.get('depth_anchors',0) <= 0:
            raise RuntimeError('depth test had no actual radar depth anchors')
    atomic_json_write(dict(status='complete',scope='Engineering smoke only; no scientific subset claims',
        batch_size=args.batch_size,gpu=torch.cuda.get_device_name(0),
        finished_unix_time=time.time(),results=results),args.output_dir/'checks.json')
    print('ALL_CUDA_CHECKS_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
