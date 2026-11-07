#!/usr/bin/env python3
"""Run one full-data GT readout; partial completion is never Slurm success."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

from random_depth_study import ROOT, TREATMENTS, load_config, require_pilot, treatment_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--treatment', choices=TREATMENTS, required=True)
    parser.add_argument('--readout', choices=['frozen_camera_bev_probe', 'frozen_fused_bev_probe',
                                             'end_to_end_segmentation'], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if (not os.environ.get('SLURM_JOB_ID') or socket.gethostname().split('.')[0] != config['node']
            or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < 4):
        raise ValueError('Readout requires a single-GPU Slurm allocation on the configured data node')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    require_pilot(config, commit)
    task = next(t for t in treatment_plan(config, args.treatment)['downstream'] if t['name'] == args.readout)
    command = task['argv']
    directory = Path(command[command.index('--run-dir')+1])
    if (directory/'config.json').exists():
        raise FileExistsError('Readout output exists; inspect before choosing an explicit restart')
    child = None
    def stop(signum, frame):
        if child is not None:
            # This is one Python trainer, NOT torchrun. Its signal handler
            # sets STOP and saves at an update boundary.
            child.send_signal(signal.SIGUSR1)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    environment = dict(os.environ, TORCH_HOME=str(ROOT/'.cache/torch'),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    child = subprocess.Popen(command, cwd=ROOT, env=environment)
    result = child.wait()
    if result:
        return result
    summary = json.loads((directory/'summary.json').read_text())
    return 0 if summary.get('status') == 'complete' else 75


if __name__ == '__main__':
    sys.exit(main())
