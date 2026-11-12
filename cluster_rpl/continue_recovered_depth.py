#!/usr/bin/env python3
"""One bounded continuation after a committed 47-hour exit; no automatic error retry."""
import argparse
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import time

CONTROLLER = Path('/Midgard/home/hongen/projects/DD2414-Project/.cache/deployments/aec677657cd0')
CONTROLLER_COMMIT = 'aec677657cd0a72b7aa1ef72510faf4f1d3bee2a'
SACCT = '/usr/local/bin/sacct'


def slurm_query_environment():
    """Resolve existing node libraries for this subprocess only, never the trainer."""
    environment = dict(os.environ)
    paths = ['/usr/local/lib', '/usr/local/lib/slurm']
    paths.extend(p for p in environment.get('LD_LIBRARY_PATH', '').split(':') if p)
    environment['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(paths))
    return environment


def parent_accounting(job):
    if not job.isdigit():
        raise ValueError('One numeric parent job ID is required')
    owner = pwd.getpwuid(os.getuid()).pw_name
    for _ in range(30):
        output = subprocess.check_output([SACCT, '-n', '-X', '-j', job,
            '--format=JobIDRaw,User,State%30,ExitCode', '-P'],
            env=slurm_query_environment(), text=True, timeout=10).strip()
        rows = [line.split('|') for line in output.splitlines() if line.strip()]
        if rows:
            if len(rows) != 1 or len(rows[0]) != 4:
                raise ValueError('Ambiguous parent accounting; refuse automatic continuation')
            identity, user, state, code = (part.strip() for part in rows[0])
            if identity != job or user != owner:
                raise ValueError('Parent accounting does not match our exact job/owner')
            if state not in ('RUNNING', 'COMPLETING', 'PENDING'):
                return dict(job_id=identity, user=user, state=state, exit_code=code)
        time.sleep(1)
    raise RuntimeError('Terminal parent accounting unavailable')


def decision(state, exit_code, summary, treatment):
    config, progress = summary.get('config', {}), summary.get('progress', {})
    if (config.get('training_profile') != 'random-metric-bev' or config.get('treatment') != treatment
            or config.get('epochs') != 50 or config.get('num_workers') != 0
            or config.get('train_samples') != 28130 or config.get('val_samples') != 6019
            or not 0 <= progress.get('epoch', -1) <= 50
            or not 0 <= progress.get('update', -1) <= 50*5626
            or progress.get('epoch') != progress.get('update', -1)//5626
            or progress.get('samples_seen', -1) != progress.get('update', -2)*5):
        raise ValueError('Unexpected committed recovery summary/protocol')
    if state == 'COMPLETED' and exit_code == '0:0' and summary.get('status') == 'complete':
        if (progress['epoch'] != 50 or progress['samples_seen'] != 50*28130
                or (summary.get('final_validation') or {}).get('samples') != 6019):
            raise ValueError('Completed parent lacks full training/final validation')
        return 'complete'
    if state == 'FAILED' and exit_code == '75:0' and summary.get('status') == 'partial':
        return 'resume'
    raise RuntimeError('Parent did not exit at the committed time boundary; do not retry its failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--treatment', choices=['dino-only','dino-metric-depth-bev'], required=True)
    parser.add_argument('--previous-job', required=True)
    parser.add_argument('--check-only', action='store_true',
        help='Verify actual node accounting and saved protocol without starting training')
    args = parser.parse_args()
    if (not args.previous_job.isdigit() or not os.environ.get('SLURM_JOB_ID')
            or socket.gethostname().split('.')[0] != 'eowyn'):
        raise ValueError('Use an explicit dependent own job in a Slurm allocation on eowyn')
    actual = subprocess.check_output(['git','-C',str(CONTROLLER),'rev-parse','HEAD'],text=True).strip()
    if actual != CONTROLLER_COMMIT:
        raise ValueError('Original admitted recovery controller changed')
    accounting=parent_accounting(args.previous_job)
    cfg=json.loads(args.config.read_text())
    summary=json.loads((Path(cfg['run_root'])/args.treatment/'pretrain/summary.json').read_text())
    result=decision(accounting['state'],accounting['exit_code'],summary,args.treatment)
    print('BOUNDED_CONTINUATION '+json.dumps(dict(parent=args.previous_job,decision=result,
        accounting=accounting,check_only=args.check_only,treatment=args.treatment,
        committed_update=summary['progress']['update'],controller_commit=actual)),flush=True)
    if args.check_only or result=='complete': return
    python=str(Path(cfg['env_prefix'])/'bin/python')
    os.execv(python,[python,str(CONTROLLER/'cluster_rpl/recover_random_depth.py'),
        '--config',str(args.config.resolve()),'--phase','pretrain','--treatment',args.treatment])


if __name__=='__main__': main()
