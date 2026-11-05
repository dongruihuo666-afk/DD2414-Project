#!/usr/bin/env python3
"""Opt-in RPL planning and one-job execution. Never SSH or publish to GitHub."""

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_official_full_data import plan as official_plan

PROTOCOL = ROOT / 'configs/full_data_official_protocol.json'
MANIFEST = ROOT / 'configs/radar_scaling_manifest_seed125.json'
BEVCAR_COMMIT = '29cacda3bc5416d47428c1d0f017527acad34f90'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def settings(path):
    cfg = read(path)
    for key in ['data_root', 'run_root', 'env_prefix', 'mamba',
                'camera_checkpoint', 'bevcar_source']:
        if not Path(cfg[key]).is_absolute() or '\n' in cfg[key]:
            raise ValueError(f'{key} must be a single absolute path')
    if cfg['precision'] not in ['bf16', 'fp16']:
        raise ValueError('precision must be explicitly bf16 or fp16')
    if not 0 <= cfg['workers'] <= cfg['cpus']:
        raise ValueError('workers must fit requested CPUs')
    if not re.fullmatch(r'[1-9][0-9]*[GM]', cfg['memory']):
        raise ValueError('memory must be an explicit Slurm amount such as 32G')
    match = re.fullmatch(r'(\d+):([0-5]\d):([0-5]\d)', cfg['time'])
    if not match or not 300 < sum(int(v) * w for v, w in zip(match.groups(), [3600, 60, 1])) <= 48*3600:
        raise ValueError('time must be greater than five minutes and at most 48 hours')
    for key in ['node', 'partition', 'account', 'qos']:
        if cfg.get(key) is not None and not re.fullmatch(r'[A-Za-z0-9_.-]+', cfg[key]):
            raise ValueError(f'invalid {key}')
    return cfg


def replace_option(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def tasks(cfg):
    protocol = read(PROTOCOL)
    rows = official_plan(Path(cfg['run_root']), Path(cfg['data_root']),
                         Path(cfg['bevcar_source']), Path(cfg['camera_checkpoint']), protocol)
    # Reverse scientific priority, but keep each required pretraining first.
    ordered = []
    for velocity in ['zero', 'full']:
        group = [r for r in rows if f'velocity_{velocity}_' in r['name']]
        ordered += [group[0], *reversed(group[1:])]
    ordered.append(rows[0])
    for row in ordered:
        command = row['command']
        command[0] = str(Path(cfg['env_prefix']) / 'bin/python')
        command[1] = str(ROOT / command[1])
        replace_option(command, '--precision', cfg['precision'])
        replace_option(command, '--num-workers', cfg['workers'])
        # Optional source-checkpoint relocation never changes exact resume rules.
        if row['requires']:
            command.append('--allow-source-relocation')
    return ordered


def validate_finished_config(source, task):
    command = task['command']
    expected = {key: command[command.index('--' + key.replace('_', '-')) + 1]
                for key in ['mode', 'velocity_mode', 'probe_location', 'precision', 'epochs', 'nsweeps']}
    for key, value in expected.items():
        if str(source.get(key)) != value:
            raise ValueError(f'{task["name"]}: completed {key} mismatch')
    expected_values = dict(image_resolution=[448,800], teacher_resolution=[448,798],
        depth_mode='none', seed=125, train_samples=28130, val_samples=6019,
        image_weight=.2, bev_weight=.8, motion_weight=.5, batch_size=1,
        camera_lr=1e-5, probe_lr=1e-3, learning_rate=1e-4 if task['mode']=='finetune' else 2e-4,
        experiment='baseline_multisweep_double_distillation')
    expected_values['manifest_sha256'] = read(PROTOCOL)['manifest_sha256']
    for key, value in expected_values.items():
        if source.get(key) != value:
            raise ValueError(f'{task["name"]}: completed {key} mismatch')
    init = source.get('official_camera_initialization') or {}
    if init.get('sha256') != read(PROTOCOL)['camera_checkpoint_sha256']:
        raise ValueError('Completed task was not initialized from the approved RGB checkpoint')


def completed(cfg, task):
    directory = Path(task['directory'])
    if (directory / 'summary.json').is_file():
        summary = read(directory / 'summary.json')
        if summary.get('status') == 'complete':
            validate_finished_config(read(directory / 'config.json'), task)
            if summary.get('final_validation', {}).get('samples') != 6019:
                raise ValueError('Completed local run lacks full validation')
            return True
    if cfg.get('completed_index'):
        index = read(cfg['completed_index'])
        record = index['completed'].get(task['name'])
        if record:
            validate_finished_config(record['config'], task)
            if record['summary'].get('status') != 'complete' or record['summary'].get('final_validation', {}).get('samples') != 6019:
                raise ValueError('Imported completion is not a complete full validation')
            return True
    return False


def parent_checkpoint(cfg, task):
    if not task['requires']:
        return None
    parent = next(r for r in tasks(cfg) if r['name'] == task['requires'])
    checkpoint = Path(parent['directory']) / 'checkpoints/latest.pt'
    if not completed(cfg, parent) or not checkpoint.is_file():
        raise ValueError('Parent pretraining is incomplete or its checkpoint has not been transferred')
    if cfg.get('completed_index'):
        record = read(cfg['completed_index'])['completed'].get(parent['name'])
        if record and digest(checkpoint) != record['checkpoint_sha256']:
            raise ValueError('Transferred parent checkpoint SHA-256 mismatch')
    return checkpoint


def submit_command(cfg, config_path, task, pilot=False, resume=False, test_only=False):
    if cfg['node'].startswith('CHOOSE_'):
        raise ValueError('Select a real node after checking GPU and disk capacity')
    directory = Path(task['directory']) if not pilot else Path(cfg['run_root']) / 'engineering_pretrain_resume'
    command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--gres=gpu:1',
        f'--nodelist={cfg["node"]}', f'--cpus-per-task={cfg["cpus"]}',
        f'--mem={cfg["memory"]}', '--time=' + ('00:30:00' if pilot else cfg['time']),
        '--signal=B:USR1@300', '--open-mode=append', '--chdir=' + str(ROOT),
        '--output=' + str(directory / 'slurm-%j.out'),
        '--error=' + str(directory / 'slurm-%j.err'), '--job-name=dd2414-' + ('pilot' if pilot else task['mode'])]
    for key in ['partition', 'account', 'qos']:
        if cfg.get(key):
            command += [f'--{key}={cfg[key]}']
    if test_only:
        command.append('--test-only')
    command += [str(ROOT / 'cluster_rpl/job.sh'), str(Path(config_path).resolve()), task['name']]
    if pilot:
        command.append('--pilot')
    if resume:
        command.append('--resume')
    return command


def check_pilot(cfg):
    marker = Path(cfg['run_root']) / 'engineering_pretrain_resume/passed.json'
    if not marker.is_file():
        raise ValueError('Submit the real three-update/new-process-resume pilot first')
    report = read(marker)
    if report['code_commit'] != git_commit() or report['settings'] != cfg:
        raise ValueError('Pilot belongs to different code/settings; use a new run root and repeat it')
    validate_asset_report(cfg)


def validate_asset_report(cfg):
    report = read(Path(cfg['run_root']) / 'assets_audit.json')
    if report.get('status') != 'passed' or report.get('settings') != cfg or report.get('code_commit') != git_commit():
        raise ValueError('Full data/assets audit missing or stale; repeat the allocated CPU audit')


def git_commit():
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def clean_checkout():
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Tracked project changes exist; commit/review them before admission or execution')


def run_child(command, directory):
    """Forward checkpoint signals; partial checkpoints are not job success."""
    directory.mkdir(parents=True, exist_ok=True)
    stopping = False
    process = None
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGUSR1)
    for sig in [signal.SIGUSR1, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    with (directory / 'console.log').open('a') as handle:
        handle.write('\nCOMMAND ' + shlex.join(command) + '\n')
        handle.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if stopping:
            process.send_signal(signal.SIGUSR1)
        code = process.wait()
    if code:
        raise RuntimeError(f'Trainer exited {code}; inspect {directory}/console.log')
    return stopping


def execute(cfg, task, pilot=False, resume=False):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run only inside an allocated Slurm job, never on moria')
    node = os.environ.get('SLURMD_NODENAME') or os.uname().nodename.split('.')[0]
    if node != cfg['node']:
        raise ValueError('Allocated node differs from data node')
    clean_checkout()
    validate_asset_report(cfg)
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one Slurm-visible GPU is required; multi-GPU training is not implemented')
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
    os.environ['DD2414_TELEMETRY_GPU_UUID'] = gpu_uuid if gpu_uuid.startswith('GPU-') else 'GPU-' + gpu_uuid
    if cfg['precision'] == 'bf16' and (torch.cuda.get_device_capability()[0] < 8 or not torch.cuda.is_bf16_supported()):
        raise ValueError('Native BF16 unavailable; choose precision before locking the study')
    directory = Path(cfg['run_root']) / 'engineering_pretrain_resume' if pilot else Path(task['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.job.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another job owns this run directory') from error
        return execute_locked(cfg, task, directory, pilot, resume)


def execute_locked(cfg, task, directory, pilot, resume):
    command = list(task['command'])
    if pilot:
        replace_option(command, '--run-dir', directory)
        previous = read(directory / 'summary.json')['progress']['update'] if resume else 0
        for update in [2, 3]:
            if previous >= update:
                continue
            current = command + ['--stop-after-updates', str(update)]
            if update == 3 or resume:
                current.append('--resume')
            if run_child(current, directory):
                raise RuntimeError('Pilot interrupted; no admission marker written')
            summary = read(directory / 'summary.json')
            latest = summary.get('progress', {}).get('latest', {})
            if summary.get('status') != 'partial' or summary['progress']['update'] != update or not all(
                    math.isfinite(latest.get(k, float('nan'))) for k in ['loss','gradient_norm']):
                raise ValueError('Pilot did not complete real finite updates')
        summary = read(directory / 'summary.json')
        if summary.get('status') != 'partial' or summary['progress']['update'] != 3 or not all(
                math.isfinite(summary['progress']['latest'].get(k, float('nan'))) for k in ['loss','gradient_norm']):
            raise ValueError('Pilot did not reach exactly three finite updates')
        write(directory / 'passed.json', dict(code_commit=git_commit(), settings=cfg,
            checkpoint_sha256=digest(directory / 'checkpoints/latest.pt'),
            scope='Three real full-manifest updates, including a new-process resume; not a scientific subset result'))
    else:
        check_pilot(cfg)
        parent_checkpoint(cfg, task)
        if completed(cfg, task):
            raise ValueError('Task is already complete; no duplicate training')
        if resume:
            command.append('--resume')
        stopped = run_child(command, directory)
        if stopped or read(directory / 'summary.json').get('status') != 'complete':
            raise RuntimeError('Partial checkpoint saved; request approval before resubmitting with --resume')


def inventory(cfg):
    rows = []
    active = read(cfg['completed_index']).get('in_progress', []) if cfg.get('completed_index') else []
    for task in tasks(cfg):
        done = completed(cfg, task)
        partial = (Path(task['directory']) / 'checkpoints/latest.pt').is_file()
        rows.append(dict(name=task['name'], mode=task['mode'], epochs=task['epochs'],
            requires=task['requires'], status='complete' if done else 'active_on_workstation_snapshot'
            if task['name'] in active else 'partial' if partial else 'not_started',
            command=shlex.join(task['command'])))
    return rows


def export_completed(study, output):
    """Snapshot only durable completed results; never copy a running checkpoint."""
    data = dict(completed={}, in_progress=[], note='Historical completion snapshot; not live queue state')
    pipeline = Path(study) / 'pipeline.json'
    if pipeline.is_file():
        state = read(pipeline)
        data['snapshot_updated_unix_time'] = state.get('updated_unix_time')
        if state.get('status') == 'running' and state.get('current_stage'):
            data['in_progress'] = [state['current_stage']]
    for directory in sorted(Path(study).iterdir()):
        if not directory.is_dir() or not (directory / 'summary.json').is_file():
            continue
        summary = read(directory / 'summary.json')
        if summary.get('status') != 'complete' or summary.get('final_validation', {}).get('samples') != 6019:
            continue
        checkpoint = directory / 'checkpoints/latest.pt'
        if not checkpoint.is_file():
            raise ValueError(f'Completed task has no durable checkpoint: {directory}')
        data['completed'][directory.name] = dict(config=read(directory / 'config.json'),
            summary=summary, checkpoint_sha256=digest(checkpoint),
            summary_sha256=digest(directory / 'summary.json'), original_directory=str(directory.resolve()))
    write(output, data)
    print(json.dumps(dict(output=str(output), completed=list(data['completed'])), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'cluster_rpl/config.example.json')
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('plan')
    export = sub.add_parser('export-completed')
    export.add_argument('--study', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    submit = sub.add_parser('submit')
    submit.add_argument('--task', required=True)
    submit.add_argument('--pilot', action='store_true')
    submit.add_argument('--resume', action='store_true')
    submit.add_argument('--test-only', action='store_true')
    submit.add_argument('--execute', action='store_true', help='One actual sbatch command, only after owner approval')
    run = sub.add_parser('run')
    run.add_argument('--task', required=True)
    run.add_argument('--pilot', action='store_true')
    run.add_argument('--resume', action='store_true')
    sub.add_parser('progress')
    args = parser.parse_args()
    if args.action == 'export-completed':
        export_completed(args.study, args.output)
        return
    cfg = settings(args.config)
    if args.action == 'plan':
        print(json.dumps(inventory(cfg), indent=2))
        return
    if args.action == 'progress':
        for task in tasks(cfg):
            status = Path(task['directory']) / 'status.json'
            print(json.dumps(dict(task=task['name'], progress=read(status) if status.is_file() else None)))
        return
    task = next((r for r in tasks(cfg) if r['name'] == args.task), None)
    if task is None:
        raise ValueError('Unknown task; copy an exact name from plan')
    if args.pilot and task['mode'] != 'pretrain':
        raise ValueError('Pilot must use a pretraining task')
    if args.action == 'run':
        execute(cfg, task, args.pilot, args.resume)
        return
    command = submit_command(cfg, args.config, task, args.pilot, args.resume, args.test_only)
    print(shlex.join(command), flush=True)
    if not args.execute:
        print('PLAN ONLY: no job submitted. Get approval for this command before --execute.')
        return
    clean_checkout()
    if not args.pilot and not args.test_only:
        check_pilot(cfg)
        parent_checkpoint(cfg, task)
        if completed(cfg, task):
            raise ValueError('Task is already complete')
        if cfg.get('completed_index') and task['name'] in read(cfg['completed_index']).get('in_progress', []):
            raise ValueError('Task was active on the workstation at the last snapshot; refresh inventory')
    if args.test_only:
        subprocess.run(command, check=True)
        return
    directory = Path(cfg['run_root']) / 'engineering_pretrain_resume' if args.pilot else Path(task['directory'])
    if not args.resume and (directory / 'checkpoints/latest.pt').exists():
        raise ValueError('Checkpoint exists; request an explicit resume instead')
    directory.mkdir(parents=True, exist_ok=True)
    job_id = subprocess.check_output(command, text=True).strip()
    print('Submitted job ' + job_id)
    write(directory / ('submission-' + job_id.split(';')[0] + '.json'),
        dict(job_id=job_id, command=command, code_commit=git_commit(), settings=cfg,
             submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
