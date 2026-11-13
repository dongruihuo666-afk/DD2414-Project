#!/usr/bin/env python3
"""Official-split experiment queue. Wait for other GPU users; never kill them."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STOP = False


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stop(signum, frame):
    global STOP
    STOP = True


def gpu_available():
    """Require an idle device and no compute clients, not just spare VRAM."""
    try:
        clients = subprocess.check_output(['nvidia-smi', '-i', '0',
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        values = subprocess.check_output(['nvidia-smi', '-i', '0',
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        free, utilization = [int(v.strip()) for v in values.splitlines()[0].split(',')]
        return not clients and free >= 22000 and utilization <= 10, dict(
            compute_pids=clients, free_mib=free, utilization_percent=utilization)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        return False, dict(error=str(error))


def plan(study, data, bevcar, camera, protocol):
    common = [sys.executable, 'scripts/train_multisweep_teacher.py',
        '--data-root', str(data), '--bevcar-source', str(bevcar),
        '--manifest', str(ROOT/'configs/radar_scaling_manifest_seed125.json'),
        '--official-camera-checkpoint', str(camera), '--nsweeps', '10',
        '--depth-mode', 'none', '--image-resolution', '448', '800',
        '--teacher-resolution', '448', '798', '--batch-size', str(protocol['batch_size']),
        '--num-workers', str(protocol['workers']), '--camera-lr', str(protocol['encoder_learning_rate']),
        '--learning-rate', str(protocol['student_learning_rate']),
        '--probe-lr', str(protocol['probe_learning_rate']), '--seed', str(protocol['seed']),
        '--image-weight', str(protocol['image_loss_weight']),
        '--bev-weight', str(protocol['bev_loss_weight']),
        '--motion-weight', str(protocol['motion_auxiliary_weight']),
        '--precision', 'bf16', '--checkpoint-every', '250', '--print-every', '10']
    stages = []
    def add(name, mode, epochs, parent=None, velocity='full', location='fused_bev', initial=False):
        directory = study/name
        command = common + ['--run-dir', str(directory), '--mode', mode,
            '--epochs', str(epochs), '--velocity-mode', velocity, '--probe-location', location]
        if parent:
            command += ['--checkpoint', str(study/parent/'checkpoints/latest.pt')]
        if initial:
            command += ['--initial-camera-probe']
        if mode == 'finetune':
            command[command.index('--learning-rate')+1] = str(protocol['finetune_learning_rate'])
        stages.append(dict(name=name, directory=str(directory), command=command,
                           requires=parent, epochs=epochs, mode=mode))
    add('official_rgb_encoder_camera_bev_frozen_probe', 'frozen-probe', protocol['probe_epochs'],
        location='camera_bev', initial=True)
    for velocity in ['full', 'zero']:
        parent = f'official_rgb_double_dino_10sweep_velocity_{velocity}_pretrain'
        add(parent, 'pretrain', protocol['pretrain_epochs'], velocity=velocity)
        for location in ['camera_bev', 'fused_bev']:
            add(parent + '_' + location + '_frozen_probe', 'frozen-probe',
                protocol['probe_epochs'], parent, velocity, location)
        add(parent + '_object_velocity_readout', 'object-motion',
            protocol['downstream_epochs'], parent, velocity)
        add(parent + '_end_to_end_segmentation', 'finetune',
            protocol['downstream_epochs'], parent, velocity)
    return stages


def validate_partial(summary, minimum_update):
    latest = summary.get('progress', {}).get('latest', {})
    return (summary.get('status') == 'partial' and
        summary['progress']['update'] == minimum_update and
        all(math.isfinite(latest.get(key, float('nan'))) for key in ['loss', 'gradient_norm']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/home/students2026/datasets/nuscenes'))
    p.add_argument('--bevcar-source', type=Path, default=ROOT/'external/BEVCar')
    p.add_argument('--camera-checkpoint', type=Path,
        default=ROOT/'checkpoints/8x5_5e-4_rgb12_22:43:46/model-000025000.pth')
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    protocol = read_json(ROOT/'configs/full_data_official_protocol.json')
    stages = plan(args.study_dir.resolve(), args.data_root, args.bevcar_source,
                  args.camera_checkpoint, protocol)
    if args.plan_only:
        print(json.dumps(stages, indent=2))
        return
    args.study_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.study_dir/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    state_path = args.study_dir/'pipeline.json'
    state = read_json(state_path) or dict(stages={}, engineering_checks={})
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if state.get('protocol_sha256', protocol_sha) != protocol_sha:
        raise RuntimeError('Queue protocol changed; use a new study directory')
    state.update(protocol=protocol, protocol_sha256=protocol_sha, plan=stages)
    def status(value, **extra):
        previous = state.get('status')
        state.update(status=value, updated_unix_time=time.time(), **extra)
        write_json(state_path, state)
        if previous != value:
            print(json.dumps(dict(status=value, current_stage=state.get('current_stage'),
                message=state.get('message'), gpu=state.get('gpu'))), flush=True)
    def wait_for_gpu():
        stable = 0
        while not STOP:
            available, telemetry = gpu_available()
            stable = stable + 1 if available else 0
            status('waiting_for_gpu', gpu=telemetry,
                   message='Other GPU jobs are never interrupted; require three idle checks.')
            if stable >= 3:
                return True
            time.sleep(20)
        return False
    def execute(command, directory):
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/'checkpoints/latest.pt').exists():
            command = command + ['--resume']
        with (directory/'console.log').open('a') as handle:
            handle.write('\nCOMMAND ' + json.dumps(command) + '\n')
            handle.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT)
            status('running', child_pid=process.pid, command=command)
            signaled = False
            while process.poll() is None:
                if STOP and not signaled:
                    process.send_signal(signal.SIGUSR1)
                    signaled = True
                state['child_status'] = read_json(directory/'status.json')
                status('stopping' if STOP else 'running')
                time.sleep(10)
        return process.returncode, read_json(directory/'summary.json')
    def record_publication_command():
        state['publication'] = dict(status='manual_only', command=[
            sys.executable, 'tools/publish_results.py', '--study', 'official',
            '--study-dir', str(args.study_dir.resolve())])
        write_json(state_path, state)
    audit_command = [sys.executable, 'scripts/audit_full_data_protocol.py',
        '--protocol', 'official_train_val', '--require-ready', '--data-root', str(args.data_root),
        '--camera-checkpoint', str(args.camera_checkpoint), '--output', str(args.study_dir/'audit.json')]
    with (args.study_dir/'audit.log').open('a') as handle:
        subprocess.run(audit_command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    audit = read_json(args.study_dir/'audit.json')
    if (audit['camera_checkpoint']['sha256'] != protocol['camera_checkpoint_sha256'] or
            audit['active_full_manifest_sha256'] != protocol['manifest_sha256']):
        raise RuntimeError('Checkpoint or manifest does not match approved protocol')
    if not wait_for_gpu():
        status('partial')
        return
    # Two real updates followed by a new-process resume to update three, on the
    # full training manifest. No small scientific subset or validation shortcut.
    engineering = args.study_dir/'engineering_pretrain_resume'
    pretrain = next(t for t in stages if t['mode'] == 'pretrain')
    for updates in [2, 3]:
        if state['engineering_checks'].get(str(updates)) == 'passed':
            continue
        command = list(pretrain['command'])
        command[command.index('--run-dir')+1] = str(engineering)
        command += ['--stop-after-updates', str(updates)]
        status('engineering_check', current_stage='full_manifest_pretrain_resume')
        code, summary = execute(command, engineering)
        if STOP:
            status('partial')
            return
        if code or not validate_partial(summary, updates):
            status('failed_engineering_check', returncode=code,
                   message='Inspect engineering_pretrain_resume/console.log; formal tasks not launched.')
            return
        state['engineering_checks'][str(updates)] = 'passed'
        status('engineering_passed')
    for task in stages:
        name, directory = task['name'], Path(task['directory'])
        summary = read_json(directory/'summary.json')
        if summary.get('status') == 'complete':
            state['stages'][name] = dict(status='complete', final_validation=summary['final_validation'])
            continue
        if STOP:
            break
        if task['requires'] and state['stages'].get(task['requires'], {}).get('status') != 'complete':
            state['stages'][name] = dict(status='blocked_by_parent')
            continue
        if shutil.disk_usage(ROOT).free < 30 * 1024**3:
            status('blocked_disk_space', message='Less than 30 GiB free; no files removed.')
            return
        if not wait_for_gpu():
            break
        status('starting', current_stage=name)
        succeeded = False
        # Bounded restart; no schema/hash/shape bypass or architecture mutation.
        for attempt in range(1, 4):
            code, summary = execute(task['command'], directory)
            succeeded = code == 0 and summary.get('status') == 'complete'
            state['stages'][name] = dict(status='complete' if succeeded else 'failed',
                attempts=attempt, returncode=code, final_validation=summary.get('final_validation'))
            status('running' if succeeded else 'stage_failed')
            if succeeded or STOP:
                break
        record_publication_command()
        if STOP:
            break
        # Fail closed on an unfixable stage. Do not report an unfinished matrix
        # as complete or continue launching potentially broken comparisons.
        if not succeeded:
            status('failed', current_stage=name)
            return
    status('partial' if STOP else 'complete')
    record_publication_command()


if __name__ == '__main__':
    main()
