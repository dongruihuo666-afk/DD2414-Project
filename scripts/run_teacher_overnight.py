#!/usr/bin/env python3
"""Persistent, restart-safe V+/V0 and frozen-probe overnight pipeline."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / 'fullsize_baseline' / 'teacher_overnight'
RUN_ROOT = REPO_ROOT / 'fullsize_baseline' / 'runs'
PROBE_ROOT = REPO_ROOT / 'fullsize_baseline' / 'probes'
FULL_RUN = RUN_ROOT / 'full28130_seed125'
ZERO_RUN = RUN_ROOT / 'full28130_zero_velocity_seed125'
MANIFEST = REPO_ROOT / 'configs' / 'radar_scaling_manifest_seed125.json'
DATA_ROOT = Path(os.environ.get('NUSCENES_ROOT', Path.home() / 'datasets' / 'nuscenes'))
BEVCAR_SOURCE = Path(os.environ.get('BEVCAR_SOURCE_DIR', REPO_ROOT / 'external' / 'BEVCar'))


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def update(stage, state='running', **extra):
    value = {
        'state': state,
        'stage': stage,
        'updated_unix_time': time.time(),
        **extra,
    }
    atomic_json(value, OUTPUT_ROOT / 'status.json')
    print(json.dumps(value, sort_keys=True), flush=True)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def user_service_state(unit):
    """Return a best-effort user-service state without making it a dependency."""
    try:
        result = subprocess.run(
            ['systemctl', '--user', 'is-active', unit],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            check=False,
        )
    except OSError:
        return None
    state = result.stdout.strip()
    return state or None


def run(command, log_name, env=None):
    log_path = OUTPUT_ROOT / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as handle:
        handle.write(f'COMMAND {json.dumps(command)}\n')
        handle.flush()
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f'command failed with exit {result.returncode}; see {log_path}'
        )


def wait_for_full_velocity():
    while True:
        summary = read_json(FULL_RUN / 'summary.json')
        status = read_json(FULL_RUN / 'status.json') or {}
        if summary and summary.get('status') == 'complete':
            if not (FULL_RUN / 'checkpoints' / 'epoch008.pt').is_file():
                raise RuntimeError('V+ summary is complete but epoch008.pt is missing')
            return
        if status.get('state') == 'failed':
            raise RuntimeError(f'V+ failed: {status.get("message")}')
        service_state = user_service_state('dd2414-fullsize-full.service')
        # The trainer updates this timestamp every 50 steps. If its known
        # systemd service has stopped and the status is stale, do not wait
        # forever on a run that can no longer produce its final summary.
        updated = float(status.get('updated_unix_time', 0.0))
        if (service_state in ('inactive', 'failed', 'unknown')
                and time.time() - updated > 300):
            raise RuntimeError(
                f'V+ service is {service_state} and its status is stale'
            )
        update(
            'waiting_for_full_velocity', velocity_full_status=status,
            velocity_full_service_state=service_state,
        )
        time.sleep(60)


def training_command(run_dir, *, scale, stop_after_steps=0,
                     stop_after_epoch=0, resume=False, val_samples=6019,
                     batch_size=5, workers=8):
    command = [
        sys.executable, 'scripts/train_fullsize_baseline.py',
        '--data-root', str(DATA_ROOT),
        '--bevcar-source', str(BEVCAR_SOURCE),
        '--manifest', str(MANIFEST),
        '--scale', scale,
        '--run-dir', str(run_dir),
        '--epochs', '8',
        '--batch-size', str(batch_size),
        '--precision', 'bf16',
        '--num-workers', str(workers),
        '--seed', '125',
        '--nsweeps', '1',
        '--radar-velocity-mode', 'zero',
        '--val-samples', str(val_samples),
    ]
    if stop_after_steps:
        command += ['--stop-after-steps', str(stop_after_steps)]
    if stop_after_epoch:
        command += ['--stop-after-epoch', str(stop_after_epoch)]
    if resume:
        command.append('--resume')
    return command


def probe_command(checkpoint, output_dir, mode, train_samples=0, val_samples=6019):
    return [
        sys.executable, 'scripts/train_fullsize_linear_probe.py',
        '--data-root', str(DATA_ROOT),
        '--bevcar-source', str(BEVCAR_SOURCE),
        '--manifest', str(MANIFEST),
        '--checkpoint', str(checkpoint),
        '--output-dir', str(output_dir),
        '--radar-velocity-mode', mode,
        '--epochs', '1',
        '--batch-size', '5' if train_samples == 0 else '1',
        '--num-workers', '8' if train_samples == 0 else '0',
        '--seed', '125',
        '--train-samples', str(train_samples),
        '--val-samples', str(val_samples),
    ]


def ensure_probe(checkpoint, output_dir, mode, log_name,
                 train_samples=0, val_samples=6019):
    summary = read_json(output_dir / 'summary.json')
    if summary and summary.get('status') == 'complete':
        return
    run(probe_command(
        checkpoint, output_dir, mode, train_samples=train_samples,
        val_samples=val_samples,
    ), log_name)


def summarize(label, full_probe, zero_probe):
    output_dir = OUTPUT_ROOT / label
    run([
        sys.executable, 'scripts/summarize_velocity_probe.py',
        '--full-summary', str(full_probe / 'summary.json'),
        '--zero-summary', str(zero_probe / 'summary.json'),
        '--output-json', str(output_dir / 'comparison.json'),
        '--output-md', str(output_dir / 'comparison.md'),
        '--label', label,
    ], f'{label}_summary.log')


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        update('waiting_for_full_velocity')
        wait_for_full_velocity()

        smoke_train = REPO_ROOT / 'fullsize_baseline' / 'smoke' / 'zero_velocity_2steps'
        smoke_status = read_json(smoke_train / 'status.json') or {}
        if smoke_status.get('state') != 'stopped':
            update('zero_velocity_training_smoke')
            run(training_command(
                smoke_train, scale='4096', stop_after_steps=2,
                val_samples=4, batch_size=2, workers=0,
            ), 'zero_velocity_training_smoke.log')
        smoke_status = read_json(smoke_train / 'status.json') or {}
        if smoke_status.get('state') != 'stopped':
            raise RuntimeError('zero-velocity training smoke did not cleanly stop')

        smoke_probe = REPO_ROOT / 'fullsize_baseline' / 'smoke' / 'probe_full_4'
        update('linear_probe_smoke')
        ensure_probe(
            FULL_RUN / 'checkpoints' / 'epoch001.pt', smoke_probe, 'full',
            'linear_probe_smoke.log', train_samples=4, val_samples=4,
        )

        zero_status = read_json(ZERO_RUN / 'status.json') or {}
        if not (zero_status.get('state') in ('stopped', 'complete')
                and float(zero_status.get('epoch_progress', 0)) >= 1.0):
            update('zero_velocity_epoch1')
            resume = (ZERO_RUN / 'checkpoints' / 'latest.pt').is_file()
            run(training_command(
                ZERO_RUN, scale='full', stop_after_epoch=1, resume=resume,
            ), 'zero_velocity_epoch1.log')
        if not (ZERO_RUN / 'checkpoints' / 'epoch001.pt').is_file():
            raise RuntimeError('V0 epoch001.pt is missing after phase one')

        full_probe_1 = PROBE_ROOT / 'velocity_full_epoch001'
        zero_probe_1 = PROBE_ROOT / 'velocity_zero_epoch001'
        update('epoch1_full_velocity_probe')
        ensure_probe(
            FULL_RUN / 'checkpoints' / 'epoch001.pt', full_probe_1, 'full',
            'epoch1_full_velocity_probe.log',
        )
        update('epoch1_zero_velocity_probe')
        ensure_probe(
            ZERO_RUN / 'checkpoints' / 'epoch001.pt', zero_probe_1, 'zero',
            'epoch1_zero_velocity_probe.log',
        )
        summarize('epoch001', full_probe_1, zero_probe_1)

        zero_summary = read_json(ZERO_RUN / 'summary.json')
        if not zero_summary or zero_summary.get('status') != 'complete':
            update('zero_velocity_epochs2_to_8')
            run(training_command(
                ZERO_RUN, scale='full', resume=True,
            ), 'zero_velocity_epochs2_to_8.log')
        zero_summary = read_json(ZERO_RUN / 'summary.json') or {}
        if zero_summary.get('status') != 'complete':
            raise RuntimeError('V0 did not complete eight epochs')

        full_probe_8 = PROBE_ROOT / 'velocity_full_epoch008'
        zero_probe_8 = PROBE_ROOT / 'velocity_zero_epoch008'
        update('epoch8_full_velocity_probe')
        ensure_probe(
            FULL_RUN / 'checkpoints' / 'epoch008.pt', full_probe_8, 'full',
            'epoch8_full_velocity_probe.log',
        )
        update('epoch8_zero_velocity_probe')
        ensure_probe(
            ZERO_RUN / 'checkpoints' / 'epoch008.pt', zero_probe_8, 'zero',
            'epoch8_zero_velocity_probe.log',
        )
        summarize('epoch008', full_probe_8, zero_probe_8)
        update(
            'complete', state='complete',
            epoch1_report=str(OUTPUT_ROOT / 'epoch001' / 'comparison.md'),
            epoch8_report=str(OUTPUT_ROOT / 'epoch008' / 'comparison.md'),
        )
        print('TEACHER_OVERNIGHT_OK', flush=True)
    except Exception as error:
        update(
            'failed', state='failed',
            message=f'{type(error).__name__}: {error}',
        )
        raise


if __name__ == '__main__':
    main()
