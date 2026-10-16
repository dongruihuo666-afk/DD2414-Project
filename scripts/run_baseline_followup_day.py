#!/usr/bin/env python3
"""Persistent sequential GPU queue with bounded retries and an absolute deadline."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullsize_training import atomic_json_write
from report_baseline_followup import build_report

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def choose_epochs(samples_per_second, hours_remaining):
    # Reserve half the available time for validation, references, probes,
    # startup, and slower batches; decide once for BOTH production jobs.
    return 3 if 2*3*28130/max(samples_per_second, .01) < hours_remaining*3600*.5 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--data-root', type=Path, default=Path.home()/'datasets/nuscenes')
    parser.add_argument('--bevcar-source', type=Path, default=ROOT/'external/BEVCar')
    args = parser.parse_args()
    study = args.study_dir.resolve()
    study.mkdir(parents=True, exist_ok=True)
    lock = (study/'queue.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGUSR1, signal.SIGINT]:
        signal.signal(sig, request_stop)
    state_path = study/'pipeline.json'
    state = read_json(state_path) or dict(started_unix_time=time.time(),
        deadline_unix_time=time.time()+args.hours*3600, stages={})
    deadline = state['deadline_unix_time']
    manifest = ROOT/'configs/radar_scaling_manifest_seed125.json'
    run_root = ROOT/'fullsize_baseline/runs'
    sources = dict(full=run_root/'full28130_seed125/checkpoints/epoch008.pt',
        zero=run_root/'full28130_zero_velocity_seed125/checkpoints/epoch008.pt')

    def update(stage, **extra):
        state.update(stage=stage, updated_unix_time=time.time(), **extra)
        atomic_json_write(state, state_path)

    def report_and_publish():
        build_report(study)
        # Git/network failures never discard a training result or stop the queue.
        # The exact blocking message remains visible for later manual recovery.
        result = subprocess.run([sys.executable, 'scripts/publish_baseline_followup.py',
            '--study-dir', str(study)], cwd=ROOT, capture_output=True, text=True)
        state['publication'] = dict(returncode=result.returncode,
            message=(result.stdout+result.stderr)[-3000:])
        atomic_json_write(state, state_path)

    def execute(name, command, expected_summary, *, retry=True, budget_seconds=None):
        if read_json(expected_summary).get('status') == 'complete':
            state['stages'][name] = dict(status='complete', reused=True)
            update(name)
            return True
        previous = state['stages'].get(name, {})
        if previous.get('status') == 'failed' and previous.get('attempts',0) >= 2:
            return False
        if STOP or time.time() >= deadline-1800:
            state['stages'][name] = dict(status='deferred', reason='deadline or stop')
            update(name)
            return False
        stage_dir = Path(expected_summary).parent
        stage_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(previous.get('attempts',0)+1, 3 if retry else 2):
            actual = list(command)
            if 'train_baseline_downstream.py' in ' '.join(actual) and (stage_dir/'checkpoints/latest.pt').is_file():
                actual.append('--resume')
            status = dict(status='running', attempts=attempt, command=actual,
                          started_unix_time=time.time())
            state['stages'][name] = status
            update(name)
            print(f'START {name}: {json.dumps(actual)}', flush=True)
            with (stage_dir/'console.log').open('a') as handle:
                handle.write('COMMAND '+json.dumps(actual)+'\n')
                handle.flush()
                process = subprocess.Popen(actual, cwd=ROOT, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True)
                status['pid'] = process.pid
                timed_out = False
                started = time.time()
                while process.poll() is None:
                    child_status = read_json(stage_dir/'status.json')
                    update(name, child_status=child_status)
                    limit = min(deadline-1200,
                                started+budget_seconds if budget_seconds else deadline)
                    heartbeat = child_status.get('updated_unix_time', started)
                    stale = time.time()-max(heartbeat, started) > 1800
                    if STOP or time.time() >= limit or stale:
                        timed_out = True
                        process.send_signal(signal.SIGUSR1 if
                            'train_baseline_downstream.py' in ' '.join(actual)
                            else signal.SIGTERM)
                        try:
                            process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                        break
                    time.sleep(20)
            result = read_json(expected_summary)
            complete = process.returncode == 0 and result.get('status') == 'complete'
            status.update(status='complete' if complete else ('partial' if timed_out else 'failed'),
                returncode=process.returncode, finished_unix_time=time.time())
            update(name)
            report_and_publish()
            if complete:
                return True
            if timed_out or STOP:
                break
            time.sleep(10)
        return False

    python = sys.executable
    common = ['--data-root', str(args.data_root), '--manifest', str(manifest)]
    # A CUDA throughput pilot is a finite gate, not a small-sample scientific result.
    pilot = study/'pilot'
    if not state.get('paired_configuration'):
        command = [python, 'scripts/train_baseline_downstream.py', *common,
            '--bevcar-source',str(args.bevcar_source), '--checkpoint',str(sources['full']),
            '--run-dir',str(pilot), '--batch-size',str(args.batch_size),
            '--accumulation','1', '--num-workers',str(args.num_workers),
            '--epochs','3','--stop-after-updates','200', '--print-every','25']
        # A deliberate pilot stop is partial by design, so run it as a separate
        # command and inspect its required finite 200-update checkpoint below.
        pilot.mkdir(exist_ok=True)
        if (pilot/'checkpoints/latest.pt').is_file():
            command += ['--resume']
        update('throughput_pilot')
        with (pilot/'console.log').open('a') as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=max(60, min(3600, deadline-time.time())))
        summary = read_json(pilot/'summary.json')
        if result.returncode or summary.get('progress',{}).get('update',0) < 200:
            update('throughput_pilot', status='failed', message='Pilot failed; production was not started')
            raise RuntimeError('CUDA throughput gate failed; see pilot/console.log')
        speed = summary['progress']['samples_seen']/summary['progress']['training_seconds']
        epochs = choose_epochs(speed, (deadline-time.time())/3600)
        state['paired_configuration'] = dict(epochs=epochs, batch_size=args.batch_size,
            accumulation=1, num_workers=args.num_workers, measured_samples_per_second=speed)
        update('throughput_pilot')
    settings = state['paired_configuration']

    for radar, label, checkpoint in [
        (False,'official_camera','8x5_5e-4_rgb12_22:43:46'),
        (True,'official_camera_radar_5sweep','8x5_5e-4_rad25_18:55:34')]:
        directory = study/label
        command = [python, 'scripts/eval_pretrained_baseline.py',*common,
            '--checkpoint',str(ROOT/'checkpoints'/checkpoint/'model-000025000.pth'),
            '--run-dir',str(directory),'--batch-size','2','--num-workers',str(args.num_workers)]
        if radar:
            command += ['--use-radar']
        execute(label, command, directory/'summary.json', budget_seconds=3*3600)

    probe_dir = study/'frozen_1x1_5sweep_1epoch'
    execute(probe_dir.name, [python,'scripts/train_fullsize_linear_probe.py',*common,
        '--bevcar-source',str(args.bevcar_source), '--checkpoint',str(run_root/
            'full28130_sweeps5_seed125/checkpoints/epoch008.pt'),
        '--output-dir',str(probe_dir),'--epochs','1','--batch-size','5',
        '--num-workers',str(args.num_workers)], probe_dir/'summary.json',budget_seconds=2*3600)

    for velocity in ['full', 'zero']:
        directory = study/f'finetune_1sweep_velocity_{velocity}_{settings["epochs"]}epochs'
        command = [python,'scripts/train_baseline_downstream.py',*common,
            '--bevcar-source',str(args.bevcar_source), '--checkpoint',str(sources[velocity]),
            '--run-dir',str(directory),'--epochs',str(settings['epochs']),
            '--batch-size',str(settings['batch_size']),'--accumulation','1',
            '--num-workers',str(settings['num_workers'])]
        execute(directory.name,command,directory/'summary.json')

    # Only run the optional matched-head pair when primary tasks have finished
    # and at least four hours remain. Each mode uses the same head LR/schedule.
    primary_ok = all(row.get('status') == 'complete' for row in state['stages'].values())
    if primary_ok and deadline-time.time() > 4*3600 and not STOP:
        for mode in ['frozen-probe','unfrozen-probe']:
            directory = study/f'same_1x1_head_{mode}_1epoch'
            command = [python,'scripts/train_baseline_downstream.py',*common,
                '--bevcar-source',str(args.bevcar_source), '--checkpoint',str(sources['full']),
                '--run-dir',str(directory),'--mode',mode,'--epochs','1',
                '--weight-decay','0','--batch-size',str(settings['batch_size']),
                '--accumulation','1','--num-workers',str(settings['num_workers'])]
            execute(directory.name,command,directory/'summary.json',budget_seconds=2*3600)
    if primary_ok and deadline-time.time() > 3600 and not STOP:
        directory = study/'depth_and_object_velocity_audit'
        execute(directory.name,[python,'scripts/audit_baseline_followup_targets.py',
            *common,'--run-dir',str(directory)], directory/'summary.json',
            retry=False,budget_seconds=1800)
    complete = all(row.get('status') == 'complete' for row in state['stages'].values())
    update('finished', status='complete' if complete else 'finished_with_partial_or_failed_stages',
           finished_unix_time=time.time(), child_status={})
    report_and_publish()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
