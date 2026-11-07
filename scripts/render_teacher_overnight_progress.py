#!/usr/bin/env python3
"""Print a compact status for the unattended teacher-requested pipeline."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def metric(value):
    return 'n/a' if value is None else f'{value:.4f}'


def describe_run(label, path):
    status = read(path / 'status.json')
    if not status:
        return f'{label}: not started'
    text = f'{label}: {status.get("state", "unknown")}'
    if 'epoch_progress' in status:
        text += f' epoch={status["epoch_progress"]:.3f}/{status.get("epochs", 8)}'
    if 'percent_complete' in status:
        text += f' {status["percent_complete"]:.1f}%'
    if status.get('message'):
        text += f' ({status["message"]})'
    return text


def describe_probe(label, path):
    summary = read(path / 'summary.json')
    if summary and summary.get('status') == 'complete':
        result = summary['final_validation']
        return (
            f'{label}: complete IoU={metric(result["vehicle_iou"])} '
            f'P={metric(result["precision"])} R={metric(result["recall"])} '
            f'F1={metric(result["f1"])}'
        )
    status = read(path / 'status.json')
    return f'{label}: {status.get("state", "not started") if status else "not started"}'


def main():
    pipeline = read(ROOT / 'fullsize_baseline' / 'teacher_overnight' / 'status.json')
    print('Teacher overnight pipeline')
    print(f'pipeline: {pipeline.get("state")} / {pipeline.get("stage")}'
          if pipeline else 'pipeline: not started')
    runs = ROOT / 'fullsize_baseline' / 'runs'
    probes = ROOT / 'fullsize_baseline' / 'probes'
    print(describe_run('V+ full velocity', runs / 'full28130_seed125'))
    print(describe_run('V0 zero velocity', runs / 'full28130_zero_velocity_seed125'))
    print(describe_probe('V+ probe epoch 1', probes / 'velocity_full_epoch001'))
    print(describe_probe('V0 probe epoch 1', probes / 'velocity_zero_epoch001'))
    print(describe_probe('V+ probe epoch 8', probes / 'velocity_full_epoch008'))
    print(describe_probe('V0 probe epoch 8', probes / 'velocity_zero_epoch008'))
    if pipeline and pipeline.get('message'):
        print(f'error: {pipeline["message"]}')


if __name__ == '__main__':
    main()
