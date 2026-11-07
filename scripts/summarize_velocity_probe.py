#!/usr/bin/env python3
"""Compare matched-budget V+ and V0 frozen linear probes."""

import argparse
import json
from pathlib import Path

from fullsize_training import atomic_json_write


METRICS = ('vehicle_iou', 'precision', 'recall', 'f1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-summary', type=Path, required=True)
    parser.add_argument('--zero-summary', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    parser.add_argument('--label', required=True)
    return parser.parse_args()


def load_summary(path, expected_mode):
    value = json.loads(path.read_text())
    if value.get('status') != 'complete':
        raise ValueError(f'incomplete probe summary: {path}')
    if value['radar_velocity_mode'] != expected_mode:
        raise ValueError(f'{path}: expected mode {expected_mode}')
    return value


def format_value(value):
    return 'n/a' if value is None else f'{value:.6f}'


def main():
    args = parse_args()
    full = load_summary(args.full_summary, 'full')
    zero = load_summary(args.zero_summary, 'zero')
    full_metrics = full['final_validation']
    zero_metrics = zero['final_validation']
    delta = {
        metric: (
            full_metrics[metric] - zero_metrics[metric]
            if full_metrics[metric] is not None and zero_metrics[metric] is not None
            else None
        )
        for metric in METRICS
    }
    result = {
        'label': args.label,
        'comparison': 'full radar velocity minus train-time zero radar velocity',
        'full': full_metrics,
        'zero': zero_metrics,
        'velocity_utility': delta,
        'matched_budget': {
            'backbone_epoch': [full['backbone_epoch'], zero['backbone_epoch']],
            'probe_epochs': [full['probe_epochs'], zero['probe_epochs']],
            'train_samples': [full['train_samples'], zero['train_samples']],
            'val_samples': [full['val_samples'], zero['val_samples']],
        },
        'interpretation': (
            'Positive IoU/F1 utility means explicit radar velocity improved linear '
            'vehicle separability under this frozen-camera controlled baseline.'
        ),
    }
    atomic_json_write(result, args.output_json)
    lines = [
        f'# Velocity-input linear-probe comparison: {args.label}',
        '',
        '| Input during backbone training | Vehicle IoU | Precision | Recall | F1 |',
        '| --- | ---: | ---: | ---: | ---: |',
        '| Full radar velocity | ' + ' | '.join(
            format_value(full_metrics[metric]) for metric in METRICS
        ) + ' |',
        '| Zero radar velocity | ' + ' | '.join(
            format_value(zero_metrics[metric]) for metric in METRICS
        ) + ' |',
        '| V+ minus V0 | ' + ' | '.join(
            format_value(delta[metric]) for metric in METRICS
        ) + ' |',
        '',
        'Positive V+ minus V0 IoU/F1 indicates that explicit radar velocity made the ',
        'frozen shared-BEV representation more linearly useful for vehicle segmentation.',
        '',
        'Limitation: both backbones use the historical frozen random camera encoder; ',
        'this is a controlled velocity-input result, not a fully trained camera-radar model.',
        '',
    ]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)
    print('VELOCITY_PROBE_SUMMARY_OK', flush=True)


if __name__ == '__main__':
    main()
