#!/usr/bin/env python3
"""Create a compact, official-metric comparison of both downstream runs."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune-summary', type=Path, required=True)
    parser.add_argument('--linear-probe-summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def load_complete(path, expected_mode):
    value = json.loads(path.read_text())
    if value.get('status') != 'complete':
        raise ValueError(f'{path} is not a complete experiment')
    config = value.get('run_config', {})
    if config.get('mode') != expected_mode:
        raise ValueError(f'{path} does not contain the {expected_mode} run')
    if not value.get('validation'):
        raise ValueError(f'{path} has no validation result')
    epoch = max(int(key) for key in value['validation'])
    return value, epoch, value['validation'][str(epoch)]


def markdown_table(report):
    rows = [
        '# Hybrid-teacher 10-sweep downstream results',
        '',
        '| Experiment | Final epoch | Vehicle BEV IoU | Segmentation CE | Precision* | Recall* | F1* |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for key in ('end_to_end_finetuning', 'frozen_linear_probe'):
        item = report[key]
        rows.append(
            f"| {item['display_name']} | {item['epoch']} | "
            f"{100.0 * item['vehicle_bev_iou']:.3f}% | "
            f"{item['segmentation_ce_loss']:.5f} | "
            f"{100.0 * item['auxiliary_precision']:.3f}% | "
            f"{100.0 * item['auxiliary_recall']:.3f}% | "
            f"{100.0 * item['auxiliary_f1']:.3f}% |"
        )
    rows.extend([
        '',
        'Vehicle BEV IoU is the primary metric and uses the original Simple-BEV '
        'mean-per-frame definition. Values marked with `*` are auxiliary global '
        'confusion-matrix diagnostics and are not substitutes for Vehicle BEV IoU.',
        '',
        'The end-to-end result measures downstream task performance after updating '
        'all student modules. The frozen 1x1-probe result measures how linearly '
        'accessible vehicle semantics were in the pretrained representation.',
    ])
    return '\n'.join(rows) + '\n'


def main():
    args = parse_args()
    finetune, finetune_epoch, finetune_result = load_complete(
        args.finetune_summary, 'finetune',
    )
    probe, probe_epoch, probe_result = load_complete(
        args.linear_probe_summary, 'linear-probe',
    )
    finetune_config = finetune['run_config']
    probe_config = probe['run_config']
    for key in ('pretrain_checkpoint_sha256', 'manifest_sha256', 'nsweeps',
                'train_samples', 'val_samples'):
        if finetune_config[key] != probe_config[key]:
            raise ValueError(f'downstream runs are unmatched for {key}')

    def record(display_name, epoch, result):
        official = result['official_simplebev']
        auxiliary = result['auxiliary_global_confusion']
        return {
            'display_name': display_name,
            'epoch': epoch,
            'vehicle_bev_iou': official['vehicle_bev_iou'],
            'segmentation_ce_loss': official['mean_segmentation_ce_loss'],
            'auxiliary_global_vehicle_iou': auxiliary['global_vehicle_iou'],
            'auxiliary_precision': auxiliary['precision'],
            'auxiliary_recall': auxiliary['recall'],
            'auxiliary_f1': auxiliary['f1'],
        }

    report = {
        'protocol': {
            'shared_pretrain_checkpoint_sha256': (
                finetune_config['pretrain_checkpoint_sha256']
            ),
            'radar_sweeps': finetune_config['nsweeps'],
            'train_samples': finetune_config['train_samples'],
            'val_samples': finetune_config['val_samples'],
            'primary_metric': (
                'official Simple-BEV mean per-frame Vehicle BEV IoU'
            ),
        },
        'end_to_end_finetuning': record(
            'All student modules trainable', finetune_epoch, finetune_result,
        ),
        'frozen_linear_probe': record(
            'Frozen representation plus 1x1 probe', probe_epoch, probe_result,
        ),
        'comparison_boundary': (
            'Published Simple-BEV numbers are directly comparable only after '
            'matching resolution, split, schedule, and evaluation code.'
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'official_metrics.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n'
    )
    (args.output_dir / 'official_metrics.md').write_text(markdown_table(report))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
