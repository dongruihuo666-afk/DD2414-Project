#!/usr/bin/env python3
"""Summarize three supervised BEVCar mini runs for a meeting figure."""

import argparse
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('camera', 'correct', 'empty', 'wrong')


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--seeds', type=int, nargs='+', default=(125, 126, 127))
    return parser.parse_args()


def read_runs(output_dir, seeds):
    runs = []
    for seed in seeds:
        name = ('bevcar_supervised_mini' if seed == 125
                else f'bevcar_supervised_mini_seed{seed}')
        with (output_dir / f'{name}.json').open() as stream:
            run = json.load(stream)
        if run['seed'] != seed or run['steps'] != 80:
            raise ValueError(f'unexpected experiment configuration for seed {seed}')
        runs.append(run)
    tokens = [[item['sample_token'] for item in run['samples']] for run in runs]
    if any(current != tokens[0] for current in tokens[1:]):
        raise ValueError('the seeds were evaluated on different validation frames')
    return runs


def make_summary(runs):
    result = {
        'seeds': [run['seed'] for run in runs],
        'training_samples': runs[0]['training_samples'],
        'steps_per_seed': runs[0]['steps'],
        'validation_frames_per_seed': len(runs[0]['samples']),
        'validation_scenes': sorted({sample['scene'] for sample in runs[0]['samples']}),
        'pretrained_camera_decoder_used': True,
        'human_box_derived_labels_used': True,
        'frozen_dinov2_used': False,
        'official_bevcar_checkpoint_used': False,
        'condition_iou_per_seed': {
            name: [run['after_summary'][name]['mean_iou'] for run in runs]
            for name in CONDITIONS
        },
        'condition_segmentation_loss_per_seed': {
            name: [run['after_summary'][name]['mean_segmentation_loss'] for run in runs]
            for name in CONDITIONS
        },
    }
    result['condition_iou_mean'] = {
        name: statistics.mean(values)
        for name, values in result['condition_iou_per_seed'].items()
    }
    result['correct_better_than_empty_cases'] = sum(
        item['correct_iou_after'] > item['empty_iou_after']
        for run in runs for item in run['samples']
    )
    result['correct_better_than_wrong_cases'] = sum(
        item['correct_iou_after'] > item['wrong_iou_after']
        for run in runs for item in run['samples']
    )
    result['total_seed_frame_cases'] = sum(len(run['samples']) for run in runs)
    result['mean_correct_empty_probability_difference'] = statistics.mean(
        item['correct_empty_mean_abs_probability']
        for run in runs for item in run['samples']
    )
    result['peak_allocated_cuda_gib'] = max(
        run['peak_allocated_cuda_gib'] for run in runs
    )
    result['mean_training_seconds_per_seed'] = statistics.mean(
        run['training_seconds'] for run in runs
    )
    return result


def render(summary, path):
    canvas = Image.new('RGB', (1200, 665), '#f7f9fc')
    draw = ImageDraw.Draw(canvas)
    fonts = '/usr/share/fonts/truetype/dejavu/'
    title = ImageFont.truetype(fonts + 'DejaVuSans-Bold.ttf', 25)
    body = ImageFont.truetype(fonts + 'DejaVuSans.ttf', 17)
    small = ImageFont.truetype(fonts + 'DejaVuSans.ttf', 14)
    draw.text((28, 22), 'Simple-BEV + BEVCar: supervised mini radar check',
              font=title, fill='#1c2736')
    draw.text((28, 64),
              '4 adaptation frames / 12 mini validation frames / 2 scenes / 3 random seeds',
              font=body, fill='#455468')
    draw.text((28, 93),
              'Frozen pretrained camera + decoder; trainable BEVCar radar + fusion; 80 updates/seed',
              font=small, fill='#455468')
    headers = ('Seed', 'Camera only', 'Correct radar', 'No radar', 'Wrong scene')
    xcoords = (45, 230, 460, 705, 920)
    draw.rectangle((28, 140, 1172, 405), fill='white', outline='#bec8d4', width=2)
    for label, x in zip(headers, xcoords):
        draw.text((x, 158), label, font=body, fill='#455468')
    for row, seed in enumerate(summary['seeds']):
        y = 215 + row * 45
        draw.text((45, y), str(seed), font=body, fill='#1c2736')
        for column, name in enumerate(CONDITIONS):
            value = summary['condition_iou_per_seed'][name][row]
            draw.text((xcoords[column + 1], y), f'{value:.3f}',
                      font=body, fill='#1c2736')
    draw.line((40, 357, 1160, 357), fill='#dce3eb', width=2)
    draw.text((45, 370), 'Mean IoU', font=body, fill='#1c2736')
    for column, name in enumerate(CONDITIONS):
        value = summary['condition_iou_mean'][name]
        draw.text((xcoords[column + 1], 370), f'{value:.3f}',
                  font=title, fill='#3678cc' if name == 'correct' else '#1c2736')
    draw.text((28, 439),
              f'Correct radar beats no radar in '
              f'{summary["correct_better_than_empty_cases"]}/'
              f'{summary["total_seed_frame_cases"]} seed-frame cases; '
              f'beats wrong-scene radar in '
              f'{summary["correct_better_than_wrong_cases"]}/'
              f'{summary["total_seed_frame_cases"]}.',
              font=body, fill='#1c2736')
    draw.text((28, 481),
              'This supports some supervised radar dependence; the gain is modest and '
              'individual frames can get worse.',
              font=body, fill='#1c2736')
    draw.text((28, 548),
              'Important: these scenes were unseen during mini adaptation, not '
              'necessarily unseen by the official camera checkpoint.',
              font=small, fill='#a14929')
    draw.text((28, 580),
              'Not a matched-budget benchmark, not a full nuScenes result, and '
              'not a self-supervised result.',
              font=small, fill='#a14929')
    canvas.save(path)


def main():
    args = arguments()
    runs = read_runs(args.output_dir, args.seeds)
    result = make_summary(runs)
    json_path = args.output_dir / 'bevcar_supervised_summary.json'
    figure_path = args.output_dir / 'bevcar_supervised_summary.png'
    json_path.write_text(json.dumps(result, indent=2) + '\n')
    render(result, figure_path)
    print(f'summary: {json_path}')
    print(f'figure: {figure_path}')
    print('SUPERVISED_BEVCar_SUMMARY_OK')


if __name__ == '__main__':
    main()
