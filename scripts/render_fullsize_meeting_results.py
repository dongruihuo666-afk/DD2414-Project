#!/usr/bin/env python3
"""Render reproducible meeting figures for the full-data velocity experiment."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from compare_motion_target_bevcar_tiny import build_model, official_voxelnet  # noqa: E402
from dinov2_bev_demo import build_loaders  # noqa: E402
from train_fullsize_linear_probe import (  # noqa: E402
    label_tensors,
    resolve_velocity_mode,
    shared_bev,
)


EPOCHS = (1, 3, 5, 8)
METRICS = ('vehicle_iou', 'precision', 'recall', 'f1')
RAW_LABEL = 'Camera + radar spatial features + raw velocity'
ZERO_LABEL = 'Camera + radar spatial features; velocity channels zeroed'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root', type=Path,
        default=Path.home() / 'datasets' / 'nuscenes',
    )
    parser.add_argument(
        '--bevcar-source', type=Path,
        default=REPO_ROOT / 'external' / 'BEVCar',
    )
    parser.add_argument(
        '--manifest', type=Path,
        default=REPO_ROOT / 'configs' / 'radar_scaling_manifest_seed125.json',
    )
    parser.add_argument(
        '--raw-run', type=Path,
        default=REPO_ROOT / 'fullsize_baseline' / 'runs' / 'full28130_seed125',
    )
    parser.add_argument(
        '--zero-run', type=Path,
        default=(REPO_ROOT / 'fullsize_baseline' / 'runs'
                 / 'full28130_zero_velocity_seed125'),
    )
    parser.add_argument(
        '--raw-probe', type=Path,
        default=(REPO_ROOT / 'fullsize_baseline' / 'probes'
                 / 'velocity_full_epoch008' / 'probe_latest.pt'),
    )
    parser.add_argument(
        '--zero-probe', type=Path,
        default=(REPO_ROOT / 'fullsize_baseline' / 'probes'
                 / 'velocity_zero_epoch008' / 'probe_latest.pt'),
    )
    parser.add_argument(
        '--comparison-root', type=Path,
        default=REPO_ROOT / 'fullsize_baseline' / 'teacher_overnight',
    )
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=125)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text())


def setup_plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'figure.titlesize': 18,
        'legend.fontsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    return plt


def save_figure(fig, path, dpi=180):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.png')
    fig.savefig(temporary, dpi=dpi, bbox_inches='tight', facecolor='white')
    temporary.replace(path)


def validation_series(run_dir):
    output = {}
    for epoch in EPOCHS:
        output[epoch] = load_json(
            Path(run_dir) / 'validation' / f'epoch{epoch:03d}_summary.json'
        )
    return output


def render_training_curves(plt, raw, zero, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    colors = ('#1261A0', '#D97706')
    for values, label, color in (
        (raw, RAW_LABEL, colors[0]),
        (zero, ZERO_LABEL, colors[1]),
    ):
        semantic = [values[e]['means']['matched']['semantic_loss'] for e in EPOCHS]
        motion = [values[e]['means']['matched']['motion_loss'] for e in EPOCHS]
        axes[0].plot(EPOCHS, semantic, marker='o', linewidth=2.5, label=label,
                     color=color)
        axes[1].plot(EPOCHS, motion, marker='o', linewidth=2.5, label=label,
                     color=color)
    axes[0].set_title('Held-out DINO-BEV semantic loss')
    axes[1].set_title('Held-out radar motion loss')
    axes[0].set_ylabel('Loss (lower is better)')
    axes[1].set_ylabel('Loss (lower is better)')
    for axis in axes:
        axis.set_xlabel('Completed training epochs')
        axis.set_xticks(EPOCHS)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].annotate(
        'Small final semantic gap', xy=(8, raw[8]['means']['matched']['semantic_loss']),
        xytext=(4.4, 0.294), arrowprops={'arrowstyle': '->', 'color': '#444'},
    )
    axes[1].annotate(
        'Raw velocity makes the motion task learnable',
        xy=(8, raw[8]['means']['matched']['motion_loss']), xytext=(3.5, 0.72),
        arrowprops={'arrowstyle': '->', 'color': '#444'},
    )
    fig.suptitle('Full nuScenes training: validation trends')
    save_figure(fig, path)
    plt.close(fig)


def render_radar_intervention(plt, raw_epoch8, zero_epoch8, path):
    modes = ('matched', 'zero_velocity', 'empty', 'wrong_scene')
    labels = ('Matched radar', 'Velocity zeroed', 'Empty radar', 'Wrong-scene radar')
    semantic = [raw_epoch8['means'][mode]['semantic_loss'] for mode in modes]
    motion = [raw_epoch8['means'][mode]['motion_loss'] for mode in modes]
    colors = ('#238B45', '#F2B134', '#C73E1D', '#7B2CBF')
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    x = np.arange(len(labels))
    bars0 = axes[0].bar(x, semantic, color=colors)
    bars1 = axes[1].bar(x, motion, color=colors)
    for axis, bars, values, title in (
        (axes[0], bars0, semantic, 'DINO-BEV semantic loss'),
        (axes[1], bars1, motion, 'Radar motion loss'),
    ):
        axis.set_xticks(x, labels, rotation=18, ha='right')
        axis.set_ylabel('Loss (lower is better)')
        axis.set_title(title)
        axis.grid(axis='y', alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value,
                      f'{value:.3f}', ha='center', va='bottom')
    zero_matched_semantic = zero_epoch8['means']['matched']['semantic_loss']
    axes[0].axhline(zero_matched_semantic, color='#D97706', linestyle='--',
                    linewidth=1.6, label=(
                        'Separately trained velocity-zeroed model '
                        f'({zero_matched_semantic:.3f})'
                    ))
    axes[0].legend(frameon=False)
    fig.suptitle('Epoch 8 input intervention: correct aligned radar is best')
    save_figure(fig, path)
    plt.close(fig)


def render_probe_summary(plt, comparison_root, path):
    reports = {
        epoch: load_json(Path(comparison_root) / f'epoch{epoch:03d}' / 'comparison.json')
        for epoch in (1, 8)
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    x = np.arange(2)
    width = 0.34
    colors = ('#1261A0', '#D97706')

    raw_iou = [100 * reports[e]['full']['vehicle_iou'] for e in (1, 8)]
    zero_iou = [100 * reports[e]['zero']['vehicle_iou'] for e in (1, 8)]
    axes[0, 0].bar(x - width / 2, raw_iou, width, label=RAW_LABEL, color=colors[0])
    axes[0, 0].bar(x + width / 2, zero_iou, width, label=ZERO_LABEL, color=colors[1])
    axes[0, 0].set_xticks(x, ('Epoch 1', 'Epoch 8'))
    axes[0, 0].set_ylabel('Vehicle IoU (%)')
    axes[0, 0].set_title('Frozen one-layer vehicle probe')
    axes[0, 0].legend(frameon=False)

    deltas = [100 * reports[e]['velocity_utility']['vehicle_iou'] for e in (1, 8)]
    bars = axes[0, 1].bar(x, deltas, color='#238B45')
    axes[0, 1].axhline(0, color='black', linewidth=0.8)
    axes[0, 1].set_xticks(x, ('Epoch 1', 'Epoch 8'))
    axes[0, 1].set_ylabel('IoU difference (percentage points)')
    axes[0, 1].set_title('Raw-velocity gain is positive but small')
    for bar, value in zip(bars, deltas):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, value,
                        f'+{value:.3f} pp', ha='center', va='bottom')

    report8 = reports[8]
    metric_labels = ('Precision', 'Recall', 'F1')
    raw_metrics = [100 * report8['full'][key] for key in ('precision', 'recall', 'f1')]
    zero_metrics = [100 * report8['zero'][key] for key in ('precision', 'recall', 'f1')]
    xx = np.arange(3)
    axes[1, 0].bar(xx - width / 2, raw_metrics, width, color=colors[0],
                   label='Raw velocity included')
    axes[1, 0].bar(xx + width / 2, zero_metrics, width, color=colors[1],
                   label='Velocity channels zeroed')
    axes[1, 0].set_xticks(xx, metric_labels)
    axes[1, 0].set_ylabel('Score (%)')
    axes[1, 0].set_title('Epoch 8: high recall, very low precision')
    axes[1, 0].legend(frameon=False)

    raw = report8['full']
    zero = report8['zero']
    fp_per_tp = (raw['fp'] / raw['tp'], zero['fp'] / zero['tp'])
    bars = axes[1, 1].bar(x, fp_per_tp, color=colors)
    axes[1, 1].set_xticks(x, ('Raw velocity\nincluded',
                              'Velocity channels\nzeroed'))
    axes[1, 1].set_ylabel('False-positive cells per true-positive cell')
    axes[1, 1].set_title('Primary failure: widespread false positives')
    for bar, value in zip(bars, fp_per_tp):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value,
                        f'{value:.1f}×', ha='center', va='bottom')
    for axis in axes.flat:
        axis.grid(axis='y', alpha=0.25)
    fig.suptitle('Downstream representation quality after frozen linear probing')
    save_figure(fig, path)
    plt.close(fig)


def ordered_unique(values):
    return list(dict.fromkeys(values))


def evenly_spaced_scene_tokens(scene_tokens, count):
    scenes = ordered_unique(scene_tokens)
    if count < 1 or count > len(scenes):
        raise ValueError('sample count must fit the validation scene count')
    positions = np.linspace(0, len(scenes) - 1, count).round().astype(int)
    return [scenes[position] for position in positions]


def item_masks(item):
    target = item[12][0, 0].bool()
    valid = item[13][0, 0].bool()
    return target, valid


def select_scene_diverse_vehicle_items(dataset, manifest, count):
    scene_tokens = manifest['val']['scene_tokens']
    selected_scenes = evenly_spaced_scene_tokens(scene_tokens, count)
    output = []
    for scene_token in selected_scenes:
        positions = [i for i, scene in enumerate(scene_tokens) if scene == scene_token]
        midpoint = (len(positions) - 1) / 2
        candidate_positions = sorted(
            positions, key=lambda position: (abs(positions.index(position) - midpoint), position)
        )
        chosen = None
        for manifest_position in candidate_positions:
            dataset_index = manifest['val']['dataset_indices'][manifest_position]
            item = dataset[dataset_index]
            target, valid = item_masks(item)
            if int((target & valid).sum()) > 0:
                chosen = {
                    'manifest_position': manifest_position,
                    'dataset_index': dataset_index,
                    'token': manifest['val']['tokens'][manifest_position],
                    'scene_token': scene_token,
                    'item': item,
                    'gt_cells': int((target & valid).sum()),
                }
                break
        if chosen is None:
            raise RuntimeError(f'no valid vehicle cells found in scene {scene_token}')
        output.append(chosen)
    return output


def load_predictions(checkpoint_path, probe_path, items, bevcar_source, seed,
                     device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    velocity_mode = resolve_velocity_mode('auto', checkpoint)
    official_class = official_voxelnet(bevcar_source)
    model = build_model(device, seed, official_class, zero_camera=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    probe_payload = torch.load(probe_path, map_location='cpu', weights_only=False)
    probe = nn.Conv2d(model.student.latent_dim, 1, kernel_size=1).to(device)
    probe.load_state_dict(probe_payload['probe'], strict=True)
    probe.eval()
    predictions = []
    with torch.inference_mode():
        for selected in items:
            batch = default_collate([selected['item']])
            features = shared_bev(
                model, batch, device, velocity_mode, torch.bfloat16,
            )
            target, valid = label_tensors(batch, device)
            probability = probe(features.float()).sigmoid()
            predictions.append({
                'probability': probability[0, 0].cpu().numpy(),
                'prediction': probability[0, 0].ge(0.5).cpu().numpy(),
                'target': target[0, 0].gt(0.5).cpu().numpy(),
                'valid': valid[0, 0].gt(0.5).cpu().numpy(),
            })
    del checkpoint, probe_payload, probe, model
    torch.cuda.empty_cache()
    return velocity_mode, predictions


def sample_iou(prediction, target, valid):
    intersection = np.logical_and(prediction, target) & valid
    union = np.logical_or(prediction, target) & valid
    denominator = int(union.sum())
    return float(intersection.sum() / denominator) if denominator else None


def binary_rgb(mask, valid, color):
    image = np.zeros((*mask.shape, 3), dtype=np.float32)
    image[~valid] = (0.18, 0.18, 0.18)
    image[valid] = (0.01, 0.01, 0.01)
    image[mask & valid] = color
    return image


def error_rgb(prediction, target, valid):
    image = np.zeros((*prediction.shape, 3), dtype=np.float32)
    image[~valid] = (0.18, 0.18, 0.18)
    tp = prediction & target & valid
    fp = prediction & ~target & valid
    fn = ~prediction & target & valid
    image[tp] = (0.15, 0.78, 0.34)
    image[fp] = (0.90, 0.20, 0.18)
    image[fn] = (0.18, 0.46, 0.95)
    return image


def radar_return_count(item):
    radar = item[16][0]
    return int(radar[:3].abs().sum(dim=0).gt(0).sum())


def render_mask_comparison(plt, selected, raw_predictions, zero_predictions,
                           path):
    columns = 5
    fig, axes = plt.subplots(
        len(selected), columns, figsize=(18, 3.65 * len(selected)),
        constrained_layout=True,
    )
    if len(selected) == 1:
        axes = axes[None, :]
    titles = (
        'nuScenes GT vehicle mask',
        'Prediction: raw velocity included',
        'Error: raw velocity included',
        'Prediction: velocity channels zeroed',
        'Error: velocity channels zeroed',
    )
    metadata = []
    for row, (sample, raw, zero) in enumerate(
            zip(selected, raw_predictions, zero_predictions)):
        target, valid = raw['target'], raw['valid']
        raw_iou = sample_iou(raw['prediction'], target, valid)
        zero_iou = sample_iou(zero['prediction'], target, valid)
        images = (
            binary_rgb(target, valid, (1.0, 1.0, 1.0)),
            binary_rgb(raw['prediction'], valid, (0.10, 0.75, 0.95)),
            error_rgb(raw['prediction'], target, valid),
            binary_rgb(zero['prediction'], valid, (0.96, 0.62, 0.12)),
            error_rgb(zero['prediction'], target, valid),
        )
        for column, image in enumerate(images):
            axis = axes[row, column]
            axis.imshow(image, origin='lower', interpolation='nearest')
            axis.scatter([99.5], [99.5], marker='^', s=24, c='white',
                         edgecolors='black', linewidths=0.5)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(titles[column])
        axes[row, 0].set_ylabel(
            f'Scene {row + 1}\nGT cells={sample["gt_cells"]}\n'
            f'Radar returns={radar_return_count(sample["item"])}',
        )
        axes[row, 1].text(
            0.02, 0.98, f'IoU={100 * raw_iou:.2f}%', transform=axes[row, 1].transAxes,
            va='top', color='white', bbox={'facecolor': 'black', 'alpha': 0.7},
        )
        axes[row, 3].text(
            0.02, 0.98, f'IoU={100 * zero_iou:.2f}%', transform=axes[row, 3].transAxes,
            va='top', color='white', bbox={'facecolor': 'black', 'alpha': 0.7},
        )
        metadata.append({
            'dataset_index': sample['dataset_index'],
            'sample_token': sample['token'],
            'scene_token': sample['scene_token'],
            'gt_vehicle_cells': sample['gt_cells'],
            'radar_returns': radar_return_count(sample['item']),
            'raw_velocity_vehicle_iou': raw_iou,
            'velocity_zeroed_vehicle_iou': zero_iou,
        })
    fig.suptitle(
        'Direct mask comparison on fixed, scene-diverse validation samples\n'
        'Error maps: green=true positive, red=false positive, blue=false negative; triangle=ego',
    )
    save_figure(fig, path, dpi=160)
    plt.close(fig)
    return metadata


def add_card(axis, xy, width, height, title, value, note, color):
    import matplotlib.patches as patches

    x, y = xy
    card = patches.FancyBboxPatch(
        (x, y), width, height, boxstyle='round,pad=0.015,rounding_size=0.025',
        transform=axis.transAxes, facecolor=color, edgecolor='none', alpha=0.96,
    )
    axis.add_patch(card)
    axis.text(x + 0.04 * width, y + 0.78 * height, title, transform=axis.transAxes,
              fontsize=12, color='white', weight='bold', va='top')
    axis.text(x + 0.04 * width, y + 0.53 * height, value, transform=axis.transAxes,
              fontsize=23, color='white', weight='bold', va='top')
    axis.text(x + 0.04 * width, y + 0.12 * height, note, transform=axis.transAxes,
              fontsize=10, color='white', va='bottom', wrap=True)


def render_overview(plt, comparison_root, raw_epoch8, zero_epoch8, path):
    report = load_json(Path(comparison_root) / 'epoch008' / 'comparison.json')
    raw = report['full']
    zero = report['zero']
    figure = plt.figure(figsize=(16, 9))
    axis = figure.add_axes((0, 0, 1, 1))
    axis.axis('off')
    axis.text(0.05, 0.94, 'Full nuScenes radar-velocity experiment', fontsize=27,
              weight='bold', va='top')
    axis.text(
        0.05, 0.885,
        '28,130 train samples × 8 epochs  →  6,019-frame validation  →  frozen 1×1 vehicle probe',
        fontsize=14, color='#444', va='top',
    )
    add_card(
        axis, (0.05, 0.56), 0.28, 0.25, 'Radar motion loss',
        f'{raw_epoch8["means"]["matched"]["motion_loss"]:.3f} vs '
        f'{zero_epoch8["means"]["matched"]["motion_loss"]:.3f}',
        'Raw velocity is clearly learned: about 94% lower held-out motion loss.',
        '#1261A0',
    )
    delta_pp = 100 * report['velocity_utility']['vehicle_iou']
    add_card(
        axis, (0.36, 0.56), 0.28, 0.25, 'Vehicle IoU',
        f'{100 * raw["vehicle_iou"]:.2f}% vs {100 * zero["vehicle_iou"]:.2f}%',
        f'Raw-velocity gain is only +{delta_pp:.3f} percentage points.',
        '#238B45',
    )
    add_card(
        axis, (0.67, 0.56), 0.28, 0.25, 'Main semantic failure',
        f'{100 * raw["precision"]:.2f}% precision',
        f'Recall is {100 * raw["recall"]:.1f}%, but false positives dominate.',
        '#C73E1D',
    )
    axis.text(0.05, 0.47, 'What the experiment establishes', fontsize=16,
              weight='bold')
    conclusions = [
        ('1', 'Aligned radar is used',
         'Empty and wrong-scene radar substantially worsen semantic and motion losses.'),
        ('2', 'Raw velocity helps motion, not much semantics',
         'The very large motion-loss benefit becomes only a tiny vehicle-IoU benefit.'),
        ('3', 'The representation is not yet a strong vehicle segmenter',
         'About 5% Vehicle IoU points to the frozen random camera encoder and sparse objective.'),
    ]
    y = 0.39
    for number, title, detail in conclusions:
        axis.text(0.06, y, number, fontsize=17, weight='bold', color='white',
                  ha='center', va='center',
                  bbox={'boxstyle': 'circle', 'facecolor': '#343A40', 'edgecolor': 'none'})
        axis.text(0.095, y + 0.015, title, fontsize=13, weight='bold', va='center')
        axis.text(0.095, y - 0.025, detail, fontsize=11, color='#444', va='center')
        y -= 0.105
    axis.text(
        0.05, 0.045,
        'Scope: one seed, one radar sweep, raw vx/vy input; camera ResNet is random-initialized and frozen. '
        'This is a controlled legacy baseline, not the final dual-level-distillation model.',
        fontsize=10.5, color='#555', va='bottom',
    )
    save_figure(figure, path, dpi=170)
    plt.close(figure)


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError('samples must be positive')
    plt = setup_plotting()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_series = validation_series(args.raw_run)
    zero_series = validation_series(args.zero_run)
    outputs = {
        'training_curves': args.output_dir / 'fullsize_training_validation_curves.png',
        'radar_intervention': args.output_dir / 'fullsize_radar_intervention.png',
        'linear_probe': args.output_dir / 'fullsize_linear_probe_summary.png',
        'mask_comparison': args.output_dir / 'fullsize_vehicle_mask_comparison.png',
        'overview': args.output_dir / 'fullsize_experiment_overview.png',
    }
    render_training_curves(plt, raw_series, zero_series, outputs['training_curves'])
    render_radar_intervention(
        plt, raw_series[8], zero_series[8], outputs['radar_intervention'],
    )
    render_probe_summary(plt, args.comparison_root, outputs['linear_probe'])
    render_overview(
        plt, args.comparison_root, raw_series[8], zero_series[8], outputs['overview'],
    )

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for prediction-mask rendering')
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = load_json(args.manifest)
    _, val_loader_raw = build_loaders(
        args.data_root, num_workers=0, nsweeps=1,
        rotate_radar_velocity=True, dset='trainval',
    )
    selected = select_scene_diverse_vehicle_items(
        val_loader_raw.dataset, manifest, args.samples,
    )
    raw_mode, raw_predictions = load_predictions(
        Path(args.raw_run) / 'checkpoints' / 'epoch008.pt', args.raw_probe,
        selected, args.bevcar_source, args.seed, device,
    )
    zero_mode, zero_predictions = load_predictions(
        Path(args.zero_run) / 'checkpoints' / 'epoch008.pt', args.zero_probe,
        selected, args.bevcar_source, args.seed, device,
    )
    if (raw_mode, zero_mode) != ('full', 'zero'):
        raise RuntimeError(f'unexpected checkpoint modes: {raw_mode}, {zero_mode}')
    sample_metadata = render_mask_comparison(
        plt, selected, raw_predictions, zero_predictions,
        outputs['mask_comparison'],
    )
    metadata = {
        'experiment': 'fullsize_raw_velocity_meeting_visuals_v1',
        'selection': (
            'Four evenly spaced validation scenes; within each scene, choose the '
            'sample nearest its temporal midpoint that has at least one valid GT '
            'vehicle cell. Selection never uses model predictions.'
        ),
        'threshold': 0.5,
        'samples': sample_metadata,
        'sources': {
            'raw_velocity_run': str(args.raw_run.resolve()),
            'velocity_zeroed_run': str(args.zero_run.resolve()),
            'raw_velocity_probe': str(args.raw_probe.resolve()),
            'velocity_zeroed_probe': str(args.zero_probe.resolve()),
            'manifest': str(args.manifest.resolve()),
        },
        'outputs': {key: str(value.resolve()) for key, value in outputs.items()},
        'limitations': [
            'One training seed and one radar sweep.',
            'Binary vehicle mask, not multi-class semantic mIoU.',
            'The camera ResNet is random-initialized and frozen in this legacy baseline.',
            'Per-sample examples are deterministic illustrations; aggregate metrics use all 6,019 validation frames.',
        ],
    }
    metadata_path = args.output_dir / 'fullsize_meeting_visuals.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    for key, value in outputs.items():
        print(f'{key}: {value}', flush=True)
    print(f'metadata: {metadata_path}', flush=True)
    print('FULLSIZE_MEETING_VISUALS_OK', flush=True)


if __name__ == '__main__':
    main()
