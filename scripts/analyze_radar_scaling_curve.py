#!/usr/bin/env python3
"""Analyze the fixed-budget full-data radar-dependence scaling curve."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ('64', 64, 'radar_scaling_scale64_seed125.json'),
    ('256', 256, 'radar_scaling_scale256_seed125.json'),
    ('1024', 1024, 'radar_scaling_scale1024_seed125.json'),
    ('4096', 4096, 'radar_scaling_scale4096_seed125.json'),
    ('full', 28130, 'radar_scaling_full_seed125.json'),
)
MODES = ('zero_velocity', 'empty', 'wrong_scene')
METRICS = (
    'semantic_loss',
    'motion_loss',
    'moving_frame_motion_loss',
    'cell_weighted_motion_loss',
)
WEIGHT_KEYS = {
    'semantic_loss': 'samples',
    'motion_loss': 'samples',
    'moving_frame_motion_loss': 'moving_frames',
    'cell_weighted_motion_loss': 'covered_cells',
}
MODE_LABELS = {
    'zero_velocity': 'Zero velocity',
    'empty': 'Empty radar',
    'wrong_scene': 'Wrong scene',
}
COLORS = {
    'zero_velocity': '#3568b8',
    'empty': '#c85b32',
    'wrong_scene': '#2c8b66',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir', type=Path,
        default=ROOT / 'artifacts' / 'trainval',
    )
    parser.add_argument(
        '--summary-out', type=Path,
        default=ROOT / 'artifacts' / 'trainval' / 'radar_scaling_curve_seed125.json',
    )
    parser.add_argument(
        '--figure-out', type=Path,
        default=ROOT / 'artifacts' / 'trainval' / 'radar_scaling_curve_seed125.png',
    )
    parser.add_argument(
        '--report-out', type=Path,
        default=ROOT / 'RADAR_SCALING_ANALYSIS.md',
    )
    parser.add_argument('--bootstrap-iterations', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261007)
    return parser.parse_args()


def load_runs(input_dir):
    runs = []
    reference_scenes = None
    for label, train_samples, filename in RUNS:
        path = input_dir / filename
        value = json.loads(path.read_text())
        config = value['run_config']
        validation = value['validation']
        if value['status'] != 'complete':
            raise ValueError(f'incomplete run: {path}')
        expected = {
            'seed': 125,
            'nsweeps': 1,
            'max_updates': 30000,
            'semantic_weight': 1.0,
            'motion_weight': 0.5,
            'train_samples': train_samples,
        }
        for key, expected_value in expected.items():
            if config[key] != expected_value:
                raise ValueError(
                    f'{path}: expected {key}={expected_value}, got {config[key]}'
                )
        if validation['samples'] != 6019 or len(validation['per_scene']) != 150:
            raise ValueError(f'{path}: expected 6,019 frames and 150 scenes')
        scenes = sorted(validation['per_scene'])
        if reference_scenes is None:
            reference_scenes = scenes
        elif scenes != reference_scenes:
            raise ValueError(f'{path}: validation scenes differ from earlier scales')
        try:
            source = str(path.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            source = str(path)
        runs.append({
            'label': label,
            'train_samples': train_samples,
            'path': source,
            'value': value,
        })
    return runs, reference_scenes


def scene_values(run, scenes, metric, mode):
    """Return paired scene penalties and their aggregation weights."""
    per_scene = run['validation']['per_scene']
    weight_key = WEIGHT_KEYS[metric]
    values = []
    weights = []
    for token in scenes:
        scene = per_scene[token]
        penalty = scene['penalties'][metric][mode]
        weight = int(scene[weight_key])
        if penalty is None:
            if weight != 0:
                raise ValueError(f'{token}: missing {metric} with nonzero weight')
            penalty = 0.0
        values.append(float(penalty))
        weights.append(float(weight))
    return np.asarray(values), np.asarray(weights)


def weighted_mean(values, weights):
    denominator = float(weights.sum())
    if denominator <= 0:
        raise ValueError('weighted mean requires positive total weight')
    return float(np.dot(values, weights) / denominator)


def bootstrap_interval(values, weights, sample_indices):
    """Return a deterministic percentile CI from scene-resampling indices."""
    sampled_weights = weights[sample_indices]
    denominator = sampled_weights.sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError('a bootstrap draw has no eligible scene weight')
    draws = (values[sample_indices] * sampled_weights).sum(axis=1) / denominator
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def earliest_persistent_scale(scales, intervals):
    """Find the earliest scale whose and all later lower CI bounds exceed zero."""
    for index, scale in enumerate(scales):
        if all(interval[0] > 0 for interval in intervals[index:]):
            return scale
    return None


def analyze(runs, scenes, iterations, seed):
    if iterations < 1000:
        raise ValueError('use at least 1,000 bootstrap iterations')
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0, len(scenes), size=(iterations, len(scenes)), endpoint=False,
    )
    scales = []
    for item in runs:
        run = item['value']
        validation = run['validation']
        scale = {
            'label': item['label'],
            'train_samples': item['train_samples'],
            'source': item['path'],
            'validation_samples': validation['samples'],
            'validation_scenes': len(validation['per_scene']),
            'matched': validation['means']['matched'],
            'penalties': {},
        }
        for metric in METRICS:
            scale['penalties'][metric] = {}
            for mode in MODES:
                values, weights = scene_values(run, scenes, metric, mode)
                reconstructed = weighted_mean(values, weights)
                estimate = float(validation['penalties'][metric][mode])
                if not np.isclose(reconstructed, estimate, atol=1e-9, rtol=1e-9):
                    raise ValueError(
                        f'{item["label"]} {metric} {mode}: scene aggregation '
                        f'{reconstructed} != stored aggregate {estimate}'
                    )
                low, high = bootstrap_interval(values, weights, sample_indices)
                eligible = weights > 0
                scale['penalties'][metric][mode] = {
                    'estimate': estimate,
                    'ci95_low': low,
                    'ci95_high': high,
                    'positive_scenes': int(np.sum((values > 0) & eligible)),
                    'eligible_scenes': int(np.sum(eligible)),
                }
        scales.append(scale)

    labels = [scale['label'] for scale in scales]
    findings = {'persistent_positive_ci_onset': {}, 'peak': {}}
    for metric in METRICS:
        findings['persistent_positive_ci_onset'][metric] = {}
        findings['peak'][metric] = {}
        for mode in MODES:
            intervals = [
                (scale['penalties'][metric][mode]['ci95_low'],
                 scale['penalties'][metric][mode]['ci95_high'])
                for scale in scales
            ]
            findings['persistent_positive_ci_onset'][metric][mode] = (
                earliest_persistent_scale(labels, intervals)
            )
            peak = max(
                scales,
                key=lambda scale: scale['penalties'][metric][mode]['estimate'],
            )
            findings['peak'][metric][mode] = {
                'scale': peak['label'],
                'estimate': peak['penalties'][metric][mode]['estimate'],
            }

    p8, full = scales[-2], scales[-1]
    findings['full_minus_4096'] = {}
    for metric in METRICS:
        findings['full_minus_4096'][metric] = {}
        for mode in MODES:
            p8_values, p8_weights = scene_values(
                runs[-2]['value'], scenes, metric, mode,
            )
            full_values, full_weights = scene_values(
                runs[-1]['value'], scenes, metric, mode,
            )
            if not np.array_equal(p8_weights, full_weights):
                raise ValueError(f'{metric}: validation weights differ across scales')
            differences = full_values - p8_values
            estimate = (full['penalties'][metric][mode]['estimate']
                        - p8['penalties'][metric][mode]['estimate'])
            reconstructed = weighted_mean(differences, full_weights)
            if not np.isclose(reconstructed, estimate, atol=1e-9, rtol=1e-9):
                raise ValueError(f'{metric} {mode}: cross-scale delta mismatch')
            low, high = bootstrap_interval(
                differences, full_weights, sample_indices,
            )
            findings['full_minus_4096'][metric][mode] = {
                'estimate': estimate,
                'ci95_low': low,
                'ci95_high': high,
            }
    findings['matched_64_to_full'] = {
        metric: {
            'scale64': scales[0]['matched'][metric],
            'full': scales[-1]['matched'][metric],
            'change': scales[-1]['matched'][metric] - scales[0]['matched'][metric],
        }
        for metric in METRICS
    }
    return {
        'schema_version': 1,
        'protocol': {
            'dataset': 'v1.0-trainval',
            'seed': 125,
            'radar_sweeps': 1,
            'updates_per_scale': 30000,
            'validation_frames': 6019,
            'validation_scenes': len(scenes),
            'bootstrap_unit': 'scene',
            'bootstrap_iterations': iterations,
            'bootstrap_seed': seed,
            'bootstrap_interval': 'percentile 95%',
            'aggregation': (
                'paired per-scene penalties, resampled by scene and weighted by '
                'frames, moving frames, or covered cells to match each stored metric'
            ),
        },
        'scales': scales,
        'findings': findings,
        'limitations': [
            'All five points use only training seed 125; scene bootstrap does not '
            'measure random-seed or training-run variability.',
            'The camera encoder is randomly initialized and frozen.',
            'Semantic and motion losses are diagnostic targets, not vehicle IoU.',
            'The fixed 30,000-update budget gives smaller subsets more repeated '
            'exposure per token than the full-data run.',
        ],
    }


def penalty_cell(scale, metric, mode):
    value = scale['penalties'][metric][mode]
    return (f'{value["estimate"]:+.5f} '
            f'[{value["ci95_low"]:+.5f}, {value["ci95_high"]:+.5f}]')


def write_report(summary, path):
    scales = summary['scales']
    onset = summary['findings']['persistent_positive_ci_onset']
    lines = [
        '# Radar dependence scaling analysis',
        '',
        '## Scope and method',
        '',
        'This report analyzes the locked seed-125, one-sweep curve at 64, 256, '
        '1,024, 4,096 and 28,130 training samples. Every point used 30,000 '
        'optimizer updates and the same 6,019-frame / 150-scene validation split. '
        'Intervals are deterministic paired scene-bootstrap percentile 95% '
        f'intervals ({summary["protocol"]["bootstrap_iterations"]:,} resamples). '
        'Scenes, rather than temporally adjacent '
        'frames, are the resampling unit.',
        '',
        'Positive penalties mean that modifying or mismatching radar makes the '
        'loss worse than matched radar. They measure radar dependence, not '
        'downstream vehicle-segmentation accuracy.',
        '',
        '![Radar dependence curve](artifacts/trainval/radar_scaling_curve_seed125.png)',
        '',
        '## Semantic cosine-loss penalties',
        '',
        '| Training samples | Matched loss | Zero velocity | Empty radar | Wrong scene |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ]
    for scale in scales:
        lines.append(
            f'| {scale["train_samples"]:,} | '
            f'{scale["matched"]["semantic_loss"]:.5f} | '
            f'{penalty_cell(scale, "semantic_loss", "zero_velocity")} | '
            f'{penalty_cell(scale, "semantic_loss", "empty")} | '
            f'{penalty_cell(scale, "semantic_loss", "wrong_scene")} |'
        )
    lines.extend([
        '',
        '## Motion-loss penalties',
        '',
        '| Training samples | Matched loss | Zero velocity | Empty radar | Wrong scene |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ])
    for scale in scales:
        lines.append(
            f'| {scale["train_samples"]:,} | '
            f'{scale["matched"]["motion_loss"]:.5f} | '
            f'{penalty_cell(scale, "motion_loss", "zero_velocity")} | '
            f'{penalty_cell(scale, "motion_loss", "empty")} | '
            f'{penalty_cell(scale, "motion_loss", "wrong_scene")} |'
        )
    lines.extend([
        '',
        '## Absolute matched-loss guardrail',
        '',
        '| Training samples | Semantic loss | Approx. weighted cosine similarity | Motion loss |',
        '| ---: | ---: | ---: | ---: |',
    ])
    for scale in scales:
        semantic = scale['matched']['semantic_loss']
        lines.append(
            f'| {scale["train_samples"]:,} | {semantic:.5f} | '
            f'{1.0 - semantic:.5f} | {scale["matched"]["motion_loss"]:.5f} |'
        )
    lines.extend([
        '',
        'Matched semantic loss improves from 0.46199 at 64 samples to 0.30256 '
        'at 4,096 and remains 0.30538 at full scale. Matched motion loss improves '
        'from 1.58663 to 0.79639. The diagnostic objectives therefore do not '
        'collapse as radar dependence grows, but these losses still do not '
        'measure vehicle segmentation.',
        '',
        '## Onset and interpretation',
        '',
        f'- Semantic zero-velocity dependence is persistently positive from '
        f'**{onset["semantic_loss"]["zero_velocity"]} samples** by the scene-bootstrap criterion.',
        f'- Semantic empty-radar dependence is persistently positive from '
        f'**{onset["semantic_loss"]["empty"]} samples**.',
        f'- Semantic wrong-scene dependence is persistently positive from '
        f'**{onset["semantic_loss"]["wrong_scene"]} samples**.',
        f'- All three motion penalties are persistently positive from '
        f'**{onset["motion_loss"]["wrong_scene"]} samples**; motion dependence '
        'is already strong at the first curve point.',
        '- Semantic and motion wrong-scene penalties rise through 4,096 samples '
        'and then decline slightly at full scale. The paired full-minus-4,096 '
        'changes are reported below; they quantify validation-scene uncertainty '
        'but not training-seed uncertainty.',
        f'- Semantic wrong-scene full-minus-4,096: '
        f'**{summary["findings"]["full_minus_4096"]["semantic_loss"]["wrong_scene"]["estimate"]:+.5f}** '
        f'[{summary["findings"]["full_minus_4096"]["semantic_loss"]["wrong_scene"]["ci95_low"]:+.5f}, '
        f'{summary["findings"]["full_minus_4096"]["semantic_loss"]["wrong_scene"]["ci95_high"]:+.5f}].',
        f'- Motion wrong-scene full-minus-4,096: '
        f'**{summary["findings"]["full_minus_4096"]["motion_loss"]["wrong_scene"]["estimate"]:+.5f}** '
        f'[{summary["findings"]["full_minus_4096"]["motion_loss"]["wrong_scene"]["ci95_low"]:+.5f}, '
        f'{summary["findings"]["full_minus_4096"]["motion_loss"]["wrong_scene"]["ci95_high"]:+.5f}].',
        '- The supported shape is an onset followed by saturation or a small '
        'late decline, not indefinite monotonic growth.',
        '- Persistence across later data scales and positive scene-bootstrap '
        'intervals show that the onset is not a single-scene or single-scale '
        'blip. Because every point uses seed 125, this is **not** evidence of '
        'repeatability across training seeds.',
        '- P11 must compare matched camera-only and fused downstream models on '
        'full-validation vehicle IoU. Without that result, the curve establishes '
        'radar dependence but not overall model utility.',
        '',
        '## Reproduce',
        '',
        '```bash',
        'MPLCONFIGDIR=.cache/matplotlib python scripts/analyze_radar_scaling_curve.py \\',
        '  --bootstrap-iterations 10000 --bootstrap-seed 20261007',
        '```',
        '',
        'The machine-readable result is '
        '`artifacts/trainval/radar_scaling_curve_seed125.json`.',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def render(summary, path):
    scales = summary['scales']
    x = np.asarray([scale['train_samples'] for scale in scales], dtype=float)
    labels = [f'{int(value):,}' for value in x]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    panels = (
        ('semantic_loss', 'Semantic cosine-loss penalty'),
        ('motion_loss', 'Motion Huber-loss penalty'),
    )
    for axis, (metric, title) in zip(axes, panels):
        for mode in MODES:
            values = np.asarray([
                scale['penalties'][metric][mode]['estimate'] for scale in scales
            ])
            low = np.asarray([
                scale['penalties'][metric][mode]['ci95_low'] for scale in scales
            ])
            high = np.asarray([
                scale['penalties'][metric][mode]['ci95_high'] for scale in scales
            ])
            axis.errorbar(
                x, values, yerr=np.vstack((values - low, high - values)),
                marker='o', markersize=5.5, linewidth=2.0, capsize=3.5,
                color=COLORS[mode], label=MODE_LABELS[mode],
            )
        axis.axhline(0.0, color='#667085', linewidth=1.0)
        axis.set_xscale('log', base=2)
        axis.set_xticks(x, labels)
        axis.set_xlabel('Training samples (fixed 30,000 updates)')
        axis.set_ylabel('Loss increase relative to matched radar')
        axis.set_title(title, fontweight='bold')
        axis.grid(True, axis='y', color='#d9dee7', linewidth=0.8)
        axis.spines[['top', 'right']].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.95), frameon=False)
    fig.suptitle('Radar dependence strengthens, then saturates at full scale',
                 fontsize=15, fontweight='bold', y=0.995)
    fig.text(
        0.5, 0.015,
        'Paired 150-scene bootstrap 95% CI · seed 125 only · diagnostic losses, not vehicle IoU',
        ha='center', fontsize=9, color='#475467',
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    runs, scenes = load_runs(args.input_dir)
    summary = analyze(
        runs, scenes, args.bootstrap_iterations, args.bootstrap_seed,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + '\n')
    write_report(summary, args.report_out)
    render(summary, args.figure_out)
    print(f'summary: {args.summary_out}')
    print(f'report: {args.report_out}')
    print(f'figure: {args.figure_out}')
    print('RADAR_SCALING_ANALYSIS_OK')


if __name__ == '__main__':
    main()
