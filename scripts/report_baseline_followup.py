#!/usr/bin/env python3
"""Make an incremental, honest report from the unattended baseline study."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullsize_training import atomic_json_write


def paired_scene_interval(first, second, draws=2000):
    if set(first) != set(second):
        raise ValueError('paired scene sets differ')
    scenes = sorted(first)
    left = np.asarray([first[key] for key in scenes], dtype=float)
    right = np.asarray([second[key] for key in scenes], dtype=float)
    rng = np.random.default_rng(125)
    differences = []
    for _ in range(draws):
        indices = rng.integers(len(scenes), size=len(scenes))
        a, b = left[indices].sum(0), right[indices].sum(0)
        differences.append(a[0]/max(a.sum(),1)-b[0]/max(b.sum(),1))
    return [float(x) for x in np.quantile(differences, [.025,.975])]


def read_scene_counts(path):
    groups, tokens = {}, set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['token'] in tokens:
            raise ValueError('duplicate frame in paired evaluation')
        tokens.add(row['token'])
        values = groups.setdefault(row['scene_token'], np.zeros(3, dtype=np.int64))
        values += np.array([row['tp'], row['fp'], row['fn']])
    return groups, tokens


def build_report(study):
    rows = []
    for path in sorted(study.glob('*/summary.json')):
        if path.parent.name in ['pilot', 'smoke']:
            continue
        data = json.loads(path.read_text())
        metric = data.get('final_validation')
        if not metric:
            validations = data.get('progress', {}).get('validations', {})
            metric = validations[max(validations, key=int)] if validations else {}
        rows.append(dict(name=path.parent.name, status=data.get('status'),
                         metrics=metric or {}, source=str(path)))
    # Keep the historical probe results visible and explicitly labelled.
    for name in ['velocity_full_epoch008', 'velocity_zero_epoch008']:
        path = ROOT/'fullsize_baseline/probes'/name/'summary.json'
        if path.is_file():
            data = json.loads(path.read_text())
            rows.append(dict(name=f'Historical frozen 1x1: {name}',
                             status=data['status'], metrics=data['final_validation'],
                             source=str(path)))
    comparison = study/'comparison'
    comparison.mkdir(parents=True, exist_ok=True)
    atomic_json_write(rows, comparison/'metrics.json')
    text = ['# Baseline meeting follow-up results', '',
        'Vehicle IoU uses full-split valid-cell intersection / union.',
        'Official references use 448x800 images; legacy downstream uses 112x192.',
        'Different resolution, heads, supervision, and initialization prevent attribution to architecture alone.', '',
        '| Experiment | Status | Vehicle IoU (%) | Precision (%) | Recall (%) | F1 (%) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        values = [row['metrics'].get(key) for key in ['vehicle_iou','precision','recall','f1']]
        formatted = ['pending' if value is None else f'{100*value:.3f}' for value in values]
        text.append('| '+' | '.join([row['name'], row['status'] or 'unknown', *formatted])+' |')
    pair = [next(study.glob(f'finetune_1sweep_velocity_{mode}_*epochs/summary.json'), None)
            for mode in ['full','zero']]
    if all(pair):
        summaries = [json.loads(path.read_text()) for path in pair]
        configs = [value['config'] for value in summaries]
        keys = ['epochs','batch_size','accumulation','image_resolution', 'manifest_sha256',
                'precision','learning_rate','camera_lr','weight_decay']
        matched = all(configs[0][key] == configs[1][key] for key in keys)
        if matched and all(value.get('status') == 'complete' for value in summaries):
            epoch = configs[0]['epochs']
            counts = [read_scene_counts(path.parent/f'validation/epoch{epoch:03d}/per_frame.jsonl')
                      for path in pair]
            if counts[0][1] != counts[1][1] or len(counts[0][1]) != 6019:
                raise ValueError('matched comparison requires the same complete 6019 tokens')
            interval = paired_scene_interval(counts[0][0], counts[1][0])
            delta = summaries[0]['final_validation']['vehicle_iou']-summaries[1]['final_validation']['vehicle_iou']
            result = dict(vehicle_iou_difference=delta, paired_scene_bootstrap_95=interval,
                          frames=6019, scenes=len(counts[0][0]), training_epochs=epoch,
                          interpretation='evaluation uncertainty; not training-seed uncertainty')
            atomic_json_write(result, comparison/'paired_finetuning.json')
            text.extend(['', f'Raw minus zero-velocity finetuning IoU: {100*delta:+.3f} percentage points;',
                f'paired scene-bootstrap 95% interval [{100*interval[0]:+.3f}, {100*interval[1]:+.3f}] points.',
                'This describes evaluation uncertainty for one trained pair, not variation across training seeds.'])
    text.extend(['', 'Inspect each config, per-frame JSONL, and fixed-token mask PNG before drawing conclusions.',
        'Raw/zero finetuning compares two pretraining recipes; one seed does not measure seed uncertainty.',
        'A complete queue can contain failed stages: consult pipeline status and individual summaries.',
        'Optional depth/object-motion outputs are diagnostics, not downstream validation improvements.', ''])
    (comparison/'REPORT.md').write_text('\n'.join(text))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = [row for row in rows if row['status'] == 'complete' and row['metrics'].get('vehicle_iou') is not None]
    if complete:
        fig, ax = plt.subplots(figsize=(12, max(4, .55*len(complete))))
        ax.barh([row['name'] for row in complete],
                [100*row['metrics']['vehicle_iou'] for row in complete])
        ax.set_xlabel('Vehicle BEV IoU (%) - global intersection / union')
        ax.set_title('Full validation; see report for resolution and protocol differences')
        fig.tight_layout()
        fig.savefig(comparison/'vehicle_iou.png', dpi=160)
        plt.close(fig)
    # New finetuning learning curves are separate from legacy pretraining losses.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plotted = False
    for path in sorted(study.glob('finetune_*/metrics.jsonl')):
        records = []
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records:
            continue
        plotted = True
        axes[0].plot([r['samples_seen']/28130 for r in records],
                      [r['loss'] for r in records], alpha=.6, label=path.parent.name)
        summary = path.parent/'summary.json'
        if summary.is_file():
            vals = json.loads(summary.read_text()).get('progress', {}).get('validations', {})
            points = sorted((int(k), 100*v['vehicle_iou']) for k,v in vals.items())
            if points:
                axes[1].plot(*zip(*points), marker='o', label=path.parent.name)
    if plotted:
        axes[0].set(xlabel='Epoch', ylabel='Supervised training loss')
        axes[1].set(xlabel='Completed epoch', ylabel='Full-val Vehicle IoU (%)')
        for axis in axes:
            if axis.lines:
                axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(comparison/'finetuning_curves.png', dpi=160)
    plt.close(fig)
    print(comparison/'REPORT.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-dir', type=Path, required=True)
    build_report(parser.parse_args().study_dir)
