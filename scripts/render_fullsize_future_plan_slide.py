#!/usr/bin/env python3
"""Render a 16:9 meeting slide explaining low Vehicle IoU and next tests."""

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--comparison', type=Path,
        default=(REPO_ROOT / 'fullsize_baseline' / 'teacher_overnight'
                 / 'epoch008' / 'comparison.json'),
    )
    parser.add_argument(
        '--output', type=Path,
        default=REPO_ROOT / 'artifacts' / 'fullsize_low_iou_future_plan.png',
    )
    return parser.parse_args()


def add_rounded_box(axis, xy, width, height, facecolor, edgecolor='none',
                    linewidth=1.0, radius=0.018):
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        xy, width, height,
        boxstyle=f'round,pad=0.012,rounding_size={radius}',
        transform=axis.transAxes, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth,
    )
    axis.add_patch(box)
    return box


def add_numbered_step(axis, number, y, title, detail, color):
    axis.text(
        0.685, y, str(number), transform=axis.transAxes, ha='center', va='center',
        fontsize=12, color='white', weight='bold',
        bbox={'boxstyle': 'circle,pad=0.35', 'facecolor': color,
              'edgecolor': 'none'},
    )
    axis.text(0.713, y + 0.018, title, transform=axis.transAxes,
              fontsize=11.7, weight='bold', va='center', color='#202428')
    axis.text(0.713, y - 0.018, detail, transform=axis.transAxes,
              fontsize=9.35, va='top', color='#555B61', linespacing=1.25)


def render_slide(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw = report['full']
    vehicle_iou = 100 * raw['vehicle_iou']
    precision = 100 * raw['precision']
    recall = 100 * raw['recall']
    fp_per_tp = raw['fp'] / raw['tp']

    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    figure = plt.figure(figsize=(16, 9), facecolor='white')
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis('off')

    axis.text(0.045, 0.945, 'Why is Vehicle IoU only about 5%?',
              fontsize=26, weight='bold', va='top', color='#202428')
    axis.text(
        0.045, 0.895,
        'Measured failure  →  testable explanations  →  controlled future plan',
        fontsize=13.5, va='top', color='#60676D',
    )

    # Left: the metric-level explanation supported directly by validation data.
    add_rounded_box(axis, (0.04, 0.185), 0.285, 0.645, '#FFF4F1', '#E36A4A', 1.6)
    axis.text(0.061, 0.795, 'MEASURED BOTTLENECK', fontsize=11.5,
              weight='bold', color='#B63B21', va='top')
    axis.text(0.061, 0.744, f'Vehicle IoU  {vehicle_iou:.2f}%', fontsize=24,
              weight='bold', color='#202428', va='top')
    axis.text(0.061, 0.677, r'$IoU = TP\, /\, (TP + FP + FN)$', fontsize=17,
              color='#33383D', va='top')
    axis.plot([0.061, 0.303], [0.635, 0.635], transform=axis.transAxes,
              color='#E8B4A7', linewidth=1.1)
    metric_rows = (
        ('Precision', f'{precision:.2f}%'),
        ('Recall', f'{recall:.1f}%'),
        ('False positives / true positive', f'{fp_per_tp:.1f}×'),
    )
    y = 0.586
    for label, value in metric_rows:
        axis.text(0.061, y, label, fontsize=11.2, color='#555B61', va='center')
        axis.text(0.298, y, value, fontsize=13.2, weight='bold', color='#B63B21',
                  va='center', ha='right')
        y -= 0.064
    add_rounded_box(axis, (0.059, 0.265), 0.247, 0.116, '#D4492D')
    axis.text(0.075, 0.343, 'Direct reason for low IoU', fontsize=11,
              color='white', weight='bold', va='top')
    axis.text(
        0.075, 0.309,
        'The model recovers many vehicles, but predicts\nfar too much background as vehicle.',
        fontsize=10.2, color='white', va='top', linespacing=1.35,
    )
    axis.text(0.061, 0.218, 'Supported by all 6,019 validation frames',
              fontsize=9.3, color='#7A4940', style='italic')

    # Middle: design facts and hypotheses, explicitly not causal proof yet.
    add_rounded_box(axis, (0.345, 0.185), 0.285, 0.645, '#F5F7F9', '#CBD2D8', 1.3)
    axis.text(0.366, 0.795, 'LIKELY CONTRIBUTORS TO TEST', fontsize=11.5,
              weight='bold', color='#3D4852', va='top')
    axis.text(0.366, 0.758, 'Design facts, not causal proof yet', fontsize=9.5,
              color='#747C83', va='top', style='italic')
    contributors = (
        ('1  Weak camera semantics',
         'The camera ResNet is random-initialized and frozen;\nit never learns DINO image semantics.'),
        ('2  Objective / task mismatch',
         'The backbone learns DINO-BEV and radar motion,\nnot vehicle masks; the downstream head is only 1×1.'),
        ('3  Probe may be under-tested',
         'One probe epoch and a fixed 0.5 threshold may\nunderstate separability or expose poor calibration.'),
        ('4  Sparse and uncertain evidence',
         'Only one radar sweep and one seed were tested;\ndensity and run-to-run variance remain unknown.'),
    )
    y = 0.699
    for title, detail in contributors:
        add_rounded_box(axis, (0.365, y - 0.09), 0.245, 0.10, 'white', '#DFE4E8', 0.9,
                        radius=0.012)
        axis.text(0.381, y - 0.012, title, fontsize=10.7, weight='bold',
                  color='#30363B', va='top')
        axis.text(0.381, y - 0.047, detail, fontsize=8.9, color='#5A6268',
                  va='top', linespacing=1.3)
        y -= 0.126

    # Right: ordered future plan; each step is a controlled question.
    add_rounded_box(axis, (0.65, 0.185), 0.31, 0.645, '#F2F8F5', '#9CC9AF', 1.3)
    axis.text(0.672, 0.795, 'FUTURE PLAN — IN THIS ORDER', fontsize=11.5,
              weight='bold', color='#22633D', va='top')
    add_numbered_step(
        axis, 1, 0.708, 'Audit the readout',
        'Threshold/PR curve; train the same frozen probe\nto convergence; verify class balance and frozen weights.',
        '#4F6D7A',
    )
    add_numbered_step(
        axis, 2, 0.586, 'Learn camera semantics',
        'DINO self-supervision for the trainable camera\nencoder + radar-masked DINO distillation in fused BEV.',
        '#287D4F',
    )
    add_numbered_step(
        axis, 3, 0.464, 'Separate radar effects',
        'Matched camera-only, radar-spatial, and radar +\nvelocity runs; identical data, budget, seed and probe.',
        '#287D4F',
    )
    add_numbered_step(
        axis, 4, 0.342, 'Test density and robustness',
        'Then compare 1/5/10 radar sweeps, add seeds, and\nreport Vehicle IoU, PR/F1 and multi-class mIoU.',
        '#287D4F',
    )
    axis.text(0.672, 0.225, 'Decision rule', fontsize=10.2, weight='bold',
              color='#22633D')
    axis.text(
        0.744, 0.225,
        'Change one factor at a time; keep full validation.',
        fontsize=9.5, color='#466353',
    )

    add_rounded_box(axis, (0.04, 0.075), 0.92, 0.068, '#25323B')
    axis.text(
        0.06, 0.109,
        'Takeaway: radar is being used, but better radar motion loss alone does not guarantee better vehicle semantics.',
        fontsize=13.1, color='white', weight='bold', va='center',
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.png')
    figure.savefig(temporary, dpi=180, facecolor='white')
    temporary.replace(output)
    plt.close(figure)


def main():
    args = parse_args()
    report = json.loads(args.comparison.read_text())
    render_slide(report, args.output)
    print(f'future_plan_slide: {args.output}')
    print('FULLSIZE_FUTURE_PLAN_SLIDE_OK')


if __name__ == '__main__':
    main()
