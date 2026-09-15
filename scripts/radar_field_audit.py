#!/usr/bin/env python3
"""Audit the 19 nuScenes radar channels used by Simple-BEV."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dinov2_bev_demo import build_loader  # noqa: E402


FIELDS = (
    'x', 'y', 'z', 'dyn_prop', 'id', 'rcs', 'vx', 'vy',
    'vx_comp', 'vy_comp', 'is_quality_valid', 'ambig_state',
    'x_rms', 'y_rms', 'invalid_state', 'pdh0', 'vx_rms', 'vy_rms',
    'time_lag',
)
CATEGORICAL = {
    3, 4, 10, 11, 12, 13, 14, 15, 16, 17,
}
ROLES = (
    'position', 'position', 'position', 'state', 'identifier', 'appearance',
    'raw velocity', 'raw velocity', 'ego-comp velocity', 'ego-comp velocity',
    'quality', 'quality', 'uncertainty code', 'uncertainty code', 'quality',
    'false-alarm code', 'uncertainty code', 'uncertainty code', 'temporal',
)
FIRST_ENCODER = {
    0: 'use, normalize', 1: 'use, normalize', 2: 'use, normalize',
    3: 'optional mask', 4: 'do not use', 5: 'use, robust norm',
    6: 'skip initially', 7: 'skip initially',
    8: 'use after rotation', 9: 'use after rotation',
    10: 'quality mask', 11: 'quality mask',
    12: 'optional later', 13: 'optional later', 14: 'quality mask',
    15: 'quality weight', 16: 'optional later', 17: 'optional later',
    18: 'use, normalize',
}
BACKGROUND = (245, 247, 250)
TEXT = (25, 30, 40)


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--nsweeps', type=int, default=1)
    return parser.parse_args()


def collect_points(loader, num_samples):
    iterator = iter(loader)
    samples = []
    counts = []
    for index in range(num_samples):
        batch = next(iterator)
        radar = batch[16][0, 0].transpose(0, 1).numpy()
        valid = np.abs(radar[:, :3]).sum(axis=1) > 0
        points = radar[valid]
        samples.append(points)
        counts.append(len(points))
        print(f'sample {index:02d}: valid_radar_points={len(points)}')
    return np.concatenate(samples, axis=0), np.asarray(counts)


def field_statistics(points):
    rows = []
    for index, name in enumerate(FIELDS):
        values = points[:, index]
        finite = values[np.isfinite(values)]
        row = {
            'index': index,
            'name': name,
            'role': ROLES[index],
            'kind': 'categorical' if index in CATEGORICAL else 'continuous',
            'finite_fraction': len(finite) / max(len(values), 1),
            'min': float(np.min(finite)),
            'p01': float(np.percentile(finite, 1)),
            'median': float(np.median(finite)),
            'mean': float(np.mean(finite)),
            'std': float(np.std(finite)),
            'p99': float(np.percentile(finite, 99)),
            'max': float(np.max(finite)),
            'unique': int(len(np.unique(finite))),
            'recommendation': FIRST_ENCODER[index],
        }
        rows.append(row)
    return rows


def draw_histogram(draw, values, box, title, unit=''):
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 27), title, font=font(16, True), fill=TEXT)
    draw.rectangle(box, fill='white', outline=(185, 190, 200), width=1)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, (1, 99))
    clipped = np.clip(finite, low, high)
    counts, _ = np.histogram(clipped, bins=32, range=(low, high))
    max_count = max(int(counts.max()), 1)
    bar_width = (x1 - x0 - 20) / len(counts)
    for index, count in enumerate(counts):
        height = (y1 - y0 - 38) * count / max_count
        bx0 = x0 + 10 + index * bar_width
        draw.rectangle(
            (bx0, y1 - 24 - height, bx0 + max(bar_width - 1, 1), y1 - 24),
            fill=(50, 135, 210),
        )
    draw.text((x0 + 8, y1 - 21), f'{low:.2f}{unit}', font=font(11), fill=(75, 80, 90))
    right = f'{high:.2f}{unit}'
    bbox = draw.textbbox((0, 0), right, font=font(11))
    draw.text((x1 - (bbox[2] - bbox[0]) - 8, y1 - 21), right,
              font=font(11), fill=(75, 80, 90))


def render(points, counts, rows, nsweeps, output_path):
    width, height = 1550, 1420
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), 'nuScenes radar field audit before building an encoder',
              font=font(29, True), fill=TEXT)
    draw.text(
        (24, 60),
        f'{len(counts)} fixed mini training samples | {nsweeps} sweep | '
        f'{len(points):,} valid points | per sample {counts.min()}-{counts.max()} (mean {counts.mean():.1f})',
        font=font(15), fill=(70, 78, 92),
    )
    draw.text(
        (24, 88),
        'Current loader disables radar filters, so quality/state fields must not be treated as ordinary continuous values.',
        font=font(14), fill=(145, 65, 35),
    )

    radar_range = np.linalg.norm(points[:, :2], axis=1)
    compensated_speed = np.linalg.norm(points[:, 8:10], axis=1)
    histograms = (
        (radar_range, 'Planar range', 'm'),
        (points[:, 5], 'Radar cross section (RCS)', ''),
        (compensated_speed, 'Compensated speed magnitude', 'm/s'),
        (points[:, 18] * 1000.0, 'Time lag', 'ms'),
    )
    for index, (values, title, unit) in enumerate(histograms):
        x = 24 + index * 380
        draw_histogram(draw, values, (x, 145, x + 350, 340), title, unit)

    table_y = 390
    columns = (
        ('idx', 24, 45), ('field', 75, 165), ('role', 245, 175),
        ('kind', 425, 115), ('min', 545, 95), ('median', 645, 95),
        ('p99', 745, 95), ('max', 845, 95), ('unique', 945, 80),
        ('first encoder', 1030, 245),
    )
    draw.rectangle((20, table_y, 1290, table_y + 37), fill=(38, 48, 65))
    for label, x, _ in columns:
        draw.text((x, table_y + 9), label, font=font(14, True), fill='white')
    for row_index, row in enumerate(rows):
        y = table_y + 37 + row_index * 45
        if row_index % 2 == 0:
            draw.rectangle((20, y, 1290, y + 45), fill=(232, 235, 241))
        values = (
            str(row['index']), row['name'], row['role'], row['kind'],
            f"{row['min']:.2f}", f"{row['median']:.2f}",
            f"{row['p99']:.2f}", f"{row['max']:.2f}",
            str(row['unique']), row['recommendation'],
        )
        for value, (_, x, _) in zip(values, columns):
            color = (145, 65, 35) if 'rotation' in value else TEXT
            draw.text((x, y + 12), value, font=font(13), fill=color)

    note_x = 1310
    draw.text((note_x, table_y), 'Key decisions', font=font(18, True), fill=TEXT)
    notes = (
        '1. XYZ decides where a point',
        '   lands in the BEV grid.',
        '',
        '2. RCS describes return',
        '   strength; robust scaling',
        '   is required.',
        '',
        '3. Prefer compensated',
        '   velocity over raw vx/vy.',
        '',
        '4. Velocity vectors must be',
        '   rotated into one common',
        '   reference frame first.',
        '',
        '5. IDs and categorical state',
        '   codes are not ordinary',
        '   real-valued measurements.',
        '',
        '6. Start with point MLP +',
        '   permutation-invariant',
        '   voxel mean/max pooling.',
    )
    for index, line in enumerate(notes):
        draw.text((note_x, table_y + 38 + index * 25), line,
                  font=font(13), fill=TEXT)

    footer_y = table_y + 37 + len(rows) * 45 + 25
    draw.text((24, footer_y),
              'Recommended minimal numeric inputs: normalized XYZ + robust-scaled RCS + rotated vx_comp/vy_comp + normalized time lag.',
              font=font(15, True), fill=TEXT)
    draw.text((24, footer_y + 30),
              'Use quality/state fields as masks, weights, or later embeddings; do not feed point ID as a physical feature.',
              font=font(14), fill=(70, 78, 92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable because VizData builds its voxel helper on GPU')
    loader = build_loader(args.data_root, num_workers=0, nsweeps=args.nsweeps)
    points, counts = collect_points(loader, args.num_samples)
    rows = field_statistics(points)
    if not np.isfinite(points).all():
        raise RuntimeError('valid radar points contain non-finite values')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'radar_field_audit.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    npz_path = args.output_dir / 'radar_field_audit.npz'
    np.savez_compressed(npz_path, points=points, valid_counts=counts)
    image_path = args.output_dir / 'radar_field_audit.png'
    render(points, counts, rows, args.nsweeps, image_path)

    print(f'total_valid_points: {len(points)}')
    print(f'valid_points_per_sample: min={counts.min()} mean={counts.mean():.1f} max={counts.max()}')
    print(f'finite_values: {np.isfinite(points).all()}')
    print(f'compensated_speed_p99_mps: {np.percentile(np.linalg.norm(points[:, 8:10], axis=1), 99):.3f}')
    print(f'rcs_p01_p99: {np.percentile(points[:, 5], 1):.3f}, {np.percentile(points[:, 5], 99):.3f}')
    print(f'time_lag_min_max_seconds: {points[:, 18].min():.6f}, {points[:, 18].max():.6f}')
    print('velocity_frame_warning: XYZ is transformed, vx/vy channels are not rotated by PointCloud.transform')
    print(f'csv: {csv_path}')
    print(f'raw_snapshot: {npz_path}')
    print(f'visualization: {image_path}')
    print('RADAR_FIELD_AUDIT_OK')


if __name__ == '__main__':
    main()
