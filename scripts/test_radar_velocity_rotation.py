#!/usr/bin/env python3
"""Verify optional radar velocity rotation on synthetic and real mini data."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import nuscenesdataset  # noqa: E402
from dinov2_bev_demo import build_loader  # noqa: E402


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
    return parser.parse_args()


def synthetic_test():
    points = np.zeros((18, 2), dtype=np.float64)
    points[6:8] = np.asarray(((1.0, 0.0), (0.0, 2.0)))
    points[8:10] = np.asarray(((3.0, 0.0), (0.0, 4.0)))
    original = points.copy()
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    result = nuscenesdataset.rotate_radar_velocity_channels(points, rotation)
    np.testing.assert_allclose(result[6:8], ((0.0, -2.0), (1.0, 0.0)), atol=1e-8)
    np.testing.assert_allclose(result[8:10], ((0.0, -4.0), (3.0, 0.0)), atol=1e-8)
    np.testing.assert_array_equal(result[:6], original[:6])
    np.testing.assert_array_equal(result[10:], original[10:])
    np.testing.assert_allclose(
        np.linalg.norm(result[6:8], axis=0),
        np.linalg.norm(original[6:8], axis=0), atol=1e-8,
    )
    np.testing.assert_allclose(
        np.linalg.norm(result[8:10], axis=0),
        np.linalg.norm(original[8:10], axis=0), atol=1e-8,
    )


def load_real_pair(data_root):
    loader = build_loader(
        data_root, num_workers=0, nsweeps=1, rotate_radar_velocity=False
    )
    dataset = loader.dataset
    data_index = int(dataset.indices[0, 0])
    record = dataset.ixes[data_index]
    common = dict(
        nusc=dataset.nusc,
        sample_rec=record,
        nsweeps=1,
        min_distance=2.2,
        use_radar_filters=False,
        dataroot=dataset.dataroot,
    )
    legacy = nuscenesdataset.get_radar_data(
        **common, rotate_radar_velocity=False
    )
    rotated = nuscenesdataset.get_radar_data(
        **common, rotate_radar_velocity=True
    )
    return legacy, rotated


def real_data_test(legacy, rotated):
    if legacy.shape != rotated.shape or legacy.shape[0] != 19:
        raise AssertionError('legacy and rotated radar shapes differ')
    np.testing.assert_allclose(legacy[:6], rotated[:6], atol=1e-6)
    np.testing.assert_allclose(legacy[10:], rotated[10:], atol=1e-6)
    for start in (6, 8):
        legacy_speed = np.linalg.norm(legacy[start:start + 2], axis=0)
        rotated_speed = np.linalg.norm(rotated[start:start + 2], axis=0)
        np.testing.assert_allclose(legacy_speed, rotated_speed, rtol=2e-5, atol=2e-5)
    if not np.isfinite(rotated).all():
        raise AssertionError('rotated radar contains NaN or Inf')


def angle_differences(legacy, rotated):
    legacy_velocity = legacy[8:10].T
    rotated_velocity = rotated[8:10].T
    speed = np.linalg.norm(legacy_velocity, axis=1)
    valid = speed > 0.1
    legacy_angle = np.arctan2(legacy_velocity[valid, 1], legacy_velocity[valid, 0])
    rotated_angle = np.arctan2(rotated_velocity[valid, 1], rotated_velocity[valid, 0])
    difference = np.arctan2(
        np.sin(rotated_angle - legacy_angle),
        np.cos(rotated_angle - legacy_angle),
    )
    return np.abs(np.degrees(difference)), valid


def draw_vector_panel(draw, points, velocity, box, title, color):
    x0, y0, x1, y1 = box
    draw.text((x0, y0 - 30), title, font=font(18, True), fill=TEXT)
    draw.rectangle(box, fill=(10, 14, 20), outline=(150, 155, 165), width=1)
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    draw.line((center_x, y0, center_x, y1), fill=(55, 62, 75), width=1)
    draw.line((x0, center_y, x1, center_y), fill=(55, 62, 75), width=1)
    scale_xy = (x1 - x0) / 100.0
    keep = (
        (np.abs(points[0]) < 50) & (np.abs(points[1]) < 50)
        & (np.linalg.norm(velocity, axis=0) > 0.15)
        & (np.linalg.norm(velocity, axis=0) < 15.0)
    )
    indices = np.where(keep)[0]
    stride = max(len(indices) // 180, 1)
    for index in indices[::stride]:
        px = center_x - points[1, index] * scale_xy
        py = center_y - points[0, index] * scale_xy
        vx, vy = velocity[:, index]
        end_x = px - vy * 3.0
        end_y = py - vx * 3.0
        draw.ellipse((px - 1.5, py - 1.5, px + 1.5, py + 1.5), fill=(220, 225, 235))
        draw.line((px, py, end_x, end_y), fill=color, width=2)
    draw.text((x0 + 8, y0 + 8), 'front', font=font(12), fill=(185, 190, 200))


def render(legacy, rotated, angle_diff, output_path):
    width, height = 1130, 690
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), 'Radar velocity frame correction: legacy vs. common frame',
              font=font(29, True), fill=TEXT)
    draw.text((24, 60),
              'Same radar points and speed magnitudes; only vx/vy direction is rotated. Translation is never applied to velocity.',
              font=font(15), fill=(70, 78, 92))
    draw_vector_panel(
        draw, legacy[:3], legacy[8:10], (24, 130, 534, 640),
        'Legacy: five sensor-frame directions', (255, 95, 75),
    )
    draw_vector_panel(
        draw, rotated[:3], rotated[8:10], (594, 130, 1104, 640),
        'Corrected: one reference-frame direction', (0, 225, 245),
    )
    draw.text((24, 655),
              f'Non-static points: {len(angle_diff)} | median direction change {np.median(angle_diff):.1f} deg | '
              f'95th percentile {np.percentile(angle_diff, 95):.1f} deg',
              font=font(14, True), fill=TEXT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable because VizData initializes on GPU')
    synthetic_test()
    legacy, rotated = load_real_pair(args.data_root)
    real_data_test(legacy, rotated)
    angle_diff, moving_mask = angle_differences(legacy, rotated)
    speed_legacy = np.linalg.norm(legacy[8:10], axis=0)
    speed_rotated = np.linalg.norm(rotated[8:10], axis=0)
    max_speed_error = float(np.max(np.abs(speed_legacy - speed_rotated)))
    changed_fraction = float(np.mean(angle_diff > 1e-3))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output_dir / 'radar_velocity_rotation.npz'
    np.savez_compressed(
        snapshot_path,
        legacy=legacy,
        rotated=rotated,
        moving_mask=moving_mask,
        absolute_angle_difference_degrees=angle_diff,
    )
    image_path = args.output_dir / 'radar_velocity_rotation.png'
    render(legacy, rotated, angle_diff, image_path)
    print('synthetic_90_degree_rotation: OK')
    print(f'real_radar_shape: {legacy.shape}')
    print(f'xyz_unchanged: {np.allclose(legacy[:3], rotated[:3])}')
    print(f'nonvelocity_metadata_unchanged: {np.allclose(legacy[10:], rotated[10:])}')
    print(f'max_compensated_speed_magnitude_error: {max_speed_error:.9f}')
    print(f'nonstatic_points: {len(angle_diff)}')
    print(f'direction_changed_fraction: {changed_fraction:.6f}')
    print(f'median_direction_change_degrees: {np.median(angle_diff):.3f}')
    print(f'p95_direction_change_degrees: {np.percentile(angle_diff, 95):.3f}')
    print(f'snapshot: {snapshot_path}')
    print(f'visualization: {image_path}')
    print('RADAR_VELOCITY_ROTATION_TEST_OK')


if __name__ == '__main__':
    main()
