#!/usr/bin/env python3
"""Visualize frozen DINOv2 features projected into the Simple-BEV frame."""

from _bootstrap import enable_source_package
enable_source_package()
from dd2414.teachers.dino import (build_loaders, build_loader, load_teacher, extract_teacher_features, camera_geometry, project_teacher_to_bev, make_radar_bev, radar_anchored_soft_targets)

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import nuscenesdataset  # noqa: E402
import utils.basic  # noqa: E402
import utils.geom  # noqa: E402
import utils.vox  # noqa: E402
from dd2414.config.geometry import Z, Y, X, bounds, scene_centroid, scene_centroid_py  # noqa: E402


CAMERA_NAMES = (
    'front (BEV reference)', 'front-left', 'front-right',
    'back-left', 'back', 'back-right',
)
TEACHER_HEIGHT = 196
TEACHER_WIDTH = 336
PATCH_SIZE = 14
BACKGROUND = (245, 247, 250)
TEXT = (25, 30, 40)


def font(size, bold=False):
    candidates = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'
        if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--nsweeps', type=int, default=1)
    parser.add_argument('--model-name', default='dinov2_vits14')
    return parser.parse_args()


















def pca_colors(camera_features, teacher_bev, coverage, soft_targets, soft_weight):
    camera_tokens = camera_features.permute(0, 2, 3, 1).reshape(
        -1, camera_features.shape[1]
    ).float().cpu().numpy()
    center = camera_tokens.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(camera_tokens - center, full_matrices=False)
    basis = vh[:3].T

    camera_rgb = ((camera_tokens - center) @ basis).reshape(
        camera_features.shape[0], camera_features.shape[2],
        camera_features.shape[3], 3,
    )
    bev_tokens = teacher_bev.permute(1, 2, 0).float().cpu().numpy()
    bev_rgb = (bev_tokens.reshape(-1, bev_tokens.shape[-1]) - center) @ basis
    bev_rgb = bev_rgb.reshape(Z, X, 3)
    soft_tokens = soft_targets.permute(1, 2, 0).float().cpu().numpy()
    soft_rgb = (soft_tokens.reshape(-1, soft_tokens.shape[-1]) - center) @ basis
    soft_rgb = soft_rgb.reshape(Z, X, 3)

    valid_bev = coverage.cpu().numpy() > 0
    valid_soft = soft_weight.cpu().numpy() > 0.05
    scale_values = np.concatenate(
        [camera_rgb.reshape(-1, 3), bev_rgb[valid_bev], soft_rgb[valid_soft]], axis=0
    )
    low = np.percentile(scale_values, 1, axis=0)
    high = np.percentile(scale_values, 99, axis=0)

    def scale(values):
        values = (values - low) / np.maximum(high - low, 1e-6)
        return (np.clip(values, 0, 1) * 255).astype(np.uint8)

    camera_rgb = scale(camera_rgb)
    bev_rgb = scale(bev_rgb)
    soft_rgb = scale(soft_rgb)
    bev_rgb[~valid_bev] = 0
    soft_rgb[~valid_soft] = 0
    return camera_rgb, bev_rgb, soft_rgb


def image_array(images):
    return images[0].clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).numpy()


def heatmap(values, color=(255, 180, 0)):
    values = values.astype(np.float32)
    values /= max(float(values.max()), 1e-6)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for channel, amount in enumerate(color):
        rgb[..., channel] = (values * amount).astype(np.uint8)
    return rgb


def overlay_radar(base, radar):
    result = base.copy()
    mask = radar > 0
    expanded = mask.copy()
    expanded[:-1] |= mask[1:]
    expanded[1:] |= mask[:-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[:, 1:] |= mask[:, :-1]
    result[expanded] = (0, 235, 255)
    result[mask] = (255, 255, 255)
    return result


def render_board(images, camera_rgb, bev_rgb, coverage, radar, soft_rgb,
                 soft_weight, ground_truth, visible_radar_points,
                 model_name, extraction_seconds, projection_seconds, peak_gib,
                 output_path):
    margin = 24
    panel_width, panel_height = 360, 210
    width = 3 * panel_width + 4 * margin
    bottom_size = 172
    height = 120 + 2 * (panel_height + 42) + 80 + bottom_size + 115
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title = font(29, bold=True)
    heading = font(18, bold=True)
    body = font(15)
    small = font(13)

    draw.text(
        (margin, 18), 'Self-supervised prototype: frozen DINOv2 semantics -> BEV',
        font=title, fill=TEXT,
    )
    draw.text(
        (margin, 60),
        'Six camera feature maps are projected with Simple-BEV geometry; no 3D box label is used by the teacher.',
        font=body, fill=(70, 78, 92),
    )
    draw.text(
        (margin, 84), 'Color = DINO principal components | cyan/white = radar returns | green = GT sanity check',
        font=small, fill=(70, 78, 92),
    )

    for index in range(6):
        row, col = divmod(index, 3)
        x = margin + col * (panel_width + margin)
        y = 120 + row * (panel_height + 42)
        original = Image.fromarray(images[index]).resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        semantics = Image.fromarray(camera_rgb[index]).resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        blended = Image.blend(original, semantics, 0.48)
        canvas.paste(blended, (x, y))
        draw.rectangle((x, y, x + panel_width, y + 25), fill=(0, 0, 0))
        draw.text((x + 7, y + 4), CAMERA_NAMES[index], font=small, fill='white')
        draw.text((x, y + panel_height + 8), 'image + frozen DINO patch features', font=small, fill=TEXT)

    coverage_rgb = heatmap(coverage)
    soft_confidence_rgb = heatmap(soft_weight, color=(255, 180, 0))
    gt_rgb = np.zeros((Z, X, 3), dtype=np.uint8)
    gt_rgb[ground_truth > 0.5] = (0, 255, 80)
    panels = (
        ('Naive DINO rays', bev_rgb),
        ('Ray coverage', coverage_rgb),
        ('Radar anchors', heatmap(radar, color=(0, 235, 255))),
        ('Soft confidence', soft_confidence_rgb),
        ('Soft DINO target', overlay_radar(soft_rgb, radar)),
        ('GT (view only)', gt_rgb),
    )
    panel_gap = (width - 2 * margin - 6 * bottom_size) // 5
    panel_y = 120 + 2 * (panel_height + 42) + 55
    for index, (label, values) in enumerate(panels):
        x = margin + index * (bottom_size + panel_gap)
        draw.text((x, panel_y - 28), label, font=heading, fill=TEXT)
        panel = Image.fromarray(values).resize(
            (bottom_size, bottom_size), Image.Resampling.NEAREST
        )
        canvas.paste(panel, (x, panel_y))

    stats_y = panel_y + bottom_size + 19
    draw.text(
        (margin, stats_y),
        f'{model_name} (frozen) | camera tokens: 6 x 384 x 14 x 24 | teacher BEV: 384 x 200 x 200',
        font=body, fill=TEXT,
    )
    draw.text(
        (margin, stats_y + 25),
        f'extract {extraction_seconds:.2f}s | geometry + soft targets {projection_seconds:.2f}s | peak VRAM {peak_gib:.2f} GiB | visible radar anchors {visible_radar_points}',
        font=body, fill=TEXT,
    )
    draw.text(
        (margin, stats_y + 52),
        'Naive rays expose depth ambiguity; radar-anchored soft regions are the actual distillation targets. No box label is used.',
        font=small, fill=(145, 65, 35),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    required = ('samples', 'sweeps', 'maps', 'v1.0-mini')
    missing = [name for name in required if not (args.data_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f'nuScenes mini is incomplete at {args.data_root}; missing: {missing}'
        )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run where WSL GPU access is allowed')

    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loader = build_loader(args.data_root, args.num_workers, args.nsweeps)
    batch = next(iter(loader))
    images = batch[0][:, 0]

    teacher = load_teacher(args.model_name, device)
    teacher_params = sum(parameter.numel() for parameter in teacher.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in teacher.parameters()
        if parameter.requires_grad
    )
    print(f'teacher_parameters: {teacher_params:,}')
    print(f'teacher_trainable_parameters: {trainable_params:,}')

    with torch.inference_mode():
        start = time.perf_counter()
        camera_features = extract_teacher_features(teacher, images, device)
        torch.cuda.synchronize()
        extraction_seconds = time.perf_counter() - start

        feature_height, feature_width = camera_features.shape[-2:]
        images_cpu, pixel_from_camera, cameras_from_camera0, cameras_from_vehicle = camera_geometry(
            batch, feature_height, feature_width, device
        )
        start = time.perf_counter()
        teacher_bev, coverage, vox_util = project_teacher_to_bev(
            camera_features, pixel_from_camera, cameras_from_camera0, device
        )
        radar_bev, radar_xyz_vehicle, radar_camera0 = make_radar_bev(
            batch, cameras_from_vehicle, vox_util, device
        )
        soft_targets, soft_weight, radar_view_count = radar_anchored_soft_targets(
            camera_features, pixel_from_camera, cameras_from_vehicle,
            radar_xyz_vehicle, radar_camera0, vox_util,
        )
        torch.cuda.synchronize()
        projection_seconds = time.perf_counter() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / 'dinov2_teacher_features.npz'
    np.savez_compressed(
        cache_path,
        features=camera_features.half().cpu().numpy(),
        intrinsics=batch[3][0, 0].numpy(),
        rotations=batch[1][0, 0].numpy(),
        translations=batch[2][0, 0].numpy(),
        model_name=np.asarray(args.model_name),
    )
    soft_cache_path = args.output_dir / 'dinov2_radar_soft_targets.npz'
    np.savez_compressed(
        soft_cache_path,
        semantic_target=soft_targets.half().cpu().numpy(),
        confidence=soft_weight.half().cpu().numpy(),
        radar_view_count=radar_view_count.byte().cpu().numpy(),
        dense_semantic_target=teacher_bev.half().cpu().numpy(),
        camera_coverage=coverage.half().cpu().numpy(),
        model_name=np.asarray(args.model_name),
    )

    camera_rgb, bev_rgb, soft_rgb = pca_colors(
        camera_features, teacher_bev, coverage, soft_targets, soft_weight
    )
    rgb_images = image_array(images_cpu)
    ground_truth = batch[12][0, 0, 0].numpy()
    coverage_np = coverage.cpu().numpy()
    radar_np = radar_bev.cpu().numpy()
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    output_path = args.output_dir / 'dinov2_bev_correspondence.png'
    render_board(
        rgb_images, camera_rgb, bev_rgb, coverage_np, radar_np,
        soft_rgb, soft_weight.cpu().numpy(), ground_truth,
        int((radar_view_count > 0).sum().item()), args.model_name, extraction_seconds,
        projection_seconds, peak_gib, output_path,
    )

    valid_radar = int((batch[16][0, 0, :3].abs().sum(dim=0) > 0).sum())
    covered_cells = int((coverage > 0).sum().item())
    print(f'teacher_feature_shape: {tuple(camera_features.shape)}')
    print(f'teacher_bev_shape: {tuple(teacher_bev.shape)}')
    print(f'covered_bev_cells: {covered_cells}/{Z * X}')
    print(f'valid_radar_points: {valid_radar}')
    print(f'camera_visible_radar_points: {int((radar_view_count > 0).sum().item())}')
    print(f'soft_target_cells: {int((soft_weight > 0.05).sum().item())}/{Z * X}')
    print(f'extraction_seconds: {extraction_seconds:.3f}')
    print(f'projection_seconds: {projection_seconds:.3f}')
    print(f'peak_cuda_memory_gib: {peak_gib:.3f}')
    print(f'feature_cache: {cache_path}')
    print(f'soft_target_cache: {soft_cache_path}')
    print(f'visualization: {output_path}')
    print('DINOV2_BEV_DEMO_OK')


if __name__ == '__main__':
    main()
