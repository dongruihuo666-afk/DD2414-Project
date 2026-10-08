#!/usr/bin/env python3
"""Evaluate official camera and radar checkpoints on a fixed mini subset."""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import nuscenesdataset  # noqa: E402
import saverloader  # noqa: E402
from nets.segnet import Segnet  # noqa: E402
from train_nuscenes import (  # noqa: E402
    SimpleLoss,
    Z,
    Y,
    X,
    bounds,
    run_model,
    scene_centroid,
    scene_centroid_py,
)


BACKGROUND = (245, 247, 250)
TEXT = (25, 30, 40)
PANEL_SIZE = 190
MARGIN = 24


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--camera-checkpoint', type=Path, required=True)
    parser.add_argument('--radar-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--nsweeps', type=int, default=5)
    return parser.parse_args()


def build_batches(data_root, num_samples, nsweeps):
    config = {
        'crop_offset': 0,
        'resize_lim': [1.0, 1.0],
        'final_dim': (112, 192),
        'H': 900,
        'W': 1600,
        'cams': [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
        ],
        'ncams': 6,
    }
    _, val_loader = nuscenesdataset.compile_data(
        'mini', str(data_root), data_aug_conf=config,
        centroid=scene_centroid_py, bounds=bounds, res_3d=(Z, Y, X),
        bsz=1, nworkers=0, nworkers_val=0, shuffle=False,
        nsweeps=nsweeps, seqlen=1, refcam_id=1, get_tids=True,
        temporal_aug=False, use_radar_filters=False, do_shuffle_cams=False,
    )
    if num_samples > len(val_loader):
        raise ValueError(f'requested {num_samples} samples, validation has {len(val_loader)}')
    iterator = iter(val_loader)
    return [next(iterator) for _ in range(num_samples)]


def evaluate(checkpoint, batches, device, use_radar):
    vox_util = nuscenesdataset.utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    model = Segnet(
        Z, Y, X, vox_util=vox_util,
        use_radar=use_radar, use_metaradar=use_radar,
        do_rgbcompress=True, encoder_type='res101', rand_flip=False,
        pretrained_backbone=False,
    ).to(device)
    saverloader.load(str(checkpoint), model)
    model.eval()
    loss_fn = SimpleLoss(2.13).to(device)
    results = []
    torch.cuda.empty_cache()

    with torch.inference_mode():
        # Exclude one-time CUDA/kernel initialization from the timing comparison.
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            run_model(
                model, loss_fn, batches[0], device=str(device),
                return_outputs=False,
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        for index, batch in enumerate(batches):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, metrics, outputs = run_model(
                    model, loss_fn, batch, device=str(device), return_outputs=True
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            prediction = torch.sigmoid(
                outputs['seg_bev_logits'][0, 0]
            ).float().cpu().numpy()
            ground_truth = outputs['seg_bev_gt'][0, 0].float().cpu().numpy()
            radar_bev = None
            if use_radar:
                radar_bev = (
                    outputs['radar_voxels'][0, 0].float().sum(dim=1)
                    .clamp(0, 1).cpu().numpy()
                )
            result = {
                'index': index,
                'iou': float(metrics['iou']),
                'loss': float(loss),
                'seconds': elapsed,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'radar_bev': radar_bev,
            }
            results.append(result)
            mode = 'camera+radar' if use_radar else 'camera-only'
            print(
                f'{mode} sample {index:02d}: '
                f'iou={result["iou"]:.4f} time={elapsed:.3f}s'
            )

    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    del model, loss_fn, vox_util
    torch.cuda.empty_cache()
    return results, peak_gib


def probability_image(values):
    values = np.clip(values, 0, 1)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * values).astype(np.uint8)
    rgb[..., 1] = (85 * np.sqrt(values)).astype(np.uint8)
    return rgb


def binary_image(values, color):
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[values > 0.5] = color
    return rgb


def radar_overlay(prediction, radar):
    rgb = probability_image(prediction)
    mask = radar > 0
    expanded = mask.copy()
    expanded[:-1] |= mask[1:]
    expanded[1:] |= mask[:-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[:, 1:] |= mask[:, :-1]
    rgb[expanded] = (0, 235, 255)
    return rgb


def paste_panel(canvas, draw, array, x, y, label):
    draw.text((x, y - 27), label, font=font(17, True), fill=TEXT)
    panel = Image.fromarray(array).resize(
        (PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST
    )
    canvas.paste(panel, (x, y))


def render_summary(camera, radar, nsweeps, peak_camera, peak_radar, output_path):
    camera_ious = np.asarray([item['iou'] for item in camera])
    radar_ious = np.asarray([item['iou'] for item in radar])
    deltas = radar_ious - camera_ious
    order = np.argsort(deltas)
    selected = (
        ('Radar helps most', int(order[-1])),
        ('Middle case', int(order[len(order) // 2])),
        ('Radar helps least', int(order[0])),
    )

    width = 5 * PANEL_SIZE + 6 * MARGIN
    row_height = PANEL_SIZE + 72
    height = 190 + 3 * row_height + 25
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 18), 'Simple-BEV fixed mini subset: camera vs. camera + radar',
              font=font(28, True), fill=TEXT)
    draw.text(
        (MARGIN, 60),
        f'First {len(camera)} validation samples | 112x192 images | {nsweeps} radar sweeps | official checkpoints',
        font=font(15), fill=(70, 78, 92),
    )
    draw.text(
        (MARGIN, 95),
        f'Camera mean IoU {camera_ious.mean():.3f} +/- {camera_ious.std():.3f}    '
        f'Camera+radar {radar_ious.mean():.3f} +/- {radar_ious.std():.3f}    '
        f'Radar higher on {(deltas > 0).sum()}/{len(deltas)} samples',
        font=font(18, True), fill=TEXT,
    )
    draw.text(
        (MARGIN, 127),
        f'Mean forward: camera {np.mean([v["seconds"] for v in camera]):.3f}s, '
        f'radar {np.mean([v["seconds"] for v in radar]):.3f}s | '
        f'peak VRAM {peak_camera:.2f}/{peak_radar:.2f} GiB',
        font=font(15), fill=(70, 78, 92),
    )
    draw.text(
        (MARGIN, 153),
        'Functionality study on a tiny low-resolution subset; not a reproduction of the paper benchmark.',
        font=font(14), fill=(145, 65, 35),
    )

    labels = ('Ground truth', 'Radar returns', 'Camera-only', 'Camera + radar', 'Radar overlay')
    for row, (description, index) in enumerate(selected):
        y = 220 + row * row_height
        camera_item, radar_item = camera[index], radar[index]
        delta = radar_item['iou'] - camera_item['iou']
        draw.text(
            (MARGIN, y - 28),
            f'{description} | sample {index:02d} | '
            f'IoU {camera_item["iou"]:.3f} -> {radar_item["iou"]:.3f} '
            f'(delta {delta:+.3f})',
            font=font(18, True), fill=TEXT,
        )
        arrays = (
            binary_image(camera_item['ground_truth'], (0, 255, 80)),
            binary_image(radar_item['radar_bev'], (0, 235, 255)),
            probability_image(camera_item['prediction']),
            probability_image(radar_item['prediction']),
            radar_overlay(radar_item['prediction'], radar_item['radar_bev']),
        )
        for col, (label, array) in enumerate(zip(labels, arrays)):
            x = MARGIN + col * (PANEL_SIZE + MARGIN)
            paste_panel(canvas, draw, array, x, y + 32, label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return selected


def main():
    args = parse_args()
    for checkpoint in (args.camera_checkpoint, args.radar_checkpoint):
        if not checkpoint.is_dir():
            raise FileNotFoundError(f'checkpoint not found: {checkpoint}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    torch.manual_seed(125)
    np.random.seed(125)
    torch.cuda.init()
    device = torch.device('cuda:0')
    batches = build_batches(args.data_root, args.num_samples, args.nsweeps)
    camera, camera_peak = evaluate(
        args.camera_checkpoint, batches, device, use_radar=False
    )
    radar, radar_peak = evaluate(
        args.radar_checkpoint, batches, device, use_radar=True
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'mini_subset_metrics.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow((
            'sample_index', 'camera_iou', 'camera_radar_iou', 'iou_delta',
            'camera_seconds', 'camera_radar_seconds',
        ))
        for camera_item, radar_item in zip(camera, radar):
            writer.writerow((
                camera_item['index'], camera_item['iou'], radar_item['iou'],
                radar_item['iou'] - camera_item['iou'],
                camera_item['seconds'], radar_item['seconds'],
            ))

    npz_path = args.output_dir / 'mini_subset_predictions.npz'
    np.savez_compressed(
        npz_path,
        ground_truth=np.stack([item['ground_truth'] for item in camera]),
        camera_prediction=np.stack([item['prediction'] for item in camera]),
        radar_prediction=np.stack([item['prediction'] for item in radar]),
        radar_bev=np.stack([item['radar_bev'] for item in radar]),
        camera_iou=np.asarray([item['iou'] for item in camera], dtype=np.float32),
        radar_iou=np.asarray([item['iou'] for item in radar], dtype=np.float32),
        camera_seconds=np.asarray([item['seconds'] for item in camera], dtype=np.float32),
        radar_seconds=np.asarray([item['seconds'] for item in radar], dtype=np.float32),
        camera_peak_gib=np.asarray(camera_peak, dtype=np.float32),
        radar_peak_gib=np.asarray(radar_peak, dtype=np.float32),
        nsweeps=np.asarray(args.nsweeps, dtype=np.int32),
    )
    image_path = args.output_dir / 'mini_subset_comparison.png'
    selected = render_summary(
        camera, radar, args.nsweeps, camera_peak, radar_peak, image_path
    )

    camera_ious = np.asarray([item['iou'] for item in camera])
    radar_ious = np.asarray([item['iou'] for item in radar])
    print(f'camera_mean_iou: {camera_ious.mean():.6f}')
    print(f'radar_mean_iou: {radar_ious.mean():.6f}')
    print(f'radar_higher_count: {int((radar_ious > camera_ious).sum())}/{len(camera)}')
    print(f'camera_mean_seconds: {np.mean([v["seconds"] for v in camera]):.3f}')
    print(f'radar_mean_seconds: {np.mean([v["seconds"] for v in radar]):.3f}')
    print(f'camera_peak_gib: {camera_peak:.3f}')
    print(f'radar_peak_gib: {radar_peak:.3f}')
    print(f'representative_samples: {selected}')
    print(f'csv: {csv_path}')
    print(f'predictions: {npz_path}')
    print(f'visualization: {image_path}')
    print('MINI_SUBSET_EVAL_OK')


if __name__ == '__main__':
    main()
