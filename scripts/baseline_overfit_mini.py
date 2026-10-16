#!/usr/bin/env python3
"""Overfit the supervised Simple-BEV baseline on four fixed mini samples."""

import argparse
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


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--nsweeps', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
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
    train_loader, _ = nuscenesdataset.compile_data(
        'mini', str(data_root), data_aug_conf=config,
        centroid=scene_centroid_py, bounds=bounds, res_3d=(Z, Y, X),
        bsz=1, nworkers=0, nworkers_val=0, shuffle=False,
        nsweeps=nsweeps, seqlen=1, refcam_id=1, get_tids=True,
        temporal_aug=False, use_radar_filters=False, do_shuffle_cams=False,
    )
    iterator = iter(train_loader)
    return [next(iterator) for _ in range(num_samples)]


def evaluate(model, loss_fn, batches, device):
    model.eval()
    results = []
    with torch.inference_mode():
        for index, batch in enumerate(batches):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, metrics, outputs = run_model(
                    model, loss_fn, batch, device=str(device), return_outputs=True
                )
            results.append({
                'index': index,
                'loss': float(loss),
                'iou': float(metrics['iou']),
                'prediction': torch.sigmoid(
                    outputs['seg_bev_logits'][0, 0]
                ).float().cpu().numpy(),
                'ground_truth': outputs['seg_bev_gt'][0, 0].float().cpu().numpy(),
                'radar': (
                    outputs['radar_voxels'][0, 0].float().sum(dim=1)
                    .clamp(0, 1).cpu().numpy()
                ),
            })
    return results


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


def draw_curve(draw, values, box):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill='white', outline=(180, 185, 195), width=2)
    values = np.asarray(values, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 1e-6)
    points = []
    for index, value in enumerate(values):
        x = x0 + 14 + index * (x1 - x0 - 28) / max(len(values) - 1, 1)
        y = y1 - 14 - (float(value) - low) / span * (y1 - y0 - 28)
        points.append((x, y))
    draw.line(points, fill=(218, 60, 50), width=3)
    draw.text((x0 + 8, y0 + 7), f'max {high:.3f}', font=font(12), fill=(80, 85, 95))
    draw.text((x0 + 8, y1 - 24), f'min {low:.3f}', font=font(12), fill=(80, 85, 95))


def render(losses, before, after, peak_gib, elapsed, output_path):
    before_iou = np.asarray([item['iou'] for item in before])
    after_iou = np.asarray([item['iou'] for item in after])
    deltas = after_iou - before_iou
    selected = (
        ('Largest IoU increase', int(np.argmax(deltas))),
        ('Smallest IoU increase', int(np.argmin(deltas))),
    )
    width, height = 1170, 1045
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 18), 'Simple-BEV supervised baseline: 4-sample overfit check',
              font=font(29, True), fill=TEXT)
    draw.text((25, 60),
              'Camera + radar official checkpoint | 50 updates | cycles over four fixed mini training samples',
              font=font(15), fill=(70, 78, 92))
    draw.text((25, 105), 'Positive task loss (segmentation + center + offset)',
              font=font(18, True), fill=TEXT)
    draw_curve(draw, losses, (25, 140, 550, 390))
    first_cycle = float(np.mean(losses[:4]))
    last_cycle = float(np.mean(losses[-4:]))
    draw.text((25, 410),
              f'First 4-step mean {first_cycle:.3f} -> last 4-step mean {last_cycle:.3f}',
              font=font(16, True), fill=TEXT)

    draw.text((600, 105), 'Same 4 training samples', font=font(20, True), fill=TEXT)
    draw.text((600, 155), f'Mean IoU before: {before_iou.mean():.3f}',
              font=font(20), fill=TEXT)
    draw.text((600, 195), f'Mean IoU after:  {after_iou.mean():.3f}',
              font=font(20), fill=TEXT)
    draw.text((600, 245), f'Elapsed: {elapsed:.1f}s', font=font(17), fill=TEXT)
    draw.text((600, 280), f'Peak VRAM: {peak_gib:.2f} GiB', font=font(17), fill=TEXT)
    draw.text((600, 335),
              'Uses human 3D-box-derived labels:', font=font(16, True), fill=(145, 65, 35))
    draw.text((600, 365), 'segmentation + center + offset losses',
              font=font(15), fill=(145, 65, 35))

    panel_size = 190
    labels = ('Ground truth', 'Radar returns', 'Before training', 'After training')
    for row, (description, index) in enumerate(selected):
        y = 500 + row * 255
        draw.text((25, y - 32),
                  f'{description} | sample {index} | IoU {before_iou[index]:.3f} -> {after_iou[index]:.3f}',
                  font=font(18, True), fill=TEXT)
        arrays = (
            binary_image(before[index]['ground_truth'], (0, 255, 80)),
            binary_image(before[index]['radar'], (0, 235, 255)),
            probability_image(before[index]['prediction']),
            probability_image(after[index]['prediction']),
        )
        for col, (label, array) in enumerate(zip(labels, arrays)):
            x = 25 + col * 275
            draw.text((x, y), label, font=font(16, True), fill=TEXT)
            panel = Image.fromarray(array).resize(
                (panel_size, panel_size), Image.Resampling.NEAREST
            )
            canvas.paste(panel, (x, y + 27))

    draw.text((25, 1015),
              'Sanity/overfit result only: improvement on memorized training samples is not validation performance.',
              font=font(14), fill=(145, 65, 35))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    torch.manual_seed(125)
    np.random.seed(125)
    torch.cuda.init()
    device = torch.device('cuda:0')
    batches = build_batches(args.data_root, args.num_samples, args.nsweeps)
    vox_util = nuscenesdataset.utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    model = Segnet(
        Z, Y, X, vox_util=vox_util,
        use_radar=True, use_metaradar=True, do_rgbcompress=True,
        encoder_type='res101', rand_flip=False, pretrained_backbone=False,
    ).to(device)
    saverloader.load(str(args.checkpoint), model)
    loss_fn = SimpleLoss(2.13).to(device)
    # Keep the checkpoint's learned task scales fixed in this tiny diagnostic;
    # otherwise their logarithmic regularizers can make total_loss negative and
    # obscure whether the actual prediction losses are decreasing.
    model.ce_weight.requires_grad_(False)
    model.center_weight.requires_grad_(False)
    model.offset_weight.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-7,
    )
    # The default 65536 scale overflows this older multi-loss checkpoint on its
    # first full-model backward. A conservative fixed starting scale is stable.
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0, growth_interval=1000)

    before = evaluate(model, loss_fn, batches, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    objective_losses = []
    gradient_norms = []
    start = time.perf_counter()
    model.train()
    for step in range(args.steps):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss, metrics = run_model(model, loss_fn, batch, device=str(device))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        task_loss = (
            metrics['ce_loss'] + metrics['center_loss'] + metrics['offset_loss']
        )
        losses.append(float(task_loss))
        objective_losses.append(float(loss.detach()))
        gradient_norms.append(float(grad_norm))
        if step < 4 or (step + 1) % 10 == 0:
            print(
                f'step {step + 1:02d}/{args.steps}: sample={step % len(batches)} '
                f'task_loss={losses[-1]:.5f} objective={objective_losses[-1]:.5f} '
                f'grad_norm={gradient_norms[-1]:.4f}'
            )

    after = evaluate(model, loss_fn, batches, device)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    if not np.isfinite(losses).all() or not np.isfinite(gradient_norms).all():
        raise RuntimeError('training produced non-finite loss or gradient')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / 'baseline_overfit_checkpoint'
    saverloader.save(
        str(checkpoint_dir), optimizer, model, args.steps, keep_latest=1
    )
    output_path = args.output_dir / 'baseline_overfit_4samples.png'
    render(losses, before, after, peak_gib, elapsed, output_path)
    metrics_path = args.output_dir / 'baseline_overfit_metrics.npz'
    np.savez_compressed(
        metrics_path,
        losses=np.asarray(losses, dtype=np.float32),
        objective_losses=np.asarray(objective_losses, dtype=np.float32),
        gradient_norms=np.asarray(gradient_norms, dtype=np.float32),
        before_iou=np.asarray([item['iou'] for item in before], dtype=np.float32),
        after_iou=np.asarray([item['iou'] for item in after], dtype=np.float32),
        ground_truth=np.stack([item['ground_truth'] for item in before]),
        radar_bev=np.stack([item['radar'] for item in before]),
        before_prediction=np.stack([item['prediction'] for item in before]),
        after_prediction=np.stack([item['prediction'] for item in after]),
        peak_cuda_memory_gib=np.asarray(peak_gib, dtype=np.float32),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float32),
    )

    first_cycle = float(np.mean(losses[:args.num_samples]))
    last_cycle = float(np.mean(losses[-args.num_samples:]))
    before_mean = float(np.mean([item['iou'] for item in before]))
    after_mean = float(np.mean([item['iou'] for item in after]))
    print(f'first_cycle_mean_loss: {first_cycle:.6f}')
    print(f'last_cycle_mean_loss: {last_cycle:.6f}')
    print(f'loss_reduction_percent: {(1 - last_cycle / first_cycle) * 100:.2f}')
    print(f'before_mean_iou: {before_mean:.6f}')
    print(f'after_mean_iou: {after_mean:.6f}')
    print(f'peak_cuda_memory_gib: {peak_gib:.3f}')
    print(f'elapsed_seconds: {elapsed:.3f}')
    print(f'metrics: {metrics_path}')
    print(f'visualization: {output_path}')
    print('BASELINE_OVERFIT_OK')


if __name__ == '__main__':
    main()
