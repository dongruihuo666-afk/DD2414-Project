#!/usr/bin/env python3
"""Radar-only four-frame learning check against cached frozen-DINOv2 targets.

This is an equal-data/equal-update optimization diagnostic, not an accuracy
benchmark or a comparison of complete camera-radar Simple-BEV models.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from dinov2_bev_demo import build_loader  # noqa: E402
from nets.bevcar_voxel_adapter import prepare_bevcar_voxels  # noqa: E402
from nets.radar_encoder import (  # noqa: E402
    RadarPointEncoder, transform_radar_to_camera_bev,
)
from semantic_distill_smoke import semantic_loss  # noqa: E402
from train_nuscenes import Z, Y, X, bounds, scene_centroid  # noqa: E402
import utils.basic  # noqa: E402
import utils.geom  # noqa: E402
import utils.vox  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--target-cache', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    return parser.parse_args()


def official_voxelnet(source):
    path = source / 'nets' / 'voxelnet.py'
    if not path.is_file():
        raise FileNotFoundError(f'official BEVCar source missing: {path}')
    spec = importlib.util.spec_from_file_location('official_bevcar_voxelnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VoxelNet


class RadarStudent(nn.Module):
    def __init__(self, branch, official_class=None):
        super().__init__()
        if branch == 'light':
            self.encoder = RadarPointEncoder(
                Z=Z, Y=Y, X=X, bounds=bounds, scene_centroid=(0, 0, 0),
                out_channels=64, use_quality_mask=True,
            )
            self.projection = nn.Identity()
        elif branch == 'bevcar':
            self.encoder = official_class(
                use_col=False, reduced_zx=False, output_dim=128,
                use_radar_occupancy_map=False,
            )
            self.projection = nn.Conv2d(128, 64, 1)
        else:
            raise ValueError(branch)
        self.head = nn.Conv2d(64, 384, 1)

    def forward(self, radar_input):
        feature = (self.encoder(*radar_input) if isinstance(radar_input, tuple)
                   else self.encoder(radar_input))
        return self.head(self.projection(feature))


def cached_target(path, batch, device):
    with np.load(path, allow_pickle=False) as cache:
        if str(cache['model_name']) != 'dinov2_vits14':
            raise ValueError('expected the frozen dinov2_vits14 target cache')
        for key, batch_index in (
                ('intrinsics', 3), ('rotations', 1), ('translations', 2)):
            actual = batch[batch_index][0, 0].numpy()
            if not np.allclose(actual, cache[key], rtol=1e-4, atol=1e-4):
                raise ValueError(f'target/sample geometry mismatch: {path}, {key}')
        target = torch.from_numpy(cache['semantic_target'].astype(np.float32)).to(device)
        confidence = torch.from_numpy(cache['confidence'].astype(np.float32)).to(device)
    if target.shape != (384, Z, X) or confidence.shape != (Z, X):
        raise ValueError(f'unexpected target geometry in {path}')
    return target[None], confidence[None]


def radar_inputs(batch, device):
    radar_vehicle = batch[16][:, 0].to(device).permute(0, 2, 1)
    rotations = batch[1][:, 0].to(device)
    translations = batch[2][:, 0].to(device)
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    B = radar_vehicle.shape[0]
    camera_from_vehicle = utils.basic.unpack_seqdim(
        utils.geom.safe_inverse(utils.basic.pack_seqdim(vehicle_from_cameras, B)), B
    )[:, 0]
    radar_camera = transform_radar_to_camera_bev(
        radar_vehicle, camera_from_vehicle
    )
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    with torch.no_grad():
        features, coords, counts, diagnostics = prepare_bevcar_voxels(
            radar_camera, vox_util, quality_filter=False,
        )
    return radar_camera, (features, coords, counts), diagnostics


def run_branch(name, inputs, targets, confidences, steps, learning_rate,
               device, official_class):
    torch.manual_seed(125)
    model = RadarStudent(name, official_class).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-5)
    branch_inputs = [entry[0 if name == 'light' else 1] for entry in inputs]
    def evaluate():
        model.eval()
        with torch.inference_mode():
            values = []
            for radar, target, confidence in zip(branch_inputs, targets, confidences):
                loss, _ = semantic_loss(model(radar), target, confidence)
                values.append(float(loss))
        return values

    initial = evaluate()
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    history = []
    radar_gradients = []
    for step in range(steps):
        index = step % len(inputs)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(branch_inputs[index])
        loss, _ = semantic_loss(prediction, targets[index], confidences[index])
        if not torch.isfinite(loss):
            raise RuntimeError(f'{name}: nonfinite loss')
        loss.backward()
        gradients = [p.grad.float().norm() for p in model.encoder.parameters()
                     if p.grad is not None]
        if not gradients:
            raise RuntimeError(f'{name}: no encoder gradients')
        grad_norm = torch.stack(gradients).norm().item()
        if not np.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError(f'{name}: invalid encoder gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.detach()))
        radar_gradients.append(grad_norm)
        print(f'{name} step {step + 1}/{steps}: loss={history[-1]:.6f} '
              f'encoder_grad={grad_norm:.5f}', flush=True)
    final = evaluate()
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    return {
        'trainable_parameters': sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        'before_losses': initial,
        'after_losses': final,
        'step_losses': history,
        'encoder_gradient_norms': radar_gradients,
        'elapsed_seconds': elapsed,
        'peak_allocated_cuda_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def plot_report(report, path):
    """Render an intentionally labelled optimization diagnostic."""
    image = Image.new('RGB', (1080, 650), '#f6f8fb')
    draw = ImageDraw.Draw(image)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    body = ImageFont.truetype(font_path, 17)
    heading = ImageFont.truetype(bold_path, 25)
    small = ImageFont.truetype(font_path, 14)
    draw.text((30, 22), 'Radar-only DINOv2 target: four-frame overfit',
              font=heading, fill='#1c2736')
    draw.text((30, 62),
              f'Same frozen-teacher targets, {report["steps_per_branch"]} updates each; random radar encoders',
              font=body, fill='#455468')
    plot = (70, 120, 1020, 455)
    draw.rectangle(plot, fill='white', outline='#b8c1ce', width=2)
    draw.text((30, 265), 'Loss', font=small, fill='#455468')
    colors = {'light': '#3678cc', 'bevcar': '#d06431'}
    for name in ('light', 'bevcar'):
        values = report['branches'][name]['step_losses']
        points = [
            (plot[0] + 20 + index * (plot[2] - plot[0] - 40) / max(len(values)-1, 1),
             plot[3] - 20 - (value - 0.25) / 0.85 * (plot[3] - plot[1] - 40))
            for index, value in enumerate(values)
        ]
        draw.line(points, fill=colors[name], width=4)
    draw.text((70, 467), 'Update 1', font=small, fill='#455468')
    draw.text((943, 467), f'Update {report["steps_per_branch"]}',
              font=small, fill='#455468')
    for index, name in enumerate(('light', 'bevcar')):
        result = report['branches'][name]
        y = 508 + index * 43
        draw.rectangle((70, y + 4, 90, y + 24), fill=colors[name])
        draw.text((102, y),
                  f'{name}: same-frame eval {np.mean(result["before_losses"]):.3f} '
                  f'-> {np.mean(result["after_losses"]):.3f}; '
                  f'peak allocated CUDA {result["peak_allocated_cuda_gib"]:.2f} GiB',
                  font=body, fill='#1c2736')
    draw.text((30, 609),
              'Training-sample loss only. Different capacity/filtering; no held-out or camera-fusion result.',
              font=small, fill='#a14929')
    image.save(path)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this small comparison')
    if not 1 <= args.samples <= 4 or args.steps < 1:
        raise ValueError('use 1-4 cached samples and at least one update')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    loader = build_loader(args.data_root, num_workers=0, nsweeps=1,
                          rotate_radar_velocity=True)
    batches = [next(iter(loader))] if args.samples == 1 else []
    if args.samples > 1:
        iterator = iter(loader)
        batches = [next(iterator) for _ in range(args.samples)]
    targets, confidences, inputs, counts = [], [], [], []
    for index, batch in enumerate(batches):
        target, confidence = cached_target(
            args.target_cache / f'sample_{index:03d}.npz', batch, device
        )
        light_input, bevcar_input, info = radar_inputs(batch, device)
        targets.append(target)
        confidences.append(confidence)
        inputs.append((light_input, bevcar_input))
        counts.append({key: int(value[0]) for key, value in info.items()})
    official_class = official_voxelnet(args.bevcar_source)
    results = {}
    for branch in ('light', 'bevcar'):
        results[branch] = run_branch(
            branch, inputs, targets, confidences, args.steps,
            args.learning_rate, device, official_class,
        )
        torch.cuda.empty_cache()
    report = {
        'scope': 'radar-only, same frozen-DINOv2 cached targets, random weights',
        'samples': args.samples, 'steps_per_branch': args.steps,
        'learning_rate': args.learning_rate,
        'target_model': 'dinov2_vits14 pretrained frozen teacher',
        'supervised_bev_checkpoint_loaded': False,
        'box_loss_used': False,
        'input_counts': counts,
        'branches': results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'radar_dino_tiny_comparison.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    figure = args.output_dir / 'radar_dino_tiny_comparison.png'
    plot_report(report, figure)
    print(f'report: {path}')
    print(f'figure: {figure}')
    for branch, result in results.items():
        print(f'{branch}: before={np.mean(result["before_losses"]):.6f} '
              f'after={np.mean(result["after_losses"]):.6f} '
              f'peak_cuda_gib={result["peak_allocated_cuda_gib"]:.3f}')
    print('RADAR_DINO_TINY_COMPARISON_OK')


if __name__ == '__main__':
    main()
