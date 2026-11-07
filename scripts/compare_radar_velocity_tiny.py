#!/usr/bin/env python3
"""Radar-velocity motion-target diagnostic against the same radar encoders.

Trains a 2-channel velocity head on top of the light/BEVCar radar encoders to
predict the ego-motion-compensated radar velocity (vx_comp, vy_comp) of MOVING
returns splatted into BEV cells, then measures how the loss reacts to ablating
the velocity columns of the input radar. Static returns dominate raw radar, so
the target is restricted to points faster than a speed threshold; this isolates
the motion signal that only radar Doppler can provide. The semantic-target
counterpart leaves the "geometry only" penalty flat (~0); a moving-velocity
target should make it large. Training-sample overfit diagnostic, not an
accuracy benchmark.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from compare_radar_dino_tiny import ablate_radar  # noqa: E402
from dinov2_bev_demo import build_loader  # noqa: E402
from nets.bevcar_voxel_adapter import prepare_bevcar_voxels  # noqa: E402
from nets.radar_encoder import (  # noqa: E402
    RadarPointEncoder, transform_radar_to_camera_bev,
)
from train_nuscenes import Z, Y, X, bounds, scene_centroid  # noqa: E402
import utils.basic  # noqa: E402
import utils.geom  # noqa: E402
import utils.vox  # noqa: E402


VELOCITY_CLIP = 15.0
MOVING_THRESHOLD = 1.0


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed-list', default='125,42,7')
    return parser.parse_args()


def official_voxelnet(source):
    path = source / 'nets' / 'voxelnet.py'
    if not path.is_file():
        raise FileNotFoundError(f'official BEVCar source missing: {path}')
    spec = importlib.util.spec_from_file_location('official_bevcar_voxelnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VoxelNet


class VelocityStudent(nn.Module):
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
        self.head = nn.Conv2d(64, 2, 1)

    def forward(self, radar_input):
        feature = (self.encoder(*radar_input) if isinstance(radar_input, tuple)
                   else self.encoder(radar_input))
        return self.head(self.projection(feature))


def velocity_target_and_input(batch, device):
    """Return (light_input, bevcar_input, target (2,Z,X), coverage (Z,X), info)."""
    radar_vehicle = batch[16][:, 0].to(device).permute(0, 2, 1)
    rotations = batch[1][:, 0].to(device)
    translations = batch[2][:, 0].to(device)
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    batch_size = radar_vehicle.shape[0]
    camera_from_vehicle = utils.basic.unpack_seqdim(
        utils.geom.safe_inverse(utils.basic.pack_seqdim(vehicle_from_cameras, batch_size)),
        batch_size,
    )[:, 0]
    radar_camera = transform_radar_to_camera_bev(radar_vehicle, camera_from_vehicle)
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    with torch.no_grad():
        features, coords, counts, _ = prepare_bevcar_voxels(
            radar_camera, vox_util, quality_filter=False,
        )
    bevcar_input = (features, coords, counts)

    xyz = radar_camera[..., :3]
    mem = vox_util.Ref2Mem(xyz, Z, Y, X, assert_cube=False)[0]  # (R, 3)
    grid_max = torch.tensor((X, Y, Z), device=device)
    velocity = radar_camera[0, :, 8:10].clamp(-VELOCITY_CLIP, VELOCITY_CLIP)  # (R, 2)
    moving = velocity.norm(dim=-1).gt(MOVING_THRESHOLD)
    valid = (
        torch.isfinite(radar_camera[0]).all(dim=-1)
        & xyz[0].abs().sum(dim=-1).gt(0)
        & (mem > -0.5).all(dim=-1)
        & (mem < grid_max - 0.5).all(dim=-1)
        & radar_camera[0, :, 10].gt(0.5)
        & radar_camera[0, :, 11].round().eq(3)
        & radar_camera[0, :, 14].round().eq(0)
        & moving
    )
    x_cell = mem[:, 0].round().long()
    z_cell = mem[:, 2].round().long()
    field = torch.zeros((2, Z, X), device=device)
    count = torch.zeros((Z, X), device=device)
    index = torch.where(valid)[0]
    if index.numel() > 0:
        flat = z_cell[index] * X + x_cell[index]
        field_flat = field.view(2, -1)
        count_flat = count.view(-1)
        field_flat[0].index_add_(0, flat, velocity[index, 0])
        field_flat[1].index_add_(0, flat, velocity[index, 1])
        count_flat.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        field = field_flat.view(2, Z, X)
        count = count_flat.view(Z, X)
    coverage = count.gt(0)
    field[0, coverage] = field[0, coverage] / count[coverage]
    field[1, coverage] = field[1, coverage] / count[coverage]
    info = {
        'radar_points': int((xyz[0].abs().sum(dim=-1).gt(0)).sum()),
        'moving_points': int(valid.sum()),
        'covered_cells': int(coverage.sum()),
    }
    return radar_camera, bevcar_input, field, coverage.float(), info


def velocity_loss(prediction, target, coverage):
    """Huber loss over radar-covered BEV cells, normalized by the cell count."""
    if prediction.dim() == 4:
        prediction = prediction.squeeze(0)
    mask = coverage.gt(0)
    if not mask.any():
        return torch.zeros((), device=prediction.device), 0
    loss = F.smooth_l1_loss(prediction[:, mask], target[:, mask], reduction='mean')
    return loss, int(mask.sum())


def run_branch(name, inputs, targets, coverages, steps, learning_rate, device,
               official_class, seed):
    torch.manual_seed(seed)
    model = VelocityStudent(name, official_class).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-5)
    branch_inputs = [entry[0 if name == 'light' else 1] for entry in inputs]

    def evaluate(source_inputs):
        model.eval()
        with torch.inference_mode():
            values = []
            for radar, target, coverage in zip(source_inputs, targets, coverages):
                loss, _ = velocity_loss(model(radar), target, coverage)
                values.append(float(loss))
        return values

    before = evaluate(branch_inputs)
    model.train()
    history = []
    for step in range(steps):
        index = step % len(branch_inputs)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = velocity_loss(model(branch_inputs[index]), targets[index],
                                coverages[index])
        if not torch.isfinite(loss):
            raise RuntimeError(f'{name}: nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.detach()))
    after = evaluate(branch_inputs)

    ablations = {}
    for mode in ('geometry_only', 'permuted_measurements', 'empty'):
        changed = [ablate_radar(radar, name, mode) for radar in branch_inputs]
        ablations[mode] = evaluate(changed)
    return {
        'before_losses': before,
        'after_losses': after,
        'step_losses': history,
        'ablation_losses': ablations,
        'trainable_parameters': sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this small comparison')
    seeds = [int(part.strip()) for part in args.seed_list.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seed-list must have distinct integer seeds')
    torch.set_num_threads(2)
    device = torch.device('cuda')

    loader = build_loader(args.data_root, num_workers=0, nsweeps=1,
                          rotate_radar_velocity=True)
    batches = [next(iter(loader))] if args.samples == 1 else []
    if args.samples > 1:
        iterator = iter(loader)
        batches = [next(iterator) for _ in range(args.samples)]

    inputs, targets, coverages, infos = [], [], [], []
    for batch in batches:
        light_input, bevcar_input, field, coverage, info = velocity_target_and_input(
            batch, device
        )
        inputs.append((light_input, bevcar_input))
        targets.append(field)
        coverages.append(coverage)
        infos.append(info)
        print(f'sample: covered_cells={info["covered_cells"]} '
              f'moving_points={info["moving_points"]} '
              f'radar_points={info["radar_points"]}', flush=True)

    official_class = official_voxelnet(args.bevcar_source)
    seed_results = {}
    for seed in seeds:
        seed_results[str(seed)] = {}
        for branch in ('light', 'bevcar'):
            seed_results[str(seed)][branch] = run_branch(
                branch, inputs, targets, coverages, args.steps,
                args.learning_rate, device, official_class, seed,
            )
            torch.cuda.empty_cache()

    report = {
        'scope': 'radar-velocity motion target, random weights, training-sample overfit',
        'samples': args.samples,
        'steps_per_branch': args.steps,
        'learning_rate': args.learning_rate,
        'velocity_clip': VELOCITY_CLIP,
        'moving_threshold': MOVING_THRESHOLD,
        'target_fields': 'vx_comp,vy_comp (cols 8,9) on moving points only',
        'input_counts': infos,
        'seeds': seeds,
        'seed_results': seed_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'radar_velocity_tiny.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    for seed in seeds:
        for branch in ('light', 'bevcar'):
            result = seed_results[str(seed)][branch]
            correct = np.mean(result['after_losses'])
            geometry = np.mean(result['ablation_losses']['geometry_only'])
            permuted = np.mean(result['ablation_losses']['permuted_measurements'])
            empty = np.mean(result['ablation_losses']['empty'])
            print(f'seed {seed} {branch}: before={np.mean(result["before_losses"]):.4f} '
                  f'correct={correct:.4f} geometry_only={geometry:.4f} '
                  f'(penalty {geometry - correct:+.4f}) '
                  f'permuted={permuted:.4f} empty={empty:.4f}')
    print(f'report: {path}')
    print('RADAR_VELOCITY_TINY_OK')


if __name__ == '__main__':
    main()
