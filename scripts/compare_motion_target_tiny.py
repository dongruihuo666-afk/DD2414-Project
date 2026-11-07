#!/usr/bin/env python3
"""Motion/Doppler target on the full fusion model — radar-measurement sensitivity probe.

Attaches a 2-channel motion head to the shared BEV of the full image+radar fusion
model (Segnet + metaradar input) and trains it to predict the ego-motion-compensated
velocity (vx_comp, vy_comp) of moving radar returns. The camera encoder is frozen so
that velocity can only reach the motion head through the radar branch (otherwise the
strong camera encoder memorizes the motion map and defeats the ablation). The semantic
DINOv2 target is kept as a control: its radar penalty is flat because it uses only
radar *position*, whereas the motion target should produce a large penalty when the
radar velocity channels are zeroed. Small-sample overfit diagnostic, not a benchmark.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import utils.basic  # noqa: E402
import utils.geom  # noqa: E402
import utils.vox  # noqa: E402
from dinov2_bev_demo import load_teacher  # noqa: E402
from eval_image_distill_heldout import build_targets  # noqa: E402
from nets.segnet import Segnet  # noqa: E402
from semantic_distill_smoke import (  # noqa: E402
    SemanticDistillationModel, prepare_inputs, semantic_loss,
)
from train_nuscenes import Z, Y, X, bounds, scene_centroid  # noqa: E402


VELOCITY_CLIP = 15.0
MOVING_THRESHOLD = 1.0
# Radar meta channels (radar cols 3:18) that hold velocity: vx, vy, vx_comp, vy_comp.
VELOCITY_META_CHANNELS = slice(3, 7)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed-list', default='125,42,7')
    parser.add_argument('--model-name', default='dinov2_vits14')
    return parser.parse_args()


class MotionDistillationModel(SemanticDistillationModel):
    """SemanticDistillationModel plus a 2-channel velocity head on the shared BEV."""

    def __init__(self, student):
        super().__init__(student, image_distill=False)
        self.motion_head = nn.Conv2d(student.latent_dim, 2, 1)

    def forward(self, rgb, pixel_from_cameras, camera0_from_cameras,
                vox_util, radar_voxels):
        fusion_prediction, _, shared_bev, _ = super().forward(
            rgb, pixel_from_cameras, camera0_from_cameras, vox_util, radar_voxels
        )
        motion_prediction = self.motion_head(shared_bev)
        return fusion_prediction, motion_prediction


def build_model(device, seed):
    torch.manual_seed(seed)
    student = Segnet(
        Z, Y, X, vox_util=utils.vox.Vox_util(
            Z, Y, X, scene_centroid=scene_centroid.to(device),
            bounds=bounds, assert_cube=False,
        ),
        use_radar=True, use_metaradar=True, do_rgbcompress=True,
        encoder_type='res101', rand_flip=False, pretrained_backbone=False,
    ).to(device)
    model = MotionDistillationModel(student).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.student.bev_compressor.parameters():
        parameter.requires_grad_(True)
    for parameter in model.semantic_head.parameters():
        parameter.requires_grad_(True)
    for parameter in model.motion_head.parameters():
        parameter.requires_grad_(True)
    # The camera encoder stays frozen: velocity must reach the motion head through
    # the radar branch, so the ablation actually tests radar-measurement dependence.
    return model


def motion_target(batch, device):
    """Splat moving-return compensated velocity into a (2, Z, X) field.

    Velocity is read from radar cols 8:10 (vx_comp, vy_comp) in the same frame as the
    metaradar input channels, and splatted at the point's camera0-frame (Z, X) cell —
    the same grid the shared BEV lives on.
    """
    radar = batch[16][:, 0].to(device).permute(0, 2, 1)  # (B, R, 19)
    rotations = batch[1][:, 0].to(device)
    translations = batch[2][:, 0].to(device)
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    batch_size = radar.shape[0]
    cameras_from_vehicle = utils.basic.unpack_seqdim(
        utils.geom.safe_inverse(utils.basic.pack_seqdim(vehicle_from_cameras, batch_size)),
        batch_size,
    )
    radar_xyz = radar[:, :, :3]
    radar_camera0 = utils.geom.apply_4x4(cameras_from_vehicle[:, 0], radar_xyz)
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    mem = vox_util.Ref2Mem(radar_camera0, Z, Y, X, assert_cube=False)[0]  # (R, 3)
    grid_max = torch.tensor((X, Y, Z), device=device)
    velocity = radar[0, :, 8:10].clamp(-VELOCITY_CLIP, VELOCITY_CLIP)  # (R, 2)
    moving = velocity.norm(dim=-1).gt(MOVING_THRESHOLD)
    valid = (
        torch.isfinite(radar[0]).all(dim=-1)
        & radar_xyz[0].abs().sum(dim=-1).gt(0)
        & (mem > -0.5).all(dim=-1)
        & (mem < grid_max - 0.5).all(dim=-1)
        & radar[0, :, 10].gt(0.5)
        & radar[0, :, 11].round().eq(3)
        & radar[0, :, 14].round().eq(0)
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
        'moving_points': int(valid.sum()),
        'covered_cells': int(coverage.sum()),
    }
    return field, coverage.float(), info


def motion_loss(prediction, target, coverage):
    """Huber loss over moving-return cells, normalized by the cell count."""
    prediction = prediction.float()
    if prediction.dim() == 4:
        prediction = prediction.squeeze(0)
    mask = coverage.gt(0)
    if not mask.any():
        return torch.zeros((), device=prediction.device), 0
    loss = F.smooth_l1_loss(prediction[:, mask], target[:, mask], reduction='mean')
    return loss, int(mask.sum())


def ablate_radar_voxels(radar_voxels, mode):
    if mode == 'correct':
        return radar_voxels
    out = radar_voxels.clone()
    if mode == 'zero_velocity':
        out[:, VELOCITY_META_CHANNELS] = 0.0
    elif mode == 'no_radar':
        out.zero_()
    else:
        raise ValueError(mode)
    return out


def train(model, inputs, fusion_targets, confidences, motion_fields, motion_coverages,
          sem_weight, motion_weight, steps, learning_rate, device):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0, growth_interval=1000)
    model.train()
    model.student.decoder.eval()
    sem_history, mot_history = [], []
    for step in range(steps):
        index = step % len(inputs)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            images, pixel_from_cameras, camera0_from_cameras, vox_util, radar_voxels = inputs[index]
            prediction, motion_prediction = model(
                images, pixel_from_cameras, camera0_from_cameras, vox_util, radar_voxels
            )
            loss = torch.zeros((), device=device)
            sem_value, mot_value = 0.0, 0.0
            if sem_weight > 0:
                sem, _ = semantic_loss(
                    prediction, fusion_targets[index][None], confidences[index][None]
                )
                loss = loss + sem_weight * sem
                sem_value = float(sem.detach())
            if motion_weight > 0:
                mot, _ = motion_loss(
                    motion_prediction, motion_fields[index], motion_coverages[index]
                )
                loss = loss + motion_weight * mot
                mot_value = float(mot.detach())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        sem_history.append(sem_value)
        mot_history.append(mot_value)
    return sem_history, mot_history


def evaluate(model, inputs, fusion_targets, confidences, motion_fields,
             motion_coverages, device):
    model.eval()
    sem, mot = {'correct': [], 'zero_velocity': [], 'no_radar': []}, \
               {'correct': [], 'zero_velocity': [], 'no_radar': []}
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for index in range(len(inputs)):
            images, pixel_from_cameras, camera0_from_cameras, vox_util, radar_voxels = inputs[index]
            for mode in ('correct', 'zero_velocity', 'no_radar'):
                radar = ablate_radar_voxels(radar_voxels, mode)
                prediction, motion_prediction = model(
                    images, pixel_from_cameras, camera0_from_cameras, vox_util, radar
                )
                sem_value, _ = semantic_loss(
                    prediction, fusion_targets[index][None], confidences[index][None]
                )
                mot_value, _ = motion_loss(
                    motion_prediction, motion_fields[index], motion_coverages[index]
                )
                sem[mode].append(float(sem_value))
                mot[mode].append(float(mot_value))
    return (
        {k: np.asarray(v, dtype=np.float32) for k, v in sem.items()},
        {k: np.asarray(v, dtype=np.float32) for k, v in mot.items()},
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    seeds = [int(part.strip()) for part in args.seed_list.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seed-list must have distinct integer seeds')
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    from dinov2_bev_demo import build_loader
    loader = build_loader(args.data_root, 0, 1, split='train')
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(args.samples)]

    teacher = load_teacher(args.model_name, device)
    for parameter in teacher.parameters():
        if parameter.requires_grad:
            raise RuntimeError('DINOv2 teacher must remain frozen')
    fusion_targets, confidences, _ = build_targets(batches, teacher, device)
    del teacher
    torch.cuda.empty_cache()

    inputs = [prepare_inputs(batch, device) for batch in batches]
    motion_fields, motion_coverages, motion_infos = [], [], []
    for batch in batches:
        field, coverage, info = motion_target(batch, device)
        motion_fields.append(field)
        motion_coverages.append(coverage)
        motion_infos.append(info)
        print(f'sample: covered_cells={info["covered_cells"]} '
              f'moving_points={info["moving_points"]}', flush=True)

    variants = (
        ('semantic_only', 1.0, 0.0),
        ('motion_only', 0.0, 1.0),
        ('joint', 1.0, 0.5),
    )
    seed_results = {}
    for seed in seeds:
        seed_results[str(seed)] = {}
        for label, sem_weight, motion_weight in variants:
            model = build_model(device, seed)
            start = time.perf_counter()
            sem_history, mot_history = train(
                model, inputs, fusion_targets, confidences,
                motion_fields, motion_coverages,
                sem_weight, motion_weight, args.steps, args.learning_rate, device,
            )
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start
            sem_eval, mot_eval = evaluate(
                model, inputs, fusion_targets, confidences,
                motion_fields, motion_coverages, device,
            )
            seed_results[str(seed)][label] = {
                'sem_history': sem_history,
                'mot_history': mot_history,
                'sem_eval': {k: v.tolist() for k, v in sem_eval.items()},
                'mot_eval': {k: v.tolist() for k, v in mot_eval.items()},
                'train_seconds': train_seconds,
            }
            del model
            torch.cuda.empty_cache()
            sem_penalty = float(sem_eval['zero_velocity'].mean() - sem_eval['correct'].mean())
            mot_penalty = float(mot_eval['zero_velocity'].mean() - mot_eval['correct'].mean())
            print(f'seed {seed} {label}: sem correct={sem_eval["correct"].mean():.4f} '
                  f'zero_vel={sem_eval["zero_velocity"].mean():.4f} '
                  f'(penalty {sem_penalty:+.4f}) no_radar={sem_eval["no_radar"].mean():.4f}')
            print(f'seed {seed} {label}: mot correct={mot_eval["correct"].mean():.4f} '
                  f'zero_vel={mot_eval["zero_velocity"].mean():.4f} '
                  f'(penalty {mot_penalty:+.4f}) no_radar={mot_eval["no_radar"].mean():.4f}')

    report = {
        'scope': 'motion target on full fusion model, frozen camera encoder, overfit',
        'samples': args.samples,
        'steps': args.steps,
        'learning_rate': args.learning_rate,
        'velocity_clip': VELOCITY_CLIP,
        'moving_threshold': MOVING_THRESHOLD,
        'velocity_meta_channels': list(range(VELOCITY_META_CHANNELS.start, VELOCITY_META_CHANNELS.stop)),
        'target_fields': 'vx_comp,vy_comp (radar cols 8,9) on moving points only',
        'variants': {label: {'sem_weight': sw, 'motion_weight': mw}
                     for label, sw, mw in variants},
        'motion_infos': motion_infos,
        'seeds': seeds,
        'seed_results': seed_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'motion_target_tiny.json'
    path.write_text(json.dumps(report, indent=2) + '\n')

    print('--- motion-target summary (mean over seeds) ---')
    for label, sem_weight, motion_weight in variants:
        for seed in seeds:
            pass
        for loss_name, key in (('sem', 'sem_eval'), ('mot', 'mot_eval')):
            correct = np.mean([np.mean(seed_results[str(s)][label][key]['correct'])
                               for s in seeds])
            zero_vel = np.mean([np.mean(seed_results[str(s)][label][key]['zero_velocity'])
                                for s in seeds])
            no_radar = np.mean([np.mean(seed_results[str(s)][label][key]['no_radar'])
                                for s in seeds])
            print(f'{label:14s} {loss_name}: correct={correct:.4f} '
                  f'zero_vel={zero_vel:.4f} (penalty {zero_vel - correct:+.4f}) '
                  f'no_radar={no_radar:.4f}')
    print(f'report: {path}')
    print('MOTION_TARGET_TINY_OK')


if __name__ == '__main__':
    main()
