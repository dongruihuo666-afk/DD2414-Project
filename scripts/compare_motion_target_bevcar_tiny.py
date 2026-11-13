#!/usr/bin/env python3
"""Motion/Doppler target on the full fusion model with a BEVCar radar branch.

This is the follow-up to compare_motion_target_tiny.py. That experiment attached a
motion head to the full image+radar fusion model whose radar branch was metaradar
("last-write-wins" voxelization), and found the velocity ablation penalty was tiny
(+0.05) because metaradar drops the sparse per-point velocity. Here the metaradar
branch is replaced by BEVCar's VoxelNet, which keeps velocity per voxel through a
learned MLP. Everything else is unchanged: the camera encoder is frozen (so velocity
can only reach the motion head through the radar branch), the semantic DINOv2 target
is kept as a control, and the probe measures how the loss reacts to zeroing the
radar velocity channels. Expectation: the motion-target penalty becomes large, of the
same order as the BEVCar result in compare_radar_velocity_tiny.py, proving the recipe
(motion target + velocity-preserving radar encoder). Small-sample overfit diagnostic.
"""
from _bootstrap import enable_source_package
enable_source_package()

import argparse
import importlib.util
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
from dinov2_bev_demo import build_loader, load_teacher  # noqa: E402
from eval_image_distill_heldout import build_targets  # noqa: E402
from nets.bevcar_voxel_adapter import prepare_bevcar_voxels  # noqa: E402
from nets.radar_encoder import transform_radar_to_camera_bev  # noqa: E402
from nets.segnet import Segnet  # noqa: E402
from semantic_distill_smoke import SemanticDistillationModel, semantic_loss  # noqa: E402
from train_nuscenes import Z, Y, X, bounds, scene_centroid  # noqa: E402


VELOCITY_CLIP = 15.0
MOVING_THRESHOLD = 1.0
# BEVCar's 7 input channels are (z_mem, y_mem, x_mem, rcs, raw_vx, raw_vz, valid_mask);
# the two velocity channels are indices 4 and 5.
BEVCAR_VELOCITY_FEATURE_CHANNELS = slice(4, 6)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed-list', default='125,42,7')
    parser.add_argument('--model-name', default='dinov2_vits14')
    parser.add_argument('--variants', default='semantic_only,motion_only,joint',
                        help='comma-separated subset of the three variants')
    parser.add_argument('--zero-camera', action='store_true',
                        help='zero the camera BEV so velocity can only come from radar')
    return parser.parse_args()


from dd2414.models.legacy_motion import official_voxelnet


from dd2414.models.legacy_motion import MotionBevcarModel


from dd2414.models.legacy_motion import build_model


from dd2414.models.legacy_motion import prepare_bevcar_inputs


from dd2414.objectives.radar_motion import motion_target


from dd2414.objectives.radar_motion import motion_loss


def ablate_bevcar_voxels(bevcar_voxels, mode):
    features, coords, counts = bevcar_voxels
    if mode == 'correct':
        return bevcar_voxels
    out_features = features.clone()
    if mode == 'zero_velocity':
        out_features[..., BEVCAR_VELOCITY_FEATURE_CHANNELS] = 0.0
    elif mode == 'no_radar':
        out_features.zero_()
    else:
        raise ValueError(mode)
    return out_features, coords, counts


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
            images, pixel_from_cameras, camera0_from_cameras, vox_util, bevcar_voxels = inputs[index]
            prediction, motion_prediction = model(
                images, pixel_from_cameras, camera0_from_cameras, vox_util, bevcar_voxels
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
            images, pixel_from_cameras, camera0_from_cameras, vox_util, bevcar_voxels = inputs[index]
            for mode in ('correct', 'zero_velocity', 'no_radar'):
                radar = ablate_bevcar_voxels(bevcar_voxels, mode)
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

    loader = build_loader(
        args.data_root, num_workers=0, nsweeps=1,
        rotate_radar_velocity=True,
    )
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(args.samples)]

    teacher = load_teacher(args.model_name, device)
    for parameter in teacher.parameters():
        if parameter.requires_grad:
            raise RuntimeError('DINOv2 teacher must remain frozen')
    fusion_targets, confidences, _ = build_targets(batches, teacher, device)
    del teacher
    torch.cuda.empty_cache()

    inputs = [prepare_bevcar_inputs(batch, device) for batch in batches]
    motion_fields, motion_coverages, motion_infos = [], [], []
    for batch in batches:
        field, coverage, info = motion_target(batch, device)
        motion_fields.append(field)
        motion_coverages.append(coverage)
        motion_infos.append(info)
        print(f'sample: covered_cells={info["covered_cells"]} '
              f'moving_points={info["moving_points"]}', flush=True)

    official_class = official_voxelnet(args.bevcar_source)
    all_variants = (
        ('semantic_only', 1.0, 0.0),
        ('motion_only', 0.0, 1.0),
        ('joint', 1.0, 0.5),
    )
    requested = [part.strip() for part in args.variants.split(',') if part.strip()]
    known = {label for label, _, _ in all_variants}
    if not requested or not set(requested).issubset(known):
        raise ValueError(f'--variants must be a subset of {sorted(known)}')
    variants = tuple(v for v in all_variants if v[0] in requested)
    seed_results = {}
    for seed in seeds:
        seed_results[str(seed)] = {}
        for label, sem_weight, motion_weight in variants:
            model = build_model(device, seed, official_class,
                                zero_camera=args.zero_camera)
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
        'scope': 'motion target on full fusion model with BEVCar radar branch, frozen camera encoder, overfit',
        'samples': args.samples,
        'steps': args.steps,
        'learning_rate': args.learning_rate,
        'zero_camera': args.zero_camera,
        'velocity_clip': VELOCITY_CLIP,
        'moving_threshold': MOVING_THRESHOLD,
        'bevcar_velocity_channels': list(range(
            BEVCAR_VELOCITY_FEATURE_CHANNELS.start, BEVCAR_VELOCITY_FEATURE_CHANNELS.stop
        )),
        'target_fields': 'vx_comp,vy_comp (radar cols 8,9) on moving points only, camera frame',
        'variants': {label: {'sem_weight': sw, 'motion_weight': mw}
                     for label, sw, mw in variants},
        'motion_infos': motion_infos,
        'seeds': seeds,
        'seed_results': seed_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = 'motion_target_bevcar_tiny_zerocam.json' if args.zero_camera \
        else 'motion_target_bevcar_tiny.json'
    path = args.output_dir / name
    path.write_text(json.dumps(report, indent=2) + '\n')

    print('--- motion-target (BEVCar) summary (mean over seeds) ---')
    for label, sem_weight, motion_weight in variants:
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
    print('MOTION_TARGET_BEVCAR_TINY_OK')


if __name__ == '__main__':
    main()
