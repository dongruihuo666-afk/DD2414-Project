#!/usr/bin/env python3
"""Held-out check for the motion target on the full fusion model (BEVCar radar branch).

compare_motion_target_bevcar_tiny.py showed the recipe on a 4-sample overfit split:
with the camera zeroed, zeroing the radar velocity channels raised the motion loss
(velocity reaches the motion head through BEVCar), but with the camera ON the penalty
vanished — on only 4 samples a frozen camera encoder is a fixed per-sample pattern that
the trainable bev_compressor can memorize, so it never has to read radar velocity.

Here the camera stays ON but training and evaluation samples are drawn from the
scene-disjoint train/val splits, so the camera BEV on a held-out val sample is a
pattern the bev_compressor has never seen and cannot memorize. If the motion head
genuinely relies on radar velocity, zeroing those channels must raise the held-out
motion loss. This is a small-sample held-out generalization probe, not a downstream
accuracy benchmark.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from compare_motion_target_bevcar_tiny import (  # noqa: E402
    BEVCAR_VELOCITY_FEATURE_CHANNELS,
    MOVING_THRESHOLD,
    VELOCITY_CLIP,
    MotionBevcarModel,
    build_model,
    evaluate,
    motion_loss,
    motion_target,
    official_voxelnet,
    prepare_bevcar_inputs,
    train,
)
from dinov2_bev_demo import build_loader, load_teacher  # noqa: E402
from eval_image_distill_heldout import build_targets  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--train-samples', type=int, default=4)
    parser.add_argument('--val-samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed-list', default='125,42,7')
    parser.add_argument('--model-name', default='dinov2_vits14')
    parser.add_argument('--variants', default='semantic_only,motion_only,joint',
                        help='comma-separated subset of the three variants')
    parser.add_argument('--tag', default='',
                        help='optional suffix for the output filename (e.g. train64)')
    return parser.parse_args()


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

    train_loader = build_loader(
        args.data_root, num_workers=0, nsweeps=1,
        rotate_radar_velocity=True, split='train',
    )
    val_loader = build_loader(
        args.data_root, num_workers=0, nsweeps=1,
        rotate_radar_velocity=True, split='val',
    )
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    train_batches = [next(train_iter) for _ in range(args.train_samples)]
    val_batches = [next(val_iter) for _ in range(args.val_samples)]

    teacher = load_teacher(args.model_name, device)
    for parameter in teacher.parameters():
        if parameter.requires_grad:
            raise RuntimeError('DINOv2 teacher must remain frozen')
    train_fusion, train_conf, _ = build_targets(train_batches, teacher, device)
    val_fusion, val_conf, _ = build_targets(val_batches, teacher, device)
    del teacher
    torch.cuda.empty_cache()

    train_inputs = [prepare_bevcar_inputs(batch, device) for batch in train_batches]
    val_inputs = [prepare_bevcar_inputs(batch, device) for batch in val_batches]

    def build_motion(batches):
        fields, coverages, infos = [], [], []
        for batch in batches:
            field, coverage, info = motion_target(batch, device)
            fields.append(field)
            coverages.append(coverage)
            infos.append(info)
            print(f'sample: covered_cells={info["covered_cells"]} '
                  f'moving_points={info["moving_points"]}', flush=True)
        return fields, coverages, infos

    train_motion, train_coverages, train_infos = build_motion(train_batches)
    val_motion, val_coverages, val_infos = build_motion(val_batches)

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
            model = build_model(device, seed, official_class, zero_camera=False)
            start = time.perf_counter()
            sem_history, mot_history = train(
                model, train_inputs, train_fusion, train_conf,
                train_motion, train_coverages,
                sem_weight, motion_weight, args.steps, args.learning_rate, device,
            )
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start
            heldout_sem, heldout_mot = evaluate(
                model, val_inputs, val_fusion, val_conf,
                val_motion, val_coverages, device,
            )
            seed_results[str(seed)][label] = {
                'sem_history': sem_history,
                'mot_history': mot_history,
                'heldout_sem_eval': {k: v.tolist() for k, v in heldout_sem.items()},
                'heldout_mot_eval': {k: v.tolist() for k, v in heldout_mot.items()},
                'train_seconds': train_seconds,
            }
            del model
            torch.cuda.empty_cache()
            sem_penalty = float(heldout_sem['zero_velocity'].mean() - heldout_sem['correct'].mean())
            mot_penalty = float(heldout_mot['zero_velocity'].mean() - heldout_mot['correct'].mean())
            print(f'seed {seed} {label}: heldout sem correct={heldout_sem["correct"].mean():.4f} '
                  f'zero_vel={heldout_sem["zero_velocity"].mean():.4f} '
                  f'(penalty {sem_penalty:+.4f}) no_radar={heldout_sem["no_radar"].mean():.4f}',
                  flush=True)
            print(f'seed {seed} {label}: heldout mot correct={heldout_mot["correct"].mean():.4f} '
                  f'zero_vel={heldout_mot["zero_velocity"].mean():.4f} '
                  f'(penalty {mot_penalty:+.4f}) no_radar={heldout_mot["no_radar"].mean():.4f}',
                  flush=True)

    report = {
        'scope': 'motion target on full fusion model with BEVCar radar branch, held-out (train on train split, eval on scene-disjoint val split)',
        'train_samples': args.train_samples,
        'val_samples': args.val_samples,
        'steps': args.steps,
        'learning_rate': args.learning_rate,
        'velocity_clip': VELOCITY_CLIP,
        'moving_threshold': MOVING_THRESHOLD,
        'bevcar_velocity_channels': list(range(
            BEVCAR_VELOCITY_FEATURE_CHANNELS.start, BEVCAR_VELOCITY_FEATURE_CHANNELS.stop
        )),
        'target_fields': 'vx_comp,vy_comp (radar cols 8,9) on moving points only, camera frame',
        'variants': {label: {'sem_weight': sw, 'motion_weight': mw}
                     for label, sw, mw in variants},
        'train_motion_infos': train_infos,
        'val_motion_infos': val_infos,
        'seeds': seeds,
        'seed_results': seed_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_{args.tag}' if args.tag else ''
    path = args.output_dir / f'motion_target_bevcar_heldout{suffix}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')

    print('--- motion-target (BEVCar) held-out summary (mean over seeds) ---')
    for label, sem_weight, motion_weight in variants:
        for loss_name, key in (('sem', 'heldout_sem_eval'), ('mot', 'heldout_mot_eval')):
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
    print('MOTION_TARGET_BEVCAR_HELDOUT_OK')


if __name__ == '__main__':
    main()
