#!/usr/bin/env python3
"""Held-out check for dual-teacher (camera DINOv2 + radar-anchored BEV) distillation.

Trains several trainable-encoder variants on a few training samples — fusion-only
baseline, 1:1 image+fusion, and dual-teacher 0.8/0.2 weightings in both
directions — then measures, on scene-disjoint validation samples, (a) each
variant's held-out fusion loss, (b) the held-out image-feature similarity of
image-distilled variants, and (c) the fusion-loss change when the radar input is
zeroed (a radar-sensitivity probe). This is a small-sample generalization probe,
not a downstream accuracy benchmark.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import utils.vox  # noqa: E402
from dinov2_bev_demo import (  # noqa: E402
    camera_geometry,
    extract_teacher_features,
    load_teacher,
    make_radar_bev,
    radar_anchored_soft_targets,
)
from heldout_data import add_dataset_args, load_probe_batches, validate_probe_args  # noqa: E402
from nets.segnet import Segnet  # noqa: E402
from semantic_distill_smoke import (  # noqa: E402
    SemanticDistillationModel,
    prepare_inputs,
    semantic_loss,
)
from train_nuscenes import Z, Y, X, bounds, scene_centroid  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    add_dataset_args(parser)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--train-samples', type=int, default=4)
    parser.add_argument('--val-samples', type=int, default=4)
    parser.add_argument('--nsweeps', type=int, default=1,
                        help='number of radar sweeps merged per sample')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed-list', default='125',
                        help='comma-separated integer seeds')
    parser.add_argument('--model-name', default='dinov2_vits14')
    parser.add_argument('--tag', default='',
                        help='optional suffix for the output filenames (e.g. train64)')
    return parser.parse_args()


def build_targets(batches, teacher, device):
    """Return (fusion_targets, confidences, image_targets) for a list of batches.

    ``image_targets`` are the raw DINOv2 patch features (S, 384, 14, 24); the
    fusion target is the radar-anchored soft BEV target (384, Z, X).
    """
    fusion_targets = []
    confidences = []
    image_targets = []
    with torch.inference_mode():
        for batch in batches:
            images = batch[0][:, 0]
            features = extract_teacher_features(teacher, images, device)
            _, pixel_from_camera, cameras_from_camera0, cameras_from_vehicle = camera_geometry(
                batch, features.shape[-2], features.shape[-1], device
            )
            vox_util = utils.vox.Vox_util(
                Z, Y, X, scene_centroid=scene_centroid.to(device),
                bounds=bounds, assert_cube=False,
            )
            _, radar_vehicle, radar_camera0 = make_radar_bev(
                batch, cameras_from_vehicle, vox_util, device
            )
            target, confidence, _ = radar_anchored_soft_targets(
                features, pixel_from_camera, cameras_from_vehicle,
                radar_vehicle, radar_camera0, vox_util,
            )
            fusion_targets.append(target.half())
            confidences.append(confidence.half())
            image_targets.append(features.half())
            del features, vox_util
    return fusion_targets, confidences, image_targets


def build_model(device, image_distill, seed):
    torch.manual_seed(seed)
    student = Segnet(
        Z, Y, X, vox_util=utils.vox.Vox_util(
            Z, Y, X, scene_centroid=scene_centroid.to(device),
            bounds=bounds, assert_cube=False,
        ),
        use_radar=True, use_metaradar=True, do_rgbcompress=True,
        encoder_type='res101', rand_flip=False, pretrained_backbone=False,
    ).to(device)
    model = SemanticDistillationModel(student, image_distill=image_distill).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.student.bev_compressor.parameters():
        parameter.requires_grad_(True)
    for parameter in model.semantic_head.parameters():
        parameter.requires_grad_(True)
    if image_distill:
        for parameter in model.image_head.parameters():
            parameter.requires_grad_(True)
    for parameter in model.student.encoder.parameters():
        parameter.requires_grad_(True)
    return model


def image_confidence_for(image_target, device):
    return torch.ones(
        image_target.shape[0], image_target.shape[-2], image_target.shape[-1],
        device=device,
    )


def train_variant(model, inputs, fusion_targets, confidences, image_targets,
                  image_distill, steps, learning_rate, device,
                  fusion_weight=1.0, image_weight=1.0):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0, growth_interval=1000)
    model.train()
    model.student.decoder.eval()
    losses = []
    for step in range(steps):
        index = step % len(inputs)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            prediction, image_prediction, _, _ = model(*inputs[index])
            loss, _ = semantic_loss(
                prediction, fusion_targets[index][None], confidences[index][None]
            )
            loss = fusion_weight * loss
            if image_distill:
                image_prediction = F.interpolate(
                    image_prediction, size=image_targets[index].shape[-2:],
                    mode='bilinear', align_corners=False,
                )
                image_loss, _ = semantic_loss(
                    image_prediction, image_targets[index],
                    image_confidence_for(image_targets[index], device),
                )
                loss = loss + image_weight * image_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    return losses


def evaluate(model, inputs, fusion_targets, confidences, image_targets, device):
    model.eval()
    fusion_losses = []
    image_losses = []
    no_radar_fusion_losses = []
    with torch.inference_mode():
        for index in range(len(inputs)):
            images, pixel_from_cameras, camera0_from_cameras, vox_util, radar_voxels = inputs[index]
            prediction, image_prediction, _, _ = model(
                images, pixel_from_cameras, camera0_from_cameras, vox_util, radar_voxels
            )
            fusion_loss, _ = semantic_loss(
                prediction, fusion_targets[index][None], confidences[index][None]
            )
            fusion_losses.append(float(fusion_loss))
            no_radar_prediction, _, _, _ = model(
                images, pixel_from_cameras, camera0_from_cameras, vox_util,
                torch.zeros_like(radar_voxels),
            )
            no_radar_fusion_loss, _ = semantic_loss(
                no_radar_prediction, fusion_targets[index][None], confidences[index][None]
            )
            no_radar_fusion_losses.append(float(no_radar_fusion_loss))
            if image_prediction is not None:
                image_prediction = F.interpolate(
                    image_prediction, size=image_targets[index].shape[-2:],
                    mode='bilinear', align_corners=False,
                )
                image_loss, _ = semantic_loss(
                    image_prediction, image_targets[index],
                    image_confidence_for(image_targets[index], device),
                )
                image_losses.append(float(image_loss))
    return (
        np.asarray(fusion_losses, dtype=np.float32),
        np.asarray(image_losses, dtype=np.float32) if image_losses else None,
        np.asarray(no_radar_fusion_losses, dtype=np.float32),
    )


def main():
    args = parse_args()
    validate_probe_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    seeds = [int(part.strip()) for part in args.seed_list.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seed-list must have distinct integer seeds')
    if args.nsweeps < 1:
        raise ValueError('nsweeps must be positive')

    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    train_batches, val_batches, sample_manifest = load_probe_batches(args)

    teacher = load_teacher(args.model_name, device)
    for parameter in teacher.parameters():
        if parameter.requires_grad:
            raise RuntimeError('DINOv2 teacher must remain frozen')

    train_fusion, train_conf, train_image = build_targets(train_batches, teacher, device)
    val_fusion, val_conf, val_image = build_targets(val_batches, teacher, device)
    del teacher
    torch.cuda.empty_cache()

    train_inputs = [prepare_inputs(batch, device) for batch in train_batches]
    val_inputs = [prepare_inputs(batch, device) for batch in val_batches]

    variants = (
        ('baseline',                False, 1.0, 1.0),
        ('image_distill',           True,  1.0, 1.0),
        ('dual_08_camera_02_radar', True,  0.2, 0.8),
        ('dual_02_camera_08_radar', True,  0.8, 0.2),
    )
    seed_results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed_results[str(seed)] = {}
        for label, image_distill, fusion_weight, image_weight in variants:
            model = build_model(device, image_distill, seed)
            start = time.perf_counter()
            train_losses = train_variant(
                model, train_inputs, train_fusion, train_conf, train_image,
                image_distill, args.steps, args.learning_rate, device,
                fusion_weight=fusion_weight, image_weight=image_weight,
            )
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start
            held_fusion, held_image, no_radar_fusion = evaluate(
                model, val_inputs, val_fusion, val_conf, val_image, device
            )
            seed_results[str(seed)][label] = {
                'train_losses': train_losses,
                'train_seconds': train_seconds,
                'heldout_fusion_losses': held_fusion,
                'heldout_image_losses': held_image,
                'no_radar_fusion_losses': no_radar_fusion,
            }
            del model
            torch.cuda.empty_cache()
            radar_penalty = float(no_radar_fusion.mean() - held_fusion.mean())
            print(f'[seed {seed} {label}] fusion_w={fusion_weight} image_w={image_weight} '
                  f'train last-cycle mean loss: {np.mean(train_losses[-args.train_samples:]):.6f}')
            print(f'[seed {seed} {label}] heldout fusion loss: mean {held_fusion.mean():.6f} '
                  f'per-sample {np.round(held_fusion, 4).tolist()}')
            print(f'[seed {seed} {label}] heldout no-radar fusion loss: mean {no_radar_fusion.mean():.6f} '
                  f'(radar penalty {radar_penalty:+.6f})', flush=True)
            if held_image is not None:
                print(f'[seed {seed} {label}] heldout image loss: mean {held_image.mean():.6f} '
                      f'per-sample {np.round(held_image, 4).tolist()}')

    summary = {}
    save_dict = {}
    for label, _, fusion_weight, image_weight in variants:
        held_fusion = np.concatenate(
            [seed_results[str(s)][label]['heldout_fusion_losses'] for s in seeds])
        no_radar = np.concatenate(
            [seed_results[str(s)][label]['no_radar_fusion_losses'] for s in seeds])
        held_images = [seed_results[str(s)][label]['heldout_image_losses'] for s in seeds]
        held_image = np.concatenate(held_images) if all(i is not None for i in held_images) else None
        last_cycle = np.mean([
            np.mean(seed_results[str(s)][label]['train_losses'][-args.train_samples:])
            for s in seeds
        ])
        summary[label] = {
            'fusion_weight': fusion_weight,
            'image_weight': image_weight,
            'heldout_fusion_mean': float(held_fusion.mean()),
            'no_radar_fusion_mean': float(no_radar.mean()),
            'radar_penalty_mean': float(no_radar.mean() - held_fusion.mean()),
            'heldout_image_mean': float(held_image.mean()) if held_image is not None else None,
            'train_last_cycle_mean': float(last_cycle),
            'train_seconds': float(np.mean([seed_results[str(s)][label]['train_seconds'] for s in seeds])),
        }
        save_dict[f'{label}_heldout_fusion'] = held_fusion
        save_dict[f'{label}_no_radar_fusion'] = no_radar
        if held_image is not None:
            save_dict[f'{label}_heldout_image'] = held_image
    summary['meta'] = {
        'dset': args.dset,
        'sample_selection': args.sample_selection,
        'sample_manifest': sample_manifest,
        'train_samples': args.train_samples,
        'val_samples': args.val_samples,
        'steps': args.steps,
        'nsweeps': args.nsweeps,
        'learning_rate': args.learning_rate,
        'seeds': seeds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (f'trainval_nsweeps{args.nsweeps}' if args.dset == 'trainval' else '')
    suffix = f'_{tag}' if tag else ''
    np.savez_compressed(args.output_dir / f'dual_teacher_heldout_metrics{suffix}.npz', **save_dict)
    with open(args.output_dir / f'dual_teacher_heldout{suffix}.json', 'w') as handle:
        json.dump(summary, handle, indent=2)

    print('--- dual-teacher heldout summary (mean over seeds) ---')
    for label, _, fusion_weight, image_weight in variants:
        row = summary[label]
        image_str = f"{row['heldout_image_mean']:.6f}" if row['heldout_image_mean'] is not None else 'n/a'
        print(f"{label:26s} fusion_w={fusion_weight} image_w={image_weight} | "
              f"fusion {row['heldout_fusion_mean']:.6f} | no_radar {row['no_radar_fusion_mean']:.6f} "
              f"(penalty {row['radar_penalty_mean']:+.6f}) | image {image_str}")
    print(f'metrics: {args.output_dir / f"dual_teacher_heldout_metrics{suffix}.npz"}')
    print(f'summary: {args.output_dir / f"dual_teacher_heldout{suffix}.json"}')
    print('DUAL_TEACHER_HELDOUT_OK')


if __name__ == '__main__':
    main()
