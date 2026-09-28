#!/usr/bin/env python3
"""Held-out check for image-level DINOv2 distillation.

Trains two trainable-encoder variants on a few training samples — one with the
image-level DINOv2 term and one without — then measures, on scene-disjoint
validation samples, (a) the held-out fusion loss of each variant and (b) the
held-out image-feature similarity of the image-distilled variant. This is a
small-sample generalization probe, not a downstream accuracy benchmark.
"""

import argparse
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
    build_loader,
    camera_geometry,
    extract_teacher_features,
    load_teacher,
    make_radar_bev,
    radar_anchored_soft_targets,
)
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
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--train-samples', type=int, default=4)
    parser.add_argument('--val-samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--model-name', default='dinov2_vits14')
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
                  image_distill, steps, learning_rate, device):
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
            if image_distill:
                image_prediction = F.interpolate(
                    image_prediction, size=image_targets[index].shape[-2:],
                    mode='bilinear', align_corners=False,
                )
                image_loss, _ = semantic_loss(
                    image_prediction, image_targets[index],
                    image_confidence_for(image_targets[index], device),
                )
                loss = loss + image_loss
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
    with torch.inference_mode():
        for index in range(len(inputs)):
            prediction, image_prediction, _, _ = model(*inputs[index])
            fusion_loss, _ = semantic_loss(
                prediction, fusion_targets[index][None], confidences[index][None]
            )
            fusion_losses.append(float(fusion_loss))
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
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

    train_loader = build_loader(args.data_root, 0, 1, split='train')
    val_loader = build_loader(args.data_root, 0, 1, split='val')
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    train_batches = [next(train_iter) for _ in range(args.train_samples)]
    val_batches = [next(val_iter) for _ in range(args.val_samples)]

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

    results = {}
    for label, image_distill in (('baseline', False), ('image_distill', True)):
        model = build_model(device, image_distill, args.seed)
        start = time.perf_counter()
        train_losses = train_variant(
            model, train_inputs, train_fusion, train_conf, train_image,
            image_distill, args.steps, args.learning_rate, device,
        )
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        held_fusion, held_image = evaluate(
            model, val_inputs, val_fusion, val_conf, val_image, device
        )
        results[label] = {
            'train_losses': train_losses,
            'train_seconds': train_seconds,
            'heldout_fusion_losses': held_fusion,
            'heldout_image_losses': held_image,
        }
        print(f'[{label}] train last-cycle mean loss: {np.mean(train_losses[-args.train_samples:]):.6f}')
        print(f'[{label}] heldout fusion loss: mean {held_fusion.mean():.6f} '
              f'per-sample {np.round(held_fusion, 4).tolist()}')
        if held_image is not None:
            print(f'[{label}] heldout image loss: mean {held_image.mean():.6f} '
                  f'per-sample {np.round(held_image, 4).tolist()}')

    base = results['baseline']['heldout_fusion_losses']
    dist = results['image_distill']['heldout_fusion_losses']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / 'image_distill_heldout_metrics.npz',
        baseline_heldout_fusion=base,
        image_distill_heldout_fusion=dist,
        image_distill_heldout_image=results['image_distill']['heldout_image_losses'],
        baseline_train_losses=np.asarray(results['baseline']['train_losses'], dtype=np.float32),
        image_distill_train_losses=np.asarray(results['image_distill']['train_losses'], dtype=np.float32),
        train_samples=np.asarray(args.train_samples),
        val_samples=np.asarray(args.val_samples),
        steps=np.asarray(args.steps),
    )
    print(f'heldout_fusion_baseline_mean: {base.mean():.6f}')
    print(f'heldout_fusion_image_distill_mean: {dist.mean():.6f}')
    print(f'heldout_fusion_delta_mean: {dist.mean() - base.mean():.6f}')
    if results['image_distill']['heldout_image_losses'] is not None:
        print(f'heldout_image_loss_mean: {results["image_distill"]["heldout_image_losses"].mean():.6f}')
    print(f'metrics: {args.output_dir / "image_distill_heldout_metrics.npz"}')
    print('IMAGE_DISTILL_HELDOUT_OK')


if __name__ == '__main__':
    main()
