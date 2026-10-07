#!/usr/bin/env python3
"""Compare the new batch path against two historical batch-one forwards."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from compare_motion_target_bevcar_tiny import (  # noqa: E402
    build_model,
    official_voxelnet,
    prepare_bevcar_inputs,
)
from dinov2_bev_demo import build_loaders, load_teacher  # noqa: E402
from eval_image_distill_heldout import build_targets  # noqa: E402
from fullsize_training import (  # noqa: E402
    atomic_json_write,
    build_motion_targets_batched,
    build_targets_batched,
    slice_batch,
)
from radar_scaling_data import IndexedSubset, scale_indices, validate_scaling_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=125)
    return parser.parse_args()


def maximum_difference(left, right):
    return float((left.float() - right.float()).abs().max())


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    indices = scale_indices(manifest, 4096)[:2]
    train_loader, _ = build_loaders(
        args.data_root, num_workers=0, nsweeps=1,
        rotate_radar_velocity=True, dset='trainval',
    )
    subset = IndexedSubset(train_loader.dataset, indices)
    _, batch = next(iter(DataLoader(subset, batch_size=2, shuffle=False)))
    device = torch.device('cuda:0')
    teacher = load_teacher('dinov2_vits14', device)
    batched_target, batched_confidence = build_targets_batched(batch, teacher, device)
    historical_targets = []
    historical_confidences = []
    for index in range(2):
        targets, confidences, _ = build_targets(
            [slice_batch(batch, index)], teacher, device,
        )
        historical_targets.append(targets[0])
        historical_confidences.append(confidences[0])
    historical_target = torch.stack(historical_targets)
    historical_confidence = torch.stack(historical_confidences)

    batched_motion, batched_coverage, _ = build_motion_targets_batched(batch, device)
    single_motion = []
    single_coverage = []
    for index in range(2):
        field, coverage, _ = build_motion_targets_batched(
            slice_batch(batch, index), device,
        )
        single_motion.append(field[0])
        single_coverage.append(coverage[0])
    single_motion = torch.stack(single_motion)
    single_coverage = torch.stack(single_coverage)

    official_class = official_voxelnet(args.bevcar_source)
    model = build_model(device, args.seed, official_class, zero_camera=False)
    model.eval()
    with torch.inference_mode():
        batched_semantic, batched_motion_prediction = model(
            *prepare_bevcar_inputs(batch, device)
        )
        single_semantic = []
        single_motion_prediction = []
        for index in range(2):
            semantic, motion = model(
                *prepare_bevcar_inputs(slice_batch(batch, index), device)
            )
            single_semantic.append(semantic[0])
            single_motion_prediction.append(motion[0])
    single_semantic = torch.stack(single_semantic)
    single_motion_prediction = torch.stack(single_motion_prediction)

    result = {
        'samples': 2,
        'target_max_abs_difference': maximum_difference(
            batched_target, historical_target,
        ),
        'confidence_max_abs_difference': maximum_difference(
            batched_confidence, historical_confidence,
        ),
        'motion_target_max_abs_difference': maximum_difference(
            batched_motion, single_motion,
        ),
        'motion_coverage_max_abs_difference': maximum_difference(
            batched_coverage, single_coverage,
        ),
        'semantic_prediction_max_abs_difference': maximum_difference(
            batched_semantic, single_semantic,
        ),
        'motion_prediction_max_abs_difference': maximum_difference(
            batched_motion_prediction, single_motion_prediction,
        ),
        'semantic_prediction_mean_abs_difference': float(
            (batched_semantic.float() - single_semantic.float()).abs().mean()
        ),
        'motion_prediction_mean_abs_difference': float(
            (batched_motion_prediction.float() - single_motion_prediction.float()).abs().mean()
        ),
    }
    if result['target_max_abs_difference'] > 2e-3:
        raise RuntimeError(f'batch target mismatch: {result}')
    if result['confidence_max_abs_difference'] > 2e-3:
        raise RuntimeError(f'batch confidence mismatch: {result}')
    if result['motion_target_max_abs_difference'] != 0:
        raise RuntimeError(f'batch motion target mismatch: {result}')
    if result['motion_coverage_max_abs_difference'] != 0:
        raise RuntimeError(f'batch motion coverage mismatch: {result}')
    if result['semantic_prediction_max_abs_difference'] > 2e-2:
        raise RuntimeError(f'batch semantic prediction mismatch: {result}')
    if result['motion_prediction_max_abs_difference'] > 2e-2:
        raise RuntimeError(f'batch motion prediction mismatch: {result}')
    atomic_json_write(result, args.output)
    print(json.dumps(result, indent=2), flush=True)
    print('FULLSIZE_BATCH_EQUIVALENCE_OK', flush=True)


if __name__ == '__main__':
    main()
