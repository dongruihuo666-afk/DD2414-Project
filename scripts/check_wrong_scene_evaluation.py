#!/usr/bin/env python3
"""Run all four radar controls on streamed trainval validation frames."""

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from compare_motion_target_bevcar_tiny import (  # noqa: E402
    build_model,
    motion_loss,
    motion_target,
    official_voxelnet,
    prepare_bevcar_inputs,
)
from dinov2_bev_demo import build_loaders, load_teacher  # noqa: E402
from eval_image_distill_heldout import build_targets  # noqa: E402
from radar_evaluation import (  # noqa: E402
    IncrementalEvaluationWriter,
    RADAR_MODES,
    radar_for_mode,
    wrong_scene_dataset_indices,
)
from radar_scaling_data import make_streaming_loader, validate_scaling_manifest  # noqa: E402
from semantic_distill_smoke import semantic_loss  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--nsweeps', type=int, default=1)
    parser.add_argument('--model-name', default='dinov2_vits14')
    return parser.parse_args()


def spread_positions(total, count):
    if not 1 <= count <= total:
        raise ValueError('sample count must fit the validation split')
    if count == 1:
        return [0]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def main():
    args = parse_args()
    if args.samples < 2:
        raise ValueError('wrong-scene check requires at least two target samples')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    wrong_indices, wrong_map = wrong_scene_dataset_indices(manifest, seed=args.seed)
    val = manifest['val']
    scene_by_token = dict(zip(val['tokens'], val['scene_tokens']))

    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    train_loader, val_loader = build_loaders(
        args.data_root, num_workers=0, nsweeps=args.nsweeps,
        rotate_radar_velocity=True, dset='trainval',
    )
    del train_loader
    dataset = val_loader.dataset
    positions = spread_positions(val['count'], args.samples)
    target_indices = [val['dataset_indices'][position] for position in positions]
    wrong_indices = [wrong_indices[position] for position in positions]
    target_stream = make_streaming_loader(dataset, target_indices, num_workers=0)
    wrong_stream = make_streaming_loader(dataset, wrong_indices, num_workers=0)

    teacher = load_teacher(args.model_name, device)
    official_class = official_voxelnet(args.bevcar_source)
    model = build_model(device, args.seed, official_class, zero_camera=False)
    model.eval()
    totals = {mode: {'semantic_loss': 0.0, 'motion_loss': 0.0}
              for mode in RADAR_MODES}

    with IncrementalEvaluationWriter(args.output) as writer:
        with torch.inference_mode():
            for offset, (batch, wrong_batch) in enumerate(zip(target_stream, wrong_stream)):
                fusion_targets, confidences, _ = build_targets([batch], teacher, device)
                inputs = prepare_bevcar_inputs(batch, device)
                wrong_inputs = prepare_bevcar_inputs(wrong_batch, device)
                motion_field, motion_coverage, motion_info = motion_target(batch, device)
                images, pixels, cameras, vox_util, matched_voxels = inputs
                modes = {}
                for mode in RADAR_MODES:
                    radar = radar_for_mode(
                        matched_voxels, mode, wrong_scene_voxels=wrong_inputs[-1],
                    )
                    semantic_prediction, motion_prediction = model(
                        images, pixels, cameras, vox_util, radar,
                    )
                    sem, _ = semantic_loss(
                        semantic_prediction, fusion_targets[0][None],
                        confidences[0][None],
                    )
                    mot, _ = motion_loss(
                        motion_prediction, motion_field, motion_coverage,
                    )
                    if not torch.isfinite(sem) or not torch.isfinite(mot):
                        raise RuntimeError(
                            f'non-finite loss for validation position {positions[offset]} '
                            f'mode {mode}: semantic={float(sem)}, motion={float(mot)}'
                        )
                    modes[mode] = {
                        'semantic_loss': float(sem),
                        'motion_loss': float(mot),
                    }
                    totals[mode]['semantic_loss'] += float(sem)
                    totals[mode]['motion_loss'] += float(mot)
                manifest_position = positions[offset]
                token = val['tokens'][manifest_position]
                wrong_token = wrong_map[token]
                writer.write({
                    'token': token,
                    'scene_token': val['scene_tokens'][manifest_position],
                    'wrong_radar_token': wrong_token,
                    'wrong_radar_scene_token': scene_by_token[wrong_token],
                    'motion_covered_cells': motion_info['covered_cells'],
                    'modes': modes,
                })

    count = args.samples
    means = {
        mode: {name: value / count for name, value in metrics.items()}
        for mode, metrics in totals.items()
    }
    summary = {
        'dataset': 'v1.0-trainval',
        'seed': args.seed,
        'nsweeps': args.nsweeps,
        'samples': count,
        'modes': list(RADAR_MODES),
        'means': means,
        'records': str(args.output),
        'mapping_is_full_val_bijection': len(set(wrong_map.values())) == val['count'],
        'all_wrong_scenes_differ': all(
            scene_by_token[target] != scene_by_token[source]
            for target, source in wrong_map.items()
        ),
        'acceptance': 'all four radar modes produced aligned incremental records',
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print('WRONG_SCENE_EVALUATION_OK', flush=True)


if __name__ == '__main__':
    main()
