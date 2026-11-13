#!/usr/bin/env python3
"""Opt-in streaming BEVCar semantic+motion trainer for the scaling study."""
from _bootstrap import enable_source_package
enable_source_package()
from dd2414.runtime.provenance import record_package_implementation

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from dd2414.models.legacy_motion import build_model, official_voxelnet, prepare_bevcar_inputs
from dd2414.objectives.radar_motion import motion_loss, motion_target
from dd2414.teachers.dino import build_loaders, load_teacher
from dd2414.teachers.dino import build_targets
from dd2414.evaluation.radar_controls import IncrementalEvaluationWriter, RADAR_MODES, apply_radar_velocity_mode, radar_for_mode, spread_positions, wrong_scene_dataset_indices
from dd2414.data.manifest import IndexedSubset, make_streaming_loader, scale_indices, validate_scaling_manifest
from dd2414.losses.semantic import semantic_loss
from dd2414.runtime.checkpoints import StatefulShuffleSampler, load_training_checkpoint, save_periodic_checkpoint


GIB = 1024 ** 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--scale', choices=('full',), default='full')
    parser.add_argument('--max-updates', type=int, required=True)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--semantic-weight', type=float, default=1.0)
    parser.add_argument('--motion-weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--nsweeps', type=int, default=1)
    parser.add_argument('--model-name', default='dinov2_vits14')
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-log', type=Path, required=True)
    parser.add_argument('--validation-records', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path, required=True)
    parser.add_argument('--val-samples', type=int, choices=(6019,), default=6019,
                        help='Full validation only.')
    parser.add_argument('--stop-after-update', type=int, default=0,
                        help='cleanly stop after this update, for resume testing')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-amp', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--validation-print-every', type=int, default=500)
    return parser.parse_args()


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from dd2414.evaluation.scaling import file_sha256


def atomic_json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


from dd2414.evaluation.scaling import prepare_train_log


from dd2414.evaluation.scaling import append_train_record


from dd2414.evaluation.scaling import trainable_optimizer


from dd2414.evaluation.scaling import _empty_metric_totals


from dd2414.evaluation.scaling import _metric_means


from dd2414.evaluation.scaling import _paired_penalties


from dd2414.evaluation.scaling import validation_summary


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if (args.max_updates < 1 or args.checkpoint_every < 1
            or args.print_every < 1 or args.validation_print_every < 1):
        raise ValueError('update/checkpoint/print intervals must be positive')
    if args.nsweeps != 1:
        raise ValueError('the primary scaling protocol is locked to one sweep')
    if args.semantic_weight != 1.0 or args.motion_weight != 0.5:
        raise ValueError('P4 objective is locked to semantic=1.0, motion=0.5')

    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    if manifest['seed'] != args.seed:
        raise ValueError('manifest seed does not match training seed')
    dataset_indices = scale_indices(manifest, 'full')
    scale_name = 'full'

    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    reset_seeds(args.seed)
    train_loader, val_loader = build_loaders(
        args.data_root, num_workers=0, nsweeps=args.nsweeps,
        rotate_radar_velocity=True, dset='trainval',
    )
    train_dataset, val_dataset = train_loader.dataset, val_loader.dataset
    del train_loader, val_loader
    subset = IndexedSubset(train_dataset, dataset_indices)
    sampler = StatefulShuffleSampler(subset, seed=args.seed)

    teacher = load_teacher(args.model_name, device)
    official_class = official_voxelnet(args.bevcar_source)
    model = build_model(device, args.seed, official_class, zero_camera=False)
    trainable, optimizer = trainable_optimizer(model, args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0, growth_interval=1000)
    run_config = {
        'dataset': 'v1.0-trainval',
        'manifest_sha256': manifest['manifest_sha256'],
        'scale': scale_name,
        'train_samples': len(dataset_indices),
        'seed': args.seed,
        'nsweeps': args.nsweeps,
        'max_updates': args.max_updates,
        'learning_rate': args.learning_rate,
        'semantic_weight': args.semantic_weight,
        'motion_weight': args.motion_weight,
        'model_name': args.model_name,
        'bevcar_voxelnet_sha256': file_sha256(args.bevcar_source / 'nets' / 'voxelnet.py'),
        'checkpoint_every': args.checkpoint_every,
        'num_workers': 0,
    }
    record_package_implementation(run_config, Path(__file__).resolve().parents[1])
    progress = {
        'epoch': 0,
        'update': 0,
        'loss_sum': 0.0,
        'semantic_sum': 0.0,
        'motion_sum': 0.0,
        'moving_updates': 0,
        'covered_cells': 0,
        'train_seconds': 0.0,
        'peak_allocated_gib': 0.0,
        'amp_overflow_retries': 0,
    }
    if args.resume:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f'resume checkpoint is missing: {args.checkpoint}')
        progress = load_training_checkpoint(
            args.checkpoint, model=model, optimizer=optimizer, scaler=scaler,
            sampler=sampler, expected_run_config=run_config, map_location=device,
        )
        progress.setdefault('amp_overflow_retries', 0)
        prepare_train_log(args.train_log, resume_update=progress['update'])
        print(f'resumed update={progress["update"]} epoch={progress["epoch"]} '
              f'position={sampler.position}', flush=True)
    else:
        if args.checkpoint.exists():
            raise FileExistsError(
                f'checkpoint already exists; pass --resume or choose a new path: {args.checkpoint}'
            )
        prepare_train_log(args.train_log)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint_seconds = 0.0
    stopped = False
    model.train()
    model.student.decoder.eval()
    while progress['update'] < args.max_updates:
        stream = DataLoader(
            subset, batch_size=1, sampler=sampler, num_workers=0,
            drop_last=False, pin_memory=False,
        )
        consumed = False
        for subset_position, batch in stream:
            consumed = True
            update_start = time.perf_counter()
            subset_position = int(subset_position.item())
            dataset_index = dataset_indices[subset_position]
            fusion_targets, confidences, _ = build_targets([batch], teacher, device)
            inputs = prepare_bevcar_inputs(batch, device)
            motion_field, motion_coverage, motion_info = motion_target(batch, device)
            overflow_retries = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    semantic_prediction, motion_prediction = model(*inputs)
                    sem, _ = semantic_loss(
                        semantic_prediction, fusion_targets[0][None], confidences[0][None],
                    )
                    mot, _ = motion_loss(
                        motion_prediction, motion_field, motion_coverage,
                    )
                    loss = args.semantic_weight * sem + args.motion_weight * mot
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f'non-finite training loss at update {progress["update"] + 1}'
                    )
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, 5.0, error_if_nonfinite=False,
                )
                scaler.step(optimizer)
                scaler.update()
                if torch.isfinite(gradient_norm):
                    break
                overflow_retries += 1
                progress['amp_overflow_retries'] += 1
                scale_after = scaler.get_scale()
                print(
                    f'amp overflow before update={progress["update"] + 1} '
                    f'retry={overflow_retries} scale={scale_before:g}->{scale_after:g}',
                    flush=True,
                )
                if scale_after >= scale_before or overflow_retries >= 16:
                    raise RuntimeError(
                        f'non-finite gradients did not recover before update '
                        f'{progress["update"] + 1}'
                    )
            torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_start
            progress['update'] += 1
            progress['epoch'] = sampler.epoch
            progress['loss_sum'] += float(loss.detach())
            progress['semantic_sum'] += float(sem.detach())
            progress['motion_sum'] += float(mot.detach())
            progress['moving_updates'] += int(motion_info['covered_cells'] > 0)
            progress['covered_cells'] += motion_info['covered_cells']
            progress['train_seconds'] += update_seconds
            progress['peak_allocated_gib'] = max(
                progress['peak_allocated_gib'],
                torch.cuda.max_memory_allocated(device) / GIB,
            )
            append_train_record(args.train_log, {
                'update': progress['update'],
                'epoch': sampler.epoch,
                'sampler_position': sampler.position,
                'dataset_index': dataset_index,
                'loss': float(loss.detach()),
                'semantic_loss': float(sem.detach()),
                'motion_loss': float(mot.detach()),
                'motion_covered_cells': motion_info['covered_cells'],
                'amp_overflow_retries': overflow_retries,
                'seconds': update_seconds,
                'allocated_gib': torch.cuda.memory_allocated(device) / GIB,
            })
            del fusion_targets, confidences, inputs, motion_field, motion_coverage

            force = (
                progress['update'] == args.max_updates
                or (args.stop_after_update > 0
                    and progress['update'] == args.stop_after_update)
            )
            if force or progress['update'] % args.checkpoint_every == 0:
                checkpoint_start = time.perf_counter()
                save_periodic_checkpoint(
                    args.checkpoint, every_updates=args.checkpoint_every,
                    force=force, model=model, optimizer=optimizer, scaler=scaler,
                    sampler=sampler, progress=progress, run_config=run_config,
                )
                checkpoint_seconds += time.perf_counter() - checkpoint_start
                print(f'checkpoint update={progress["update"]} '
                      f'path={args.checkpoint}', flush=True)
            if progress['update'] % args.print_every == 0 or force:
                print(f'update={progress["update"]}/{args.max_updates} '
                      f'loss={float(loss):.4f} sem={float(sem):.4f} '
                      f'mot={float(mot):.4f} seconds={update_seconds:.3f}', flush=True)
            if args.stop_after_update > 0 and progress['update'] == args.stop_after_update:
                stopped = True
                break
            if progress['update'] >= args.max_updates:
                break
        if stopped or progress['update'] >= args.max_updates:
            break
        if not consumed:
            continue

    status = 'stopped_for_resume' if stopped else 'complete'
    validation = None
    if not stopped and args.val_samples:
        validation = validation_summary(
            model, teacher, val_dataset, manifest, args.val_samples, device,
            args.seed, args.validation_records, args.eval_amp,
            print_every=args.validation_print_every,
        )
    summary = {
        'status': status,
        'run_config': run_config,
        'progress': progress,
        'mean_train_loss': progress['loss_sum'] / progress['update'],
        'mean_semantic_loss': progress['semantic_sum'] / progress['update'],
        'mean_motion_loss': progress['motion_sum'] / progress['update'],
        'updates_per_second': progress['update'] / progress['train_seconds'],
        'estimated_30000_update_hours': (
            30000 * progress['train_seconds'] / progress['update'] / 3600
        ),
        'checkpoint_seconds_this_process': checkpoint_seconds,
        'checkpoint_bytes': args.checkpoint.stat().st_size,
        'teacher_feature_cache': None,
        'validation': validation,
    }
    atomic_json_write(summary, args.summary_out)
    console_summary = dict(summary)
    if validation is not None:
        console_summary['validation'] = {
            key: value for key, value in validation.items()
            if key not in ('positions', 'per_scene')
        }
        console_summary['validation']['position_count'] = len(validation['positions'])
        console_summary['validation']['scene_count'] = len(validation['per_scene'])
    print(json.dumps(console_summary, indent=2, allow_nan=False), flush=True)
    print('RADAR_SCALING_STOPPED' if stopped else 'RADAR_SCALING_TRAIN_OK', flush=True)


if __name__ == '__main__':
    main()
