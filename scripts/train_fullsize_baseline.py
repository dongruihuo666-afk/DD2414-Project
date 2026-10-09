#!/usr/bin/env python3
"""Epoch-native 4,096/full trainer for the unchanged radar-scaling model."""

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from compare_motion_target_bevcar_tiny import (  # noqa: E402
    build_model,
    official_voxelnet,
    prepare_bevcar_inputs,
)
from dinov2_bev_demo import build_loaders, load_teacher  # noqa: E402
from fullsize_training import (  # noqa: E402
    AcknowledgedShuffleSampler,
    atomic_json_write,
    build_motion_targets_batched,
    build_targets_batched,
    build_targets_from_features_batched,
    extract_teacher_features_batched,
    learning_rate_at_step,
    motion_loss_batched,
    progress_status,
    query_nvidia_smi,
    semantic_loss_batched,
)
from radar_scaling_data import IndexedSubset, scale_indices, validate_scaling_manifest  # noqa: E402
from radar_evaluation import apply_radar_velocity_mode  # noqa: E402
from train_radar_scaling import (  # noqa: E402
    file_sha256,
    prepare_train_log,
    validation_summary,
)
from training_checkpoint import (  # noqa: E402
    load_training_checkpoint,
    save_periodic_checkpoint,
    save_training_checkpoint,
)


GIB = 1024 ** 3
SUPPORTED_RADAR_SWEEPS = (1, 5, 10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--scale', choices=('4096', 'full'), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--minimum-lr-ratio', type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=float, default=0.25)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--nsweeps', type=int, choices=SUPPORTED_RADAR_SWEEPS,
                        default=1)
    parser.add_argument('--semantic-weight', type=float, default=1.0)
    parser.add_argument('--motion-weight', type=float, default=0.5)
    parser.add_argument('--radar-velocity-mode', choices=('full', 'zero'),
                        default='full')
    parser.add_argument('--model-name', default='dinov2_vits14')
    parser.add_argument('--checkpoint-every-steps', type=int, default=1000)
    parser.add_argument('--print-every-steps', type=int, default=50)
    parser.add_argument('--validation-epochs', default='1,3,5,8')
    parser.add_argument('--val-samples', type=int, default=6019)
    parser.add_argument('--validation-print-every', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-steps', type=int, default=0,
                        help='engineering-only clean stop for resume checks')
    parser.add_argument('--stop-after-epoch', type=int, default=0,
                        help='cleanly stop after this epoch and its validation')
    parser.add_argument('--profile-timing', action='store_true',
                        help='synchronize CUDA and record engineering phase timings')
    return parser.parse_args()


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_validation_epochs(value, total_epochs):
    epochs = tuple(sorted({int(part.strip()) for part in value.split(',') if part.strip()}))
    if any(epoch < 1 or epoch > total_epochs for epoch in epochs):
        raise ValueError('validation epochs must fall within the training run')
    return epochs


def default_learning_rate(batch_size):
    return min(2e-4 * math.sqrt(batch_size), 5e-4)


def append_json_line(handle, value):
    handle.write(json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ) + '\n')


def ema_update(values, current, decay=0.98):
    if not values:
        return dict(current)
    return {
        key: decay * values[key] + (1.0 - decay) * current[key]
        for key in current
    }


def refresh_wall_time(progress, base_wall_seconds, started_at):
    progress['wall_train_seconds'] = base_wall_seconds + max(
        time.monotonic() - started_at, 0.0,
    )


def write_status(args, progress, total_steps, dataset_size, latest, ema,
                 learning_rate, state='running', message=None):
    elapsed = progress.get('wall_train_seconds', 0.0)
    status = progress_status(
        state=state, scale=args.scale, epochs=args.epochs,
        dataset_size=dataset_size, batch_size=args.batch_size,
        optimizer_step=progress['update'], total_steps=total_steps,
        samples_seen=progress['samples_seen'], elapsed_seconds=elapsed,
        latest=latest, ema=ema, learning_rate=learning_rate,
        gpu=query_nvidia_smi(), message=message,
    )
    status['radar_velocity_mode'] = args.radar_velocity_mode
    status['nsweeps'] = args.nsweeps
    atomic_json_write(status, args.run_dir / 'status.json')
    return status


def compact_validation(summary):
    return {
        'samples': summary['samples'],
        'moving_frames': summary['moving_frames'],
        'covered_cells': summary['covered_cells'],
        'seconds': summary['seconds'],
        'frames_per_second': summary['frames_per_second'],
        'means': summary['means'],
        'penalties': summary['penalties'],
        'records': summary['records'],
    }


def run_epoch_validation(*, args, completed_epoch, model, teacher, val_dataset,
                         manifest, device, progress, total_steps, dataset_size,
                         latest, ema, learning_rate, writer, checkpoint_path,
                         checkpoint_dir, validation_dir, optimizer, scaler,
                         sampler, run_config, base_wall_seconds, started_at):
    epoch_checkpoint = checkpoint_dir / f'epoch{completed_epoch:03d}.pt'
    if not epoch_checkpoint.exists():
        shutil.copy2(checkpoint_path, epoch_checkpoint)
    refresh_wall_time(progress, base_wall_seconds, started_at)
    write_status(
        args, progress, total_steps, dataset_size, latest, ema,
        learning_rate, state='validating',
        message=f'full validation for epoch {completed_epoch}',
    )
    validation_path = validation_dir / f'epoch{completed_epoch:03d}.jsonl'
    result = validation_summary(
        model, teacher, val_dataset, manifest, args.val_samples,
        device, args.seed, validation_path, use_amp=True,
        print_every=args.validation_print_every,
        radar_velocity_mode=args.radar_velocity_mode,
    )
    atomic_json_write(
        result, validation_dir / f'epoch{completed_epoch:03d}_summary.json',
    )
    for metric, modes in result['penalties'].items():
        for mode, value in modes.items():
            if value is not None:
                writer.add_scalar(
                    f'validation/{metric}_{mode}_penalty', value, completed_epoch,
                )
    progress['completed_validation_epochs'] = sorted(
        set(progress['completed_validation_epochs']) | {completed_epoch}
    )
    refresh_wall_time(progress, base_wall_seconds, started_at)
    save_training_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, scaler=scaler,
        sampler=sampler, progress=progress, run_config=run_config,
    )
    model.train()
    model.student.decoder.eval()
    return compact_validation(result)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('epochs and batch size must be positive')
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError('num-workers must be nonnegative and prefetch-factor positive')
    if args.checkpoint_every_steps < 1 or args.print_every_steps < 1:
        raise ValueError('checkpoint and print intervals must be positive')
    if args.nsweeps not in SUPPORTED_RADAR_SWEEPS:
        raise ValueError(
            f'nsweeps must be one of {SUPPORTED_RADAR_SWEEPS}'
        )
    if args.semantic_weight != 1.0 or args.motion_weight != 0.5:
        raise ValueError('the epoch baseline keeps semantic=1.0 and motion=0.5')
    if args.stop_after_steps and args.stop_after_epoch:
        raise ValueError('choose at most one clean-stop condition')
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        raise ValueError('stop-after-epoch must fall within the training run')
    validation_epochs = parse_validation_epochs(args.validation_epochs, args.epochs)
    learning_rate = args.learning_rate or default_learning_rate(args.batch_size)

    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    if manifest['seed'] != args.seed:
        raise ValueError('manifest seed does not match training seed')
    dataset_indices = scale_indices(manifest, args.scale)
    dataset_size = len(dataset_indices)
    steps_per_epoch = math.ceil(dataset_size / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(steps_per_epoch * args.warmup_epochs)

    args.run_dir = args.run_dir.resolve()
    checkpoint_dir = args.run_dir / 'checkpoints'
    checkpoint_path = checkpoint_dir / 'latest.pt'
    metrics_path = args.run_dir / 'metrics.jsonl'
    validation_dir = args.run_dir / 'validation'
    summary_path = args.run_dir / 'summary.json'
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    reset_seeds(args.seed)

    # Build only the dataset objects here. The epoch stream below owns the
    # measured worker/prefetch configuration and acknowledged resume cursor.
    train_loader, val_loader = build_loaders(
        args.data_root, num_workers=0, nsweeps=args.nsweeps,
        rotate_radar_velocity=True, dset='trainval',
    )
    train_dataset, val_dataset = train_loader.dataset, val_loader.dataset
    del train_loader, val_loader
    subset = IndexedSubset(train_dataset, dataset_indices)
    sampler = AcknowledgedShuffleSampler(subset, seed=args.seed)

    teacher = load_teacher(args.model_name, device)
    official_class = official_voxelnet(args.bevcar_source)
    model = build_model(device, args.seed, official_class, zero_camera=False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16',
                                  init_scale=128.0, growth_interval=1000)
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    run_config = {
        'experiment': 'fullsize_epoch_baseline_v1',
        'dataset': 'v1.0-trainval',
        'manifest_sha256': manifest['manifest_sha256'],
        'scale': args.scale,
        'train_samples': dataset_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'steps_per_epoch': steps_per_epoch,
        'total_optimizer_steps': total_steps,
        'seed': args.seed,
        'nsweeps': args.nsweeps,
        'learning_rate': learning_rate,
        'minimum_lr_ratio': args.minimum_lr_ratio,
        'warmup_epochs': args.warmup_epochs,
        'weight_decay': args.weight_decay,
        'optimizer': 'AdamW',
        'precision': args.precision,
        'semantic_weight': args.semantic_weight,
        'motion_weight': args.motion_weight,
        'model_name': args.model_name,
        'bevcar_voxelnet_sha256': file_sha256(args.bevcar_source / 'nets' / 'voxelnet.py'),
        'validation_epochs': validation_epochs,
        'val_samples': args.val_samples,
        'num_workers': args.num_workers,
        'prefetch_factor': args.prefetch_factor,
    }
    # Preserve byte-for-byte run-config compatibility with the already active
    # historical full-velocity run. The new field is required only for V0.
    if args.radar_velocity_mode != 'full':
        run_config['radar_velocity_mode'] = args.radar_velocity_mode
    progress = {
        'epoch': 0,
        'update': 0,
        'samples_seen': 0,
        'loss_sum': 0.0,
        'semantic_sum': 0.0,
        'motion_sum': 0.0,
        'covered_cells': 0,
        'moving_samples': 0,
        'wall_train_seconds': 0.0,
        'peak_allocated_gib': 0.0,
        'amp_overflow_retries': 0,
        'completed_validation_epochs': [],
        'latest': {},
        'ema': {},
    }
    latest = {}
    ema = {}
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'resume checkpoint is missing: {checkpoint_path}')
        progress = load_training_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, scaler=scaler,
            sampler=sampler, expected_run_config=run_config, map_location=device,
        )
        prepare_train_log(metrics_path, resume_update=progress['update'])
    else:
        if checkpoint_path.exists() or metrics_path.exists():
            raise FileExistsError(
                f'run output already exists; use --resume or a new run directory: {args.run_dir}'
            )
        prepare_train_log(metrics_path)
    latest = dict(progress.get('latest', {}))
    ema = dict(progress.get('ema', {}))
    atomic_json_write(run_config, args.run_dir / 'config.json')

    writer = SummaryWriter(
        log_dir=str(args.run_dir / 'tensorboard'), purge_step=progress['update'],
    )
    started_at = time.monotonic()
    base_wall_seconds = progress['wall_train_seconds']
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    state = 'running'
    message = None
    validation_results = {}
    model.train()
    model.student.decoder.eval()
    initial_lr = learning_rate_at_step(
        learning_rate, progress['update'], total_steps, warmup_steps,
        args.minimum_lr_ratio,
    )
    refresh_wall_time(progress, base_wall_seconds, started_at)
    write_status(
        args, progress, total_steps, dataset_size, latest, ema, initial_lr,
        state='running', message='initializing first training batch',
    )

    for completed_epoch in progress['completed_validation_epochs']:
        stored_summary = validation_dir / f'epoch{completed_epoch:03d}_summary.json'
        if not stored_summary.is_file():
            raise FileNotFoundError(
                f'checkpoint records missing validation summary: {stored_summary}'
            )
        validation_results[str(completed_epoch)] = compact_validation(
            json.loads(stored_summary.read_text())
        )

    # A process can be interrupted during full validation after its epoch
    # checkpoint is durable. Complete any such scheduled validation before
    # consuming the next training sample.
    pending_validation_epochs = [
        epoch for epoch in validation_epochs
        if epoch <= sampler.epoch
        and epoch not in progress['completed_validation_epochs']
    ]
    for completed_epoch in pending_validation_epochs:
        validation_results[str(completed_epoch)] = run_epoch_validation(
            args=args, completed_epoch=completed_epoch, model=model,
            teacher=teacher, val_dataset=val_dataset, manifest=manifest,
            device=device, progress=progress, total_steps=total_steps,
            dataset_size=dataset_size, latest=latest, ema=ema,
            learning_rate=initial_lr, writer=writer,
            checkpoint_path=checkpoint_path, checkpoint_dir=checkpoint_dir,
            validation_dir=validation_dir, optimizer=optimizer, scaler=scaler,
            sampler=sampler, run_config=run_config,
            base_wall_seconds=base_wall_seconds, started_at=started_at,
        )

    with metrics_path.open('a') as metrics_handle:
        try:
            while sampler.epoch < args.epochs:
                stream = DataLoader(
                    subset, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.num_workers, drop_last=False,
                    pin_memory=True,
                    persistent_workers=args.num_workers > 0,
                    prefetch_factor=(
                        args.prefetch_factor if args.num_workers > 0 else None
                    ),
                )
                epoch_before = sampler.epoch
                for subset_positions, batch in stream:
                    step_started = time.perf_counter()
                    current_lr = learning_rate_at_step(
                        learning_rate, progress['update'], total_steps,
                        warmup_steps, args.minimum_lr_ratio,
                    )
                    for group in optimizer.param_groups:
                        group['lr'] = current_lr
                    phase_timings = {}
                    if args.profile_timing:
                        torch.cuda.synchronize(device)
                        phase_started = time.perf_counter()
                        with torch.inference_mode():
                            teacher_features = extract_teacher_features_batched(
                                teacher, batch[0][:, 0], device,
                            )
                        torch.cuda.synchronize(device)
                        phase_timings['dino_seconds'] = (
                            time.perf_counter() - phase_started
                        )
                        phase_started = time.perf_counter()
                        fusion_targets, confidences = \
                            build_targets_from_features_batched(
                                batch, teacher_features, device,
                            )
                        torch.cuda.synchronize(device)
                        phase_timings['target_projection_seconds'] = (
                            time.perf_counter() - phase_started
                        )
                        del teacher_features
                    else:
                        fusion_targets, confidences = build_targets_batched(
                            batch, teacher, device,
                        )
                    if args.profile_timing:
                        phase_started = time.perf_counter()
                    inputs = prepare_bevcar_inputs(batch, device)
                    inputs = (*inputs[:-1], apply_radar_velocity_mode(
                        inputs[-1], args.radar_velocity_mode,
                    ))
                    motion_fields, motion_coverages, motion_infos = \
                        build_motion_targets_batched(batch, device)
                    if args.profile_timing:
                        torch.cuda.synchronize(device)
                        phase_timings['input_motion_seconds'] = (
                            time.perf_counter() - phase_started
                        )
                        phase_started = time.perf_counter()
                    overflow_retries = 0
                    while True:
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda', dtype=amp_dtype):
                            semantic_prediction, motion_prediction = model(*inputs)
                            sem = semantic_loss_batched(
                                semantic_prediction, fusion_targets, confidences,
                            )
                            mot, covered_cells = motion_loss_batched(
                                motion_prediction, motion_fields, motion_coverages,
                            )
                            loss = args.semantic_weight * sem + args.motion_weight * mot
                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f'non-finite loss before optimizer step {progress["update"] + 1}'
                            )
                        if args.precision == 'fp16':
                            scale_before = scaler.get_scale()
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                        else:
                            loss.backward()
                            scale_before = None
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            trainable, 5.0, error_if_nonfinite=False,
                        )
                        if torch.isfinite(gradient_norm):
                            if args.precision == 'fp16':
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            break
                        if args.precision != 'fp16':
                            raise RuntimeError('non-finite BF16 gradients')
                        scaler.update()
                        overflow_retries += 1
                        progress['amp_overflow_retries'] += 1
                        if scaler.get_scale() >= scale_before or overflow_retries >= 16:
                            raise RuntimeError('FP16 gradient overflow did not recover')
                    if args.profile_timing:
                        torch.cuda.synchronize(device)
                        phase_timings['optimization_seconds'] = (
                            time.perf_counter() - phase_started
                        )

                    batch_size = int(batch[0].shape[0])
                    sampler.acknowledge(batch_size)
                    progress['update'] += 1
                    progress['samples_seen'] += batch_size
                    progress['loss_sum'] += float(loss.detach()) * batch_size
                    progress['semantic_sum'] += float(sem.detach()) * batch_size
                    progress['motion_sum'] += float(mot.detach()) * batch_size
                    progress['covered_cells'] += covered_cells
                    progress['moving_samples'] += sum(
                        int(info['covered_cells'] > 0) for info in motion_infos
                    )
                    progress['peak_allocated_gib'] = max(
                        progress['peak_allocated_gib'],
                        torch.cuda.max_memory_allocated(device) / GIB,
                    )
                    step_seconds = time.perf_counter() - step_started
                    current = {
                        'loss': float(loss.detach()),
                        'semantic_loss': float(sem.detach()),
                        'motion_loss': float(mot.detach()),
                    }
                    latest = current
                    ema = ema_update(ema, current)
                    progress['latest'] = latest
                    progress['ema'] = ema
                    refresh_wall_time(progress, base_wall_seconds, started_at)
                    elapsed = progress['wall_train_seconds']
                    epoch_progress = progress['samples_seen'] / dataset_size
                    record = {
                        'optimizer_step': progress['update'],
                        'update': progress['update'],
                        'epoch_progress': epoch_progress,
                        'sampler_epoch': sampler.epoch,
                        'sampler_position': sampler.position,
                        'subset_positions': [
                            int(value) for value in subset_positions.tolist()
                        ],
                        'dataset_indices': [
                            int(dataset_indices[int(value)])
                            for value in subset_positions.tolist()
                        ],
                        'batch_size': batch_size,
                        'loss': current['loss'],
                        'semantic_loss': current['semantic_loss'],
                        'motion_loss': current['motion_loss'],
                        'ema_loss': ema['loss'],
                        'ema_semantic_loss': ema['semantic_loss'],
                        'ema_motion_loss': ema['motion_loss'],
                        'learning_rate': current_lr,
                        'gradient_norm': float(gradient_norm),
                        'covered_cells': covered_cells,
                        'seconds': step_seconds,
                        'samples_per_second': progress['samples_seen'] / max(elapsed, 1e-9),
                        'allocated_gib': torch.cuda.memory_allocated(device) / GIB,
                        'amp_overflow_retries': overflow_retries,
                    }
                    record.update(phase_timings)
                    append_json_line(metrics_handle, record)
                    writer.add_scalar('train/loss', current['loss'], progress['update'])
                    writer.add_scalar('train/semantic_loss', current['semantic_loss'], progress['update'])
                    writer.add_scalar('train/motion_loss', current['motion_loss'], progress['update'])
                    writer.add_scalar('train/ema_loss', ema['loss'], progress['update'])
                    writer.add_scalar('train/learning_rate', current_lr, progress['update'])
                    writer.add_scalar('system/samples_per_second', record['samples_per_second'], progress['update'])
                    writer.add_scalar('system/allocated_gib', record['allocated_gib'], progress['update'])

                    periodic = progress['update'] % args.checkpoint_every_steps == 0
                    stopped = (
                        args.stop_after_steps > 0
                        and progress['update'] >= args.stop_after_steps
                    )
                    if periodic or stopped:
                        progress['epoch'] = sampler.epoch
                        refresh_wall_time(progress, base_wall_seconds, started_at)
                        save_periodic_checkpoint(
                            checkpoint_path,
                            every_updates=args.checkpoint_every_steps,
                            force=stopped,
                            model=model, optimizer=optimizer, scaler=scaler,
                            sampler=sampler, progress=progress, run_config=run_config,
                        )
                        metrics_handle.flush()
                        writer.flush()
                    if (progress['update'] % args.print_every_steps == 0
                            or stopped):
                        status = write_status(
                            args, progress, total_steps, dataset_size, latest, ema,
                            current_lr,
                            state='stopped' if stopped else 'running',
                            message='clean engineering stop' if stopped else None,
                        )
                        print(
                            f"epoch={status['epoch_progress']:.3f}/{args.epochs} "
                            f"step={progress['update']}/{total_steps} "
                            f"loss={current['loss']:.4f} sem={current['semantic_loss']:.4f} "
                            f"mot={current['motion_loss']:.4f} "
                            f"speed={status['samples_per_second']:.2f} samples/s "
                            f"eta={status['eta_seconds']:.0f}s",
                            flush=True,
                        )
                    del fusion_targets, confidences, inputs
                    del motion_fields, motion_coverages, motion_infos
                    if stopped:
                        state = 'stopped'
                        message = 'clean engineering stop'
                        break
                if state == 'stopped':
                    break
                if sampler.epoch != epoch_before + 1:
                    raise RuntimeError('sampler did not complete exactly one epoch')
                progress['epoch'] = sampler.epoch
                refresh_wall_time(progress, base_wall_seconds, started_at)
                save_training_checkpoint(
                    checkpoint_path, model=model, optimizer=optimizer, scaler=scaler,
                    sampler=sampler, progress=progress, run_config=run_config,
                )
                metrics_handle.flush()
                writer.flush()
                completed_epoch = sampler.epoch
                if completed_epoch in validation_epochs:
                    validation_results[str(completed_epoch)] = run_epoch_validation(
                        args=args, completed_epoch=completed_epoch, model=model,
                        teacher=teacher, val_dataset=val_dataset, manifest=manifest,
                        device=device, progress=progress, total_steps=total_steps,
                        dataset_size=dataset_size, latest=latest, ema=ema,
                        learning_rate=current_lr, writer=writer,
                        checkpoint_path=checkpoint_path,
                        checkpoint_dir=checkpoint_dir,
                        validation_dir=validation_dir, optimizer=optimizer,
                        scaler=scaler, sampler=sampler, run_config=run_config,
                        base_wall_seconds=base_wall_seconds, started_at=started_at,
                    )
                    state = 'running'
                if (args.stop_after_epoch > 0
                        and completed_epoch >= args.stop_after_epoch):
                    state = 'stopped'
                    message = f'clean stop after epoch {completed_epoch}'
                    refresh_wall_time(progress, base_wall_seconds, started_at)
                    write_status(
                        args, progress, total_steps, dataset_size, latest, ema,
                        current_lr, state=state, message=message,
                    )
                    break
        except Exception as error:
            state = 'failed'
            message = f'{type(error).__name__}: {error}'
            current_lr = optimizer.param_groups[0]['lr']
            refresh_wall_time(progress, base_wall_seconds, started_at)
            write_status(
                args, progress, total_steps, dataset_size, latest, ema,
                current_lr, state=state, message=message,
            )
            raise
        finally:
            refresh_wall_time(progress, base_wall_seconds, started_at)
            metrics_handle.flush()
            writer.flush()
            writer.close()

    if state != 'stopped':
        state = 'complete'
        message = f'{args.epochs}-epoch run complete'
    current_lr = optimizer.param_groups[0]['lr']
    status = progress_status(
        state=state, scale=args.scale, epochs=args.epochs,
        dataset_size=dataset_size, batch_size=args.batch_size,
        optimizer_step=progress['update'], total_steps=total_steps,
        samples_seen=progress['samples_seen'],
        elapsed_seconds=progress['wall_train_seconds'], latest=latest, ema=ema,
        learning_rate=current_lr, gpu=query_nvidia_smi(), message=message,
    )
    status['radar_velocity_mode'] = args.radar_velocity_mode
    status['nsweeps'] = args.nsweeps
    atomic_json_write(status, args.run_dir / 'status.json')
    summary = {
        'status': state,
        'run_config': run_config,
        'progress': progress,
        'mean_train_loss': progress['loss_sum'] / max(progress['samples_seen'], 1),
        'mean_semantic_loss': progress['semantic_sum'] / max(progress['samples_seen'], 1),
        'mean_motion_loss': progress['motion_sum'] / max(progress['samples_seen'], 1),
        'samples_per_second': (
            progress['samples_seen'] / max(progress['wall_train_seconds'], 1e-9)
        ),
        'validation': validation_results,
    }
    atomic_json_write(summary, summary_path)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print('FULLSIZE_STOPPED' if state == 'stopped' else 'FULLSIZE_TRAIN_OK', flush=True)


if __name__ == '__main__':
    main()
