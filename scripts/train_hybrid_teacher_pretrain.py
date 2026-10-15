#!/usr/bin/env python3
"""Resumable full-data hybrid-teacher pretraining with BEVCar and multi-sweep radar."""

import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from dinov2_bev_demo import build_loaders, load_teacher  # noqa: E402
from fullsize_training import (  # noqa: E402
    AcknowledgedShuffleSampler,
    atomic_json_write,
    build_motion_targets_batched,
    build_targets_from_features_batched,
    extract_teacher_features_batched,
    learning_rate_at_step,
    motion_loss_batched,
    query_nvidia_smi,
    semantic_loss_batched,
)
from hybrid_teacher_full import (  # noqa: E402
    build_hybrid_teacher_model,
    configure_pretraining,
    file_sha256,
    image_distillation_loss,
    prepare_inputs,
    reset_seeds,
)
from radar_scaling_data import IndexedSubset, scale_indices, validate_scaling_manifest  # noqa: E402
from training_checkpoint import (  # noqa: E402
    load_training_checkpoint,
    save_training_checkpoint,
)


GIB = 1024 ** 3
STOP_REQUESTED = False


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f'RECEIVED_SIGNAL_{signum}; stopping after current optimizer step', flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--scale', choices=('4096', 'full'), default='full')
    parser.add_argument('--train-samples', type=int, default=0,
                        help='engineering override; 0 uses the selected scale')
    parser.add_argument('--val-samples', type=int, default=6019)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--minimum-lr-ratio', type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=float, default=0.25)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--nsweeps', type=int, choices=(1, 5, 10), default=10)
    parser.add_argument('--image-weight', type=float, default=0.2)
    parser.add_argument('--bev-weight', type=float, default=0.8)
    parser.add_argument('--motion-weight', type=float, default=0.5)
    parser.add_argument('--model-name', default='dinov2_vits14')
    parser.add_argument('--validation-epochs', default='1,3,5,8')
    parser.add_argument('--checkpoint-every-updates', type=int, default=1000)
    parser.add_argument('--print-every-updates', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int, default=0)
    return parser.parse_args()


def parse_epochs(value, total):
    result = tuple(sorted({int(part) for part in value.split(',') if part.strip()}))
    if any(epoch < 1 or epoch > total for epoch in result):
        raise ValueError('validation epochs must be within the training run')
    return result


def append_json_line(handle, value):
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def make_status(args, progress, total_updates, dataset_size, state, latest,
                message=None):
    elapsed = progress['wall_seconds']
    total_exposures = args.epochs * dataset_size
    remaining = max(total_exposures - progress['samples_seen'], 0)
    rate = progress['samples_seen'] / max(elapsed, 1e-9)
    return {
        'experiment': 'hybrid_teacher_pretrain_v1',
        'state': state,
        'message': message,
        'epochs': args.epochs,
        'epoch_progress': progress['samples_seen'] / dataset_size,
        'optimizer_update': progress['update'],
        'total_optimizer_updates': total_updates,
        'samples_seen': progress['samples_seen'],
        'total_sample_exposures': total_exposures,
        'percent_complete': 100.0 * progress['samples_seen'] / total_exposures,
        'samples_per_second': rate,
        'eta_seconds': remaining / max(rate, 1e-9),
        'latest': latest,
        'peak_allocated_gib': progress['peak_allocated_gib'],
        'nsweeps': args.nsweeps,
        'gpu': query_nvidia_smi(),
        'updated_unix_time': time.time(),
    }


def prepare_group(iterator, accumulation):
    group = []
    for _ in range(accumulation):
        try:
            group.append(next(iterator))
        except StopIteration:
            break
    return group


@torch.inference_mode()
def validate(model, teacher, dataset, indices, args, device, amp_dtype):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=False,
    )
    model.eval()
    sums = {'image_loss': 0.0, 'bev_loss': 0.0, 'motion_loss': 0.0}
    samples = 0
    covered_cells = 0
    started = time.monotonic()
    for batch in loader:
        teacher_features = extract_teacher_features_batched(
            teacher, batch[0][:, 0], device,
        )
        bev_targets, confidences = build_targets_from_features_batched(
            batch, teacher_features, device,
        )
        inputs = prepare_inputs(batch, device)
        motion_targets, motion_coverage, _ = build_motion_targets_batched(
            batch, device,
        )
        with torch.autocast('cuda', dtype=amp_dtype):
            outputs = model.pretrain_forward(*inputs)
            image_loss = image_distillation_loss(
                outputs['image_prediction'], teacher_features,
            )
            bev_loss = semantic_loss_batched(
                outputs['bev_prediction'], bev_targets, confidences,
            )
            motion_loss, cells = motion_loss_batched(
                outputs['motion_prediction'], motion_targets, motion_coverage,
            )
        batch_size = int(batch[0].shape[0])
        samples += batch_size
        covered_cells += cells
        sums['image_loss'] += float(image_loss) * batch_size
        sums['bev_loss'] += float(bev_loss) * batch_size
        sums['motion_loss'] += float(motion_loss) * batch_size
    model.train()
    model.student.decoder.eval()
    return {
        'samples': samples,
        'image_distillation_loss': sums['image_loss'] / max(samples, 1),
        'radar_masked_bev_distillation_loss': sums['bev_loss'] / max(samples, 1),
        'motion_auxiliary_loss': sums['motion_loss'] / max(samples, 1),
        'motion_covered_cells': covered_cells,
        'seconds': time.monotonic() - started,
    }


def main():
    args = parse_args()
    for name in ('epochs', 'batch_size', 'gradient_accumulation',
                 'checkpoint_every_updates', 'print_every_updates'):
        if getattr(args, name.replace('-', '_'), None) is not None and getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError('worker settings are invalid')
    if min(args.image_weight, args.bev_weight, args.motion_weight) < 0:
        raise ValueError('loss weights must be nonnegative')
    if args.image_weight + args.bev_weight <= 0:
        raise ValueError('hybrid teacher requires a positive distillation weight')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)

    validation_epochs = parse_epochs(args.validation_epochs, args.epochs)
    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    if int(manifest['seed']) != args.seed:
        raise ValueError('manifest seed does not match training seed')
    dataset_indices = scale_indices(manifest, args.scale)
    if args.train_samples:
        if not 1 <= args.train_samples <= len(dataset_indices):
            raise ValueError('train-samples does not fit the selected scale')
        dataset_indices = dataset_indices[:args.train_samples]
    dataset_size = len(dataset_indices)
    if not 1 <= args.val_samples <= int(manifest['val']['count']):
        raise ValueError('val-samples must fit the complete validation split')
    effective_batch = args.batch_size * args.gradient_accumulation
    updates_per_epoch = math.ceil(dataset_size / effective_batch)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = round(updates_per_epoch * args.warmup_epochs)

    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_path = checkpoint_dir / 'latest.pt'
    metrics_path = run_dir / 'metrics.jsonl'
    validation_dir = run_dir / 'validation'
    summary_path = run_dir / 'summary.json'
    for directory in (run_dir, checkpoint_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    reset_seeds(args.seed)
    train_raw, val_raw = build_loaders(
        args.data_root, num_workers=0, nsweeps=args.nsweeps,
        rotate_radar_velocity=True, dset='trainval',
    )
    train_dataset, val_dataset = train_raw.dataset, val_raw.dataset
    del train_raw, val_raw
    subset = IndexedSubset(train_dataset, dataset_indices)
    sampler = AcknowledgedShuffleSampler(subset, seed=args.seed)

    teacher = load_teacher(args.model_name, device)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError('DINOv2 teacher must remain frozen')
    model = build_hybrid_teacher_model(
        device, args.seed, args.bevcar_source, pretrained_camera=False,
    )
    trainable = configure_pretraining(model)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        'cuda', enabled=args.precision == 'fp16', init_scale=128.0,
        growth_interval=1000,
    )
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    run_config = {
        'experiment': 'hybrid_teacher_pretrain_v1',
        'dataset': 'v1.0-trainval',
        'manifest_sha256': manifest['manifest_sha256'],
        'scale': args.scale,
        'train_samples': dataset_size,
        'val_samples': args.val_samples,
        'epochs': args.epochs,
        'physical_batch_size': args.batch_size,
        'gradient_accumulation': args.gradient_accumulation,
        'effective_batch_size': effective_batch,
        'updates_per_epoch': updates_per_epoch,
        'total_optimizer_updates': total_updates,
        'seed': args.seed,
        'nsweeps': args.nsweeps,
        'learning_rate': args.learning_rate,
        'minimum_lr_ratio': args.minimum_lr_ratio,
        'warmup_epochs': args.warmup_epochs,
        'weight_decay': args.weight_decay,
        'precision': args.precision,
        'image_weight': args.image_weight,
        'bev_weight': args.bev_weight,
        'motion_weight': args.motion_weight,
        'model_name': args.model_name,
        'camera_encoder': 'resnet101_random_initialization_trainable',
        'hybrid_teacher': (
            'image_dino_plus_radar_masked_fused_bev_dino'
        ),
        'bevcar_voxelnet_sha256': file_sha256(
            args.bevcar_source / 'nets' / 'voxelnet.py'
        ),
        'validation_epochs': validation_epochs,
        'num_workers': args.num_workers,
        'prefetch_factor': args.prefetch_factor,
    }
    progress = {
        'epoch': 0,
        'update': 0,
        'samples_seen': 0,
        'wall_seconds': 0.0,
        'peak_allocated_gib': 0.0,
        'loss_sum': 0.0,
        'image_loss_sum': 0.0,
        'bev_loss_sum': 0.0,
        'motion_loss_sum': 0.0,
        'latest': {},
        'completed_validation_epochs': [],
    }
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'missing resume checkpoint: {checkpoint_path}')
        progress = load_training_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, scaler=scaler,
            sampler=sampler, expected_run_config=run_config,
            map_location=device,
        )
    elif checkpoint_path.exists() or metrics_path.exists():
        raise FileExistsError('run output exists; use --resume or a new run-dir')
    atomic_json_write(run_config, run_dir / 'config.json')

    writer = SummaryWriter(
        log_dir=str(run_dir / 'tensorboard'), purge_step=progress['update'],
    )
    started = time.monotonic()
    prior_wall = float(progress['wall_seconds'])
    state = 'running'
    message = None
    latest = dict(progress.get('latest', {}))
    validation_results = {}
    for epoch in progress['completed_validation_epochs']:
        path = validation_dir / f'epoch{epoch:03d}.json'
        if path.is_file():
            validation_results[str(epoch)] = json.loads(path.read_text())

    model.train()
    model.student.decoder.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    def validate_pending_epoch():
        epoch = sampler.epoch
        if (epoch not in validation_epochs
                or epoch in progress['completed_validation_epochs']):
            return
        val_indices = manifest['val']['dataset_indices'][:args.val_samples]
        result = validate(
            model, teacher, val_dataset, val_indices, args, device, amp_dtype,
        )
        validation_results[str(epoch)] = result
        atomic_json_write(result, validation_dir / f'epoch{epoch:03d}.json')
        progress['completed_validation_epochs'].append(epoch)
        progress['epoch'] = sampler.epoch
        save_training_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, scaler=scaler,
            sampler=sampler, progress=progress, run_config=run_config,
        )
        writer.add_scalar(
            'validation/image_distillation_loss',
            result['image_distillation_loss'], epoch,
        )
        writer.add_scalar(
            'validation/radar_masked_bev_distillation_loss',
            result['radar_masked_bev_distillation_loss'], epoch,
        )
        writer.add_scalar(
            'validation/motion_auxiliary_loss',
            result['motion_auxiliary_loss'], epoch,
        )
        writer.flush()

    mode = 'a' if args.resume else 'w'
    with metrics_path.open(mode) as metrics_handle:
        try:
            validate_pending_epoch()
            while sampler.epoch < args.epochs:
                stream = DataLoader(
                    subset, batch_size=args.batch_size, sampler=sampler,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=False,
                    persistent_workers=args.num_workers > 0,
                    prefetch_factor=(args.prefetch_factor
                                     if args.num_workers > 0 else None),
                )
                epoch_before = sampler.epoch
                iterator = iter(stream)
                while True:
                    group = prepare_group(iterator, args.gradient_accumulation)
                    if not group:
                        break
                    group_samples = sum(int(batch[0].shape[0])
                                        for _, batch in group)
                    current_lr = learning_rate_at_step(
                        args.learning_rate, progress['update'], total_updates,
                        warmup_updates, args.minimum_lr_ratio,
                    )
                    for parameter_group in optimizer.param_groups:
                        parameter_group['lr'] = current_lr
                    optimizer.zero_grad(set_to_none=True)
                    totals = {'loss': 0.0, 'image_loss': 0.0,
                              'bev_loss': 0.0, 'motion_loss': 0.0}
                    covered_cells = 0
                    dataset_batch_indices = []
                    step_started = time.perf_counter()
                    for subset_positions, batch in group:
                        batch_size = int(batch[0].shape[0])
                        fraction = batch_size / group_samples
                        with torch.inference_mode():
                            teacher_features = extract_teacher_features_batched(
                                teacher, batch[0][:, 0], device,
                            )
                            bev_targets, confidences = \
                                build_targets_from_features_batched(
                                    batch, teacher_features, device,
                                )
                        inputs = prepare_inputs(batch, device)
                        motion_targets, motion_coverage, _ = \
                            build_motion_targets_batched(batch, device)
                        with torch.autocast('cuda', dtype=amp_dtype):
                            outputs = model.pretrain_forward(*inputs)
                            image_loss = image_distillation_loss(
                                outputs['image_prediction'], teacher_features,
                            )
                            bev_loss = semantic_loss_batched(
                                outputs['bev_prediction'], bev_targets,
                                confidences,
                            )
                            motion_loss, cells = motion_loss_batched(
                                outputs['motion_prediction'], motion_targets,
                                motion_coverage,
                            )
                            loss = (
                                args.image_weight * image_loss
                                + args.bev_weight * bev_loss
                                + args.motion_weight * motion_loss
                            )
                        if not torch.isfinite(loss):
                            raise RuntimeError('non-finite hybrid-teacher loss')
                        scaled_loss = loss * fraction
                        if args.precision == 'fp16':
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        totals['loss'] += float(loss.detach()) * batch_size
                        totals['image_loss'] += float(image_loss.detach()) * batch_size
                        totals['bev_loss'] += float(bev_loss.detach()) * batch_size
                        totals['motion_loss'] += float(motion_loss.detach()) * batch_size
                        covered_cells += cells
                        dataset_batch_indices.extend(
                            int(dataset_indices[int(value)])
                            for value in subset_positions.tolist()
                        )
                    if args.precision == 'fp16':
                        scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, 5.0, error_if_nonfinite=False,
                    )
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError('non-finite hybrid-teacher gradient')
                    if args.precision == 'fp16':
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    for _, batch in group:
                        sampler.acknowledge(int(batch[0].shape[0]))
                    progress['update'] += 1
                    progress['samples_seen'] += group_samples
                    for key in totals:
                        totals[key] /= group_samples
                    progress['loss_sum'] += totals['loss'] * group_samples
                    progress['image_loss_sum'] += totals['image_loss'] * group_samples
                    progress['bev_loss_sum'] += totals['bev_loss'] * group_samples
                    progress['motion_loss_sum'] += totals['motion_loss'] * group_samples
                    progress['peak_allocated_gib'] = max(
                        progress['peak_allocated_gib'],
                        torch.cuda.max_memory_allocated(device) / GIB,
                    )
                    progress['wall_seconds'] = prior_wall + time.monotonic() - started
                    latest = {**totals, 'gradient_norm': float(gradient_norm),
                              'covered_cells': covered_cells}
                    progress['latest'] = latest
                    record = {
                        'optimizer_update': progress['update'],
                        'epoch_progress': progress['samples_seen'] / dataset_size,
                        'sampler_epoch': sampler.epoch,
                        'sampler_position': sampler.position,
                        'dataset_indices': dataset_batch_indices,
                        'samples': group_samples,
                        **latest,
                        'learning_rate': current_lr,
                        'seconds': time.perf_counter() - step_started,
                        'allocated_gib': torch.cuda.memory_allocated(device) / GIB,
                    }
                    append_json_line(metrics_handle, record)
                    for key in ('loss', 'image_loss', 'bev_loss', 'motion_loss'):
                        writer.add_scalar(f'train/{key}', totals[key], progress['update'])
                    writer.add_scalar('train/gradient_norm', float(gradient_norm),
                                      progress['update'])
                    writer.add_scalar('system/allocated_gib', record['allocated_gib'],
                                      progress['update'])

                    requested_stop = (
                        STOP_REQUESTED
                        or (args.stop_after_updates > 0
                            and progress['update'] >= args.stop_after_updates)
                    )
                    checkpoint_due = (
                        progress['update'] % args.checkpoint_every_updates == 0
                        or requested_stop
                    )
                    if checkpoint_due:
                        progress['epoch'] = sampler.epoch
                        save_training_checkpoint(
                            checkpoint_path, model=model, optimizer=optimizer,
                            scaler=scaler, sampler=sampler, progress=progress,
                            run_config=run_config,
                        )
                        metrics_handle.flush()
                        writer.flush()
                    if (progress['update'] % args.print_every_updates == 0
                            or requested_stop):
                        status = make_status(
                            args, progress, total_updates, dataset_size,
                            'stopped' if requested_stop else 'running', latest,
                            'checkpointed stop request' if requested_stop else None,
                        )
                        atomic_json_write(status, run_dir / 'status.json')
                        print(
                            f"epoch={status['epoch_progress']:.3f}/{args.epochs} "
                            f"update={progress['update']}/{total_updates} "
                            f"loss={totals['loss']:.4f} "
                            f"image={totals['image_loss']:.4f} "
                            f"bev={totals['bev_loss']:.4f} "
                            f"motion={totals['motion_loss']:.4f}",
                            flush=True,
                        )
                    if requested_stop:
                        state = 'stopped'
                        message = 'checkpointed stop request'
                        break
                if state == 'stopped':
                    break
                if sampler.epoch != epoch_before + 1:
                    raise RuntimeError('sampler did not complete exactly one epoch')
                progress['epoch'] = sampler.epoch
                progress['wall_seconds'] = prior_wall + time.monotonic() - started
                save_training_checkpoint(
                    checkpoint_path, model=model, optimizer=optimizer,
                    scaler=scaler, sampler=sampler, progress=progress,
                    run_config=run_config,
                )
                shutil.copy2(
                    checkpoint_path,
                    checkpoint_dir / f'epoch{sampler.epoch:03d}.pt',
                )
                validate_pending_epoch()
        except Exception as error:
            state = 'failed'
            message = f'{type(error).__name__}: {error}'
            progress['wall_seconds'] = prior_wall + time.monotonic() - started
            atomic_json_write(
                make_status(args, progress, total_updates, dataset_size,
                            state, latest, message),
                run_dir / 'status.json',
            )
            raise
        finally:
            metrics_handle.flush()
            writer.flush()
            writer.close()

    if state == 'running':
        state = 'complete'
        message = 'hybrid-teacher pretraining complete'
    progress['wall_seconds'] = prior_wall + time.monotonic() - started
    atomic_json_write(
        make_status(args, progress, total_updates, dataset_size, state,
                    latest, message),
        run_dir / 'status.json',
    )
    summary = {
        'status': state,
        'run_config': run_config,
        'progress': progress,
        'mean_train_loss': progress['loss_sum'] / max(progress['samples_seen'], 1),
        'mean_image_distillation_loss': (
            progress['image_loss_sum'] / max(progress['samples_seen'], 1)
        ),
        'mean_radar_masked_bev_distillation_loss': (
            progress['bev_loss_sum'] / max(progress['samples_seen'], 1)
        ),
        'mean_motion_auxiliary_loss': (
            progress['motion_loss_sum'] / max(progress['samples_seen'], 1)
        ),
        'validation': validation_results,
        'limitations': [
            'Pretraining losses monitor optimization and are not downstream metrics.',
            'Official Simple-BEV Vehicle BEV IoU is measured only in downstream stages.',
            'Ten radar sweeps aggregate history but do not add temporal camera attention.',
        ],
    }
    atomic_json_write(summary, summary_path)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print('HYBRID_PRETRAIN_STOPPED' if state == 'stopped'
          else 'HYBRID_PRETRAIN_OK', flush=True)


if __name__ == '__main__':
    main()
