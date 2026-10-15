#!/usr/bin/env python3
"""Official-metric downstream evaluation of a hybrid-teacher checkpoint.

The two modes deliberately start from the same self-supervised checkpoint:

* ``finetune`` updates the complete Simple-BEV + BEVCar student with the
  original Simple-BEV segmentation, center and offset objectives.
* ``linear-probe`` freezes the complete representation and updates only a
  one-channel 1x1 vehicle-segmentation probe.

DINO distillation and the motion objective are not continued in either mode.
That keeps the downstream Vehicle BEV IoU directly interpretable and avoids
silently changing the requested two-experiment protocol.
"""

import argparse
import json
import math
import shutil
import signal
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from dinov2_bev_demo import build_loaders  # noqa: E402
from fullsize_training import (  # noqa: E402
    AcknowledgedShuffleSampler,
    atomic_json_write,
    learning_rate_at_step,
    query_nvidia_smi,
)
from hybrid_teacher_full import (  # noqa: E402
    auxiliary_metrics_from_counts,
    binary_counts,
    build_hybrid_teacher_model,
    configure_end_to_end_finetuning,
    configure_linear_probe,
    file_sha256,
    label_tensors,
    module_state_sha256,
    prepare_inputs,
    reset_seeds,
    simplebev_downstream_loss,
    simplebev_per_frame_iou,
    simplebev_segmentation_loss,
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
    parser.add_argument('--mode', choices=('finetune', 'linear-probe'), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pretrain-checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--train-samples', type=int, default=0,
                        help='0 uses all 28,130 manifest training samples')
    parser.add_argument('--val-samples', type=int, default=6019)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--minimum-lr-ratio', type=float, default=0.1)
    parser.add_argument('--warmup-epochs', type=float, default=0.25)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--precision', choices=('bf16', 'fp16'), default='bf16')
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--validation-epochs', default='1,3,5,8')
    parser.add_argument('--checkpoint-every-updates', type=int, default=1000)
    parser.add_argument('--print-every-updates', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int, default=0)
    parser.add_argument('--allow-incomplete-pretrain', action='store_true',
                        help='engineering smoke tests only')
    return parser.parse_args()


def parse_epochs(value, total):
    result = tuple(sorted({int(part) for part in value.split(',') if part.strip()}))
    if any(epoch < 1 or epoch > total for epoch in result):
        raise ValueError('validation epochs must be within the downstream run')
    return result


def add_counts(total, current):
    for key in total:
        total[key] += current[key]


def append_json_line(handle, value):
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def prepare_group(iterator, accumulation):
    group = []
    for _ in range(accumulation):
        try:
            group.append(next(iterator))
        except StopIteration:
            break
    return group


def load_pretrained_model(args, device):
    payload = torch.load(
        args.pretrain_checkpoint, map_location='cpu', weights_only=False,
    )
    config = payload.get('run_config', {})
    if config.get('experiment') != 'hybrid_teacher_pretrain_v1':
        raise ValueError('checkpoint is not a hybrid-teacher pretraining checkpoint')
    if int(config.get('nsweeps', 0)) != 10:
        raise ValueError('the complete experiment requires exactly 10 radar sweeps')
    if int(config.get('seed', -1)) != args.seed:
        raise ValueError('pretraining and downstream seeds must match')
    completed_epochs = int(payload.get('sampler', {}).get('epoch', -1))
    requested_epochs = int(config.get('epochs', -1))
    if not args.allow_incomplete_pretrain and completed_epochs != requested_epochs:
        raise ValueError(
            'pretraining checkpoint is incomplete; finish it or use '
            '--allow-incomplete-pretrain only for an engineering smoke test'
        )
    model = build_hybrid_teacher_model(
        device, args.seed, args.bevcar_source, pretrained_camera=False,
    )
    model.load_state_dict(payload['model'], strict=True)
    return model, config, completed_epochs


def forward_and_loss(model, mode, inputs, labels, segmentation_loss):
    if mode == 'finetune':
        outputs = model.downstream_forward(*inputs)
        total, components = simplebev_downstream_loss(
            model, outputs, labels, segmentation_loss,
        )
        return total, components, outputs['segmentation']
    logits = model(inputs)
    raw_ce = segmentation_loss(logits, labels['segmentation'], labels['valid'])
    scaled_ce = 10.0 * raw_ce
    return scaled_ce, {'segmentation_ce_loss': scaled_ce}, logits


@torch.inference_mode()
def validate(model, mode, dataset, indices, args, device, amp_dtype,
             segmentation_loss):
    loader = DataLoader(
        IndexedSubset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=False,
    )
    model.eval()
    sums = {
        'total_loss': 0.0,
        'segmentation_ce_loss': 0.0,
        'center_loss': 0.0,
        'offset_loss': 0.0,
        'uncertainty_regularization': 0.0,
    }
    counts = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    iou_sum = 0.0
    samples = 0
    started = time.monotonic()
    for _, batch in loader:
        inputs = prepare_inputs(batch, device)
        labels = label_tensors(batch, device)
        with torch.autocast('cuda', dtype=amp_dtype):
            total, components, logits = forward_and_loss(
                model, mode, inputs, labels, segmentation_loss,
            )
        batch_size = int(logits.shape[0])
        samples += batch_size
        sums['total_loss'] += float(total) * batch_size
        for key, value in components.items():
            sums[key] += float(value) * batch_size
        iou_sum += float(simplebev_per_frame_iou(
            logits, labels['segmentation'], labels['valid'],
        ).sum())
        add_counts(counts, binary_counts(
            logits, labels['segmentation'], labels['valid'],
        ))
    official = {
        'vehicle_bev_iou': iou_sum / max(samples, 1),
        'mean_total_loss': sums['total_loss'] / max(samples, 1),
        'mean_segmentation_ce_loss': (
            sums['segmentation_ce_loss'] / max(samples, 1)
        ),
    }
    if mode == 'finetune':
        official.update({
            'mean_center_loss': sums['center_loss'] / max(samples, 1),
            'mean_offset_loss': sums['offset_loss'] / max(samples, 1),
            'mean_uncertainty_regularization': (
                sums['uncertainty_regularization'] / max(samples, 1)
            ),
        })
    return {
        'metric_definition': (
            'official Simple-BEV mean of per-frame binary Vehicle BEV IoU'
        ),
        'samples': samples,
        'official_simplebev': official,
        'auxiliary_global_confusion': {
            **auxiliary_metrics_from_counts(counts), **counts,
        },
        'seconds': time.monotonic() - started,
    }


def make_status(args, progress, total_updates, dataset_size, state, latest,
                message=None):
    elapsed = progress['wall_seconds']
    total_exposures = args.epochs * dataset_size
    remaining = max(total_exposures - progress['samples_seen'], 0)
    rate = progress['samples_seen'] / max(elapsed, 1e-9)
    return {
        'experiment': f'hybrid_teacher_{args.mode}_v1',
        'mode': args.mode,
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
        'gpu': query_nvidia_smi(),
        'updated_unix_time': time.time(),
    }


def main():
    args = parse_args()
    for name in ('epochs', 'batch_size', 'gradient_accumulation',
                 'checkpoint_every_updates', 'print_every_updates'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError('worker settings are invalid')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)

    validation_epochs = parse_epochs(args.validation_epochs, args.epochs)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-4 if args.mode == 'finetune' else 1e-3
    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    all_train_indices = scale_indices(manifest, 'full')
    train_count = args.train_samples or len(all_train_indices)
    if not 1 <= train_count <= len(all_train_indices):
        raise ValueError('train-samples must fit the full training split')
    if not 1 <= args.val_samples <= int(manifest['val']['count']):
        raise ValueError('val-samples must fit the complete validation split')
    train_indices = all_train_indices[:train_count]
    dataset_size = len(train_indices)
    effective_batch = args.batch_size * args.gradient_accumulation
    updates_per_epoch = math.ceil(dataset_size / effective_batch)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = round(updates_per_epoch * args.warmup_epochs)

    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_path = checkpoint_dir / 'latest.pt'
    validation_dir = run_dir / 'validation'
    metrics_path = run_dir / 'metrics.jsonl'
    for directory in (run_dir, checkpoint_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    reset_seeds(args.seed)
    backbone, pretrain_config, pretrain_epochs = load_pretrained_model(args, device)
    if args.mode == 'finetune':
        model = backbone
        trainable = configure_end_to_end_finetuning(model)
        frozen_backbone_sha256 = None
    else:
        model, trainable = configure_linear_probe(backbone)
        model = model.to(device)
        frozen_backbone_sha256 = module_state_sha256(model.backbone)
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        'cuda', enabled=args.precision == 'fp16', init_scale=128.0,
        growth_interval=1000,
    )
    segmentation_loss = simplebev_segmentation_loss(device)

    train_raw, val_raw = build_loaders(
        args.data_root, num_workers=0, nsweeps=10,
        rotate_radar_velocity=True, dset='trainval',
    )
    train_dataset, val_dataset = train_raw.dataset, val_raw.dataset
    del train_raw, val_raw
    subset = IndexedSubset(train_dataset, train_indices)
    sampler = AcknowledgedShuffleSampler(subset, seed=args.seed)

    run_config = {
        'experiment': f'hybrid_teacher_{args.mode}_v1',
        'mode': args.mode,
        'dataset': 'v1.0-trainval',
        'manifest_sha256': manifest['manifest_sha256'],
        'train_samples': dataset_size,
        'val_samples': args.val_samples,
        'epochs': args.epochs,
        'physical_batch_size': args.batch_size,
        'gradient_accumulation': args.gradient_accumulation,
        'effective_batch_size': effective_batch,
        'updates_per_epoch': updates_per_epoch,
        'total_optimizer_updates': total_updates,
        'seed': args.seed,
        'nsweeps': 10,
        'learning_rate': learning_rate,
        'minimum_lr_ratio': args.minimum_lr_ratio,
        'warmup_epochs': args.warmup_epochs,
        'weight_decay': args.weight_decay,
        'precision': args.precision,
        'validation_epochs': validation_epochs,
        'pretrain_checkpoint': str(args.pretrain_checkpoint.resolve()),
        'pretrain_checkpoint_sha256': file_sha256(args.pretrain_checkpoint),
        'pretrain_completed_epochs': pretrain_epochs,
        'pretrain_experiment': pretrain_config['experiment'],
        'downstream_objective': (
            'official_simplebev_segmentation_center_offset'
            if args.mode == 'finetune'
            else 'official_simplebev_segmentation_only_1x1_probe'
        ),
        'distillation_during_downstream': False,
        'motion_auxiliary_during_downstream': False,
        'trainable_parameter_count': sum(value.numel() for value in trainable),
        'frozen_backbone_initial_sha256': frozen_backbone_sha256,
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
    if (args.mode == 'linear-probe'
            and module_state_sha256(model.backbone) != frozen_backbone_sha256):
        raise RuntimeError('frozen backbone differs from the pretraining checkpoint')
    atomic_json_write(run_config, run_dir / 'config.json')

    writer = SummaryWriter(
        log_dir=str(run_dir / 'tensorboard'), purge_step=progress['update'],
    )
    validation_results = {}
    for epoch in progress['completed_validation_epochs']:
        path = validation_dir / f'epoch{epoch:03d}.json'
        if path.is_file():
            validation_results[str(epoch)] = json.loads(path.read_text())
    started = time.monotonic()
    prior_wall = float(progress['wall_seconds'])
    state = 'running'
    message = None
    latest = dict(progress.get('latest', {}))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    def validate_pending_epoch():
        epoch = sampler.epoch
        if (epoch not in validation_epochs
                or epoch in progress['completed_validation_epochs']):
            return
        result = validate(
            model, args.mode, val_dataset,
            manifest['val']['dataset_indices'][:args.val_samples],
            args, device, amp_dtype, segmentation_loss,
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
            'validation/official_vehicle_bev_iou',
            result['official_simplebev']['vehicle_bev_iou'], epoch,
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
                    drop_last=False, persistent_workers=args.num_workers > 0,
                    prefetch_factor=(args.prefetch_factor
                                     if args.num_workers > 0 else None),
                )
                epoch_before = sampler.epoch
                iterator = iter(stream)
                while True:
                    group = prepare_group(iterator, args.gradient_accumulation)
                    if not group:
                        break
                    group_samples = sum(int(batch[0].shape[0]) for _, batch in group)
                    current_lr = learning_rate_at_step(
                        learning_rate, progress['update'], total_updates,
                        warmup_updates, args.minimum_lr_ratio,
                    )
                    for parameter_group in optimizer.param_groups:
                        parameter_group['lr'] = current_lr
                    optimizer.zero_grad(set_to_none=True)
                    total_value = 0.0
                    component_sums = {}
                    iou_sum = 0.0
                    dataset_batch_indices = []
                    step_started = time.perf_counter()
                    model.train()
                    if args.mode == 'linear-probe':
                        model.backbone.eval()
                    for subset_positions, batch in group:
                        batch_size = int(batch[0].shape[0])
                        fraction = batch_size / group_samples
                        inputs = prepare_inputs(batch, device)
                        labels = label_tensors(batch, device)
                        with torch.autocast('cuda', dtype=amp_dtype):
                            total, components, logits = forward_and_loss(
                                model, args.mode, inputs, labels,
                                segmentation_loss,
                            )
                        if not torch.isfinite(total):
                            raise RuntimeError('non-finite downstream loss')
                        scaled = total * fraction
                        if args.precision == 'fp16':
                            scaler.scale(scaled).backward()
                        else:
                            scaled.backward()
                        total_value += float(total.detach()) * batch_size
                        for key, value in components.items():
                            component_sums[key] = (
                                component_sums.get(key, 0.0)
                                + float(value.detach()) * batch_size
                            )
                        iou_sum += float(simplebev_per_frame_iou(
                            logits.detach(), labels['segmentation'],
                            labels['valid'],
                        ).sum())
                        dataset_batch_indices.extend(
                            int(train_indices[int(value)])
                            for value in subset_positions.tolist()
                        )
                    if args.precision == 'fp16':
                        scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, 5.0, error_if_nonfinite=False,
                    )
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError('non-finite downstream gradient')
                    if args.precision == 'fp16':
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    for _, batch in group:
                        sampler.acknowledge(int(batch[0].shape[0]))
                    progress['update'] += 1
                    progress['samples_seen'] += group_samples
                    total_value /= group_samples
                    components = {
                        key: value / group_samples
                        for key, value in component_sums.items()
                    }
                    train_iou = iou_sum / group_samples
                    progress['loss_sum'] += total_value * group_samples
                    progress['peak_allocated_gib'] = max(
                        progress['peak_allocated_gib'],
                        torch.cuda.max_memory_allocated(device) / GIB,
                    )
                    progress['wall_seconds'] = prior_wall + time.monotonic() - started
                    latest = {
                        'loss': total_value,
                        'official_vehicle_bev_iou': train_iou,
                        'gradient_norm': float(gradient_norm),
                        **components,
                    }
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
                    writer.add_scalar('train/loss', total_value, progress['update'])
                    writer.add_scalar(
                        'train/official_vehicle_bev_iou', train_iou,
                        progress['update'],
                    )
                    writer.add_scalar(
                        'train/gradient_norm', float(gradient_norm),
                        progress['update'],
                    )
                    writer.add_scalar(
                        'system/allocated_gib', record['allocated_gib'],
                        progress['update'],
                    )

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
                            f"mode={args.mode} "
                            f"epoch={status['epoch_progress']:.3f}/{args.epochs} "
                            f"update={progress['update']}/{total_updates} "
                            f"loss={total_value:.4f} iou={train_iou:.4f}",
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
        message = f'{args.mode} downstream experiment complete'
    frozen_integrity = None
    if args.mode == 'linear-probe':
        final_hash = module_state_sha256(model.backbone)
        frozen_integrity = {
            'initial_sha256': frozen_backbone_sha256,
            'final_sha256': final_hash,
            'unchanged': final_hash == frozen_backbone_sha256,
        }
        if not frozen_integrity['unchanged']:
            raise RuntimeError('linear-probe training changed the frozen backbone')
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
        'validation': validation_results,
        'primary_metric': (
            'official Simple-BEV mean per-frame Vehicle BEV IoU on nuScenes val'
        ),
        'interpretation': (
            'end-to-end task performance with all student parameters trainable'
            if args.mode == 'finetune'
            else 'linear accessibility of vehicle semantics in a frozen representation'
        ),
        'frozen_backbone_integrity': frozen_integrity,
        'limitations': [
            'Precision, recall, F1, and global IoU are auxiliary diagnostics only.',
            'A direct claim against published Simple-BEV still requires a matched '
            'input resolution, schedule, data split, and evaluation implementation.',
        ],
    }
    atomic_json_write(summary, run_dir / 'summary.json')
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print('HYBRID_DOWNSTREAM_STOPPED' if state == 'stopped'
          else 'HYBRID_DOWNSTREAM_OK', flush=True)


if __name__ == '__main__':
    main()
