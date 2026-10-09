#!/usr/bin/env python3
"""Frozen one-layer vehicle-segmentation probe for a full-size checkpoint."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from compare_motion_target_bevcar_tiny import (  # noqa: E402
    build_model,
    official_voxelnet,
    prepare_bevcar_inputs,
)
from dinov2_bev_demo import build_loaders  # noqa: E402
from fullsize_training import atomic_json_write, query_nvidia_smi  # noqa: E402
from radar_evaluation import apply_radar_velocity_mode  # noqa: E402
from radar_scaling_data import scale_indices, validate_scaling_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--radar-velocity-mode', choices=('auto', 'full', 'zero'),
                        default='auto')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--train-samples', type=int, default=0,
                        help='0 uses all 28,130 manifest training samples')
    parser.add_argument('--val-samples', type=int, default=6019)
    parser.add_argument('--print-every-steps', type=int, default=100)
    return parser.parse_args()


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def balanced_binary_loss(logits, target, valid):
    """Average foreground and background BCE without empty-class NaNs."""
    raw = F.binary_cross_entropy_with_logits(logits.float(), target.float(),
                                              reduction='none')
    valid_mask = valid.gt(0.5)
    terms = []
    for class_mask in (target.gt(0.5), target.le(0.5)):
        mask = valid_mask & class_mask
        if mask.any():
            terms.append(raw[mask].mean())
    if not terms:
        raise ValueError('probe batch has no valid BEV cells')
    return torch.stack(terms).mean()


def binary_segmentation_counts(logits, target, valid):
    prediction = logits.sigmoid().ge(0.5)
    truth = target.gt(0.5)
    mask = valid.gt(0.5)
    return {
        'tp': int((prediction & truth & mask).sum()),
        'fp': int((prediction & ~truth & mask).sum()),
        'fn': int((~prediction & truth & mask).sum()),
        'tn': int((~prediction & ~truth & mask).sum()),
    }


def metrics_from_counts(counts):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    iou_denominator = tp + fp + fn
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    return {
        'vehicle_iou': tp / iou_denominator if iou_denominator else None,
        'precision': tp / precision_denominator if precision_denominator else None,
        'recall': tp / recall_denominator if recall_denominator else None,
        'f1': 2 * tp / f1_denominator if f1_denominator else None,
        **counts,
    }


def add_counts(total, current):
    for key in total:
        total[key] += current[key]


def checkpoint_velocity_mode(payload):
    return payload['run_config'].get('radar_velocity_mode', 'full')


def resolve_velocity_mode(requested, payload):
    stored = checkpoint_velocity_mode(payload)
    if requested == 'auto':
        return stored
    if requested != stored:
        raise ValueError(
            f'probe velocity mode {requested} does not match checkpoint mode {stored}'
        )
    return requested


def shared_bev(model, batch, device, velocity_mode, amp_dtype):
    inputs = prepare_bevcar_inputs(batch, device)
    images, pixels, cameras, vox_util, radar = inputs
    radar = apply_radar_velocity_mode(radar, velocity_mode)
    with torch.inference_mode(), torch.autocast('cuda', dtype=amp_dtype):
        features, _, _ = model.student(
            rgb_camXs=images,
            pix_T_cams=pixels,
            cam0_T_camXs=cameras,
            vox_util=vox_util,
            bevcar_voxels=radar,
            return_shared_bev=True,
        )
    return features.detach()


def label_tensors(batch, device):
    target = batch[12][:, 0].to(device, non_blocking=True)
    valid = batch[13][:, 0].to(device, non_blocking=True)
    return target, valid


def evaluate(model, probe, loader, device, velocity_mode, amp_dtype,
             output_dir, status_base):
    model.eval()
    probe.eval()
    counts = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    loss_sum = 0.0
    samples = 0
    started = time.monotonic()
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            features = shared_bev(model, batch, device, velocity_mode, amp_dtype)
            target, valid = label_tensors(batch, device)
            logits = probe(features.float())
            loss = balanced_binary_loss(logits, target, valid)
            batch_size = int(target.shape[0])
            loss_sum += float(loss) * batch_size
            samples += batch_size
            add_counts(counts, binary_segmentation_counts(logits, target, valid))
            if step % 100 == 0 or step == len(loader):
                atomic_json_write({
                    **status_base,
                    'state': 'validating',
                    'validation_samples': samples,
                    'validation_total': len(loader.dataset),
                    'elapsed_seconds': time.monotonic() - started,
                    'gpu': query_nvidia_smi(),
                }, output_dir / 'status.json')
    return {
        'mean_balanced_bce': loss_sum / max(samples, 1),
        'samples': samples,
        **metrics_from_counts(counts),
    }


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('epochs/batch-size must be positive and workers nonnegative')
    if args.print_every_steps < 1:
        raise ValueError('print interval must be positive')
    args.output_dir = args.output_dir.resolve()
    summary_path = args.output_dir / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get('status') != 'complete':
            raise ValueError('existing probe summary is incomplete')
        print('LINEAR_PROBE_ALREADY_COMPLETE', flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reset_seeds(args.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    amp_dtype = torch.bfloat16

    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    all_train_indices = scale_indices(manifest, 'full')
    train_count = args.train_samples or len(all_train_indices)
    if not 1 <= train_count <= len(all_train_indices):
        raise ValueError('train-samples must fit the full manifest split')
    if not 1 <= args.val_samples <= manifest['val']['count']:
        raise ValueError('val-samples must fit the validation split')

    # Keep optimizer/scaler tensors from the training checkpoint on host RAM;
    # only the model state is copied into the newly built CUDA backbone.
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    velocity_mode = resolve_velocity_mode(args.radar_velocity_mode, payload)
    run_config = payload['run_config']
    nsweeps = int(run_config['nsweeps'])
    if run_config['seed'] != args.seed or nsweeps not in (1, 5, 10):
        raise ValueError('checkpoint seed/sweeps do not match the supported probe')
    official_class = official_voxelnet(args.bevcar_source)
    model = build_model(device, args.seed, official_class, zero_camera=False)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    probe = nn.Conv2d(model.student.latent_dim, 1, kernel_size=1).to(device)
    trainable = list(probe.parameters())
    if set(trainable) & set(model.parameters()):
        raise RuntimeError('probe unexpectedly shares parameters with the backbone')
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    train_loader_raw, val_loader_raw = build_loaders(
        args.data_root, num_workers=0, nsweeps=nsweeps,
        rotate_radar_velocity=True, dset='trainval',
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(train_loader_raw.dataset, all_train_indices[:train_count]),
        batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=False,
    )
    val_indices = manifest['val']['dataset_indices'][:args.val_samples]
    val_loader = DataLoader(
        Subset(val_loader_raw.dataset, val_indices),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
        persistent_workers=False,
    )
    del train_loader_raw, val_loader_raw

    status_base = {
        'experiment': 'frozen_one_layer_vehicle_probe_v1',
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'backbone_epoch': int(payload['sampler']['epoch']),
        'radar_velocity_mode': velocity_mode,
        'nsweeps': nsweeps,
        'probe_epochs': args.epochs,
        'train_samples': train_count,
        'val_samples': args.val_samples,
    }
    started = time.monotonic()
    updates = 0
    epoch_records = []
    try:
        for epoch in range(1, args.epochs + 1):
            probe.train()
            loss_sum = 0.0
            samples = 0
            for batch in train_loader:
                features = shared_bev(model, batch, device, velocity_mode, amp_dtype)
                target, valid = label_tensors(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = probe(features.float())
                loss = balanced_binary_loss(logits, target, valid)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'non-finite probe loss at update {updates + 1}')
                loss.backward()
                optimizer.step()
                batch_size = int(target.shape[0])
                loss_sum += float(loss.detach()) * batch_size
                samples += batch_size
                updates += 1
                if updates % args.print_every_steps == 0:
                    status = {
                        **status_base,
                        'state': 'training',
                        'epoch': epoch,
                        'updates': updates,
                        'samples_seen_this_epoch': samples,
                        'mean_train_loss': loss_sum / max(samples, 1),
                        'elapsed_seconds': time.monotonic() - started,
                        'gpu': query_nvidia_smi(),
                    }
                    atomic_json_write(status, args.output_dir / 'status.json')
                    print(
                        f'probe_epoch={epoch}/{args.epochs} update={updates} '
                        f'samples={samples}/{train_count} '
                        f'loss={status["mean_train_loss"]:.6f}', flush=True,
                    )
            validation = evaluate(
                model, probe, val_loader, device, velocity_mode, amp_dtype,
                args.output_dir, status_base,
            )
            epoch_record = {
                'epoch': epoch,
                'mean_train_loss': loss_sum / max(samples, 1),
                'validation': validation,
            }
            epoch_records.append(epoch_record)
            atomic_torch_save({
                'probe': probe.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': status_base,
            }, args.output_dir / 'probe_latest.pt')
            print(json.dumps(epoch_record, sort_keys=True), flush=True)
    except Exception as error:
        atomic_json_write({
            **status_base,
            'state': 'failed',
            'updates': updates,
            'message': f'{type(error).__name__}: {error}',
            'elapsed_seconds': time.monotonic() - started,
        }, args.output_dir / 'status.json')
        raise

    summary = {
        'status': 'complete',
        **status_base,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'updates': updates,
        'elapsed_seconds': time.monotonic() - started,
        'epochs': epoch_records,
        'final_validation': epoch_records[-1]['validation'],
        'limitations': [
            'The backbone is frozen; this measures linear separability, not end-to-end task capacity.',
            'The source backbone keeps the historical frozen random camera encoder.',
            'One training seed does not measure training-seed uncertainty.',
        ],
    }
    atomic_json_write(summary, summary_path)
    atomic_json_write({
        **status_base,
        'state': 'complete',
        'updates': updates,
        'elapsed_seconds': summary['elapsed_seconds'],
        'final_validation': summary['final_validation'],
        'gpu': query_nvidia_smi(),
    }, args.output_dir / 'status.json')
    print('FULLSIZE_LINEAR_PROBE_OK', flush=True)


if __name__ == '__main__':
    main()
