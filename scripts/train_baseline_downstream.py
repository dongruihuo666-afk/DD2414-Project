#!/usr/bin/env python3
"""Resumable full-data supervised finetuning of legacy baseline checkpoints."""

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from baseline_followup import (code_commit, digest_file, evaluate_predictions,
                               forward_loss, load_datasets, load_legacy, state_digest)
from fullsize_training import (AcknowledgedShuffleSampler, atomic_json_write,
                              learning_rate_at_step, query_nvidia_smi)
from radar_scaling_data import scale_indices, validate_scaling_manifest
from train_fullsize_linear_probe import reset_seeds
from train_nuscenes import SimpleLoss
from training_checkpoint import save_training_checkpoint, load_training_checkpoint

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['finetune', 'frozen-probe', 'unfrozen-probe'], default='finetune')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--precision', choices=['bf16', 'fp16'], default='bf16')
    parser.add_argument('--camera-lr', type=float, default=1e-5)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--probe-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--train-samples', type=int, default=0)
    parser.add_argument('--val-samples', type=int, default=6019)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--print-every', type=int, default=25)
    parser.add_argument('--stop-after-updates', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def parameter_groups(model, args):
    camera_ids = {id(p) for p in model.backbone.student.encoder.parameters()}
    camera, rest, probe = [], [], []
    probe_ids = {id(p) for p in model.probe.parameters()} if model.probe else set()
    for parameter in model.parameters():
        if parameter.requires_grad:
            (camera if id(parameter) in camera_ids else
             probe if id(parameter) in probe_ids else rest).append(parameter)
    groups = []
    for name, parameters, lr in [('camera', camera, args.camera_lr),
                                  ('student', rest, args.learning_rate),
                                  ('probe', probe, args.probe_lr)]:
        if parameters:
            groups.append(dict(params=parameters, lr=lr, base_lr=lr, name=name))
    return groups


def main():
    args = parse_args()
    for key in ['epochs', 'batch_size', 'accumulation', 'checkpoint_every', 'print_every']:
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, request_stop)
    reset_seeds(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda:0')
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('native BF16 is unavailable; choose fp16 before the paired runs')
    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    model, source_config = load_legacy(args, device)
    initial_digest = state_digest(model.backbone)
    datasets = load_datasets(args, manifest, source_config['nsweeps'])
    count = args.train_samples or 28130
    if not 1 <= count <= 28130 or not 1 <= args.val_samples <= 6019:
        raise ValueError('sample counts outside official split')
    indices = scale_indices(manifest, 'full')[:count]
    subset = Subset(datasets[0], indices)
    sampler = AcknowledgedShuffleSampler(subset, args.seed)
    groups = parameter_groups(model, args)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    trainable = [p for group in groups for p in group['params']]
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16', init_scale=128)
    criterion = SimpleLoss(2.13).to(device)
    config = {key: (str(value.resolve()) if isinstance(value, Path) else value)
              for key, value in vars(args).items()
              if key not in ['resume', 'stop_after_updates', 'print_every', 'checkpoint_every']}
    config.update(experiment='legacy_supervised_downstream',
                  source_checkpoint_sha256=digest_file(args.checkpoint),
                  source_manifest_sha256=source_config['manifest_sha256'],
                  manifest_sha256=manifest['manifest_sha256'],
                  radar_velocity_mode=source_config.get('radar_velocity_mode', 'full'),
                  nsweeps=source_config['nsweeps'], image_resolution=[112, 192],
                  camera_initialization='legacy random frozen encoder, then unfrozen for finetuning',
                  normalization='training BatchNorm for trainable student; frozen eval for frozen probe',
                  train_samples=count, val_samples=args.val_samples,
                  validation_epochs=sorted({1, args.epochs}),
                  frozen_initial_sha256=initial_digest if args.mode == 'frozen-probe' else None,
                  effective_batch=args.batch_size * args.accumulation,
                  implementation_sha256={name: digest_file(ROOT/'scripts'/name)
                      for name in ['train_baseline_downstream.py', 'baseline_followup.py']},
                  source_code_commit=code_commit(),
                  loss='official Simple-BEV multi-task' if args.mode == 'finetune' else 'legacy balanced BCE')
    if config['source_manifest_sha256'] != config['manifest_sha256']:
        raise ValueError('source checkpoint and downstream manifest differ')
    # The source commit is recorded once, not a reason to invalidate an otherwise
    # compatible resume after a documentation-only commit.
    run_dir = args.run_dir
    checkpoint_path = run_dir / 'checkpoints/latest.pt'
    config_path = run_dir / 'config.json'
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        old_config = json.loads(config_path.read_text())
        config['source_code_commit'] = old_config['source_code_commit']
    elif checkpoint_path.exists() or config_path.exists():
        raise FileExistsError('output exists; use --resume or another directory')
    progress = dict(epoch=0, update=0, samples_seen=0, wall_seconds=0.,
                    training_seconds=0., validations={}, latest={})
    if args.resume:
        progress = load_training_checkpoint(checkpoint_path, model=model,
            optimizer=optimizer, scaler=scaler, sampler=sampler,
            expected_run_config=config, map_location='cpu')
    atomic_json_write(config, config_path)
    writer = SummaryWriter(str(run_dir / 'tensorboard'), purge_step=progress['update'])
    start, prior_wall = time.monotonic(), progress['wall_seconds']
    state = 'running'

    def status(state='running', **extra):
        elapsed = prior_wall + time.monotonic() - start
        speed = progress['samples_seen'] / max(progress['training_seconds'], 1e-9)
        value = dict(state=state, updated_unix_time=time.time(),
            epoch_progress=progress['samples_seen']/count, epochs=args.epochs,
            update=progress['update'], samples_seen=progress['samples_seen'],
            samples_per_second=speed, wall_seconds=elapsed,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            eta_training_seconds=max(args.epochs*count-progress['samples_seen'], 0)/max(speed, 1e-9),
            latest=progress['latest'], gpu=query_nvidia_smi(), **extra)
        atomic_json_write(value, run_dir / 'status.json')
        return value

    def save():
        progress['epoch'] = sampler.epoch
        progress['wall_seconds'] = prior_wall + time.monotonic() - start
        save_training_checkpoint(checkpoint_path, model=model, optimizer=optimizer,
            scaler=scaler, sampler=sampler, progress=progress, run_config=config)

    def validate_pending():
        epoch = sampler.epoch
        if (sampler.position or epoch not in config['validation_epochs']
                or str(epoch) in progress['validations']):
            return
        model.eval()
        def predict(batch):
            with torch.autocast('cuda', dtype=dtype):
                output = model(batch, device, config['radar_velocity_mode'])
            return output[2] if args.mode == 'finetune' else output
        result = evaluate_predictions(predict, datasets[1],
            manifest['val']['dataset_indices'][:args.val_samples], args, manifest,
            run_dir / 'validation' / f'epoch{epoch:03d}',
            lambda n: status('validating', validation_samples=n))
        progress['validations'][str(epoch)] = result
        writer.add_scalar('validation/vehicle_iou', result['vehicle_iou'], epoch)
        save()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    updates_per_epoch = math.ceil(count / (args.batch_size * args.accumulation))
    total_updates = updates_per_epoch * args.epochs
    try:
        validate_pending()
        status()
        with (run_dir / 'metrics.jsonl').open('a') as log:
            while sampler.epoch < args.epochs and not STOP:
                loader = DataLoader(subset, sampler=sampler, batch_size=args.batch_size,
                    num_workers=args.num_workers, pin_memory=True,
                    generator=torch.Generator().manual_seed(args.seed+sampler.epoch),
                    prefetch_factor=2 if args.num_workers else None)
                iterator = iter(loader)
                epoch_before = sampler.epoch
                while not STOP:
                    step_start = time.monotonic()
                    batches = []
                    for _ in range(args.accumulation):
                        try:
                            batches.append(next(iterator))
                        except StopIteration:
                            break
                    if not batches:
                        break
                    samples = sum(batch[0].shape[0] for batch in batches)
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate_at_step(group['base_lr'],
                            progress['update'], total_updates, round(.25*updates_per_epoch))
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum, components = 0., {}
                    for batch in batches:
                        with torch.autocast('cuda', dtype=dtype):
                            loss, parts, _ = forward_loss(model, batch, device,
                                config['radar_velocity_mode'], criterion)
                        if not torch.isfinite(loss):
                            raise RuntimeError('non-finite training loss')
                        fraction = batch[0].shape[0] / samples
                        scaler.scale(loss * fraction).backward()
                        loss_sum += float(loss.detach()) * fraction
                        for key, value in parts.items():
                            components[key] = components.get(key, 0.) + float(value.detach())*fraction
                    scaler.unscale_(optimizer)
                    gradient = torch.nn.utils.clip_grad_norm_(trainable, 5., error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    sampler.acknowledge(samples)
                    progress['update'] += 1
                    progress['samples_seen'] += samples
                    duration = time.monotonic() - step_start
                    progress['training_seconds'] += duration
                    progress['latest'] = dict(loss=loss_sum, gradient_norm=float(gradient),
                        **components, seconds=duration,
                        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups})
                    log.write(json.dumps(dict(update=progress['update'],
                        samples_seen=progress['samples_seen'], epoch=sampler.epoch,
                        position=sampler.position, **progress['latest']))+'\n')
                    writer.add_scalar('train/loss', loss_sum, progress['update'])
                    if progress['update'] % args.checkpoint_every == 0:
                        save()
                    if progress['update'] % args.print_every == 0:
                        print(json.dumps(status()), flush=True)
                        log.flush()
                        writer.flush()
                    if args.stop_after_updates and progress['update'] >= args.stop_after_updates:
                        state = 'partial'
                        break
                del iterator, loader
                save()
                if sampler.epoch != epoch_before:
                    validate_pending()
                    save_training_checkpoint(run_dir / f'checkpoints/epoch{sampler.epoch:03d}.pt',
                        model=model, optimizer=optimizer, scaler=scaler, sampler=sampler,
                        progress=progress, run_config=config)
                if state == 'partial':
                    break
            if STOP:
                state = 'partial'
            elif sampler.epoch >= args.epochs:
                state = 'complete'
            save()
        if args.mode == 'frozen-probe' and state_digest(model.backbone) != initial_digest:
            raise RuntimeError('frozen backbone parameters/buffers changed')
        value = status(state)
        atomic_json_write(dict(status=state, config=config, progress=progress,
            telemetry=value, final_validation=progress['validations'].get(str(args.epochs))),
            run_dir / 'summary.json')
        print('BASELINE_DOWNSTREAM_' + state.upper(), flush=True)
    except Exception as error:
        status('failed', message=f'{type(error).__name__}: {error}')
        # Keep the last durable checkpoint: do not save a half-completed update.
        raise
    finally:
        writer.close()


if __name__ == '__main__':
    main()
