#!/usr/bin/env python3
"""CUDA interrupted-resume equivalence check for training checkpoints."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from training_checkpoint import (  # noqa: E402
    StatefulShuffleSampler,
    load_training_checkpoint,
    save_training_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--total-updates', type=int, default=8)
    parser.add_argument('--interrupt-after', type=int, default=4)
    return parser.parse_args()


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_components(dataset, device, seed):
    reset_seeds(seed)
    model = nn.Sequential(
        nn.Linear(4, 16), nn.GELU(), nn.Dropout(p=0.25), nn.Linear(16, 2),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0, growth_interval=2)
    sampler = StatefulShuffleSampler(dataset, seed=seed)
    return model, optimizer, scaler, sampler


def train_updates(model, optimizer, scaler, sampler, dataset, updates, device):
    model.train()
    completed = 0
    losses = []
    while completed < updates:
        loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            num_workers=0, drop_last=False)
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Exercise all three host RNGs plus CUDA dropout. The scalar factors
            # make a missing RNG restore change the final weights.
            host_factor = 0.9 + 0.05 * random.random() + 0.05 * np.random.random()
            inputs = inputs * host_factor + 0.01 * torch.rand_like(inputs)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                prediction = model(inputs)
                loss = nn.functional.smooth_l1_loss(prediction, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            completed += 1
            if completed >= updates:
                break
    return losses


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the AMP/GradScaler resume check')
    if not 0 < args.interrupt_after < args.total_updates:
        raise ValueError('interrupt-after must be between zero and total-updates')
    device = torch.device('cuda:0')
    generator = torch.Generator().manual_seed(2026)
    inputs = torch.randn(5, 4, generator=generator)
    targets = torch.randn(5, 2, generator=generator)
    dataset = TensorDataset(inputs, targets)
    run_config = {
        'purpose': 'checkpoint-resume-equivalence',
        'seed': args.seed,
        'dataset_size': len(dataset),
        'total_updates': args.total_updates,
    }

    reference = build_components(dataset, device, args.seed)
    reference_losses = train_updates(*reference, dataset, args.total_updates, device)
    torch.cuda.synchronize(device)
    reference_state = clone_state(reference[0])

    interrupted = build_components(dataset, device, args.seed)
    first_losses = train_updates(*interrupted, dataset, args.interrupt_after, device)
    progress = {
        'epoch': interrupted[3].epoch,
        'update': args.interrupt_after,
    }
    config_sha256 = save_training_checkpoint(
        args.checkpoint, model=interrupted[0], optimizer=interrupted[1],
        scaler=interrupted[2], sampler=interrupted[3], progress=progress,
        run_config=run_config,
    )

    resumed = build_components(dataset, device, args.seed)
    reset_seeds(args.seed + 999)
    resumed_progress = load_training_checkpoint(
        args.checkpoint, model=resumed[0], optimizer=resumed[1], scaler=resumed[2],
        sampler=resumed[3], expected_run_config=run_config, map_location=device,
    )
    remaining = args.total_updates - resumed_progress['update']
    second_losses = train_updates(*resumed, dataset, remaining, device)
    torch.cuda.synchronize(device)
    resumed_state = clone_state(resumed[0])

    differences = {
        name: float((reference_state[name] - resumed_state[name]).abs().max())
        for name in reference_state
    }
    max_difference = max(differences.values(), default=0.0)
    loss_difference = max(
        abs(left - right)
        for left, right in zip(reference_losses, first_losses + second_losses)
    )
    tolerance = 1e-7
    if max_difference > tolerance or loss_difference > tolerance:
        raise RuntimeError(
            f'resume mismatch: parameter={max_difference}, loss={loss_difference}'
        )
    if resumed[3].epoch != reference[3].epoch or resumed[3].position != reference[3].position:
        raise RuntimeError('resumed sampler did not reach the reference position')

    summary = {
        'device': torch.cuda.get_device_name(device),
        'seed': args.seed,
        'total_updates': args.total_updates,
        'interrupt_after': args.interrupt_after,
        'checkpoint': str(args.checkpoint),
        'checkpoint_config_sha256': config_sha256,
        'resumed_from': resumed_progress,
        'final_sampler': resumed[3].state_dict(),
        'max_parameter_abs_difference': max_difference,
        'max_loss_abs_difference': loss_difference,
        'tolerance': tolerance,
        'parameter_differences': differences,
        'acceptance': 'interrupted AMP training matches uninterrupted training',
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print('TRAINING_RESUME_OK', flush=True)


if __name__ == '__main__':
    main()
