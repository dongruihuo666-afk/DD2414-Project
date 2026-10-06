#!/usr/bin/env python3
"""Generate the trainval scaling manifest and exercise real streaming batches."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from dinov2_bev_demo import build_loaders  # noqa: E402
from radar_scaling_data import (  # noqa: E402
    DEFAULT_SCALE_COUNTS,
    build_scaling_manifest,
    make_streaming_loader,
    scale_indices,
    write_manifest,
)


MIB = 1024 ** 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest-out', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--scale-counts', default='64,256,1024,4096')
    parser.add_argument('--stream-scale', default='1024')
    parser.add_argument('--check-batches', type=int, default=512)
    parser.add_argument('--nsweeps', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    counts = tuple(int(part) for part in args.scale_counts.split(',') if part)
    if counts != DEFAULT_SCALE_COUNTS:
        raise ValueError(f'P1 requires locked scales {DEFAULT_SCALE_COUNTS}')
    if args.check_batches < 512:
        raise ValueError('P1 acceptance requires at least 512 streaming batches')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the current VizData implementation')

    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    loaders = build_loaders(
        args.data_root, num_workers=0, nsweeps=args.nsweeps,
        rotate_radar_velocity=True, dset='trainval',
    )
    train_dataset, val_dataset = (loader.dataset for loader in loaders)
    manifest = build_scaling_manifest(
        train_dataset, val_dataset, seed=args.seed, scale_counts=counts,
    )
    write_manifest(manifest, args.manifest_out)

    indices = scale_indices(manifest, args.stream_scale)
    if args.check_batches > len(indices):
        raise ValueError('check-batches exceeds the selected single-pass scale')
    stream = make_streaming_loader(train_dataset, indices, num_workers=0)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    allocations = []
    radar_points = 0
    start = time.perf_counter()
    for batch_index, batch in enumerate(stream, start=1):
        radar = batch[16]
        radar_points += int(radar[:, :, :3].abs().sum(dim=2).gt(0).sum())
        del radar, batch
        if batch_index % 64 == 0:
            gc.collect()
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            allocations.append(allocated)
            print(f'batches={batch_index} allocated_mib={allocated / MIB:.3f}', flush=True)
        if batch_index >= args.check_batches:
            break
    elapsed = time.perf_counter() - start

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    final_allocated = torch.cuda.memory_allocated(device)
    final_reserved = torch.cuda.memory_reserved(device)
    tolerance = 16 * MIB
    if final_allocated > baseline_allocated + tolerance:
        raise RuntimeError(
            f'CUDA allocation grew by {(final_allocated - baseline_allocated) / MIB:.3f} MiB'
        )
    if allocations and max(allocations[-2:]) > min(allocations[:2]) + tolerance:
        raise RuntimeError('periodic CUDA allocations show retained-batch growth')

    summary = {
        'dataset': 'v1.0-trainval',
        'seed': args.seed,
        'nsweeps': args.nsweeps,
        'manifest': str(args.manifest_out),
        'manifest_sha256': manifest['manifest_sha256'],
        'train_samples': manifest['scales']['full']['count'],
        'train_scenes': manifest['scales']['full']['unique_scenes'],
        'val_samples': manifest['val']['count'],
        'val_scenes': manifest['val']['unique_scenes'],
        'scale_unique_scenes': {
            name: value['unique_scenes'] for name, value in manifest['scales'].items()
        },
        'stream_scale': str(args.stream_scale),
        'checked_batches': args.check_batches,
        'radar_points': radar_points,
        'elapsed_seconds': elapsed,
        'batches_per_second': args.check_batches / elapsed,
        'baseline_allocated_mib': baseline_allocated / MIB,
        'final_allocated_mib': final_allocated / MIB,
        'baseline_reserved_mib': baseline_reserved / MIB,
        'final_reserved_mib': final_reserved / MIB,
        'periodic_allocated_mib': [value / MIB for value in allocations],
        'acceptance': '512+ batches streamed with no retained CUDA allocation growth',
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print('RADAR_SCALING_STREAM_OK', flush=True)


if __name__ == '__main__':
    main()
