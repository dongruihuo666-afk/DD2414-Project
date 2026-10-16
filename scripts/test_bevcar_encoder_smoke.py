#!/usr/bin/env python3
"""Isolated one-frame smoke test using an external official BEVCar checkout.

The released BEVCar source is imported from --bevcar-source at runtime. It is
not copied into this repository and no supervised BEVCar weights are loaded.
"""

import argparse
import importlib.util
import resource
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from nets.bevcar_voxel_adapter import prepare_bevcar_voxels  # noqa: E402
from test_bevcar_voxel_adapter import (  # noqa: E402
    first_mini_train_radar, make_vox_util,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--cpu-threads', type=int, default=2)
    return parser.parse_args()


def load_official_voxelnet(source_dir):
    source_file = source_dir / 'nets' / 'voxelnet.py'
    if not source_file.is_file():
        raise FileNotFoundError(
            f'BEVCar source not found at {source_file}; clone the official '
            'repository and pass --bevcar-source'
        )
    spec = importlib.util.spec_from_file_location('official_bevcar_voxelnet', source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VoxelNet


def main():
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable in this environment')
    device = torch.device(
        'cuda' if args.device == 'cuda' or
        (args.device == 'auto' and torch.cuda.is_available()) else 'cpu'
    )

    radar_camera, sample_token = first_mini_train_radar(args.data_root)
    features, coords, counts, info = prepare_bevcar_voxels(
        radar_camera, make_vox_util(), quality_filter=False,
    )
    if info['truncated_points'].item() != 0:
        raise AssertionError('this audit expects no point truncation')
    if torch.any((coords[0, :counts[0]] == 0).all(dim=-1)):
        raise AssertionError('the official encoder reserves voxel (0,0,0) for padding')

    voxelnet_cls = load_official_voxelnet(args.bevcar_source)
    torch.manual_seed(13)
    model = voxelnet_cls(
        use_col=False, reduced_zx=False, output_dim=128,
        use_radar_occupancy_map=False,
    ).to(device).train()
    features = features.to(device)
    coords = coords.to(device)
    counts = counts.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.monotonic()
    output = model(features, coords, counts)
    if output.shape != (1, 128, 200, 200):
        raise AssertionError(f'unexpected BEVCar output shape: {tuple(output.shape)}')
    if not torch.isfinite(output).all():
        raise AssertionError('BEVCar output contains non-finite values')
    loss = output.square().mean()
    loss.backward()
    svfe_grad = model.svfe.vfe_1.fcn.linear.weight.grad
    cml_grad = model.cml.conv3d_1.conv.weight.grad
    if svfe_grad is None or cml_grad is None:
        raise AssertionError('radar encoder did not receive gradients')
    svfe_norm = svfe_grad.float().norm().item()
    cml_norm = cml_grad.float().norm().item()
    if not (svfe_norm > 0 and cml_norm > 0):
        raise AssertionError('radar encoder gradients must be nonzero')
    if not (torch.isfinite(svfe_grad).all() and torch.isfinite(cml_grad).all()):
        raise AssertionError('radar encoder gradients must be finite')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start

    print(f'device: {device}')
    print(f'sample_token: {sample_token}')
    print(f'BEVCar_source: {args.bevcar_source.resolve()}')
    print(f'input_features_shape: {tuple(features.shape)}')
    print(f'input_coords_shape: {tuple(coords.shape)}')
    print(f'occupied_voxels: {counts.tolist()}')
    print(f'output_shape: {tuple(output.shape)}')
    print(f'loss: {loss.item():.9g}')
    print(f'SVFE_first_layer_gradient_norm: {svfe_norm:.9g}')
    print(f'CML_first_layer_gradient_norm: {cml_norm:.9g}')
    print(f'forward_backward_seconds: {elapsed:.3f}')
    if device.type == 'cuda':
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        print(f'peak_allocated_cuda_gib: {peak_gib:.3f}')
    else:
        peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        print(f'peak_process_rss_gib: {peak_gib:.3f}')
        print('gpu_memory: NOT MEASURED (CUDA unavailable)')
    print('encoder_forward_backward: OK')


if __name__ == '__main__':
    main()
