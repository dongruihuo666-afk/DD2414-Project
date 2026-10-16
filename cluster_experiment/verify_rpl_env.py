#!/usr/bin/env python3
"""Verify the RPL Python environment without confusing CPU and CUDA checks."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DINOV2_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
EXPECTED_DINOV2_WEIGHT_SHA256 = (
    'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
)
EXPECTED_BEV_CAR_COMMIT = '29cacda3bc5416d47428c1d0f017527acad34f90'
EXPECTED_BEV_CAR_VOXELNET_SHA256 = (
    '43ed0ee30f70c328bd47885cbc68272e7f5e3625c61b325b54f713a851f3e6ac'
)
EXPECTED_PACKAGES = {
    'numpy': '1.26.4',
    'torch': '2.5.1',
    'torchvision': '0.20.1',
    'nuscenes-devkit': '1.2.0',
    'Pillow': '10.4.0',
    'opencv-python-headless': '4.10.0.84',
    'matplotlib': '3.8.4',
    'scipy': '1.11.4',
    'scikit-learn': '1.5.2',
    'scikit-image': '0.22.0',
    'Shapely': '2.0.3',
    'pyquaternion': '0.9.9',
    'efficientnet-pytorch': '0.7.1',
    'fire': '0.7.1',
    'tensorboard': '2.18.0',
    'tensorboardX': '2.6.2.2',
    'protobuf': '4.25.5',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--precision', choices=('auto', 'bf16', 'fp16'),
                        default='auto')
    parser.add_argument('--bevcar-source', type=Path,
                        default=(Path(os.environ['BEVCAR_SOURCE_DIR'])
                                 if os.environ.get('BEVCAR_SOURCE_DIR') else None))
    parser.add_argument('--require-dinov2-cache', action='store_true')
    parser.add_argument('--json-output', type=Path)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_head(path):
    path = Path(path)
    if not (path / '.git').exists():
        return None
    try:
        result = subprocess.run(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'], check=True,
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


class Audit:
    def __init__(self):
        self.checks = []

    def add(self, name, status, detail, required=True):
        self.checks.append({
            'name': name,
            'status': status,
            'required': required,
            'detail': detail,
        })

    def pass_(self, name, detail, required=True):
        self.add(name, 'pass', detail, required)

    def warn(self, name, detail):
        self.add(name, 'warning', detail, False)

    def fail(self, name, detail):
        self.add(name, 'fail', detail, True)

    @property
    def failures(self):
        return [item for item in self.checks
                if item['required'] and item['status'] == 'fail']


def check_versions(audit):
    if sys.version_info[:2] == (3, 10):
        audit.pass_('python_version', sys.version.split()[0])
    else:
        audit.fail('python_version',
                   f'{sys.version.split()[0]} (expected Python 3.10)')
    for distribution, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            audit.fail(f'package:{distribution}', 'not installed')
            continue
        base = actual.split('+', 1)[0]
        if base == expected:
            audit.pass_(f'package:{distribution}', actual)
        else:
            audit.fail(f'package:{distribution}',
                       f'{actual} (expected {expected})')


def check_imports(audit):
    modules = (
        'numpy', 'torch', 'torchvision', 'PIL', 'cv2', 'matplotlib',
        'skimage', 'sklearn', 'scipy', 'shapely', 'pyquaternion',
        'efficientnet_pytorch', 'fire', 'tensorboard', 'tensorboardX',
        'nuscenes',
    )
    for module in modules:
        try:
            importlib.import_module(module)
            audit.pass_(f'import:{module}', 'ok')
        except Exception as error:  # import failures need their original type.
            audit.fail(f'import:{module}', f'{type(error).__name__}: {error}')

    scripts_path = str(REPO_ROOT / 'scripts')
    root_path = str(REPO_ROOT)
    for value in (scripts_path, root_path):
        if value not in sys.path:
            sys.path.insert(0, value)
    project_modules = (
        'nuscenesdataset',
        'nets.segnet',
        'dinov2_bev_demo',
        'hybrid_teacher_full',
        'train_hybrid_teacher_pretrain',
        'train_hybrid_teacher_downstream',
        'summarize_hybrid_teacher_cluster',
    )
    for module in project_modules:
        try:
            importlib.import_module(module)
            audit.pass_(f'project_import:{module}', 'ok')
        except Exception as error:
            audit.fail(f'project_import:{module}',
                       f'{type(error).__name__}: {error}')


def check_torch_build(audit):
    try:
        import torch
        import torchvision
    except Exception as error:
        audit.fail('torch_cuda_build',
                   f'torch/torchvision import failed: {error}')
        return None
    if torch.version.cuda == '12.4':
        audit.pass_('torch_cuda_build', 'CUDA 12.4 wheel')
    else:
        audit.fail('torch_cuda_build',
                   f'{torch.version.cuda!r} (expected CUDA 12.4, not CPU-only)')
    vision_cuda = getattr(torchvision.version, 'cuda', None)
    if vision_cuda == 12040 and torchvision.extension._HAS_OPS:
        audit.pass_('torchvision_cuda_build',
                    'CUDA 12.4 extension operators are available')
    else:
        audit.fail(
            'torchvision_cuda_build',
            f'CUDA tag {vision_cuda!r}, extension ops '
            f'{torchvision.extension._HAS_OPS!r} (expected CUDA 12.4)',
        )
    return torch


def check_bevcar(audit, source):
    if source is None:
        audit.warn('bevcar_voxelnet',
                   'not checked; pass --bevcar-source after cloning BEVCar')
        return
    source = source.resolve()
    voxelnet = source / 'nets' / 'voxelnet.py'
    if not voxelnet.is_file():
        audit.fail('bevcar_voxelnet', f'missing {voxelnet}')
        return
    digest = sha256(voxelnet)
    if digest != EXPECTED_BEV_CAR_VOXELNET_SHA256:
        audit.fail('bevcar_voxelnet_sha256',
                   f'{digest} (expected pinned VoxelNet source)')
        return
    audit.pass_('bevcar_voxelnet_sha256', digest)
    commit = git_head(source)
    if commit is not None and commit != EXPECTED_BEV_CAR_COMMIT:
        audit.fail('bevcar_git_commit',
                   f'{commit} (expected {EXPECTED_BEV_CAR_COMMIT})')
    elif commit is not None:
        audit.pass_('bevcar_git_commit', commit)
    else:
        audit.warn('bevcar_git_commit',
                   'source is not a Git checkout; VoxelNet SHA-256 did match')
    try:
        from compare_motion_target_bevcar_tiny import official_voxelnet
        cls = official_voxelnet(source)
        model = cls(
            use_col=False, reduced_zx=False, output_dim=128,
            use_radar_occupancy_map=False,
        )
        parameter_count = sum(value.numel() for value in model.parameters())
        audit.pass_('bevcar_voxelnet_load',
                    f'VoxelNet instantiated on CPU; {parameter_count} parameters')
    except Exception as error:
        audit.fail('bevcar_voxelnet_load',
                   f'{type(error).__name__}: {error}')


def check_dinov2(audit, torch, required):
    if torch is None:
        audit.fail('dinov2_cache', 'cannot inspect cache without torch')
        return
    hub_root = Path(torch.hub.get_dir())
    source = hub_root / 'facebookresearch_dinov2_main'
    weight = hub_root / 'checkpoints' / 'dinov2_vits14_pretrain.pth'
    if not (source / 'hubconf.py').is_file():
        message = f'missing DINOv2 source cache: {source}'
        audit.fail('dinov2_source_cache', message) if required else audit.warn(
            'dinov2_source_cache', message)
    else:
        commit = git_head(source)
        if commit == EXPECTED_DINOV2_COMMIT:
            audit.pass_('dinov2_source_commit', commit)
        elif commit is None:
            message = 'source exists but is not a pinned Git checkout'
            audit.fail('dinov2_source_commit', message) if required else audit.warn(
                'dinov2_source_commit', message)
        else:
            message = f'{commit} (expected {EXPECTED_DINOV2_COMMIT})'
            audit.fail('dinov2_source_commit', message) if required else audit.warn(
                'dinov2_source_commit', message)
        try:
            model = torch.hub.load(
                str(source), 'dinov2_vits14', source='local', pretrained=False,
            )
            audit.pass_('dinov2_source_load',
                        f'ViT-S/14 instantiated; {sum(p.numel() for p in model.parameters())} parameters')
            del model
        except Exception as error:
            audit.fail('dinov2_source_load',
                       f'{type(error).__name__}: {error}')
    if not weight.is_file():
        message = f'missing DINOv2 weights: {weight}'
        audit.fail('dinov2_weight_cache', message) if required else audit.warn(
            'dinov2_weight_cache', message)
    else:
        digest = sha256(weight)
        if digest == EXPECTED_DINOV2_WEIGHT_SHA256:
            audit.pass_('dinov2_weight_sha256', digest)
        else:
            audit.fail('dinov2_weight_sha256',
                       f'{digest} (expected {EXPECTED_DINOV2_WEIGHT_SHA256})')


def check_gpu(audit, torch, precision):
    if torch is None:
        audit.fail('cuda_available', 'torch is unavailable')
        return
    if not torch.cuda.is_available():
        audit.fail('cuda_available',
                   'False; CPU checks do not count as CUDA verification')
        return
    try:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        audit.pass_('cuda_available', True)
        audit.pass_('gpu_identity', {
            'index': index,
            'name': properties.name,
            'compute_capability': f'{properties.major}.{properties.minor}',
            'total_memory_gib': properties.total_memory / (1024 ** 3),
        })
        audit.pass_('native_bf16_supported', bf16_supported)
        selected = precision
        if selected == 'auto':
            selected = 'bf16' if bf16_supported else 'fp16'
        if selected == 'bf16' and not bf16_supported:
            audit.fail('requested_precision',
                       'BF16 requested but this GPU/PyTorch/driver combination does not support it; use PRECISION=fp16')
            return
        dtype = torch.bfloat16 if selected == 'bf16' else torch.float16
        left = torch.randn((256, 256), device='cuda', dtype=dtype)
        right = torch.randn((256, 256), device='cuda', dtype=dtype)
        output = left @ right
        torch.cuda.synchronize()
        if not torch.isfinite(output).all():
            audit.fail('cuda_tensor_operation', 'matrix multiply produced non-finite values')
            return
        audit.pass_('requested_precision', selected)
        audit.pass_('cuda_tensor_operation',
                    f'finite 256x256 matrix multiply using {selected}')
    except Exception as error:
        audit.fail('cuda_tensor_operation', f'{type(error).__name__}: {error}')


def main():
    args = parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    audit = Audit()
    check_versions(audit)
    check_imports(audit)
    torch = check_torch_build(audit)
    check_bevcar(audit, args.bevcar_source)
    cache_required = args.require_dinov2_cache or args.mode == 'gpu'
    check_dinov2(audit, torch, cache_required)
    if args.mode == 'gpu':
        check_gpu(audit, torch, args.precision)
        cuda_check = next(
            (item for item in audit.checks
             if item['name'] == 'cuda_available'),
            None,
        )
        cuda_status = (
            'passed' if cuda_check and cuda_check['status'] == 'pass'
            else 'failed'
        )
    else:
        audit.warn('cuda_validation',
                   'not requested; run --mode gpu inside a Slurm GPU allocation')
        cuda_status = 'not_tested_cpu_mode'

    report = {
        'mode': args.mode,
        'python_executable': sys.executable,
        'python_prefix': sys.prefix,
        'cuda_validation': cuda_status,
        'success': not audit.failures,
        'checks': audit.checks,
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    print(payload)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + '\n')
    raise SystemExit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
