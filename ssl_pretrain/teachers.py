"""Frozen image teachers: DINOv2 (semantics) and Depth-Anything-V2 metric (geometry)."""

import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14


def _freeze(model, device):
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _normalize(imgs01):
    mean = imgs01.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = imgs01.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (imgs01 - mean) / std


def teacher_size(H, W, short=448):
    """Largest patch-multiple size close to (H, W) scaled to `short` rows."""
    h = short // PATCH * PATCH
    w = int(round(W * short / H / PATCH)) * PATCH
    return h, w


def load_dino(name='dinov2_vits14_reg', device='cuda', hub_dir=None):
    if hub_dir:
        model = torch.hub.load(hub_dir, name, source='local')
    else:
        model = torch.hub.load('facebookresearch/dinov2', name)
    return _freeze(model, device)


@torch.no_grad()
def dino_features(model, imgs01, size):
    """imgs01: (N, 3, H, W) in [0, 1]. Returns patch tokens (N, D, h/14, w/14)."""
    x = F.interpolate(imgs01.float(), size=size, mode='bicubic', align_corners=False,
                      antialias=True).clamp(0, 1)
    tokens = model.forward_features(_normalize(x))['x_norm_patchtokens']
    h, w = size[0] // PATCH, size[1] // PATCH
    return tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)


def load_depth_anything_metric(source_dir, weights, encoder='vits', max_depth=80.0, device='cuda'):
    """Depth-Anything-V2 metric model (official repo, `metric_depth` subfolder).

    Use the outdoor (VKITTI) checkpoint. Its absolute scale is tied to the
    training camera, so callers must re-align the scale (see
    `targets.align_depth_scale`).
    """
    root = Path(source_dir) / 'metric_depth'
    if not (root / 'depth_anything_v2' / 'dpt.py').is_file():
        raise FileNotFoundError(f'Depth-Anything-V2 metric_depth source not found under {root}')
    for name in [n for n in sys.modules if n == 'depth_anything_v2' or n.startswith('depth_anything_v2.')]:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        dpt = importlib.import_module('depth_anything_v2.dpt')
    finally:
        sys.path.remove(str(root))
    configs = {'vits': dict(features=64, out_channels=[48, 96, 192, 384]),
               'vitb': dict(features=128, out_channels=[96, 192, 384, 768]),
               'vitl': dict(features=256, out_channels=[256, 512, 1024, 1024])}
    model = dpt.DepthAnythingV2(encoder=encoder, max_depth=max_depth, **configs[encoder])
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return _freeze(model, device)


@torch.no_grad()
def metric_depth(model, imgs01, size, chunk=6):
    """imgs01: (N, 3, H, W) in [0, 1]. Returns metric depth (N, 1, *size)."""
    x = _normalize(F.interpolate(imgs01.float(), size=size, mode='bilinear', align_corners=False))
    out = [model(x[i:i + chunk]).float() for i in range(0, x.shape[0], chunk)]
    return torch.cat(out, 0).unsqueeze(1)
