"""Opt-in metric depth probability lifting; the official Vox_util stays unchanged.

The student predicts a distribution along each viewing ray. Both treatments use
that distribution for lifting. Only the treatment uses frozen metric pseudo
depth, compared AFTER calibrated sampling into the camera BEV volume and BEFORE
radar fusion. This is geometry distillation, not independent depth ground truth.
"""
import importlib
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

from baseline_followup import digest_file
from multisweep_teacher import MultiSweepTeacher, build_model
import utils.basic
import utils.geom

PROFILE = 'random-metric-bev'
TREATMENTS = ('dino-only', 'dino-metric-depth-bev')
DEPTH_BINS = 80
DEPTH_MIN, DEPTH_MAX = 1., 80.
METRIC_WEIGHT_SHA256 = '9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb'


def ray_grid(base, features, pixels, cameras, z, y, x, xyz_camA=None,
             assert_cube=False):
    """Keep legacy image XY sampling, with exact align_corners=False depth centers."""
    b, _, h, w = features.shape
    if min(h, w) < 2:
        raise ValueError('Camera feature maps must have at least two pixels per axis')
    if xyz_camA is None:
        xyz_camA = base.Mem2Ref(utils.basic.gridcloud3d(
            b, z, y, x, norm=False, device=features.device), z, y, x,
            assert_cube=assert_cube)
    camera_z = utils.geom.apply_4x4(cameras.float(), xyz_camA.float())[..., 2]
    projected = utils.geom.apply_4x4(pixels.float(), xyz_camA.float())
    xy = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
    px, py = xy.unbind(-1)
    ny, nx = utils.basic.normalize_grid2d(py, px, h, w)
    depth_index = (camera_z - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * (DEPTH_BINS - 1)
    nz = 2 * (depth_index + .5) / DEPTH_BINS - 1
    valid = ((px > -.5) & (px < w - .5) & (py > -.5) & (py < h - .5)
             & (camera_z >= DEPTH_MIN) & (camera_z <= DEPTH_MAX))
    grid = torch.stack((nx, ny, nz), -1).reshape(b, z, y, x, 3)
    return grid.detach(), valid.reshape(b, 1, z, y, x).detach()


def sample_ray_volume(volume, grid):
    return F.grid_sample(volume.float(), grid.float(), align_corners=False,
                         mode='bilinear', padding_mode='border')


def project_teacher_depth(depth, grid, valid, feature_size, sigma_m=2.):
    """Soft metric target; no ground-truth labels or student-dependent validity."""
    if sigma_m <= 0:
        raise ValueError('Metric target width must be positive')
    with torch.no_grad():
        finite = torch.isfinite(depth) & (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX)
        # Nearest avoids interpolating invalid depths into valid observations.
        depth = F.interpolate(torch.nan_to_num(depth.float(), nan=0., posinf=0., neginf=0.),
                              feature_size, mode='nearest')
        finite = F.interpolate(finite.float(), feature_size, mode='nearest')
        centers = torch.linspace(DEPTH_MIN, DEPTH_MAX, DEPTH_BINS, device=depth.device)
        q = torch.softmax(-.5 * ((centers[None, :, None, None] - depth) / sigma_m).square(), 1)
        q = q * finite
        weights = sample_ray_volume(q[:, None], grid) * valid
    return weights.detach()


def projected_depth_loss(log_probability_bev, target_weights):
    """Per-camera mass-normalized cross entropy on projected camera geometry.

    Softmax competition along depth suppresses other locations on the SAME ray.
    Sampling log probabilities directly avoids log(clamp(p)) dead gradients.
    Targets are pseudo observations, not occupied/empty voxel labels.
    """
    target_weights = target_weights.detach()
    mass = target_weights.flatten(1).sum(1)
    value = -(target_weights * log_probability_bev).flatten(1).sum(1)
    usable = mass > 1e-8
    normalized = value / mass.clamp_min(1e-8)
    return (normalized * usable).sum() / usable.sum().clamp_min(1)


class DepthProbabilityVox:
    def __init__(self, base, head, retain_geometry=False):
        self.base, self.head, self.retain_geometry = base, head, retain_geometry
        self.geometry = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def unproject_image_to_mem(self, features, pixels, cameras, z, y, x,
                               assert_cube=False, xyz_camA=None):
        logits = self.head(features).float()
        log_probability = F.log_softmax(logits, dim=1)
        grid, valid = ray_grid(self.base, features, pixels, cameras, z, y, x,
                              xyz_camA, assert_cube)
        weights = sample_ray_volume(log_probability.exp()[:, None], grid) * valid
        volume = self.base.unproject_image_to_mem(features, pixels, cameras,
            z, y, x, assert_cube=assert_cube, xyz_camA=xyz_camA)
        # A tiny floor preserves the legacy nonzero-feature camera visibility
        # mask under probability underflow; geometric validity remains explicit.
        volume = volume * (weights.clamp_min(1e-6) * valid)
        if self.retain_geometry:
            self.geometry = dict(log_probability_bev=sample_ray_volume(
                log_probability[:, None], grid), grid=grid, valid=valid,
                feature_size=features.shape[-2:])
        return volume


class MetricBEVStudent(MultiSweepTeacher):
    def __init__(self, student):
        super().__init__(student, depth_mode='metric-bev')
        self.depth_head = nn.Conv2d(128, DEPTH_BINS, 1)
        self.geometry = None

    def features(self, inputs, location='fused_bev', retain_geometry=False):
        images, pixels, cameras, vox, radar = inputs
        wrapped = DepthProbabilityVox(vox, self.depth_head, retain_geometry)
        shared, camera, image = self.student(rgb_camXs=images, pix_T_cams=pixels,
            cam0_T_camXs=cameras, vox_util=wrapped, bevcar_voxels=radar,
            return_shared_bev=True)
        self.geometry = wrapped.geometry
        if location == 'camera_bev':
            b, cy, z, x = camera.shape
            shared = camera.reshape(b, 128, cy // 128, z, x).mean(2)
        elif location != 'fused_bev':
            raise ValueError('Unknown probe location: ' + location)
        return shared, image, None


def build_metric_model(device, seed, bevcar_source):
    # The existing factory explicitly uses pretrained_backbone=False. Neither
    # official RGB nor a previous student checkpoint is loaded by this factory.
    legacy = build_model(device, seed, bevcar_source, 'none')
    return MetricBEVStudent(legacy.student).to(device)


def configure_metric_pretraining(model):
    model.requires_grad_(False)
    for module in [model.student.encoder, model.student.bevcar_encoder,
                   model.student.bev_compressor, model.image_head, model.bev_head,
                   model.depth_head]:
        module.requires_grad_(True)


def load_metric_teacher(source, weights, expected_sha256, device):
    if digest_file(weights) != expected_sha256:
        raise ValueError('Metric Depth Anything weight SHA-256 mismatch')
    metric_root = (Path(source) / 'metric_depth').resolve()
    module_path = metric_root / 'depth_anything_v2/dpt.py'
    if not module_path.is_file():
        raise FileNotFoundError('Metric Depth Anything source is missing: ' + str(module_path))
    for name, module in list(sys.modules.items()):
        if name == 'depth_anything_v2' or name.startswith('depth_anything_v2.'):
            origin = getattr(module, '__file__', None)
            if origin and not Path(origin).resolve().is_relative_to(metric_root):
                raise RuntimeError('Relative-depth package already imported; use a clean metric-teacher process')
    sys.path.insert(0, str(metric_root))
    cls = importlib.import_module('depth_anything_v2.dpt').DepthAnythingV2
    teacher = cls(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=80)
    teacher.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return teacher.to(device).eval().requires_grad_(False)


@torch.no_grad()
def metric_teacher_depth(teacher, images, device, resolution):
    rgb = images.flatten(0, 1).to(device).float()
    rgb = F.interpolate(rgb, tuple(resolution), mode='bilinear', align_corners=False)
    mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
    std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
    # Sequential camera inference bounds teacher activation memory on 10GB GPUs.
    return torch.cat([teacher(frame[None].sub(mean).div(std)).float()[:, None]
                      for frame in rgb], dim=0).detach()
