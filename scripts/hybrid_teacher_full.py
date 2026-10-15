#!/usr/bin/env python3
"""Shared model and metric helpers for the opt-in full hybrid-teacher study."""

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import utils.vox  # noqa: E402
import utils.basic  # noqa: E402
from compare_motion_target_bevcar_tiny import (  # noqa: E402
    official_voxelnet,
    prepare_bevcar_inputs,
)
from nets.segnet import Segnet  # noqa: E402
from train_nuscenes import (  # noqa: E402
    SimpleLoss,
    X,
    Y,
    Z,
    balanced_mse_loss,
    bounds,
    scene_centroid,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module_state_sha256(module):
    """Hash parameters and buffers for a frozen-backbone integrity check."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HybridTeacherModel(nn.Module):
    """Simple-BEV + BEVCar student with two DINO heads and a motion head.

    The frozen DINOv2 network intentionally lives in the training wrapper, not
    in this deployable student checkpoint. ``image_head`` implements the first
    distillation level and ``bev_head`` implements the radar-masked fused-BEV
    level. The motion head remains a separately reported auxiliary objective.
    """

    def __init__(self, student, teacher_dim=384):
        super().__init__()
        self.student = student
        self.image_head = nn.Sequential(
            nn.Conv2d(student.feat2d_dim, student.feat2d_dim, 3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(student.feat2d_dim),
            nn.GELU(),
            nn.Conv2d(student.feat2d_dim, teacher_dim, 1),
        )
        self.bev_head = nn.Sequential(
            nn.Conv2d(student.latent_dim, student.latent_dim, 3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(student.latent_dim),
            nn.GELU(),
            nn.Conv2d(student.latent_dim, teacher_dim, 1),
        )
        self.motion_head = nn.Conv2d(student.latent_dim, 2, 1)

    def pretrain_forward(self, images, pixel_from_cameras,
                         camera0_from_cameras, vox_util, bevcar_voxels):
        shared_bev, _, packed_image_features = self.student(
            rgb_camXs=images,
            pix_T_cams=pixel_from_cameras,
            cam0_T_camXs=camera0_from_cameras,
            vox_util=vox_util,
            bevcar_voxels=bevcar_voxels,
            return_shared_bev=True,
        )
        return {
            'image_prediction': self.image_head(packed_image_features),
            'bev_prediction': self.bev_head(shared_bev),
            'motion_prediction': self.motion_head(shared_bev),
            'shared_bev': shared_bev,
        }

    def shared_bev(self, images, pixel_from_cameras, camera0_from_cameras,
                   vox_util, bevcar_voxels):
        shared_bev, _, _ = self.student(
            rgb_camXs=images,
            pix_T_cams=pixel_from_cameras,
            cam0_T_camXs=camera0_from_cameras,
            vox_util=vox_util,
            bevcar_voxels=bevcar_voxels,
            return_shared_bev=True,
        )
        return shared_bev

    def downstream_forward(self, images, pixel_from_cameras,
                           camera0_from_cameras, vox_util, bevcar_voxels):
        raw, features, segmentation, center, offset = self.student(
            rgb_camXs=images,
            pix_T_cams=pixel_from_cameras,
            cam0_T_camXs=camera0_from_cameras,
            vox_util=vox_util,
            bevcar_voxels=bevcar_voxels,
            return_shared_bev=False,
        )
        return {
            'raw': raw,
            'features': features,
            'segmentation': segmentation,
            'center': center,
            'offset': offset,
        }


class FrozenLinearProbe(nn.Module):
    """A frozen full backbone plus the only trainable 1x1 segmentation head."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.probe = nn.Conv2d(backbone.student.latent_dim, 1, kernel_size=1)

    def forward(self, inputs):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone.shared_bev(*inputs)
        return self.probe(features.detach().float())


def build_hybrid_teacher_model(device, seed, bevcar_source,
                               pretrained_camera=False):
    reset_seeds(seed)
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    voxelnet_class = official_voxelnet(Path(bevcar_source))
    radar_encoder = voxelnet_class(
        use_col=False, reduced_zx=False, output_dim=128,
        use_radar_occupancy_map=False,
    )
    student = Segnet(
        Z, Y, X, vox_util=vox_util,
        use_bevcar_encoder=True, bevcar_encoder=radar_encoder,
        do_rgbcompress=True, encoder_type='res101', rand_flip=False,
        pretrained_backbone=pretrained_camera,
    ).to(device)
    return HybridTeacherModel(student).to(device)


def configure_pretraining(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = (
        model.student.encoder,
        model.student.bevcar_encoder,
        model.student.bev_compressor,
        model.image_head,
        model.bev_head,
        model.motion_head,
    )
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.student.decoder.eval()
    return [parameter for parameter in model.parameters()
            if parameter.requires_grad]


def configure_end_to_end_finetuning(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.student.parameters():
        parameter.requires_grad_(True)
    return [parameter for parameter in model.student.parameters()
            if parameter.requires_grad]


def configure_linear_probe(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    probe_model = FrozenLinearProbe(model)
    return probe_model, list(probe_model.probe.parameters())


def prepare_inputs(batch, device):
    return prepare_bevcar_inputs(batch, device)


def image_distillation_loss(prediction, teacher_features):
    """Dense cosine loss for packed student cameras and BxS DINO maps."""
    if teacher_features.ndim != 5:
        raise ValueError('teacher image features must be (B,S,C,H,W)')
    batch, cameras, channels, height, width = teacher_features.shape
    if prediction.shape[0] != batch * cameras:
        raise ValueError('packed student image batch does not match DINO target')
    target = teacher_features.reshape(batch * cameras, channels, height, width)
    prediction = F.interpolate(
        prediction.float(), size=(height, width), mode='bilinear',
        align_corners=False,
    )
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target.float(), dim=1)
    return (1.0 - (prediction * target).sum(dim=1)).mean()


def label_tensors(batch, device):
    return {
        'segmentation': batch[12][:, 0].to(device, non_blocking=True),
        'valid': batch[13][:, 0].to(device, non_blocking=True),
        'center': batch[14][:, 0].to(device, non_blocking=True),
        'offset': batch[15][:, 0].to(device, non_blocking=True),
    }


def simplebev_per_frame_iou(logits, target, valid):
    """Exact Simple-BEV binary vehicle-IoU formula, one value per frame."""
    prediction = torch.sigmoid(logits).round()
    intersection = (prediction * target * valid).sum(dim=(1, 2, 3))
    union = ((prediction + target) * valid).clamp(0, 1).sum(dim=(1, 2, 3))
    return intersection / (1e-4 + union)


def binary_counts(logits, target, valid):
    prediction = torch.sigmoid(logits).ge(0.5)
    truth = target.gt(0.5)
    mask = valid.gt(0.5)
    return {
        'tp': int((prediction & truth & mask).sum()),
        'fp': int((prediction & ~truth & mask).sum()),
        'fn': int((~prediction & truth & mask).sum()),
        'tn': int((~prediction & ~truth & mask).sum()),
    }


def auxiliary_metrics_from_counts(counts):
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    return {
        'global_vehicle_iou': tp / max(tp + fp + fn, 1),
        'precision': tp / max(tp + fp, 1),
        'recall': tp / max(tp + fn, 1),
        'f1': 2 * tp / max(2 * tp + fp + fn, 1),
    }


def simplebev_downstream_loss(model, outputs, labels, segmentation_loss):
    """Reproduce the three supervised losses in train_nuscenes.run_model."""
    student = getattr(model, 'student', model)
    raw_ce = segmentation_loss(
        outputs['segmentation'], labels['segmentation'], labels['valid'],
    )
    raw_center = balanced_mse_loss(outputs['center'], labels['center'])
    raw_offset = torch.abs(outputs['offset'] - labels['offset']).sum(
        dim=1, keepdim=True,
    )
    raw_offset = utils.basic.reduce_masked_mean(
        raw_offset, labels['segmentation'] * labels['valid'],
    )

    ce_loss = 10.0 * raw_ce / torch.exp(student.ce_weight)
    center_loss = raw_center / (2.0 * torch.exp(student.center_weight))
    offset_loss = raw_offset / (2.0 * torch.exp(student.offset_weight))
    uncertainty = 0.5 * (
        student.ce_weight + student.center_weight + student.offset_weight
    )
    total = ce_loss + center_loss + offset_loss + uncertainty
    return total, {
        'segmentation_ce_loss': ce_loss,
        'center_loss': center_loss,
        'offset_loss': offset_loss,
        'uncertainty_regularization': uncertainty,
    }


def simplebev_segmentation_loss(device):
    return SimpleLoss(2.13).to(device)
