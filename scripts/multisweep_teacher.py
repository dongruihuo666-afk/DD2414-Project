"""Opt-in baseline double distillation, depth-aware lifting and object readout.

No changes to Segnet or the official Simple-BEV execution/checkpoint path.
The depth proxy only wraps this experiment's Vox_util instance.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from baseline_followup import ROOT, digest_file
from nets.bevcar_voxel_adapter import prepare_bevcar_voxels
from nets.bevcar_radar_bridge import official_voxelnet
from nets.radar_encoder import transform_radar_to_camera_bev
from nets.segnet import Segnet
from train_fullsize_linear_probe import reset_seeds
from train_nuscenes import Z, Y, X, bounds, scene_centroid
import utils.basic
import utils.geom
import utils.vox


def prepare_bevcar_inputs(batch,device):
    """Legacy transforms; common 7,000-voxel safety bound for 5/10 sweeps.

    VoxelNet uses dynamic occupied counts. The old 3,500 guard was safe for
    at most 700*5 raw points but could reject a dense ten-sweep frame. This
    bound admits every possible ten-sweep input without truncating evidence.
    It changes no VoxelNet parameter or spatial feature resolution.
    """
    images = batch[0][:,0].float().to(device)
    rotations = batch[1][:,0].to(device)
    translations = batch[2][:,0].to(device)
    intrinsics = batch[3][:,0].to(device)
    radar_vehicle = batch[16][:,0].to(device).permute(0,2,1)
    b = images.shape[0]
    pack = lambda value: utils.basic.pack_seqdim(value,b)
    unpack = lambda value: utils.basic.unpack_seqdim(value,b)
    pixels = unpack(utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(pack(intrinsics))))
    ego_from_cameras = utils.geom.merge_rtlist(rotations,translations)
    cameras_from_ego = unpack(utils.geom.safe_inverse(pack(ego_from_cameras)))
    cameras = utils.geom.get_camM_T_camXs(ego_from_cameras,ind=0)
    radar_camera = transform_radar_to_camera_bev(radar_vehicle,cameras_from_ego[:,0])
    vox = utils.vox.Vox_util(Z,Y,X,scene_centroid=scene_centroid.to(device),
                             bounds=bounds,assert_cube=False)
    features,coords,counts,_ = prepare_bevcar_voxels(radar_camera,vox,
        max_voxels=7000,quality_filter=False)
    return images-.5,pixels,cameras,vox,(features,coords,counts)


def image_cosine_loss(prediction, target):
    target = target.flatten(0, 1).float()
    prediction = F.interpolate(prediction.float(), target.shape[-2:],
                               mode='bilinear', align_corners=False)
    return (1 - (F.normalize(prediction, dim=1) *
                 F.normalize(target, dim=1)).sum(1)).mean()


def depth_gate(log_predicted_depth, camera_depth, log_sigma=.6):
    """Soft metric depth likelihood; no hard argmax or GT depth at inference."""
    return torch.exp(-.5*((camera_depth.clamp_min(.1).log() -
                          log_predicted_depth)/log_sigma).square())


class DepthAwareVox:
    def __init__(self, base, head):
        self.base, self.head = base, head
        self.prediction = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def unproject_image_to_mem(self, features, pixels, cameras, z, y, x,
                               assert_cube=False, xyz_camA=None):
        # The head predicts log metric camera-Z. Initialize to 20 m, and bound
        # predictions to [0.5, 120] m without discontinuous clipping gradients.
        logits = self.head(features).float()
        self.prediction = math.log(.5) + torch.sigmoid(logits)*math.log(240.)
        packed = torch.cat((features.float(), self.prediction), 1)
        volume = self.base.unproject_image_to_mem(packed, pixels, cameras,
                    z, y, x, assert_cube=assert_cube, xyz_camA=xyz_camA)
        if xyz_camA is None:
            xyz_camA = self.base.Mem2Ref(utils.basic.gridcloud3d(
                features.shape[0], z, y, x, norm=False, device=features.device),
                z, y, x, assert_cube=assert_cube)
        camera_z = utils.geom.apply_4x4(cameras.float(), xyz_camA.float())[..., 2]
        gate = depth_gate(volume[:, -1:], camera_z.reshape(-1, 1, z, y, x))
        return volume[:, :-1] * gate


class MultiSweepTeacher(nn.Module):
    def __init__(self, student, depth_mode='none'):
        super().__init__()
        self.student, self.depth_mode = student, depth_mode
        self.image_head = nn.Conv2d(128, 384, 1)
        self.bev_head = nn.Conv2d(128, 384, 1)
        self.motion_head = nn.Conv2d(128, 2, 1)
        self.depth_head = nn.Conv2d(128, 1, 1)
        nn.init.zeros_(self.depth_head.weight)
        probability = math.log(40.)/math.log(240.)
        nn.init.constant_(self.depth_head.bias, math.log(probability/(1-probability)))

    def features(self, inputs):
        images, pixels, cameras, vox, radar = inputs
        depth_vox = DepthAwareVox(vox, self.depth_head) if self.depth_mode != 'none' else vox
        shared, _, images_features = self.student(rgb_camXs=images,
            pix_T_cams=pixels, cam0_T_camXs=cameras, vox_util=depth_vox,
            bevcar_voxels=radar, return_shared_bev=True)
        return shared, images_features, getattr(depth_vox, 'prediction', None)


def build_model(device, seed, bevcar_source, depth_mode):
    reset_seeds(seed)
    vox = utils.vox.Vox_util(Z, Y, X, scene_centroid=scene_centroid.to(device),
                            bounds=bounds, assert_cube=False)
    cls = official_voxelnet(bevcar_source)
    radar = cls(use_col=False, reduced_zx=False, output_dim=128,
                use_radar_occupancy_map=False)
    student = Segnet(Z, Y, X, vox_util=vox, use_bevcar_encoder=True,
        bevcar_encoder=radar, do_rgbcompress=True, encoder_type='res101',
        pretrained_backbone=False, rand_flip=False)
    return MultiSweepTeacher(student, depth_mode).to(device)


def configure(model, mode):
    model.requires_grad_(False)
    if mode == 'pretrain':
        modules = [model.student.encoder, model.student.bevcar_encoder,
                   model.student.bev_compressor, model.image_head, model.bev_head,
                   model.motion_head]
        if model.depth_mode != 'none':
            modules.append(model.depth_head)
    elif mode == 'finetune':
        modules = [model.student]
        if model.depth_mode != 'none':
            modules.append(model.depth_head)
    else:
        modules = []
    for module in modules:
        module.requires_grad_(True)


def projected_radar_depth(batch, height, width, device, max_age=.1):
    """Camera-Z anchors, using only recent returns to limit moving-object ghosts.

    Multi-sweep encoder input is unchanged. Supervision uses compensated ego
    coordinates and per-camera calibration; no box/segmentation labels enter.
    """
    images = batch[0][:, 0]
    b, s = images.shape[:2]
    rots, trans, intrinsics = (batch[i][:, 0].to(device).float() for i in [1, 2, 3])
    camera_from_ego = utils.geom.safe_inverse(
        utils.geom.merge_rtlist(rots, trans).flatten(0, 1)).reshape(b, s, 4, 4)
    radar = batch[16][:, 0].to(device).float().transpose(1, 2)
    targets = []
    for bi in range(b):
        valid = (radar[bi, :, :3].abs().sum(-1) > 0) & torch.isfinite(radar[bi]).all(-1)
        # Last field is relative sensor timestamp in seconds, not a feature ID.
        valid &= radar[bi, :, 18].abs() <= max_age
        xyz = radar[bi:bi+1, valid, :3]
        for si in range(s):
            points = utils.geom.apply_4x4(camera_from_ego[bi:bi+1, si], xyz)[0]
            # Dataset batches store homogeneous 4x4 intrinsics; synthetic
            # callers may use 3x3. Only the calibrated camera K block is needed.
            pixel = points @ intrinsics[bi, si, :3, :3].T
            uv = pixel[:, :2]/pixel[:, 2:3].clamp_min(1e-5)
            uv = uv * uv.new_tensor([width/images.shape[-1], height/images.shape[-2]])
            z = points[:, 2]
            keep = (z > 1) & (z < 100) & (uv[:, 0] >= 0) & (uv[:, 0] < width-1)
            keep &= (uv[:, 1] >= 0) & (uv[:, 1] < height-1)
            targets.append((uv[keep], z[keep]))
    return targets


def recent_radar_target_batch(batch, max_age=.1):
    """Keep teacher targets identical between 5/10-sweep input experiments."""
    output = list(batch)
    radar = batch[16].clone()
    recent = radar[:,:,18:19].abs() <= max_age
    output[16] = torch.where(recent,radar,torch.zeros_like(radar))
    return tuple(output)


def sparse_depth_loss(log_depth, anchors):
    """Per-camera log-Z SmoothL1; empty coverage is a graph-connected zero."""
    losses, absolute, squared, relative = [], [], [], []
    for index, (uv, z) in enumerate(anchors):
        if not len(z):
            continue
        h, w = log_depth.shape[-2:]
        grid = uv * uv.new_tensor([2/(w-1), 2/(h-1)])-1
        values = F.grid_sample(log_depth[index:index+1].float(),
            grid[None, None], align_corners=True).flatten()
        losses.append(F.smooth_l1_loss(values, z.log()))
        error = values.detach().exp()-z
        absolute.append(error.abs())
        squared.append(error.square())
        relative.append(error.abs()/z)
    if not losses:
        return log_depth.sum()*0, dict(depth_anchors=0, depth_mae_m=None,
                                      depth_rmse_m=None, depth_abs_rel=None)
    return torch.stack(losses).mean(), dict(depth_anchors=sum(len(v) for v in absolute),
        depth_mae_m=float(torch.cat(absolute).mean()),
        depth_rmse_m=float(torch.cat(squared).mean().sqrt()),
        depth_abs_rel=float(torch.cat(relative).mean()))


def load_depth_teacher(device):
    source = ROOT/'.cache/Depth-Anything-V2'
    weights = ROOT/'.cache/depth_anything_v2_vits.pth'
    if not (source/'depth_anything_v2/dpt.py').is_file() or not weights.is_file():
        raise FileNotFoundError('Prepare the official Depth Anything V2 Small source/weights cache first')
    sys.path.insert(0, str(source))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def relative_depth_target(teacher, images, device):
    # Preserve the wide-camera aspect ratio, dimensions divisible by ViT patch 14.
    inputs = F.interpolate(images.flatten(0, 1).to(device), (196, 336),
                           mode='bilinear', align_corners=False)
    mean = inputs.new_tensor([.485,.456,.406])[None,:,None,None]
    std = inputs.new_tensor([.229,.224,.225])[None,:,None,None]
    return teacher((inputs-mean)/std).detach().float()


def relative_depth_loss(log_depth, target):
    """Scale/shift-invariant inverse-depth shape, NEVER treat DAV2 as meters."""
    prediction = (-log_depth.float()).exp()[:, 0]
    target = F.interpolate(target[:,None], prediction.shape[-2:], mode='bilinear',
                           align_corners=False)[:,0]
    def normalize(value):
        flat = value.flatten(1)
        centered = flat-flat.median(1, keepdim=True).values
        return centered/centered.abs().mean(1, keepdim=True).clamp_min(1e-5)
    return (normalize(prediction)-normalize(target)).abs().mean()


def object_targets(dataset, token, vox, device):
    """GT-location readout targets: centers/box extents and independent box velocity.

    Positions are used ONLY by this supervised diagnostic readout. Velocities
    are temporal nuScenes annotation derivatives, never radar-derived targets.
    """
    nusc = dataset.nusc
    rec = nusc.get('sample', token)
    lidar = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    ego = nusc.get('ego_pose', lidar['ego_pose_token'])
    cam_sd = nusc.get('sample_data', rec['data']['CAM_FRONT'])
    cam = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
    camera_from_global = (transform_matrix(cam['translation'], Quaternion(cam['rotation']), inverse=True)
        @ transform_matrix(ego['translation'], Quaternion(ego['rotation']), inverse=True))
    records = []
    for tok in rec['anns']:
        annotation = nusc.get('sample_annotation', tok)
        if 'vehicle' not in annotation['category_name'] or int(annotation['visibility_token']) == 1:
            continue
        velocity = nusc.box_velocity(tok)
        if not np.isfinite(velocity).all():
            continue
        center = camera_from_global[:3,:3] @ np.asarray(annotation['translation']) + camera_from_global[:3,3]
        vel_camera = camera_from_global[:3,:3] @ velocity
        if not (-50 < center[0] < 50 and -50 < center[2] < 50):
            continue
        # Oriented BEV ROI grid using the GT box's longitudinal/lateral axes.
        rotation = camera_from_global[:3,:3] @ Quaternion(annotation['rotation']).rotation_matrix
        extent = np.asarray(annotation['size'])[[1,0,2]]
        samples = np.array([[u*extent[0], v*extent[1], 0]
                            for u in [-.4,0,.4] for v in [-.4,0,.4]])
        points = samples @ rotation.T + center
        memory = vox.Ref2Mem(torch.tensor(points, dtype=torch.float32, device=device)[None],
                            Z,Y,X, assert_cube=False)[0]
        grid = memory[:, [0,2]] * memory.new_tensor([2/(X-1),2/(Z-1)])-1
        records.append(dict(annotation_token=tok, grid=grid,
            velocity=torch.tensor(vel_camera[[0,2]], dtype=torch.float32, device=device),
            speed=float(np.linalg.norm(velocity[:2])), center=center[[0,2]].tolist()))
    return records


def pool_objects(features, records):
    if not records:
        return features.new_empty((0, features.shape[1]))
    grid = torch.stack([record['grid'] for record in records])[None]
    # One GT object = nine oriented interior samples; no velocity in input.
    return F.grid_sample(features.float(), grid, align_corners=True)[0].mean(-1).T


class Downstream(nn.Module):
    def __init__(self, backbone, mode):
        super().__init__()
        self.backbone, self.mode = backbone, mode
        configure(backbone, mode)
        self.probe = nn.Conv2d(128, 1, 1) if mode in ['frozen-probe','unfrozen-probe'] else None
        self.motion = nn.Linear(128, 2) if mode == 'object-motion' else None
        if mode == 'unfrozen-probe':
            for module in [backbone.student.encoder, backbone.student.bevcar_encoder,
                           backbone.student.bev_compressor]:
                module.requires_grad_(True)
            if backbone.depth_mode != 'none':
                backbone.depth_head.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.mode in ['frozen-probe','object-motion']:
            self.backbone.eval()
        elif self.mode != 'finetune':
            self.backbone.student.decoder.eval()
        return self

    def forward(self, inputs):
        if self.mode in ['frozen-probe','object-motion']:
            with torch.no_grad():
                shared, image, depth = self.backbone.features(inputs)
        else:
            shared, image, depth = self.backbone.features(inputs)
        if self.mode == 'finetune':
            outputs = self.backbone.student.decoder(shared, None)
            return (outputs['raw_feat'], outputs['feat'], outputs['segmentation'],
                    outputs['instance_center'], outputs['instance_offset'])
        return self.probe(shared.float()) if self.probe is not None else shared.detach().float()
