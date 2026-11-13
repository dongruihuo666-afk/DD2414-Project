"""Existing frozen-camera research model; initialization and attributes are unchanged."""
from dd2414.data.batch import as_batch
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
import utils.basic
import utils.geom
import utils.vox
from nets.bevcar_voxel_adapter import prepare_bevcar_voxels
from nets.radar_encoder import transform_radar_to_camera_bev
from nets.segnet import Segnet
from dd2414.config.geometry import Z,Y,X,bounds,scene_centroid
from dd2414.models.distillation import SemanticDistillationModel

def official_voxelnet(source):
    path = source / 'nets' / 'voxelnet.py'
    if not path.is_file():
        raise FileNotFoundError(f'official BEVCar source missing: {path}')
    spec = importlib.util.spec_from_file_location('official_bevcar_voxelnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VoxelNet


class MotionBevcarModel(SemanticDistillationModel):
    """SemanticDistillationModel plus a 2-channel velocity head on the shared BEV."""

    def __init__(self, student):
        super().__init__(student, image_distill=False)
        self.motion_head = nn.Conv2d(student.latent_dim, 2, 1)

    def forward(self, rgb, pixel_from_cameras, camera0_from_cameras,
                vox_util, bevcar_voxels):
        shared_bev, _, image_features = self.student(
            rgb_camXs=rgb,
            pix_T_cams=pixel_from_cameras,
            cam0_T_camXs=camera0_from_cameras,
            vox_util=vox_util,
            bevcar_voxels=bevcar_voxels,
            return_shared_bev=True,
        )
        fusion_prediction = self.semantic_head(shared_bev)
        motion_prediction = self.motion_head(shared_bev)
        return fusion_prediction, motion_prediction


def build_model(device, seed, official_class, zero_camera=False):
    torch.manual_seed(seed)
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    bevcar_encoder = official_class(
        use_col=False, reduced_zx=False, output_dim=128,
        use_radar_occupancy_map=False,
    )
    student = Segnet(
        Z, Y, X, vox_util=vox_util,
        use_bevcar_encoder=True, bevcar_encoder=bevcar_encoder,
        do_rgbcompress=True, encoder_type='res101', rand_flip=False,
        pretrained_backbone=False, zero_camera_bev=zero_camera,
    ).to(device)
    model = MotionBevcarModel(student).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Trainable: the radar branch (VoxelNet + compressor) and both heads. The camera
    # encoder stays frozen so velocity must reach the motion head through the radar
    # branch, which is what the ablation tests.
    for parameter in model.student.bevcar_encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.student.bev_compressor.parameters():
        parameter.requires_grad_(True)
    for parameter in model.semantic_head.parameters():
        parameter.requires_grad_(True)
    for parameter in model.motion_head.parameters():
        parameter.requires_grad_(True)
    return model


def prepare_bevcar_inputs(batch, device):
    images = as_batch(batch).images[:, 0].float().to(device)
    rotations = as_batch(batch).camera_rotations[:, 0].to(device)
    translations = as_batch(batch).camera_translations[:, 0].to(device)
    intrinsics = as_batch(batch).intrinsics[:, 0].to(device)
    radar_vehicle = as_batch(batch).radar[:, 0].to(device).permute(0, 2, 1)
    batch_size = images.shape[0]
    pack = lambda value: utils.basic.pack_seqdim(value, batch_size)
    unpack = lambda value: utils.basic.unpack_seqdim(value, batch_size)

    pixel_from_cameras = unpack(utils.geom.merge_intrinsics(
        *utils.geom.split_intrinsics(pack(intrinsics))
    ))
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    cameras_from_vehicle = unpack(utils.geom.safe_inverse(pack(vehicle_from_cameras)))
    camera0_from_cameras = utils.geom.get_camM_T_camXs(vehicle_from_cameras, ind=0)

    # BEVCar expects XYZ and velocity already rotated into the reference-camera frame.
    radar_camera = transform_radar_to_camera_bev(radar_vehicle, cameras_from_vehicle[:, 0])
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    features, coords, counts, _ = prepare_bevcar_voxels(
        radar_camera, vox_util, quality_filter=False,
    )
    bevcar_voxels = (features, coords, counts)
    return (
        images - 0.5, pixel_from_cameras, camera0_from_cameras,
        vox_util, bevcar_voxels,
    )
