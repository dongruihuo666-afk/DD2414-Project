"""Frozen DINO teacher and calibrated BEV targets, extracted from the visualization script."""
from dd2414.data.batch import as_batch
from pathlib import Path
import torch
import torch.nn.functional as F
import utils.basic
import utils.geom
import utils.vox
from dd2414.config.geometry import Z,Y,X,bounds,scene_centroid,scene_centroid_py
TEACHER_HEIGHT = 196
TEACHER_WIDTH = 336
PATCH_SIZE = 14

def build_loaders(data_root, num_workers, nsweeps, rotate_radar_velocity=False,
                  dset='mini'):
    import nuscenesdataset
    if dset not in ('mini', 'trainval'):
        raise ValueError('dset must be mini or trainval')
    if nsweeps < 1:
        raise ValueError('nsweeps must be positive')
    data_aug_conf = {
        'crop_offset': 0,
        'resize_lim': [1.0, 1.0],
        'final_dim': (112, 192),
        'H': 900,
        'W': 1600,
        'cams': [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
        ],
        'ncams': 6,
    }
    train_loader, val_loader = nuscenesdataset.compile_data(
        dset, str(data_root), data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py, bounds=bounds, res_3d=(Z, Y, X),
        bsz=1, nworkers=num_workers, nworkers_val=num_workers,
        shuffle=False, nsweeps=nsweeps, seqlen=1, refcam_id=1,
        get_tids=True, temporal_aug=False, use_radar_filters=False,
        do_shuffle_cams=False,
        rotate_radar_velocity=rotate_radar_velocity,
    )
    return train_loader, val_loader


def build_loader(data_root, num_workers, nsweeps, rotate_radar_velocity=False,
                 split='train', dset='mini'):
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
    loaders = build_loaders(data_root, num_workers, nsweeps,
                            rotate_radar_velocity=rotate_radar_velocity, dset=dset)
    return loaders[0] if split == 'train' else loaders[1]


def load_teacher(model_name, device):
    hub_dir = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
    if hub_dir.exists():
        teacher = torch.hub.load(str(hub_dir), model_name, source='local')
    else:
        teacher = torch.hub.load('facebookresearch/dinov2', model_name)
    teacher.eval().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def extract_teacher_features(teacher, images, device):
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    features = []
    for camera in images[0]:
        resized = F.interpolate(
            camera[None].to(device), size=(TEACHER_HEIGHT, TEACHER_WIDTH),
            mode='bicubic', align_corners=False, antialias=True,
        )
        output = teacher.forward_features((resized - mean) / std)
        tokens = output['x_norm_patchtokens']
        feature = tokens.transpose(1, 2).reshape(
            1, tokens.shape[-1], TEACHER_HEIGHT // PATCH_SIZE,
            TEACHER_WIDTH // PATCH_SIZE,
        )
        features.append(feature)
    return torch.cat(features, dim=0)


def camera_geometry(batch, feature_height, feature_width, device):
    images = as_batch(batch).images[:, 0]
    rotations = as_batch(batch).camera_rotations[:, 0].to(device)
    translations = as_batch(batch).camera_translations[:, 0].to(device)
    intrinsics = as_batch(batch).intrinsics[:, 0].to(device)
    batch_size, cameras, _, image_height, image_width = images.shape
    assert batch_size == 1

    pack = lambda value: utils.basic.pack_seqdim(value, batch_size)
    unpack = lambda value: utils.basic.unpack_seqdim(value, batch_size)
    pixel_from_camera = utils.geom.merge_intrinsics(
        *utils.geom.split_intrinsics(pack(intrinsics))
    )
    pixel_from_camera = utils.geom.scale_intrinsics(
        pixel_from_camera,
        feature_width / image_width,
        feature_height / image_height,
    )
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    camera0_from_cameras = utils.geom.get_camM_T_camXs(
        vehicle_from_cameras, ind=0
    )
    cameras_from_camera0 = unpack(
        utils.geom.safe_inverse(pack(camera0_from_cameras))
    )
    cameras_from_vehicle = unpack(
        utils.geom.safe_inverse(pack(vehicle_from_cameras))
    )
    assert cameras_from_camera0.shape[1] == cameras
    return images, pixel_from_camera, cameras_from_camera0, cameras_from_vehicle


def project_teacher_to_bev(features, pixel_from_camera, cameras_from_camera0, device):
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    channels = features.shape[1]
    feature_sum = torch.zeros((1, channels, Z, Y, X), device=device)
    coverage_sum = torch.zeros((1, 1, Z, Y, X), device=device)
    xyz_mem = utils.basic.gridcloud3d(1, Z, Y, X, norm=False, device=device)
    xyz_camera0 = vox_util.Mem2Ref(xyz_mem, Z, Y, X, assert_cube=False)

    for camera_index in range(features.shape[0]):
        camera_feature = features[camera_index:camera_index + 1].float()
        camera_from_camera0 = cameras_from_camera0[:, camera_index]
        projection = utils.basic.matmul2(
            pixel_from_camera[camera_index:camera_index + 1],
            camera_from_camera0,
        )
        volume = vox_util.unproject_image_to_mem(
            camera_feature, projection, camera_from_camera0,
            Z, Y, X, assert_cube=False, xyz_camA=xyz_camera0,
        )
        valid = volume.abs().sum(dim=1, keepdim=True).gt(0).float()
        feature_sum.add_(volume)
        coverage_sum.add_(valid)
        del volume, valid

    mean_volume = feature_sum / coverage_sum.clamp_min(1.0)
    height_valid = coverage_sum.gt(0).float()
    teacher_bev = (mean_volume * height_valid).sum(dim=3)
    teacher_bev /= height_valid.sum(dim=3).clamp_min(1.0)
    bev_coverage = coverage_sum.sum(dim=3).squeeze(0).squeeze(0)
    return teacher_bev.squeeze(0), bev_coverage, vox_util


def make_radar_bev(batch, cameras_from_vehicle, vox_util, device):
    radar = as_batch(batch).radar[:, 0].to(device).permute(0, 2, 1)
    valid_points = radar[:, :, :3].abs().sum(dim=2).gt(0)
    radar_xyz_vehicle = radar[:, valid_points[0], :3]
    radar_camera0 = utils.geom.apply_4x4(
        cameras_from_vehicle[:, 0], radar_xyz_vehicle
    )
    radar_volume = vox_util.voxelize_xyz(
        radar_camera0, Z, Y, X, assert_cube=False
    )
    radar_bev = radar_volume[0, 0].sum(dim=1).clamp(0, 1)
    return radar_bev, radar_xyz_vehicle, radar_camera0


def radar_anchored_soft_targets(features, pixel_from_camera,
                                cameras_from_vehicle, radar_xyz_vehicle,
                                radar_camera0, vox_util):
    """Sample DINO at measured radar depths, then splat local soft BEV regions."""
    _, channels, height, width = features.shape
    point_feature_sum = torch.zeros(
        (radar_xyz_vehicle.shape[1], channels), device=features.device
    )
    point_view_count = torch.zeros(
        (radar_xyz_vehicle.shape[1], 1), device=features.device
    )

    for camera_index in range(features.shape[0]):
        xyz_camera = utils.geom.apply_4x4(
            cameras_from_vehicle[:, camera_index], radar_xyz_vehicle
        )
        xyz_pixel = utils.geom.apply_4x4(
            pixel_from_camera[camera_index:camera_index + 1], xyz_camera
        )
        depth = xyz_camera[0, :, 2]
        pixel_x = xyz_pixel[0, :, 0] / xyz_pixel[0, :, 2].clamp_min(1e-6)
        pixel_y = xyz_pixel[0, :, 1] / xyz_pixel[0, :, 2].clamp_min(1e-6)
        visible = (
            depth.gt(0.0)
            & pixel_x.gt(-0.5) & pixel_x.lt(width - 0.5)
            & pixel_y.gt(-0.5) & pixel_y.lt(height - 0.5)
        )
        normalized_y, normalized_x = utils.basic.normalize_grid2d(
            pixel_y[None], pixel_x[None], height, width
        )
        grid = torch.stack((normalized_x, normalized_y), dim=-1).view(1, 1, -1, 2)
        sampled = F.grid_sample(
            features[camera_index:camera_index + 1].float(), grid,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )[0, :, 0].transpose(0, 1)
        point_feature_sum[visible] += sampled[visible]
        point_view_count[visible] += 1.0

    point_features = point_feature_sum / point_view_count.clamp_min(1.0)
    point_features = F.normalize(point_features, dim=1)
    radar_mem = vox_util.Ref2Mem(radar_camera0, Z, Y, X, assert_cube=False)[0]
    soft_sum = torch.zeros((channels, Z, X), device=features.device)
    soft_weight = torch.zeros((Z, X), device=features.device)
    visible_points = point_view_count[:, 0].gt(0)

    # Vectorized soft splat: one Gaussian per visible point scattered onto the
    # BEV grid, avoiding the per-point GPU->CPU syncs of a Python loop.
    mem_x = radar_mem[:, 0]
    mem_z = radar_mem[:, 2]
    metric_range = torch.linalg.vector_norm(radar_camera0[0, :, (0, 2)], dim=1)
    sigma = 2.0 + metric_range / 25.0
    view_confidence = point_view_count[:, 0].clamp(max=2.0) / 2.0
    in_bounds = (
        (mem_x > -0.5) & (mem_x < X - 0.5) & (mem_z > -0.5) & (mem_z < Z - 0.5)
    )
    keep = visible_points & in_bounds
    if keep.any():
        mem_x = mem_x[keep]
        mem_z = mem_z[keep]
        sigma = sigma[keep]
        view_confidence = view_confidence[keep]
        kept_features = point_features[keep]
        grid_x = torch.arange(X, device=features.device, dtype=torch.float32)
        grid_z = torch.arange(Z, device=features.device, dtype=torch.float32)
        center_x = torch.round(mem_x)
        center_z = torch.round(mem_z)
        radius = torch.ceil(2.5 * sigma)
        dx = grid_x[None, None, :] - mem_x[:, None, None]
        dz = grid_z[None, :, None] - mem_z[:, None, None]
        gaussian = torch.exp(
            -(dx * dx + dz * dz) / (2.0 * sigma[:, None, None] ** 2)
        )
        mask = (
            (grid_x[None, None, :] >= center_x[:, None, None] - radius[:, None, None])
            & (grid_x[None, None, :] <= center_x[:, None, None] + radius[:, None, None])
            & (grid_z[None, :, None] >= center_z[:, None, None] - radius[:, None, None])
            & (grid_z[None, :, None] <= center_z[:, None, None] + radius[:, None, None])
        )
        gaussian = gaussian * mask * view_confidence[:, None, None]
        soft_sum = torch.einsum("kc,kzx->czx", kept_features, gaussian)
        soft_weight = gaussian.sum(dim=0)

    soft_targets = soft_sum / soft_weight.clamp_min(1e-6)[None]
    return soft_targets, soft_weight, point_view_count[:, 0]


def build_targets(batches, teacher, device):
    """Return (fusion_targets, confidences, image_targets) for a list of batches.

    ``image_targets`` are the raw DINOv2 patch features (S, 384, 14, 24); the
    fusion target is the radar-anchored soft BEV target (384, Z, X).
    """
    fusion_targets = []
    confidences = []
    image_targets = []
    with torch.inference_mode():
        for batch in batches:
            images = as_batch(batch).images[:, 0]
            features = extract_teacher_features(teacher, images, device)
            _, pixel_from_camera, cameras_from_camera0, cameras_from_vehicle = camera_geometry(
                batch, features.shape[-2], features.shape[-1], device
            )
            vox_util = utils.vox.Vox_util(
                Z, Y, X, scene_centroid=scene_centroid.to(device),
                bounds=bounds, assert_cube=False,
            )
            _, radar_vehicle, radar_camera0 = make_radar_bev(
                batch, cameras_from_vehicle, vox_util, device
            )
            target, confidence, _ = radar_anchored_soft_targets(
                features, pixel_from_camera, cameras_from_vehicle,
                radar_vehicle, radar_camera0, vox_util,
            )
            fusion_targets.append(target.half())
            confidences.append(confidence.half())
            image_targets.append(features.half())
            del features, vox_util
    return fusion_targets, confidences, image_targets
