#!/usr/bin/env python3
"""Visualize frozen DINOv2 features projected into the Simple-BEV frame."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import nuscenesdataset  # noqa: E402
import utils.basic  # noqa: E402
import utils.geom  # noqa: E402
import utils.vox  # noqa: E402
from train_nuscenes import Z, Y, X, bounds, scene_centroid, scene_centroid_py  # noqa: E402


CAMERA_NAMES = (
    'front (BEV reference)', 'front-left', 'front-right',
    'back-left', 'back', 'back-right',
)
TEACHER_HEIGHT = 196
TEACHER_WIDTH = 336
PATCH_SIZE = 14
BACKGROUND = (245, 247, 250)
TEXT = (25, 30, 40)


def font(size, bold=False):
    candidates = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'
        if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--nsweeps', type=int, default=1)
    parser.add_argument('--model-name', default='dinov2_vits14')
    return parser.parse_args()


def build_loader(data_root, num_workers, nsweeps, rotate_radar_velocity=False,
                 split='train'):
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
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
        'mini', str(data_root), data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py, bounds=bounds, res_3d=(Z, Y, X),
        bsz=1, nworkers=num_workers, nworkers_val=num_workers,
        shuffle=False, nsweeps=nsweeps, seqlen=1, refcam_id=1,
        get_tids=True, temporal_aug=False, use_radar_filters=False,
        do_shuffle_cams=False,
        rotate_radar_velocity=rotate_radar_velocity,
    )
    return train_loader if split == 'train' else val_loader


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
    images = batch[0][:, 0]
    rotations = batch[1][:, 0].to(device)
    translations = batch[2][:, 0].to(device)
    intrinsics = batch[3][:, 0].to(device)
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
    radar = batch[16][:, 0].to(device).permute(0, 2, 1)
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

    for point_index in torch.where(visible_points)[0].tolist():
        mem_x = float(radar_mem[point_index, 0])
        mem_z = float(radar_mem[point_index, 2])
        if not (-0.5 < mem_x < X - 0.5 and -0.5 < mem_z < Z - 0.5):
            continue
        metric_range = float(torch.linalg.vector_norm(
            radar_camera0[0, point_index, (0, 2)]
        ))
        # BEV cells are 0.5 m. The uncertainty region grows gently with range.
        sigma = 2.0 + metric_range / 25.0
        radius = int(np.ceil(2.5 * sigma))
        center_x, center_z = int(round(mem_x)), int(round(mem_z))
        x0, x1 = max(0, center_x - radius), min(X, center_x + radius + 1)
        z0, z1 = max(0, center_z - radius), min(Z, center_z + radius + 1)
        grid_z = torch.arange(z0, z1, device=features.device)[:, None]
        grid_x = torch.arange(x0, x1, device=features.device)[None, :]
        gaussian = torch.exp(
            -((grid_x - mem_x) ** 2 + (grid_z - mem_z) ** 2)
            / (2.0 * sigma ** 2)
        )
        view_confidence = min(float(point_view_count[point_index, 0]), 2.0) / 2.0
        gaussian *= view_confidence
        soft_sum[:, z0:z1, x0:x1] += (
            point_features[point_index, :, None, None] * gaussian
        )
        soft_weight[z0:z1, x0:x1] += gaussian

    soft_targets = soft_sum / soft_weight.clamp_min(1e-6)[None]
    return soft_targets, soft_weight, point_view_count[:, 0]


def pca_colors(camera_features, teacher_bev, coverage, soft_targets, soft_weight):
    camera_tokens = camera_features.permute(0, 2, 3, 1).reshape(
        -1, camera_features.shape[1]
    ).float().cpu().numpy()
    center = camera_tokens.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(camera_tokens - center, full_matrices=False)
    basis = vh[:3].T

    camera_rgb = ((camera_tokens - center) @ basis).reshape(
        camera_features.shape[0], camera_features.shape[2],
        camera_features.shape[3], 3,
    )
    bev_tokens = teacher_bev.permute(1, 2, 0).float().cpu().numpy()
    bev_rgb = (bev_tokens.reshape(-1, bev_tokens.shape[-1]) - center) @ basis
    bev_rgb = bev_rgb.reshape(Z, X, 3)
    soft_tokens = soft_targets.permute(1, 2, 0).float().cpu().numpy()
    soft_rgb = (soft_tokens.reshape(-1, soft_tokens.shape[-1]) - center) @ basis
    soft_rgb = soft_rgb.reshape(Z, X, 3)

    valid_bev = coverage.cpu().numpy() > 0
    valid_soft = soft_weight.cpu().numpy() > 0.05
    scale_values = np.concatenate(
        [camera_rgb.reshape(-1, 3), bev_rgb[valid_bev], soft_rgb[valid_soft]], axis=0
    )
    low = np.percentile(scale_values, 1, axis=0)
    high = np.percentile(scale_values, 99, axis=0)

    def scale(values):
        values = (values - low) / np.maximum(high - low, 1e-6)
        return (np.clip(values, 0, 1) * 255).astype(np.uint8)

    camera_rgb = scale(camera_rgb)
    bev_rgb = scale(bev_rgb)
    soft_rgb = scale(soft_rgb)
    bev_rgb[~valid_bev] = 0
    soft_rgb[~valid_soft] = 0
    return camera_rgb, bev_rgb, soft_rgb


def image_array(images):
    return images[0].clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).numpy()


def heatmap(values, color=(255, 180, 0)):
    values = values.astype(np.float32)
    values /= max(float(values.max()), 1e-6)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    for channel, amount in enumerate(color):
        rgb[..., channel] = (values * amount).astype(np.uint8)
    return rgb


def overlay_radar(base, radar):
    result = base.copy()
    mask = radar > 0
    expanded = mask.copy()
    expanded[:-1] |= mask[1:]
    expanded[1:] |= mask[:-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[:, 1:] |= mask[:, :-1]
    result[expanded] = (0, 235, 255)
    result[mask] = (255, 255, 255)
    return result


def render_board(images, camera_rgb, bev_rgb, coverage, radar, soft_rgb,
                 soft_weight, ground_truth, visible_radar_points,
                 model_name, extraction_seconds, projection_seconds, peak_gib,
                 output_path):
    margin = 24
    panel_width, panel_height = 360, 210
    width = 3 * panel_width + 4 * margin
    bottom_size = 172
    height = 120 + 2 * (panel_height + 42) + 80 + bottom_size + 115
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title = font(29, bold=True)
    heading = font(18, bold=True)
    body = font(15)
    small = font(13)

    draw.text(
        (margin, 18), 'Self-supervised prototype: frozen DINOv2 semantics -> BEV',
        font=title, fill=TEXT,
    )
    draw.text(
        (margin, 60),
        'Six camera feature maps are projected with Simple-BEV geometry; no 3D box label is used by the teacher.',
        font=body, fill=(70, 78, 92),
    )
    draw.text(
        (margin, 84), 'Color = DINO principal components | cyan/white = radar returns | green = GT sanity check',
        font=small, fill=(70, 78, 92),
    )

    for index in range(6):
        row, col = divmod(index, 3)
        x = margin + col * (panel_width + margin)
        y = 120 + row * (panel_height + 42)
        original = Image.fromarray(images[index]).resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        semantics = Image.fromarray(camera_rgb[index]).resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        blended = Image.blend(original, semantics, 0.48)
        canvas.paste(blended, (x, y))
        draw.rectangle((x, y, x + panel_width, y + 25), fill=(0, 0, 0))
        draw.text((x + 7, y + 4), CAMERA_NAMES[index], font=small, fill='white')
        draw.text((x, y + panel_height + 8), 'image + frozen DINO patch features', font=small, fill=TEXT)

    coverage_rgb = heatmap(coverage)
    soft_confidence_rgb = heatmap(soft_weight, color=(255, 180, 0))
    gt_rgb = np.zeros((Z, X, 3), dtype=np.uint8)
    gt_rgb[ground_truth > 0.5] = (0, 255, 80)
    panels = (
        ('Naive DINO rays', bev_rgb),
        ('Ray coverage', coverage_rgb),
        ('Radar anchors', heatmap(radar, color=(0, 235, 255))),
        ('Soft confidence', soft_confidence_rgb),
        ('Soft DINO target', overlay_radar(soft_rgb, radar)),
        ('GT (view only)', gt_rgb),
    )
    panel_gap = (width - 2 * margin - 6 * bottom_size) // 5
    panel_y = 120 + 2 * (panel_height + 42) + 55
    for index, (label, values) in enumerate(panels):
        x = margin + index * (bottom_size + panel_gap)
        draw.text((x, panel_y - 28), label, font=heading, fill=TEXT)
        panel = Image.fromarray(values).resize(
            (bottom_size, bottom_size), Image.Resampling.NEAREST
        )
        canvas.paste(panel, (x, panel_y))

    stats_y = panel_y + bottom_size + 19
    draw.text(
        (margin, stats_y),
        f'{model_name} (frozen) | camera tokens: 6 x 384 x 14 x 24 | teacher BEV: 384 x 200 x 200',
        font=body, fill=TEXT,
    )
    draw.text(
        (margin, stats_y + 25),
        f'extract {extraction_seconds:.2f}s | geometry + soft targets {projection_seconds:.2f}s | peak VRAM {peak_gib:.2f} GiB | visible radar anchors {visible_radar_points}',
        font=body, fill=TEXT,
    )
    draw.text(
        (margin, stats_y + 52),
        'Naive rays expose depth ambiguity; radar-anchored soft regions are the actual distillation targets. No box label is used.',
        font=small, fill=(145, 65, 35),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    required = ('samples', 'sweeps', 'maps', 'v1.0-mini')
    missing = [name for name in required if not (args.data_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f'nuScenes mini is incomplete at {args.data_root}; missing: {missing}'
        )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run where WSL GPU access is allowed')

    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loader = build_loader(args.data_root, args.num_workers, args.nsweeps)
    batch = next(iter(loader))
    images = batch[0][:, 0]

    teacher = load_teacher(args.model_name, device)
    teacher_params = sum(parameter.numel() for parameter in teacher.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in teacher.parameters()
        if parameter.requires_grad
    )
    print(f'teacher_parameters: {teacher_params:,}')
    print(f'teacher_trainable_parameters: {trainable_params:,}')

    with torch.inference_mode():
        start = time.perf_counter()
        camera_features = extract_teacher_features(teacher, images, device)
        torch.cuda.synchronize()
        extraction_seconds = time.perf_counter() - start

        feature_height, feature_width = camera_features.shape[-2:]
        images_cpu, pixel_from_camera, cameras_from_camera0, cameras_from_vehicle = camera_geometry(
            batch, feature_height, feature_width, device
        )
        start = time.perf_counter()
        teacher_bev, coverage, vox_util = project_teacher_to_bev(
            camera_features, pixel_from_camera, cameras_from_camera0, device
        )
        radar_bev, radar_xyz_vehicle, radar_camera0 = make_radar_bev(
            batch, cameras_from_vehicle, vox_util, device
        )
        soft_targets, soft_weight, radar_view_count = radar_anchored_soft_targets(
            camera_features, pixel_from_camera, cameras_from_vehicle,
            radar_xyz_vehicle, radar_camera0, vox_util,
        )
        torch.cuda.synchronize()
        projection_seconds = time.perf_counter() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / 'dinov2_teacher_features.npz'
    np.savez_compressed(
        cache_path,
        features=camera_features.half().cpu().numpy(),
        intrinsics=batch[3][0, 0].numpy(),
        rotations=batch[1][0, 0].numpy(),
        translations=batch[2][0, 0].numpy(),
        model_name=np.asarray(args.model_name),
    )
    soft_cache_path = args.output_dir / 'dinov2_radar_soft_targets.npz'
    np.savez_compressed(
        soft_cache_path,
        semantic_target=soft_targets.half().cpu().numpy(),
        confidence=soft_weight.half().cpu().numpy(),
        radar_view_count=radar_view_count.byte().cpu().numpy(),
        model_name=np.asarray(args.model_name),
    )

    camera_rgb, bev_rgb, soft_rgb = pca_colors(
        camera_features, teacher_bev, coverage, soft_targets, soft_weight
    )
    rgb_images = image_array(images_cpu)
    ground_truth = batch[12][0, 0, 0].numpy()
    coverage_np = coverage.cpu().numpy()
    radar_np = radar_bev.cpu().numpy()
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    output_path = args.output_dir / 'dinov2_bev_correspondence.png'
    render_board(
        rgb_images, camera_rgb, bev_rgb, coverage_np, radar_np,
        soft_rgb, soft_weight.cpu().numpy(), ground_truth,
        int((radar_view_count > 0).sum().item()), args.model_name, extraction_seconds,
        projection_seconds, peak_gib, output_path,
    )

    valid_radar = int((batch[16][0, 0, :3].abs().sum(dim=0) > 0).sum())
    covered_cells = int((coverage > 0).sum().item())
    print(f'teacher_feature_shape: {tuple(camera_features.shape)}')
    print(f'teacher_bev_shape: {tuple(teacher_bev.shape)}')
    print(f'covered_bev_cells: {covered_cells}/{Z * X}')
    print(f'valid_radar_points: {valid_radar}')
    print(f'camera_visible_radar_points: {int((radar_view_count > 0).sum().item())}')
    print(f'soft_target_cells: {int((soft_weight > 0.05).sum().item())}/{Z * X}')
    print(f'extraction_seconds: {extraction_seconds:.3f}')
    print(f'projection_seconds: {projection_seconds:.3f}')
    print(f'peak_cuda_memory_gib: {peak_gib:.3f}')
    print(f'feature_cache: {cache_path}')
    print(f'soft_target_cache: {soft_cache_path}')
    print(f'visualization: {output_path}')
    print('DINOV2_BEV_DEMO_OK')


if __name__ == '__main__':
    main()
