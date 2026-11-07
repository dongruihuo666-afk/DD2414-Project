#!/usr/bin/env python3
"""Acceptance tests and a presentation image for the minimal radar encoder."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import utils.basic  # noqa: E402
import utils.geom  # noqa: E402
from dinov2_bev_demo import build_loader  # noqa: E402
from nets.radar_encoder import (  # noqa: E402
    RadarPointEncoder, transform_radar_to_camera_bev,
)
from train_nuscenes import Z, Y, X, bounds, scene_centroid_py  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def camera_from_vehicle(batch, device):
    rotations = batch[1][:, 0].to(device)
    translations = batch[2][:, 0].to(device)
    batch_size = rotations.shape[0]
    pack = lambda value: utils.basic.pack_seqdim(value, batch_size)
    unpack = lambda value: utils.basic.unpack_seqdim(value, batch_size)
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    return unpack(utils.geom.safe_inverse(pack(vehicle_from_cameras)))[:, 0]


def make_synthetic(device):
    radar = torch.zeros((1, 4, 19), device=device)
    # First two points deliberately collide in one voxel.
    radar[0, 0, :3] = torch.tensor((2.0, 1.0, 12.0), device=device)
    radar[0, 1, :3] = torch.tensor((2.1, 1.0, 12.1), device=device)
    radar[0, 2, :3] = torch.tensor((-8.0, 1.0, 20.0), device=device)
    radar[0, :3, 5] = torch.tensor((2.0, 18.0, 8.0), device=device)
    radar[0, :3, 8:10] = torch.tensor(
        ((1.0, 0.0), (4.0, -2.0), (0.5, 1.0)), device=device
    )
    radar[0, :3, 10] = 1.0
    radar[0, :3, 11] = 3.0
    radar[0, :3, 14] = 0.0
    return radar


def camera_frame_transform_test(device):
    radar = torch.zeros((1, 2, 19), device=device)
    radar[0, 0, :3] = torch.tensor((1.0, 2.0, 3.0), device=device)
    radar[0, 0, 6:8] = torch.tensor((1.0, 0.0), device=device)
    radar[0, 0, 8:10] = torch.tensor((2.0, 0.0), device=device)
    transform = torch.eye(4, device=device).unsqueeze(0)
    transform[0, :3, :3] = torch.tensor(
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
        device=device,
    )
    transform[0, :3, 3] = torch.tensor((10.0, 20.0, 30.0), device=device)
    result = transform_radar_to_camera_bev(radar, transform)
    torch.testing.assert_close(
        result[0, 0, :3], torch.tensor((13.0, 22.0, 29.0), device=device)
    )
    torch.testing.assert_close(
        result[0, 0, 6:10], torch.tensor((0.0, -1.0, 0.0, -2.0), device=device)
    )
    assert torch.count_nonzero(result[0, 1]).item() == 0


def normalize_image(array):
    array = np.asarray(array, dtype=np.float32)
    positive = array[array > 0]
    high = np.percentile(positive, 99) if positive.size else 1.0
    return np.clip(array / max(float(high), 1e-8), 0, 1)


def heatmap(array):
    value = normalize_image(array)
    red = np.clip(2.2 * value, 0, 1)
    green = np.clip(2.2 * value - 0.55, 0, 1)
    blue = np.clip(2.2 * value - 1.25, 0, 1)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((360, 360), Image.Resampling.NEAREST)


def get_font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def render(raw_map, filtered_map, activation, diagnostics, output_path):
    canvas = Image.new('RGB', (1180, 570), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), 'Minimal radar point encoder: one real nuScenes mini frame',
              font=get_font(28, True), fill=(25, 30, 40))
    draw.text((24, 57),
              'No vehicle labels: 7 numeric fields -> point MLP -> voxel mean/max -> 64-channel BEV',
              font=get_font(16), fill=(65, 72, 84))
    panels = (
        ('1. All radar points in ROI', raw_map),
        ('2. Quality-filtered points', filtered_map),
        ('3. Learned BEV activation', activation),
    )
    for index, (title, value) in enumerate(panels):
        left = 24 + index * 385
        canvas.paste(heatmap(value), (left, 112))
        draw.rectangle((left, 112, left + 360, 472), outline=(120, 126, 136))
        draw.text((left, 83), title, font=get_font(17, True), fill=(25, 30, 40))
        draw.text((left + 8, 450), 'front', font=get_font(12), fill=(230, 235, 242))
    summary = (
        f"input {diagnostics['input_points']} points  |  retained {diagnostics['valid_points']}  |  "
        f"3D voxels {diagnostics['occupied_voxels']}  |  BEV cells {diagnostics['occupied_bev_cells']}"
    )
    draw.text((24, 500), summary, font=get_font(16, True), fill=(25, 30, 40))
    draw.text((24, 530),
              'Checks passed: shape, finite values, empty input, point-order invariance, collision pooling, gradients',
              font=get_font(15), fill=(28, 115, 75))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera_frame_transform_test(device)
    loader = build_loader(
        args.data_root, num_workers=args.num_workers, nsweeps=1,
        rotate_radar_velocity=True,
    )
    batch = next(iter(loader))
    radar_vehicle = batch[16][:, 0].to(device).permute(0, 2, 1)
    radar_camera = transform_radar_to_camera_bev(
        radar_vehicle, camera_from_vehicle(batch, device)
    )

    torch.manual_seed(13)
    encoder = RadarPointEncoder(
        Z=Z, Y=Y, X=X, bounds=bounds,
        scene_centroid=scene_centroid_py, out_channels=64,
        use_quality_mask=True,
    ).to(device).eval()
    bev, diagnostics = encoder(radar_camera, return_diagnostics=True)
    assert bev.shape == (1, 64, Z, X)
    assert torch.isfinite(bev).all()
    assert diagnostics['valid_points'].item() > 0
    assert diagnostics['occupied_voxels'].item() > 0
    assert diagnostics['occupied_bev_cells'].item() > 0

    empty_bev, empty_diagnostics = encoder(
        torch.zeros_like(radar_camera), return_diagnostics=True
    )
    assert torch.count_nonzero(empty_bev).item() == 0
    assert empty_diagnostics['valid_points'].item() == 0
    assert empty_diagnostics['occupied_voxels'].item() == 0

    permutation = torch.randperm(radar_camera.shape[1], device=device)
    shuffled_bev = encoder(radar_camera[:, permutation])
    permutation_error = (bev - shuffled_bev).abs().max().item()
    assert torch.allclose(bev, shuffled_bev, rtol=1e-5, atol=1e-5)

    synthetic = make_synthetic(device)
    synthetic_bev, synthetic_diagnostics = encoder(
        synthetic, return_diagnostics=True
    )
    assert synthetic_diagnostics['valid_points'].item() == 3
    assert synthetic_diagnostics['occupied_voxels'].item() == 2
    assert torch.isfinite(synthetic_bev).all()

    encoder.train()
    encoder.zero_grad(set_to_none=True)
    gradient_bev = encoder(radar_camera)
    gradient_bev.square().mean().backward()
    gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
    gradient_norm = torch.sqrt(sum(g.float().square().sum() for g in gradients))
    assert torch.isfinite(gradient_norm) and gradient_norm.item() > 0

    with torch.no_grad():
        memory = encoder._memory_coordinates(radar_camera[..., :3])
        rounded = memory.round().long()
        raw_valid = (
            radar_camera[..., :3].abs().sum(dim=-1).gt(0)
            & torch.isfinite(radar_camera).all(dim=-1)
            & (rounded >= 0).all(dim=-1)
            & (rounded <= encoder.grid_size_xyz.long() - 1).all(dim=-1)
        )
        quality_valid = encoder._valid_mask(radar_camera, memory)
        raw_map = torch.zeros((Z, X), device=device)
        filtered_map = torch.zeros((Z, X), device=device)
        for mask, target in ((raw_valid, raw_map), (quality_valid, filtered_map)):
            xyz_index = rounded[0, mask[0]]
            target.index_put_(
                (xyz_index[:, 2], xyz_index[:, 0]),
                torch.ones(xyz_index.shape[0], device=device), accumulate=True,
            )
        activation = bev[0].square().mean(dim=0).sqrt()

    values = {key: int(value.item()) for key, value in diagnostics.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / 'radar_point_encoder_test.png'
    snapshot_path = args.output_dir / 'radar_point_encoder_test.npz'
    render(
        raw_map.cpu().numpy(), filtered_map.cpu().numpy(),
        activation.detach().cpu().numpy(), values, image_path,
    )
    np.savez_compressed(
        snapshot_path,
        bev_activation=activation.detach().cpu().numpy(),
        raw_occupancy=raw_map.cpu().numpy(),
        filtered_occupancy=filtered_map.cpu().numpy(),
        permutation_max_error=np.asarray(permutation_error),
        gradient_norm=np.asarray(gradient_norm.item()),
        **{key: np.asarray(value) for key, value in values.items()},
    )

    print(f'device: {device}')
    print('camera_frame_transform: OK')
    print(f'output_shape: {tuple(bev.shape)}')
    for key, value in values.items():
        print(f'{key}: {value}')
    print(f'permutation_max_error: {permutation_error:.9g}')
    print(f'collision_test_occupied_voxels: {synthetic_diagnostics["occupied_voxels"].item()}')
    print(f'gradient_norm: {gradient_norm.item():.9g}')
    print(f'visualization: {image_path}')
    print(f'snapshot: {snapshot_path}')
    print('RADAR_POINT_ENCODER_TEST_OK')


if __name__ == '__main__':
    main()
