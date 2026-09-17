#!/usr/bin/env python3
"""One-frame geometry audit for the optional BEVCar-shaped radar input."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import nuscenesdataset  # noqa: E402
import utils.vox  # noqa: E402
from nets.bevcar_voxel_adapter import (  # noqa: E402
    FEATURE_NAMES, prepare_bevcar_voxels,
)
from nets.radar_encoder import (  # noqa: E402
    RadarPointEncoder, transform_radar_to_camera_bev,
)
from train_nuscenes import Z, Y, X, bounds, scene_centroid, scene_centroid_py  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    return parser.parse_args()


def make_vox_util():
    return utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid,
        bounds=bounds, assert_cube=False,
    )


def first_mini_train_radar(data_root):
    """Read the same first training key frame without the GPU-only VizData."""
    standard_root = data_root
    legacy_root = data_root / 'mini'
    nusc_root = standard_root if (standard_root / 'v1.0-mini').is_dir() else legacy_root
    nusc = NuScenes(version='v1.0-mini', dataroot=str(nusc_root), verbose=False)
    train_scenes = set(create_splits_scenes()['mini_train'])
    samples = [
        sample for sample in nusc.sample
        if nusc.get('scene', sample['scene_token'])['name'] in train_scenes
    ]
    samples.sort(key=lambda sample: (sample['scene_token'], sample['timestamp']))
    sample = samples[0]
    radar = nuscenesdataset.get_radar_data(
        nusc, sample, nsweeps=1, min_distance=2.2,
        use_radar_filters=False, dataroot=str(nusc_root),
        rotate_radar_velocity=True,
    )
    camera_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    camera_calibration = nusc.get(
        'calibrated_sensor', camera_data['calibrated_sensor_token']
    )
    vehicle_from_camera = transform_matrix(
        camera_calibration['translation'],
        Quaternion(camera_calibration['rotation']), inverse=False,
    )
    camera_from_vehicle = torch.from_numpy(
        np.linalg.inv(vehicle_from_camera)
    ).float()[None]
    radar_vehicle = torch.from_numpy(radar.T.copy()).float()[None]
    radar_camera = transform_radar_to_camera_bev(
        radar_vehicle, camera_from_vehicle,
    )
    return radar_camera, sample['token']


def synthetic_checks(vox_util):
    radar = torch.zeros((1, 5, 19))
    radar[0, 0, :3] = torch.tensor((2.0, 1.0, 12.0))
    radar[0, 1, :3] = torch.tensor((2.1, 1.0, 12.1))
    radar[0, 2, :3] = torch.tensor((-8.0, 1.0, 20.0))
    radar[0, 3, :3] = torch.tensor((80.0, 1.0, 20.0))
    radar[0, :4, 10] = 1.0
    radar[0, :4, 11] = 3.0
    features, coords, counts, info = prepare_bevcar_voxels(
        radar, vox_util, max_points_per_voxel=2,
    )
    assert features.shape == (1, 2, 2, 7)
    assert coords.shape == (1, 2, 3)
    assert counts.tolist() == [2]
    assert info['retained_points'].tolist() == [3]
    assert info['truncated_points'].tolist() == [0]
    assert torch.count_nonzero(features[0, 1, 1]).item() == 0

    _, _, _, clipped = prepare_bevcar_voxels(
        radar, vox_util, max_points_per_voxel=1,
    )
    assert clipped['truncated_points'].tolist() == [1]
    empty_features, empty_coords, empty_counts, empty_info = prepare_bevcar_voxels(
        torch.zeros_like(radar), vox_util,
    )
    assert empty_features.shape == (1, 1, 16, 7)
    assert torch.count_nonzero(empty_features).item() == 0
    assert torch.count_nonzero(empty_coords).item() == 0
    assert empty_counts.tolist() == [0]
    assert empty_info['retained_points'].tolist() == [0]

    batched_features, batched_coords, batched_counts, _ = prepare_bevcar_voxels(
        torch.cat((radar, torch.zeros_like(radar)), dim=0), vox_util,
    )
    assert batched_features.shape == (2, 2, 16, 7)
    assert batched_coords.shape == (2, 2, 3)
    assert batched_counts.tolist() == [2, 0]
    assert torch.count_nonzero(batched_features[1]).item() == 0

    try:
        prepare_bevcar_voxels(radar, vox_util, max_voxels=1)
    except ValueError as error:
        assert 'max_voxels' in str(error)
    else:
        raise AssertionError('voxel overflow must fail explicitly')

    # A known camera-frame voxel centre proves axis order: increasing camera
    # X changes the last coordinate, and increasing camera Z changes the first.
    centres = torch.tensor([[[0.25, 1.625, 0.25],
                             [1.25, 1.625, 0.25],
                             [0.25, 1.625, 1.25]]])
    mem = vox_util.Ref2Mem(centres, Z, Y, X).round().long()
    assert mem[0].tolist() == [[100, 4, 100], [102, 4, 100], [100, 4, 102]]
    assert coords[0, 0, 0].item() >= 0


def count_map(points, dtype=np.int32):
    image = np.zeros((Z, X), dtype=dtype)
    for z, x in points:
        image[int(z), int(x)] += 1
    return image


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def panel(array, color):
    visible = array > 0
    rgb = np.zeros((Z, X, 3), dtype=np.uint8)
    rgb[visible] = color
    return Image.fromarray(rgb).resize((360, 360), Image.Resampling.NEAREST)


def render(raw_map, expected_map, adapter_map, values, output_path):
    canvas = Image.new('RGB', (1170, 555), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), 'BEVCar-shaped voxel adapter: one nuScenes mini frame',
              font=font(26, True), fill=(25, 30, 40))
    draw.text((24, 52),
              'Camera-reference BEV grid: 200 x 8 x 200, 100 m x 100 m, no labels or model weights',
              font=font(15), fill=(60, 65, 75))
    panels = (
        ('1. All in-range radar points', raw_map, (95, 145, 255)),
        ('2. Quality-filtered point cells', expected_map, (255, 175, 75)),
        ('3. Adapter voxel cells', adapter_map, (75, 220, 160)),
    )
    for index, (title, data, color) in enumerate(panels):
        left = 24 + 382 * index
        canvas.paste(panel(data, color), (left, 105))
        draw.rectangle((left, 105, left + 360, 465), outline=(110, 120, 130))
        draw.text((left, 80), title, font=font(16, True), fill=(25, 30, 40))
        draw.text((left + 8, 445), '+Z / front', font=font(12), fill=(225, 230, 235))
    draw.text((24, 488),
              f"{values['input_points']} returns; {values['in_range_points']} in ROI; "
              f"{values['retained_points']} kept in {values['occupied_voxels']} voxels; "
              f"{values['truncated_points']} truncated",
              font=font(16, True), fill=(25, 30, 40))
    draw.text((24, 518),
              'Geometry check: point-cell and adapter-voxel maps agree exactly',
              font=font(16), fill=(28, 115, 75))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    vox_util = make_vox_util()
    synthetic_checks(vox_util)

    radar_camera, sample_token = first_mini_train_radar(args.data_root)
    features, coords, counts, info = prepare_bevcar_voxels(
        radar_camera, vox_util,
    )

    assert features.shape == (1, int(counts[0]), 16, 7)
    assert coords.shape == (1, int(counts[0]), 3)
    assert torch.isfinite(features).all()
    assert info['truncated_points'].item() == 0

    light = RadarPointEncoder(
        Z=Z, Y=Y, X=X, bounds=bounds,
        scene_centroid=scene_centroid_py, out_channels=64,
    )
    memory = vox_util.Ref2Mem(radar_camera[..., :3], Z, Y, X)
    torch.testing.assert_close(
        memory, light._memory_coordinates(radar_camera[..., :3]),
        rtol=1e-5, atol=1e-5,
    )
    selected = light._valid_mask(radar_camera, memory)
    rounded = memory.round().long()
    xyz = rounded[0, selected[0]]
    expected = count_map(torch.stack((xyz[:, 2], xyz[:, 0]), dim=-1).tolist())
    voxel_map = count_map(coords[0, :int(counts[0])][:, [0, 2]].tolist())
    assert np.array_equal(expected > 0, voxel_map > 0)
    assert int(selected.sum()) == int(info['retained_points'][0])
    assert int(voxel_map.sum()) == int(counts[0])

    raw_mask = (
        torch.isfinite(radar_camera).all(dim=-1)
        & radar_camera[..., :3].abs().sum(dim=-1).gt(0)
        & ((memory > -0.5) &
           (memory < memory.new_tensor((X, Y, Z)) - 0.5)).all(dim=-1)
    )
    raw_xyz = rounded[0, raw_mask[0]]
    raw_map = count_map(torch.stack((raw_xyz[:, 2], raw_xyz[:, 0]), dim=-1).tolist())

    values = {key: int(value[0]) for key, value in info.items()}
    image_path = args.output_dir / 'bevcar_voxel_adapter_audit.png'
    render(raw_map, expected, voxel_map, values, image_path)
    print('device: cpu')
    print(f'sample_token: {sample_token}')
    print(f'feature_names: {FEATURE_NAMES}')
    print(f'features_shape: {tuple(features.shape)}')
    print(f'coords_shape: {tuple(coords.shape)} (z, y, x)')
    for key, value in values.items():
        print(f'{key}: {value}')
    print(f'BEV_cell_mismatch: {int(np.count_nonzero((expected > 0) != (voxel_map > 0)))}')
    print('synthetic_collision_empty_and_axis_checks: OK')
    print(f'image: {image_path}')


if __name__ == '__main__':
    main()
