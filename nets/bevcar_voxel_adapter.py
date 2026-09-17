"""Prepare Simple-BEV radar points for a BEVCar-shaped VoxelNet input.

This is an independent preprocessing prototype, not BEVCar's released data
pipeline or radar encoder. Geometry uses Simple-BEV's reference-camera grid.
BEVCar's VoxelNet expects (point features, voxel coordinates, voxel counts),
with voxel coordinates ordered (z, y, x). The exact seventh feature and
training-time point sampling still need comparison with upstream preprocessing.
"""

from typing import Dict, Tuple

import torch


# BEVCar's paper names XYZ, uncompensated planar velocity and RCS. Its released
# VoxelNet accepts seven input channels; this prototype uses time lag as the
# explicit seventh channel. These values must not be called pretrained-
# checkpoint-compatible until the upstream data preprocessing is matched.
FEATURE_NAMES = ('x', 'y', 'z', 'raw_vx', 'raw_vz', 'rcs', 'time_lag')
FEATURE_INDICES = (0, 1, 2, 6, 7, 5, 18)


def prepare_bevcar_voxels(
        radar_camera: torch.Tensor,
        vox_util,
        grid: Tuple[int, int, int] = (200, 8, 200),
        max_points_per_voxel: int = 16,
        max_voxels: int = 4096,
        quality_filter: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Group camera-frame radar returns into padded BEVCar-shaped tensors.

    Args:
        radar_camera: ``(B,R,19)`` tensor after XYZ and velocity rotation into
            the Simple-BEV reference camera. Zero rows are padding.
        vox_util: The *same* Simple-BEV ``Vox_util`` used for camera BEV.
        grid: ``(Z,Y,X)`` resolution, matching the camera BEV grid.
        max_points_per_voxel: Retain the first P points in each occupied voxel.
            This deterministic audit policy is not BEVCar's training-time
            random sampling when a voxel is overfull.
        max_voxels: Maximum occupied voxels per sample; overflow is rejected
            rather than silently changing the spatial evidence.
        quality_filter: Match the current lightweight radar encoder's
            conservative point filter. Turn off for an all-in-range audit.

    Returns:
        ``features``: ``(B,K,P,7)``; padded point slots are exactly zero.
        ``coords``: ``(B,K,3)`` integer ``(z,y,x)`` voxel coordinates.
        ``counts``: ``(B,)`` actual occupied voxel counts, before K padding.
        ``diagnostics``: input, in-range, retained and overflow point counts.
        K is at least one to keep the downstream tensor rank stable.
    """
    if radar_camera.ndim != 3 or radar_camera.shape[-1] != 19:
        raise ValueError('radar_camera must have shape (B, R, 19)')
    if not radar_camera.is_floating_point():
        raise TypeError('radar_camera must be floating point')
    if len(grid) != 3 or min(grid) <= 0:
        raise ValueError('grid must contain positive (Z, Y, X) dimensions')
    if max_points_per_voxel <= 0 or max_voxels <= 0:
        raise ValueError('max_points_per_voxel and max_voxels must be positive')

    Z, Y, X = grid
    if (vox_util.Z, vox_util.Y, vox_util.X) != grid:
        raise ValueError('vox_util and grid resolutions differ')

    B, _, _ = radar_camera.shape
    xyz = radar_camera[..., :3]
    memory = vox_util.Ref2Mem(xyz, Z, Y, X, assert_cube=False)
    finite = torch.isfinite(radar_camera).all(dim=-1)
    nonpadding = xyz.abs().sum(dim=-1).gt(0)
    limits = memory.new_tensor((X, Y, Z))
    inbounds = ((memory > -0.5) & (memory < limits - 0.5)).all(dim=-1)
    spatial = finite & nonpadding & inbounds
    if quality_filter:
        quality = (
            radar_camera[..., 10].gt(0.5)
            & radar_camera[..., 11].round().eq(3)
            & radar_camera[..., 14].round().eq(0)
        )
        retained = spatial & quality
    else:
        retained = spatial

    rounded = memory.round().long()
    feature_rows = radar_camera[..., FEATURE_INDICES]
    sample_features = []
    sample_coords = []
    occupied_counts = []
    overflow_counts = []
    for batch_index in range(B):
        points = feature_rows[batch_index, retained[batch_index]]
        xyz_index = rounded[batch_index, retained[batch_index]]
        if xyz_index.numel() == 0:
            sample_features.append(points.new_zeros((0, max_points_per_voxel, 7)))
            sample_coords.append(torch.empty((0, 3), dtype=torch.long, device=points.device))
            occupied_counts.append(0)
            overflow_counts.append(0)
            continue

        x_index, y_index, z_index = xyz_index.unbind(dim=-1)
        linear = z_index * (Y * X) + y_index * X + x_index
        voxel_keys, inverse = torch.unique(linear, sorted=True, return_inverse=True)
        if voxel_keys.numel() > max_voxels:
            raise ValueError('occupied voxel count exceeds max_voxels')

        count = voxel_keys.numel()
        grouped = points.new_zeros((count, max_points_per_voxel, 7))
        overflow = 0
        for voxel_index in range(count):
            point_group = points[inverse == voxel_index]
            selected = point_group[:max_points_per_voxel]
            grouped[voxel_index, :selected.shape[0]] = selected
            overflow += point_group.shape[0] - selected.shape[0]
        coords = torch.stack((
            voxel_keys // (Y * X),
            (voxel_keys // X) % Y,
            voxel_keys % X,
        ), dim=-1)
        sample_features.append(grouped)
        sample_coords.append(coords)
        occupied_counts.append(count)
        overflow_counts.append(overflow)

    K = max(1, *occupied_counts)
    features = radar_camera.new_zeros((B, K, max_points_per_voxel, 7))
    coords = torch.zeros((B, K, 3), dtype=torch.long, device=radar_camera.device)
    for batch_index, count in enumerate(occupied_counts):
        features[batch_index, :count] = sample_features[batch_index]
        coords[batch_index, :count] = sample_coords[batch_index]

    counts = torch.tensor(occupied_counts, dtype=torch.long, device=radar_camera.device)
    diagnostics = {
        'input_points': nonpadding.sum(dim=1),
        'in_range_points': spatial.sum(dim=1),
        'retained_points': retained.sum(dim=1),
        'occupied_voxels': counts,
        'truncated_points': torch.tensor(
            overflow_counts, dtype=torch.long, device=radar_camera.device
        ),
    }
    return features, coords, counts, diagnostics
