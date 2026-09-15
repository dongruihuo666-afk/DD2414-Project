"""A small, standalone point encoder for nuScenes radar.

The encoder deliberately does not contain camera fusion or a supervised loss.
It converts a padded ``(B, R, 19)`` radar tensor, already expressed in the
Simple-BEV reference-camera frame, into a learned ``(B, C, Z, X)`` BEV map.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

import utils.geom


RADAR_NUMERIC_FIELDS = (0, 1, 2, 5, 8, 9, 18)


def transform_radar_to_camera_bev(
        radar_vehicle: torch.Tensor,
        camera_from_vehicle: torch.Tensor) -> torch.Tensor:
    """Transform radar positions and planar velocities to a camera frame.

    Args:
        radar_vehicle: ``(B, R, 19)`` tensor. XYZ and the velocity pairs in
            rows 6:10 must already share the reference ego/vehicle frame.
        camera_from_vehicle: ``(B, 4, 4)`` rigid transform.

    Returns:
        A cloned ``(B, R, 19)`` tensor. XYZ is in camera coordinates. Each
        velocity pair stores camera-plane ``(v_camera_x, v_camera_z)`` so its
        axes agree with the BEV plane. Padded rows remain exactly zero.
    """
    if radar_vehicle.ndim != 3 or radar_vehicle.shape[-1] != 19:
        raise ValueError('radar_vehicle must have shape (B, R, 19)')
    if camera_from_vehicle.shape != (radar_vehicle.shape[0], 4, 4):
        raise ValueError('camera_from_vehicle must have shape (B, 4, 4)')

    valid = (
        torch.isfinite(radar_vehicle).all(dim=-1)
        & radar_vehicle[..., :3].abs().sum(dim=-1).gt(0)
    )
    result = radar_vehicle.clone()
    result[..., :3] = utils.geom.apply_4x4(
        camera_from_vehicle, radar_vehicle[..., :3]
    )

    rotation = camera_from_vehicle[:, :3, :3]
    for start in (6, 8):
        planar_vehicle = radar_vehicle[..., start:start + 2]
        velocity_vehicle = torch.cat(
            (planar_vehicle, torch.zeros_like(planar_vehicle[..., :1])), dim=-1
        )
        velocity_camera = torch.matmul(
            rotation[:, None], velocity_vehicle.unsqueeze(-1)
        ).squeeze(-1)
        result[..., start] = velocity_camera[..., 0]
        result[..., start + 1] = velocity_camera[..., 2]

    return result.masked_fill(~valid.unsqueeze(-1), 0.0)


class RadarPointEncoder(nn.Module):
    """Encode continuous radar measurements and pool them into BEV voxels."""

    def __init__(
            self,
            Z: int = 200,
            Y: int = 8,
            X: int = 200,
            bounds: Tuple[float, float, float, float, float, float] = (
                -50.0, 50.0, -5.0, 5.0, -50.0, 50.0,
            ),
            scene_centroid: Tuple[float, float, float] = (0.0, 1.0, 0.0),
            out_channels: int = 64,
            use_quality_mask: bool = True,
    ):
        super().__init__()
        if min(Z, Y, X, out_channels) <= 0:
            raise ValueError('grid dimensions and out_channels must be positive')
        bounds_tensor = torch.as_tensor(bounds).flatten()
        centroid_tensor = torch.as_tensor(scene_centroid).flatten()
        if bounds_tensor.numel() != 6 or centroid_tensor.numel() != 3:
            raise ValueError('bounds must have 6 values and scene_centroid 3')

        self.Z, self.Y, self.X = Z, Y, X
        self.out_channels = out_channels
        self.use_quality_mask = use_quality_mask

        xmin, xmax, ymin, ymax, zmin, zmax = bounds_tensor
        cx, cy, cz = centroid_tensor
        minimum = torch.stack((xmin + cx, ymin + cy, zmin + cz))
        maximum = torch.stack((xmax + cx, ymax + cy, zmax + cz))
        if not torch.all(maximum > minimum):
            raise ValueError('each upper bound must exceed its lower bound')
        self.register_buffer('xyz_minimum', minimum.float())
        self.register_buffer('xyz_maximum', maximum.float())
        self.register_buffer(
            'grid_size_xyz', torch.tensor((X, Y, Z), dtype=torch.float32)
        )

        self.point_mlp = nn.Sequential(
            nn.Linear(len(RADAR_NUMERIC_FIELDS), 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        # Bias-free projection guarantees that an empty voxel remains zero.
        self.voxel_fusion = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels, bias=False),
            nn.GELU(),
        )

    def _memory_coordinates(self, xyz: torch.Tensor) -> torch.Tensor:
        voxel_size = (
            (self.xyz_maximum - self.xyz_minimum) / self.grid_size_xyz
        )
        return (xyz - self.xyz_minimum) / voxel_size - 0.5

    def _valid_mask(self, radar: torch.Tensor, xyz_memory: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(radar).all(dim=-1)
        nonpadding = radar[..., :3].abs().sum(dim=-1).gt(0)
        # Match Vox_util.get_inbounds(): voxel centres are integer memory
        # coordinates and the valid volume runs from -0.5 to size-0.5.
        inbounds = (
            (xyz_memory > -0.5)
            & (xyz_memory < self.grid_size_xyz - 0.5)
        ).all(dim=-1)
        valid = finite & nonpadding & inbounds
        if self.use_quality_mask:
            # A conservative first version: retain valid, unambiguous returns.
            quality = (
                radar[..., 10].gt(0.5)
                & radar[..., 11].round().eq(3)
                & radar[..., 14].round().eq(0)
            )
            valid = valid & quality
        return valid

    def _normalize_features(self, radar: torch.Tensor) -> torch.Tensor:
        xyz_center = (self.xyz_minimum + self.xyz_maximum) / 2
        xyz_half_extent = (self.xyz_maximum - self.xyz_minimum) / 2
        xyz = (radar[..., :3] - xyz_center) / xyz_half_extent
        rcs = radar[..., 5:6].clamp(-5.0, 30.0)
        rcs = (rcs - 12.5) / 17.5
        velocity = radar[..., 8:10].clamp(-15.0, 15.0) / 15.0
        time_lag = radar[..., 18:19].clamp(-0.5, 0.5) / 0.5
        return torch.cat((xyz, rcs, velocity, time_lag), dim=-1)

    def forward(self, radar: torch.Tensor, return_diagnostics: bool = False):
        """Return a learned BEV map and, optionally, simple point counts."""
        if radar.ndim != 3 or radar.shape[-1] != 19:
            raise ValueError('radar must have shape (B, R, 19)')
        if not radar.is_floating_point():
            raise TypeError('radar must be a floating-point tensor')

        batch_size, point_count, _ = radar.shape
        xyz_memory = self._memory_coordinates(radar[..., :3])
        valid = self._valid_mask(radar, xyz_memory)
        numeric = self._normalize_features(radar)
        # Invalid values are replaced before the MLP so NaNs cannot leak through
        # multiplication by a zero mask.
        numeric = torch.where(valid.unsqueeze(-1), numeric, torch.zeros_like(numeric))
        point_features = self.point_mlp(numeric)

        voxel_count = self.Z * self.Y * self.X
        flat_size = batch_size * voxel_count
        feature_sum = radar.new_zeros((flat_size, self.out_channels))
        feature_max = radar.new_full(
            (flat_size, self.out_channels), -torch.inf
        )
        counts = radar.new_zeros((flat_size, 1))

        rounded = xyz_memory.round().long()
        x_index, y_index, z_index = rounded.unbind(dim=-1)
        local_index = z_index * (self.Y * self.X) + y_index * self.X + x_index
        batch_offset = (
            torch.arange(batch_size, device=radar.device)[:, None] * voxel_count
        )
        flat_index = (local_index + batch_offset)[valid]
        valid_features = point_features[valid]

        if flat_index.numel() > 0:
            feature_sum.index_add_(0, flat_index, valid_features)
            counts.index_add_(
                0, flat_index, radar.new_ones((flat_index.numel(), 1))
            )
            expanded_index = flat_index[:, None].expand_as(valid_features)
            feature_max.scatter_reduce_(
                0, expanded_index, valid_features,
                reduce='amax', include_self=True,
            )

        occupied = counts.squeeze(-1).gt(0)
        feature_mean = feature_sum / counts.clamp_min(1.0)
        feature_max = torch.where(
            occupied.unsqueeze(-1), feature_max, torch.zeros_like(feature_max)
        )
        voxel_features = self.voxel_fusion(
            torch.cat((feature_mean, feature_max), dim=-1)
        )
        voxel_features = voxel_features * occupied.unsqueeze(-1)
        volume = voxel_features.view(
            batch_size, self.Z, self.Y, self.X, self.out_channels
        ).permute(0, 4, 1, 2, 3)
        bev = volume.amax(dim=3)

        if not return_diagnostics:
            return bev
        diagnostics: Dict[str, torch.Tensor] = {
            'input_points': radar[..., :3].abs().sum(dim=-1).gt(0).sum(dim=1),
            'valid_points': valid.sum(dim=1),
            'occupied_voxels': occupied.view(batch_size, voxel_count).sum(dim=1),
            'occupied_bev_cells': bev.abs().sum(dim=1).gt(0).sum(dim=(1, 2)),
        }
        return bev, diagnostics
