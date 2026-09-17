"""Optional BEVCar radar branch for Simple-BEV, without vendoring BEVCar code.

The caller must supply a separate official BEVCar checkout. This branch takes
the same camera-frame radar tensor used by the lightweight Simple-BEV branch.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from nets.bevcar_voxel_adapter import prepare_bevcar_voxels


def official_voxelnet(source_dir):
    source_file = Path(source_dir) / 'nets' / 'voxelnet.py'
    if not source_file.is_file():
        raise FileNotFoundError(f'official BEVCar VoxelNet not found: {source_file}')
    spec = importlib.util.spec_from_file_location('official_bevcar_voxelnet', source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VoxelNet


class BEVCarRadarBridge(nn.Module):
    """Convert camera-frame radar into the existing 64-channel fusion interface."""

    def __init__(self, vox_util, source_dir, out_channels=64):
        super().__init__()
        self.vox_util = vox_util
        voxelnet = official_voxelnet(source_dir)
        self.voxelnet = voxelnet(
            use_col=False, reduced_zx=False, output_dim=128,
            use_radar_occupancy_map=False,
        )
        self.projection = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, radar_camera):
        with torch.no_grad():
            features, coords, counts, _ = prepare_bevcar_voxels(
                radar_camera, self.vox_util, quality_filter=False,
            )
        radar_bev = self.voxelnet(features, coords, counts)
        return self.projection(radar_bev)
