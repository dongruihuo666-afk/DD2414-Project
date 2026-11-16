"""Grid constants and data loading shared by train_ssl.py and probe_bev.py.

The grid matches Simple-BEV exactly (reference-camera frame, 100 m x 100 m,
0.5 m cells, 8 height bins), so features are comparable with the official
checkpoints.
"""
import numpy as np
import torch

import nuscenesdataset
import utils.vox

scene_centroid_py = np.array([0.0, 1.0, 0.0]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()
bounds = (-50, 50, -5, 5, -50, 50)
Z, Y, X = 200, 8, 200
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def make_vox(device='cpu'):
    return utils.vox.Vox_util(Z, Y, X, scene_centroid=scene_centroid.to(device), bounds=bounds,
                              assert_cube=False)


def make_loaders(data_dir, dset='trainval', res_scale=2, batch_size=1, nworkers=4, nsweeps=5,
                 augment=True, shuffle=True, do_shuffle_cams=True, nworkers_val=None,
                 rotate_radar_velocity=True):
    """Simple-BEV loaders. Radar velocities are rotated into the ego frame by
    default; pass rotate_radar_velocity=False for official Simple-BEV
    checkpoints, which were trained on unrotated per-sensor velocities."""
    final_dim = (int(224 * res_scale), int(400 * res_scale))
    resize_lim = [0.8, 1.2] if augment else [1.0, 1.0]
    data_aug_conf = {
        'crop_offset': int(final_dim[0] * (1 - resize_lim[0])),
        'resize_lim': resize_lim, 'final_dim': final_dim, 'H': 900, 'W': 1600,
        'cams': CAMS, 'ncams': 6,
    }
    return nuscenesdataset.compile_data(
        dset, data_dir, data_aug_conf=data_aug_conf, centroid=scene_centroid_py, bounds=bounds,
        res_3d=(Z, Y, X), bsz=batch_size, nworkers=nworkers,
        nworkers_val=nworkers if nworkers_val is None else nworkers_val, shuffle=shuffle,
        use_radar_filters=False, seqlen=1, nsweeps=nsweeps, do_shuffle_cams=do_shuffle_cams,
        get_tids=True, rotate_radar_velocity=rotate_radar_velocity)


class Teachers:
    """Bundles the two frozen teachers behind the interface pretrain_step expects."""

    def __init__(self, dino_model, depth_model, image_hw, short_side=448):
        from ssl_pretrain.teachers import teacher_size
        self.dino_model, self.depth_model = dino_model, depth_model
        self.size = teacher_size(*image_hw, short=short_side)

    def dino(self, imgs01):
        from ssl_pretrain.teachers import dino_features
        return dino_features(self.dino_model, imgs01, self.size)

    def depth(self, imgs01):
        from ssl_pretrain.teachers import metric_depth
        return metric_depth(self.depth_model, imgs01, self.size)
