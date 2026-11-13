"""Original compensated-radar auxiliary target and masked Huber loss."""
import torch
import torch.nn.functional as F
import utils.basic
import utils.geom
import utils.vox
from nets.radar_encoder import transform_radar_to_camera_bev
from dd2414.config.geometry import Z,Y,X,bounds,scene_centroid
VELOCITY_CLIP = 15.0
MOVING_THRESHOLD = 1.0

def motion_target(batch, device):
    """Splat moving-return compensated velocity into a (2, Z, X) field.

    Velocity is read from radar cols 8:10 (vx_comp, vy_comp) after rotation into the
    reference-camera frame, matching the BEVCar input axes, and splatted at the point's
    camera0-frame (Z, X) cell — the same grid the shared BEV lives on.
    """
    radar_vehicle = batch[16][:, 0].to(device).permute(0, 2, 1)
    rotations = batch[1][:, 0].to(device)
    translations = batch[2][:, 0].to(device)
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    batch_size = radar_vehicle.shape[0]
    cameras_from_vehicle = utils.basic.unpack_seqdim(
        utils.geom.safe_inverse(utils.basic.pack_seqdim(vehicle_from_cameras, batch_size)),
        batch_size,
    )
    radar_camera = transform_radar_to_camera_bev(
        radar_vehicle, cameras_from_vehicle[:, 0]
    )
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    xyz = radar_camera[..., :3]
    mem = vox_util.Ref2Mem(xyz, Z, Y, X, assert_cube=False)[0]  # (R, 3)
    grid_max = torch.tensor((X, Y, Z), device=device)
    velocity = radar_camera[0, :, 8:10].clamp(-VELOCITY_CLIP, VELOCITY_CLIP)  # (R, 2)
    moving = velocity.norm(dim=-1).gt(MOVING_THRESHOLD)
    valid = (
        torch.isfinite(radar_camera[0]).all(dim=-1)
        & xyz[0].abs().sum(dim=-1).gt(0)
        & (mem > -0.5).all(dim=-1)
        & (mem < grid_max - 0.5).all(dim=-1)
        & radar_camera[0, :, 10].gt(0.5)
        & radar_camera[0, :, 11].round().eq(3)
        & radar_camera[0, :, 14].round().eq(0)
        & moving
    )
    x_cell = mem[:, 0].round().long()
    z_cell = mem[:, 2].round().long()
    field = torch.zeros((2, Z, X), device=device)
    count = torch.zeros((Z, X), device=device)
    index = torch.where(valid)[0]
    if index.numel() > 0:
        flat = z_cell[index] * X + x_cell[index]
        field_flat = field.view(2, -1)
        count_flat = count.view(-1)
        field_flat[0].index_add_(0, flat, velocity[index, 0])
        field_flat[1].index_add_(0, flat, velocity[index, 1])
        count_flat.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        field = field_flat.view(2, Z, X)
        count = count_flat.view(Z, X)
    coverage = count.gt(0)
    field[0, coverage] = field[0, coverage] / count[coverage]
    field[1, coverage] = field[1, coverage] / count[coverage]
    info = {
        'moving_points': int(valid.sum()),
        'covered_cells': int(coverage.sum()),
    }
    return field, coverage.float(), info


def motion_loss(prediction, target, coverage):
    """Huber loss over moving-return cells, normalized by the cell count."""
    prediction = prediction.float()
    if prediction.dim() == 4:
        prediction = prediction.squeeze(0)
    mask = coverage.gt(0)
    if not mask.any():
        # Keep the graph connected with a true zero: a plain torch.zeros(()) is a leaf
        # with no grad, which crashes .backward() when it is the only loss term
        # (motion_only on a sample with no moving points).
        return prediction.sum() * 0.0, 0
    loss = F.smooth_l1_loss(prediction[:, mask], target[:, mask], reduction='mean')
    return loss, int(mask.sum())
