"""Pretraining losses. Every loss returns a scalar that is 0 (graph-connected)
when its mask is empty, so branch/modality dropout never breaks backward."""

import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS = dict(
    image=1.0,           # student image features -> DINO patch tokens
    depth=1.0,           # depth bins -> scale-aligned teacher depth (image space)
    radar_depth=0.2,     # expected depth at radar anchors (image space)
    sem_cam=1.0,         # camera BEV -> lifted DINO target
    sem_fused=1.0,       # fused BEV -> lifted DINO target
    sem_rad=0.5,         # radar BEV -> lifted DINO target (camera -> radar)
    occ_cam=1.0,         # camera BEV -> teacher + held-out radar occupancy (radar -> camera)
    occ_rad=1.0,         # radar BEV -> teacher + held-out radar occupancy (camera -> radar)
    occ_fused=1.0,
    vel=0.2,             # fused BEV radial velocity at held-out radar returns
)


def _masked_mean(values, weight):
    den = weight.sum()
    return (values * weight).sum() / den.clamp_min(1e-6) if den > 0 else values.sum() * 0.0


def image_distill(pred, target):
    """Cosine between student image head (N, D, h, w) and DINO (N, D, hd, wd)."""
    pred = F.interpolate(pred.float(), size=target.shape[-2:], mode='bilinear', align_corners=False)
    return (1.0 - F.cosine_similarity(pred, target.float(), dim=1)).mean()


def depth_ce(depth_logp, q, mask):
    """Soft-label CE over depth bins at feature resolution."""
    ce = -(q * depth_logp).sum(1, keepdim=True)
    return _masked_mean(ce, mask)


def radar_anchor_depth(depth_logp, bins, anchors):
    """Huber on log expected depth at radar anchor pixels.

    anchors: dict with n (index into B*S), u, v (feature-res pixel coords),
    z (metric depth), w (weight).
    """
    if anchors is None or anchors['n'].numel() == 0:
        return depth_logp.sum() * 0.0
    N, D, h, w = depth_logp.shape
    log_c = torch.log(bins.centers(depth_logp.device))
    exp_log = (depth_logp.exp() * log_c.view(1, -1, 1, 1)).sum(1)              # (N, h, w)
    u = anchors['u'].round().long().clamp(0, w - 1)
    v = anchors['v'].round().long().clamp(0, h - 1)
    pred = exp_log[anchors['n'], v, u]
    loss = F.smooth_l1_loss(pred, torch.log(anchors['z']), reduction='none', beta=0.1)
    return _masked_mean(loss, anchors['w'])


def bev_semantic(pred, target, coverage, keep=None, extra_weight=None):
    weight = coverage
    if extra_weight is not None:
        weight = weight * extra_weight
    if keep is not None:
        weight = weight * keep.float().view(-1, 1, 1, 1)
    cos = F.cosine_similarity(pred.float(), target.float(), dim=1).unsqueeze(1)
    return _masked_mean(1.0 - cos, weight)


def heteroscedastic_bce(logit, logvar, target, weight, keep=None):
    """Kendall & Gal style: exp(-s) * BCE + s / 2, per cell."""
    if keep is not None:
        weight = weight * keep.float().view(-1, 1, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logit.float(), target, reduction='none')
    return _masked_mean(torch.exp(-logvar) * bce + 0.5 * logvar, weight)


def radial_velocity(vel_map, vt, grid, keep=None):
    """Huber between predicted velocity projected on the ray and Doppler."""
    sel = vt['valid']
    if keep is not None:
        sel = sel & keep.view(-1, 1)
    if not sel.any():
        return vel_map.sum() * 0.0
    B = vel_map.shape[0]
    gx, gz = grid.to_cell(vt['x'], vt['z'])
    nx = 2 * gx / (grid.X - 1) - 1
    nz = 2 * gz / (grid.Z - 1) - 1
    samp = F.grid_sample(vel_map.float(), torch.stack((nx, nz), -1).unsqueeze(1),
                         align_corners=True).squeeze(2).permute(0, 2, 1)       # (B, R, 2)
    pred_r = (samp * vt['ur']).sum(-1)
    loss = F.smooth_l1_loss(pred_r, vt['v_r'], reduction='none', beta=0.5)
    return _masked_mean(loss[sel], vt['weight'][sel])


def total_loss(parts, weights=None):
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    loss = sum(weights[k] * v for k, v in parts.items() if weights.get(k, 0) != 0)
    return loss
