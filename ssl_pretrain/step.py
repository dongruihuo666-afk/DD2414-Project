"""One self-supervised pretraining step: geometry -> teachers -> targets -> losses."""

from dataclasses import dataclass, field

import torch

import utils.basic
import utils.geom
from nets.radar_uncertainty import F_TIME, decode_uncertainty, radar_ego_to_cam0
from ssl_pretrain import losses as L
from ssl_pretrain import targets as T


@dataclass
class StepConfig:
    radar_input_frac: float = 0.7
    radar_target_max_age: float = 0.1
    p_drop_camera: float = 0.2
    p_drop_radar: float = 0.3
    max_depth: float = 60.0
    weights: dict = field(default_factory=dict)


def unpack_batch(sample, device):
    """Simple-BEV dataset tuple -> tensors + calibrations (time dim removed)."""
    imgs, rots, trans, intrins = (sample[i][:, 0].to(device) for i in range(4))
    seg_bev, valid_bev, center_bev, offset_bev = (sample[i][:, 0].to(device) for i in range(12, 16))
    radar = sample[16][:, 0].to(device).float()
    B = imgs.shape[0]
    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    pix_T_cams = __u(utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(__p(intrins))))
    velo_T_cams = utils.geom.merge_rtlist(rots, trans)
    cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))
    cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
    return dict(imgs=imgs.float(), pix_T_cams=pix_T_cams.float(), cam0_T_camXs=cam0_T_camXs.float(),
                cam0_T_ego=cams_T_velo[:, 0].float(), ego_T_cam0=velo_T_cams[:, 0].float(),
                radar_ego=radar, seg_bev=seg_bev, valid_bev=valid_bev,
                center_bev=center_bev, offset_bev=offset_bev)


def radar_anchors(radar, valid, cam0_T_camXs, pix_T_cams, image_hw, feat_hw, max_depth):
    """Radar returns projected into every camera, at feature resolution."""
    B, S = cam0_T_camXs.shape[:2]
    H, W = image_hw
    h, w = feat_hw
    u, v, z = T.project_to_cameras(radar[..., :3], cam0_T_camXs, pix_T_cams)
    _, prior = decode_uncertainty(radar)
    ok = valid[:, None] & (z > 1.0) & (z < max_depth) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    b, s, p = torch.nonzero(ok, as_tuple=True)
    return dict(n=b * S + s, u=u[b, s, p] * (w / W), v=v[b, s, p] * (h / H), z=z[b, s, p],
                w=prior[b, p])


def pretrain_step(model, teachers, sample, grid, cfg, device, generator=None):
    """Returns (loss, metrics, extras). `teachers` exposes .dino(imgs01) -> (N, D, hd, wd)
    and .depth(imgs01) -> (N, 1, Ht, Wt); `model` is SSLBEVNet (possibly wrapped)."""
    net = model.module if hasattr(model, 'module') else model
    g = unpack_batch(sample, device)
    imgs = g['imgs']
    B, S, _, H, W = imgs.shape

    radar, valid = radar_ego_to_cam0(g['radar_ego'], g['cam0_T_ego'])
    in_valid, tgt_valid = T.split_radar(radar, valid, cfg.radar_input_frac,
                                        cfg.radar_target_max_age, generator)
    recent = valid & (radar[..., F_TIME].abs() <= cfg.radar_target_max_age)

    with torch.no_grad():
        flat = imgs.flatten(0, 1)
        dino = teachers.dino(flat)
        depth = teachers.depth(flat)
        Cd, hd, wd = dino.shape[1:]
        depth = depth.view(B, S, 1, *depth.shape[-2:])
        scale, n_anchor = T.align_depth_scale(depth, radar, recent, g['cam0_T_camXs'],
                                              g['pix_T_cams'], (H, W), max_depth=cfg.max_depth)
        depth = depth * scale.view(B, S, 1, 1, 1)
        sem_t, cov = T.semantic_bev_target(dino.view(B, S, Cd, hd, wd), depth, g['pix_T_cams'],
                                           g['cam0_T_camXs'], (H, W), grid, cfg.max_depth)
        occ_t, occ_w, occ_rad = T.occupancy_target(depth, g['pix_T_cams'], g['cam0_T_camXs'],
                                                   g['ego_T_cam0'], (H, W), grid, radar, tgt_valid)
        vel_t = T.radial_velocity_targets(radar, tgt_valid)

    r = torch.rand((B,), generator=generator).to(device)
    drop_cam = r < cfg.p_drop_camera
    drop_rad = (~drop_cam) & (torch.rand((B,), generator=generator).to(device) < cfg.p_drop_radar)
    if not net.use_radar:
        drop_cam = torch.zeros_like(drop_cam)

    out = model(imgs - 0.5, g['pix_T_cams'], g['cam0_T_camXs'], radar=radar, radar_valid=in_valid,
                drop_camera=drop_cam, drop_radar=drop_rad if net.use_radar else None, mode='ssl')

    h, w = out['depth_logp'].shape[-2:]
    q, dmask = T.depth_bin_target(depth.flatten(0, 1), (h, w), net.bins, max_depth=cfg.max_depth)
    anchors = radar_anchors(radar, recent, g['cam0_T_camXs'], g['pix_T_cams'], (H, W), (h, w),
                            cfg.max_depth)

    parts = dict(
        image=L.image_distill(out['image_sem'], dino),
        depth=L.depth_ce(out['depth_logp'], q, dmask),
        radar_depth=L.radar_anchor_depth(out['depth_logp'], net.bins, anchors),
        sem_cam=L.bev_semantic(out['sem_cam'], sem_t, cov, keep=~drop_cam),
        sem_fused=L.bev_semantic(out['sem_fused'], sem_t, cov),
        occ_cam=L.heteroscedastic_bce(out['occ_cam'], out['occ_logvar_cam'], occ_t, occ_w, keep=~drop_cam),
        occ_fused=L.heteroscedastic_bce(out['occ_fused'], out['occ_logvar_fused'], occ_t, occ_w),
    )
    if net.use_radar:
        parts['sem_rad'] = L.bev_semantic(out['sem_rad'], sem_t, cov, keep=~drop_rad,
                                          extra_weight=out['rad_conf'].detach())
        parts['occ_rad'] = L.heteroscedastic_bce(out['occ_rad'], out['occ_logvar_rad'], occ_t, occ_w,
                                                 keep=~drop_rad)
        parts['vel'] = L.radial_velocity(out['vel'], vel_t, grid, keep=~drop_rad)
    loss = L.total_loss(parts, cfg.weights)

    metrics = {k: float(v.detach()) for k, v in parts.items()}
    metrics.update(
        loss=float(loss.detach()),
        depth_scale_mean=float(scale.mean()), depth_scale_anchored_frac=float((n_anchor > 0).float().mean()),
        sem_coverage=float(cov.mean()), occ_known_frac=float((occ_w > 0).float().mean()),
        occ_pos_frac=float((occ_t * (occ_w > 0)).sum() / (occ_w > 0).sum().clamp_min(1)),
        radar_in=float(in_valid.sum()) / B, radar_target=float(tgt_valid.sum()) / B,
    )
    if net.use_radar:
        metrics.update(gate_cam=float(out['gates'][:, 0].mean()), gate_rad=float(out['gates'][:, 1].mean()),
                       radar_conf_mean=float(out['point_conf'][in_valid].mean()) if in_valid.any() else 0.0)
    extras = dict(out=out, sem_t=sem_t, cov=cov, occ_t=occ_t, occ_w=occ_w, occ_rad=occ_rad,
                  depth=depth, drop_cam=drop_cam, drop_rad=drop_rad, in_valid=in_valid,
                  tgt_valid=tgt_valid, radar=radar, geom=g)
    return loss, metrics, extras
