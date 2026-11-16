"""Self-supervised targets built from frozen teachers and held-out radar.

Design rule: every target is built from information the supervised branch
does NOT receive as input, so it cannot be satisfied by copying the input.

* Semantic BEV target: DINOv2 patch features placed at a *metric 3D point*
  using (radar-scale-aligned) Depth-Anything depth, so each feature lands in
  one BEV region instead of being smeared along its camera ray.
* Occupancy target: teacher-depth surfaces (occupied) + the space the camera
  rays traverse before them (free) + held-out radar returns (occupied,
  weighted by decoded radar confidence).
* Radar split: the current-sweep returns are split into an input set and a
  disjoint held-out target set; held-out returns supervise occupancy and
  radial velocity.
"""

import torch
import torch.nn.functional as F

import utils.geom
from nets.radar_uncertainty import (F_TIME, F_VX_COMP, F_VZ_COMP, F_X, F_Z,
                                    decode_uncertainty)


# --------------------------------------------------------------------- radar
def split_radar(radar, valid, input_frac=0.7, target_max_age=0.1, generator=None):
    """Disjoint input/target split. Only recent returns can become targets."""
    recent = radar[..., F_TIME].abs() <= target_max_age
    r = torch.rand(valid.shape, generator=generator, device='cpu').to(valid.device)
    target = valid & recent & (r >= input_frac)
    return valid & ~target, target


def project_to_cameras(xyz_cam0, cam0_T_camXs, pix_T_cams):
    """xyz_cam0: (B, P, 3). Returns u, v, depth: (B, S, P)."""
    B, S = cam0_T_camXs.shape[:2]
    P = xyz_cam0.shape[1]
    camXs_T_cam0 = utils.geom.safe_inverse(cam0_T_camXs.reshape(B * S, 4, 4))
    xyz = xyz_cam0[:, None].expand(B, S, P, 3).reshape(B * S, P, 3)
    xyz_cam = utils.geom.apply_4x4(camXs_T_cam0, xyz)
    pix = utils.geom.apply_4x4(pix_T_cams.reshape(B * S, 4, 4), xyz_cam)
    z = xyz_cam[..., 2]
    u = pix[..., 0] / pix[..., 2].clamp_min(1e-6)
    v = pix[..., 1] / pix[..., 2].clamp_min(1e-6)
    return u.view(B, S, P), v.view(B, S, P), z.view(B, S, P)


# --------------------------------------------------------------------- depth
def downsample_depth(depth, size):
    """Area-pool inverse depth (favours the nearer surface at edges)."""
    inv = 1.0 / depth.clamp_min(1e-3)
    return 1.0 / F.interpolate(inv, size=size, mode='area').clamp_min(1e-6)


def align_depth_scale(depth, radar, anchor_valid, cam0_T_camXs, pix_T_cams, image_hw,
                      min_anchors=5, clamp=(0.5, 2.0), max_depth=60.0):
    """Per-camera metric scale for teacher depth from radar anchors.

    depth: (B, S, 1, Ht, Wt) teacher depth. Radar has no elevation, so each
    anchor only constrains depth along its image column at mounting height;
    a robust median of ratios over all anchors is used. Cameras with too
    few anchors take the median scale of the other cameras of that sample,
    else 1. Returns scale (B, S) and anchor count (B, S).
    """
    B, S, _, Ht, Wt = depth.shape
    H, W = image_hw
    u, v, z = project_to_cameras(radar[..., :3], cam0_T_camXs, pix_T_cams)
    ut, vt = u * (Wt / W), v * (Ht / H)
    inside = (ut >= 0) & (ut <= Wt - 1) & (vt >= 0) & (vt <= Ht - 1) & (z > 2) & (z < max_depth)
    inside = inside & anchor_valid[:, None]
    scale = depth.new_ones((B, S))
    count = torch.zeros((B, S), dtype=torch.long, device=depth.device)
    for b in range(B):
        for s in range(S):
            m = inside[b, s]
            if m.sum() < min_anchors:
                continue
            d_t = depth[b, s, 0, vt[b, s, m].round().long(), ut[b, s, m].round().long()]
            ok = (d_t > 1.0) & (d_t < max_depth)
            if ok.sum() < min_anchors:
                continue
            scale[b, s] = (z[b, s, m][ok] / d_t[ok]).median().clamp(*clamp)
            count[b, s] = int(ok.sum())
        has = count[b] > 0
        if has.any() and not has.all():
            scale[b, ~has] = scale[b, has].median()
    return scale, count


def depth_bin_target(depth, size, bins, sigma_bins=1.0, max_depth=60.0):
    """Soft one-dimensional bin distribution per pixel at feature resolution.

    depth: (N, 1, Ht, Wt) aligned metric depth. Returns q (N, D, h, w) and
    mask (N, 1, h, w) (teacher depth inside [dmin, min(dmax, max_depth)]).
    """
    d = downsample_depth(depth, size)
    mask = (d >= bins.dmin) & (d <= min(bins.dmax, max_depth))
    k = torch.arange(bins.n, device=d.device, dtype=d.dtype).view(1, -1, 1, 1)
    q = torch.softmax(-0.5 * ((bins.index(d) - k) / sigma_bins) ** 2, dim=1)
    return q, mask.float()


# ---------------------------------------------------------------- BEV helpers
class BEVGrid:
    """Metric <-> BEV-cell mapping of the cam0 grid (cell centres at integers)."""

    def __init__(self, vox_util, Z, X):
        self.Z, self.X = Z, X
        self.xmin, self.xmax = vox_util.XMIN, vox_util.XMAX
        self.zmin, self.zmax = vox_util.ZMIN, vox_util.ZMAX
        self.ymin, self.ymax = vox_util.YMIN, vox_util.YMAX
        self.cx = (self.xmax - self.xmin) / X
        self.cz = (self.zmax - self.zmin) / Z

    def to_cell(self, x, z):
        return (x - self.xmin) / self.cx - 0.5, (z - self.zmin) / self.cz - 0.5

    def bilinear_splat(self, batch_idx, x, z, values, weights, B):
        """Scatter values (P, C) with weights (P,) bilinearly. Returns sums."""
        C = values.shape[1]
        gx, gz = self.to_cell(x, z)
        x0, z0 = gx.floor(), gz.floor()
        num = values.new_zeros((B * self.Z * self.X, C))
        den = values.new_zeros((B * self.Z * self.X,))
        for dx in (0, 1):
            for dz in (0, 1):
                xi, zi = x0 + dx, z0 + dz
                w = weights * (1 - (gx - xi).abs()) * (1 - (gz - zi).abs())
                ok = (xi >= 0) & (xi < self.X) & (zi >= 0) & (zi < self.Z) & (w > 0)
                flat = (batch_idx[ok] * self.Z + zi[ok].long()) * self.X + xi[ok].long()
                num.index_add_(0, flat, values[ok] * w[ok, None])
                den.index_add_(0, flat, w[ok])
        num = num.view(B, self.Z, self.X, C).permute(0, 3, 1, 2)
        return num, den.view(B, 1, self.Z, self.X)

    def scatter_max(self, batch_idx, x, z, values, B):
        gx, gz = self.to_cell(x, z)
        xi, zi = gx.round().long(), gz.round().long()
        ok = (xi >= 0) & (xi < self.X) & (zi >= 0) & (zi < self.Z)
        out = values.new_zeros((B * self.Z * self.X,))
        flat = (batch_idx[ok] * self.Z + zi[ok]) * self.X + xi[ok]
        out = out.scatter_reduce(0, flat, values[ok], reduce='amax', include_self=True)
        return out.view(B, 1, self.Z, self.X)


def gaussian_blur(x, sigma=1.0, radius=2):
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), padding=(0, radius), groups=C)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), padding=(radius, 0), groups=C)


def backproject(depth, pix_T_cams, cam0_T_camXs, image_hw):
    """Pixel-centre rays of a (B, S, 1, h, w) depth map -> cam0 points.

    Returns xyz (B, S, h*w, 3) in cam0, depth (B, S, h*w), and the per-ray
    unit-depth direction in cam0 plus the camera origin (for ray marching).
    """
    B, S, _, h, w = depth.shape
    H, W = image_hw
    dev = depth.device
    jj, ii = torch.meshgrid(torch.arange(w, device=dev, dtype=depth.dtype),
                            torch.arange(h, device=dev, dtype=depth.dtype), indexing='xy')
    u = (jj + 0.5) * (W / w) - 0.5
    v = (ii + 0.5) * (H / h) - 0.5
    fx, fy, x0, y0 = utils.geom.split_intrinsics(pix_T_cams.reshape(B * S, 4, 4))
    dx = (u.reshape(1, -1) - x0[:, None]) / fx[:, None]
    dy = (v.reshape(1, -1) - y0[:, None]) / fy[:, None]
    dirs = torch.stack((dx, dy, torch.ones_like(dx)), -1)                  # (BS, P, 3)
    rot = cam0_T_camXs.reshape(B * S, 4, 4)[:, :3, :3]
    origin = cam0_T_camXs.reshape(B * S, 4, 4)[:, :3, 3]
    dirs_cam0 = torch.einsum('nij,npj->npi', rot, dirs)
    d = depth.reshape(B * S, -1)
    xyz = origin[:, None] + dirs_cam0 * d[..., None]
    P = h * w
    return (xyz.view(B, S, P, 3), d.view(B, S, P), dirs_cam0.view(B, S, P, 3),
            origin.view(B, S, 3))


# ----------------------------------------------------------- semantic target
@torch.no_grad()
def semantic_bev_target(dino, depth, pix_T_cams, cam0_T_camXs, image_hw, grid,
                        max_depth=60.0, blur_sigma=1.0):
    """Place DINO patch features at their teacher-depth 3D location.

    dino: (B, S, Cd, hd, wd); depth: (B, S, 1, Ht, Wt) aligned metric depth.
    Returns target (B, Cd, Z, X) (unit norm where covered) and coverage
    (B, 1, Z, X) in [0, 1].
    """
    B, S, Cd, hd, wd = dino.shape
    d = downsample_depth(depth.flatten(0, 1), (hd, wd)).view(B, S, 1, hd, wd)
    xyz, dd, _, _ = backproject(d, pix_T_cams, cam0_T_camXs, image_hw)
    feats = F.normalize(dino.float(), dim=2).permute(0, 1, 3, 4, 2).reshape(B, S, hd * wd, Cd)
    keep = (dd > 1.0) & (dd < max_depth) & (xyz[..., 1] > grid.ymin) & (xyz[..., 1] < grid.ymax)
    b_idx = torch.arange(B, device=dino.device).view(B, 1, 1).expand_as(keep)
    num, den = grid.bilinear_splat(b_idx[keep], xyz[..., 0][keep], xyz[..., 2][keep],
                                   feats[keep], torch.ones_like(dd[keep]), B)
    num, den = gaussian_blur(num, blur_sigma), gaussian_blur(den, blur_sigma)
    target = F.normalize(num, dim=1)
    coverage = 1.0 - torch.exp(-den)
    return target, coverage


# ---------------------------------------------------------- occupancy target
@torch.no_grad()
def occupancy_target(depth, pix_T_cams, cam0_T_camXs, ego_T_cam0, image_hw, grid,
                     radar=None, radar_target_valid=None, ray_hw=(28, 50),
                     max_depth=50.0, step=0.5, min_height=0.3, max_height=2.5,
                     free_margin=1.0):
    """Occupied / free BEV target from teacher depth and held-out radar.

    Heights are measured in the ego frame (nuScenes ego origin is on the
    ground), so ground returns are not counted as obstacles. Returns
    target (B, 1, Z, X) in {0, 1}, weight (B, 1, Z, X) in [0, 1] (0 = unknown)
    and the radar-only evidence map (B, 1, Z, X).
    """
    B, S = cam0_T_camXs.shape[:2]
    dev = depth.device
    d = downsample_depth(depth.flatten(0, 1), ray_hw).view(B, S, 1, *ray_hw)
    xyz, dd, dirs, origin = backproject(d, pix_T_cams, cam0_T_camXs, image_hw)
    P = xyz.shape[2]

    def height(points):  # points (B, N, 3) cam0 -> ego z
        return utils.geom.apply_4x4(ego_T_cam0, points)[..., 2]

    pts = xyz.reshape(B, S * P, 3)
    h_surf = height(pts)
    d_flat = dd.reshape(B, S * P)
    surf = (d_flat > 1.0) & (d_flat < max_depth) & (h_surf > min_height) & (h_surf < max_height)
    b_idx = torch.arange(B, device=dev).view(B, 1).expand(B, S * P)
    occ_cam = grid.scatter_max(b_idx[surf], pts[..., 0][surf], pts[..., 2][surf],
                               torch.ones_like(d_flat[surf]), B)

    # March every ray up to its surface; samples in the obstacle height band
    # before the surface (minus a margin) are free space.
    t = torch.arange(1.0, max_depth, step, device=dev)                 # (T,)
    samples = origin[:, :, None, None] + dirs[:, :, :, None] * t.view(1, 1, 1, -1, 1)
    samples = samples.reshape(B, -1, 3)                               # (B, S*P*T, 3)
    h_s = height(samples)
    lim = (dd.clamp(max=max_depth) - free_margin)[..., None].expand(B, S, P, t.numel()).reshape(B, -1)
    free_m = (t.view(1, 1, 1, -1).expand(B, S, P, -1).reshape(B, -1) < lim) \
        & (h_s > min_height) & (h_s < max_height)
    b2 = torch.arange(B, device=dev).view(B, 1).expand_as(free_m)
    free = grid.scatter_max(b2[free_m], samples[..., 0][free_m], samples[..., 2][free_m],
                            torch.ones_like(h_s[free_m]), B)

    occ_rad = torch.zeros_like(occ_cam)
    if radar is not None and radar_target_valid is not None and radar_target_valid.any():
        _, prior = decode_uncertainty(radar)
        sel = radar_target_valid
        rb = torch.arange(B, device=dev).view(B, 1).expand_as(sel)
        occ_rad = grid.scatter_max(rb[sel], radar[..., F_X][sel], radar[..., F_Z][sel], prior[sel], B)

    occupied = (occ_cam > 0) | (occ_rad > 0)
    target = occupied.float()
    weight = torch.where(occupied, torch.maximum(occ_cam, occ_rad), free)
    return target, weight, occ_rad


# ----------------------------------------------------------- velocity target
@torch.no_grad()
def radial_velocity_targets(radar, target_valid):
    """Held-out returns: BEV position, radial unit vector, Doppler speed, weight.

    Doppler only measures the radial component; the compensated pair
    (vx_comp, vz_comp) is that radial velocity resolved into cam0 axes.
    """
    std, prior = decode_uncertainty(radar)
    x, z = radar[..., F_X], radar[..., F_Z]
    rng = torch.sqrt(x * x + z * z).clamp_min(1e-3)
    ur = torch.stack((x / rng, z / rng), -1)
    v = torch.stack((radar[..., F_VX_COMP], radar[..., F_VZ_COMP]), -1)
    v_r = (v * ur).sum(-1)
    weight = prior / (1.0 + std['vlong'] ** 2)
    return dict(x=x, z=z, ur=ur, v_r=v_r, weight=weight * target_valid.float(), valid=target_valid)
