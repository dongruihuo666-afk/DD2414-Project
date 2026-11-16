"""Uncertainty-aware radar point encoder for Simple-BEV style BEV grids.

nuScenes stores the raw Continental ARS408-21 cluster fields (see the
nuScenes devkit ``RadarPointCloud`` docstring). Row layout of the 19-row
radar tensor produced by ``nuscenesdataset.get_radar_data``:

    0 x  1 y  2 z  3 dyn_prop  4 id  5 rcs  6 vx  7 vy  8 vx_comp  9 vy_comp
    10 is_quality_valid  11 ambig_state  12 x_rms  13 y_rms  14 invalid_state
    15 pdh0  16 vx_rms  17 vy_rms  18 time_lag (s, >= 0 for older sweeps)

The *_rms and pdh0 fields are enum codes, not metric values. They are used
twice: (1) decoded with the ARS408 tables below into metric standard
deviations / a false-alarm prior, which drive the splat extent and the point
confidence prior; (2) as raw codes through learned embeddings, so the network
can still use them if a table value is off.

All tensors handed to this module are expected in the reference camera frame
(cam0, X right, Y down, Z forward) with planar velocities rotated to
(v_x, v_z); see ``radar_ego_to_cam0``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.geom

F_X, F_Y, F_Z = 0, 1, 2
F_DYN, F_RCS = 3, 5
F_VX, F_VZ = 6, 7            # raw planar velocity (after rotation to cam0)
F_VX_COMP, F_VZ_COMP = 8, 9  # ego-motion compensated planar velocity
F_QUALITY, F_AMBIG = 10, 11
F_X_RMS, F_Y_RMS = 12, 13
F_INVALID, F_PDH0 = 14, 15
F_VX_RMS, F_VY_RMS = 16, 17
F_TIME = 18
NUM_FIELDS = 19

# ARS408-21 Cluster_2_Quality: DistLong_rms, DistLat_rms, VrelLong_rms,
# VrelLat_rms share one 5-bit table (metres or m/s). Code 31 is "invalid";
# it is mapped to the largest value so it can only *increase* uncertainty.
RMS_TABLE = (
    0.005, 0.006, 0.008, 0.011, 0.014, 0.018, 0.023, 0.029,
    0.038, 0.049, 0.063, 0.081, 0.105, 0.135, 0.174, 0.224,
    0.288, 0.371, 0.478, 0.616, 0.794, 1.023, 1.317, 1.697,
    2.187, 2.817, 3.630, 4.676, 6.025, 7.762, 10.000, 10.000,
)
# Cluster_2_Quality Pdh0 (false-alarm probability): 0 invalid, 1 <25%,
# 2 50%, 3 75%, 4 90%, 5 99%, 6 99.9%, 7 <=100%. Stored as P(real target),
# taking the bin midpoint; 0 (invalid) is treated as uninformative.
PDH0_P_REAL = (0.5, 0.875, 0.625, 0.375, 0.175, 0.055, 0.0055, 0.0005)
# ambig_state: 0 invalid, 1 ambiguous, 2 staggered ramp, 3 unambiguous,
# 4 stationary candidates.
AMBIG_P_REAL = (0.3, 0.5, 0.6, 1.0, 0.9)


def radar_ego_to_cam0(radar_ego, cam0_T_ego):
    """(B, 19, R) ego-frame radar -> (B, R, 19) cam0-frame radar.

    XYZ is transformed rigidly. The raw and compensated planar velocity pairs
    (rows 6:8, 8:10) are rotated and stored as (v_x, v_z) of cam0, i.e. the
    BEV plane. Padding rows (all-zero XYZ) stay exactly zero.
    """
    radar = radar_ego.permute(0, 2, 1).contiguous().float()
    valid = radar[..., :3].abs().sum(-1).gt(0) & torch.isfinite(radar).all(-1)
    out = radar.clone()
    out[..., :3] = utils.geom.apply_4x4(cam0_T_ego, radar[..., :3])
    rot = cam0_T_ego[:, :3, :3]
    for start in (F_VX, F_VX_COMP):
        planar = radar[..., start:start + 2]
        vel = torch.cat((planar, torch.zeros_like(planar[..., :1])), -1)
        vel_cam = torch.einsum('bij,brj->bri', rot, vel)
        out[..., start] = vel_cam[..., 0]
        out[..., start + 1] = vel_cam[..., 2]
    out = torch.where(valid[..., None], out, torch.zeros_like(out))
    return out, valid


def _codes(values, n):
    return values.round().long().clamp(0, n - 1)


def decode_uncertainty(radar):
    """Decode ARS408 enums. radar: (..., 19). Returns metric stds and prior."""
    table = radar.new_tensor(RMS_TABLE)
    std = {name: table[_codes(radar[..., idx], 32)] for name, idx in
           (('long', F_X_RMS), ('lat', F_Y_RMS), ('vlong', F_VX_RMS), ('vlat', F_VY_RMS))}
    p_pdh0 = radar.new_tensor(PDH0_P_REAL)[_codes(radar[..., F_PDH0], 8)]
    p_ambig = radar.new_tensor(AMBIG_P_REAL)[_codes(radar[..., F_AMBIG], 5)]
    p_invalid = torch.where(radar[..., F_INVALID].round().eq(0),
                            radar.new_tensor(1.0), radar.new_tensor(0.5))
    p_quality = torch.where(radar[..., F_QUALITY].gt(0.5),
                            radar.new_tensor(1.0), radar.new_tensor(0.5))
    prior = (p_pdh0 * p_ambig * p_invalid * p_quality).clamp(1e-3, 1 - 1e-3)
    return std, prior


class RadarUncertaintyEncoder(nn.Module):
    """Per-point MLP -> (feature, confidence) -> anisotropic Gaussian BEV splat.

    Radar has no usable elevation, so the output is a BEV map (B, C, Z, X).
    Each return is spread with a covariance built from its decoded radial /
    tangential position std (plus a motion term for older sweeps), weighted
    by its learned confidence. Outputs the normalised feature map and an
    evidence-based confidence map c_r = 1 - exp(-sum of weights).
    """

    def __init__(self, Z, X, xmin, xmax, zmin, zmax, out_channels=128,
                 hidden=64, window=4, min_std_m=0.25, max_std_m=2.0,
                 motion_compensate=True, unknown_tangential_speed=2.0):
        super().__init__()
        self.Z, self.X = Z, X
        self.xmin, self.xmax, self.zmin, self.zmax = map(float, (xmin, xmax, zmin, zmax))
        self.cell_x = (self.xmax - self.xmin) / X
        self.cell_z = (self.zmax - self.zmin) / Z
        self.window = window
        self.min_std_m, self.max_std_m = min_std_m, max_std_m
        self.motion_compensate = motion_compensate
        self.unknown_tangential_speed = unknown_tangential_speed
        self.out_channels = out_channels

        self.emb_dyn = nn.Embedding(8, 4)
        self.emb_ambig = nn.Embedding(5, 4)
        self.emb_invalid = nn.Embedding(32, 4)
        self.emb_pdh0 = nn.Embedding(8, 4)
        self.emb_quality = nn.Embedding(2, 2)
        self.emb_rms = nn.ModuleList([nn.Embedding(32, 4) for _ in range(4)])
        n_numeric = 15
        n_in = n_numeric + 4 * 4 + 2 + 4 * 4
        self.point_mlp = nn.Sequential(
            nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 2 * hidden), nn.LayerNorm(2 * hidden), nn.GELU(),
        )
        self.feat_out = nn.Linear(2 * hidden, out_channels)
        self.conf_out = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.conf_out.weight)
        nn.init.zeros_(self.conf_out.bias)
        offsets = torch.arange(-window, window + 1, dtype=torch.float32)
        dz, dx = torch.meshgrid(offsets, offsets, indexing='ij')
        self.register_buffer('offsets', torch.stack((dx.flatten(), dz.flatten()), -1), persistent=False)

    def point_inputs(self, radar):
        std, prior = decode_uncertainty(radar)
        x, y, z = radar[..., F_X], radar[..., F_Y], radar[..., F_Z]
        rng = torch.sqrt(x * x + z * z).clamp_min(1e-3)
        log_std = lambda s: (torch.log(s.clamp_min(1e-3)) - math.log(0.1)) / 2.0
        numeric = torch.stack((
            x / 50.0, z / 50.0, y / 5.0, rng / 50.0,
            (radar[..., F_RCS].clamp(-10, 40) - 10.0) / 20.0,
            radar[..., F_VX].clamp(-30, 30) / 15.0, radar[..., F_VZ].clamp(-30, 30) / 15.0,
            radar[..., F_VX_COMP].clamp(-30, 30) / 15.0, radar[..., F_VZ_COMP].clamp(-30, 30) / 15.0,
            radar[..., F_TIME].clamp(-1, 1) / 0.5,
            log_std(std['long']), log_std(std['lat']), log_std(std['vlong']), log_std(std['vlat']),
            torch.logit(prior),
        ), -1)
        emb = [self.emb_dyn(_codes(radar[..., F_DYN], 8)),
               self.emb_ambig(_codes(radar[..., F_AMBIG], 5)),
               self.emb_invalid(_codes(radar[..., F_INVALID], 32)),
               self.emb_pdh0(_codes(radar[..., F_PDH0], 8)),
               self.emb_quality(_codes(radar[..., F_QUALITY], 2))]
        emb += [e(_codes(radar[..., idx], 32)) for e, idx in
                zip(self.emb_rms, (F_X_RMS, F_Y_RMS, F_VX_RMS, F_VY_RMS))]
        return torch.cat([numeric] + emb, -1), std, prior

    def positions_and_covariance(self, radar, std):
        """BEV position (m) at the reference time and 2x2 covariance (m^2)."""
        x, z = radar[..., F_X], radar[..., F_Z]
        dt = radar[..., F_TIME].clamp(0, 1)
        if self.motion_compensate:
            # Older sweeps are ego-motion compensated but not object-motion
            # compensated; advance them by the compensated radial velocity.
            x = x + radar[..., F_VX_COMP] * dt
            z = z + radar[..., F_VZ_COMP] * dt
        rng = torch.sqrt(x * x + z * z).clamp_min(1e-3)
        ur = torch.stack((x / rng, z / rng), -1)           # radial unit vector
        ut = torch.stack((-ur[..., 1], ur[..., 0]), -1)    # tangential unit
        s_r = std['long'].clamp(self.min_std_m, self.max_std_m)
        # Tangential speed is unobserved by Doppler: grow lateral std with age.
        s_t = torch.sqrt(std['lat'] ** 2 + (self.unknown_tangential_speed * dt) ** 2)
        s_t = s_t.clamp(self.min_std_m, self.max_std_m)
        cov = (s_r[..., None, None] ** 2) * ur[..., :, None] * ur[..., None, :] \
            + (s_t[..., None, None] ** 2) * ut[..., :, None] * ut[..., None, :]
        return torch.stack((x, z), -1), cov

    def forward(self, radar, valid):
        """radar: (B, R, 19) cam0 frame; valid: (B, R) bool.

        Returns dict with 'feat' (B, C, Z, X), 'conf' (B, 1, Z, X),
        'evidence' (B, 1, Z, X) and 'point_conf' (B, R).
        """
        B, R, _ = radar.shape
        C = self.out_channels
        dev = radar.device
        feat_map = radar.new_zeros((B * self.Z * self.X, C))
        weight_map = radar.new_zeros((B * self.Z * self.X,))
        point_conf = radar.new_zeros((B, R))
        idx_b, idx_r = torch.nonzero(valid, as_tuple=True)
        if idx_b.numel() > 0:
            pts = radar[idx_b, idx_r]
            inputs, std, prior = self.point_inputs(pts)
            hidden = self.point_mlp(inputs)
            feats = self.feat_out(hidden)
            conf = torch.sigmoid(self.conf_out(hidden)[:, 0] + torch.logit(prior))
            point_conf = point_conf.index_put((idx_b, idx_r), conf)

            pos, cov = self.positions_and_covariance(pts, std)
            # Continuous cell coordinates (cell centres at integers).
            cx = (pos[:, 0] - self.xmin) / self.cell_x - 0.5
            cz = (pos[:, 1] - self.zmin) / self.cell_z - 0.5
            cell = torch.stack((cx.round(), cz.round()), -1)        # (P, 2)
            nb = cell[:, None, :] + self.offsets[None]              # (P, K, 2)
            d_m = torch.stack(((nb[..., 0] - cx[:, None]) * self.cell_x,
                               (nb[..., 1] - cz[:, None]) * self.cell_z), -1)
            # Add the cell's own discretisation variance so sub-cell stds do
            # not fall between cell centres, then normalise each kernel so a
            # return contributes exactly `conf` evidence (mass conserving).
            cov = cov + torch.diag(cov.new_tensor([self.cell_x ** 2, self.cell_z ** 2]) / 4.0)
            inv = torch.linalg.inv(cov)                             # (P, 2, 2)
            maha = torch.einsum('pki,pij,pkj->pk', d_m, inv, d_m)
            kern = torch.exp(-0.5 * maha)
            kern = kern / kern.sum(1, keepdim=True).clamp_min(1e-6)
            inb = (nb[..., 0] >= 0) & (nb[..., 0] < self.X) & (nb[..., 1] >= 0) & (nb[..., 1] < self.Z)
            w = kern * conf[:, None] * inb
            flat = (idx_b[:, None] * self.Z + nb[..., 1].clamp(0, self.Z - 1).long()) * self.X \
                + nb[..., 0].clamp(0, self.X - 1).long()
            flat, w = flat.flatten(), w.flatten()
            weight_map = weight_map.index_add(0, flat, w)
            feat_map = feat_map.index_add(
                0, flat, (w[:, None] * feats.repeat_interleave(self.offsets.shape[0], 0)))
        evidence = weight_map.view(B, 1, self.Z, self.X)
        feat = feat_map.view(B, self.Z, self.X, C).permute(0, 3, 1, 2)
        feat = feat / (evidence + 1e-4)
        conf_map = 1.0 - torch.exp(-evidence)
        return dict(feat=feat * conf_map, conf=conf_map, evidence=evidence,
                    point_conf=point_conf)


def flip_radar_points(radar, valid, flip_x, flip_z, x_center, z_center):
    """Mirror cam0 radar points (and velocities) for BEV flip augmentation."""
    radar = radar.clone()
    for flags, pos, vels, centre in ((flip_x, F_X, (F_VX, F_VX_COMP), x_center),
                                     (flip_z, F_Z, (F_VZ, F_VZ_COMP), z_center)):
        if flags is None or not flags.any():
            continue
        sel = flags[:, None] & valid
        radar[..., pos] = torch.where(sel, 2 * centre - radar[..., pos], radar[..., pos])
        for v in vels:
            radar[..., v] = torch.where(sel, -radar[..., v], radar[..., v])
    return radar
