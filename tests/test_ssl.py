"""CPU tests for the SSL camera+radar pipeline on a synthetic 6-camera rig.

Run from the repo root:  python -m pytest -q tests
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.geom  # noqa: E402
import utils.vox  # noqa: E402
from nets import radar_uncertainty as RU  # noqa: E402
from nets.ssl_bev import SSLBEVNet  # noqa: E402
from ssl_pretrain import losses as L  # noqa: E402
from ssl_pretrain import targets as T  # noqa: E402
from ssl_pretrain.step import StepConfig, pretrain_step, unpack_batch  # noqa: E402

Z, Y, X = 40, 4, 40
H, W = 64, 112
S = 6
TD = 16  # teacher dim for tests
BOUNDS = (-20, 20, -5, 5, -20, 20)
CENTROID = torch.tensor([[0.0, 1.0, 0.0]])


def make_vox():
    return utils.vox.Vox_util(Z, Y, X, scene_centroid=CENTROID, bounds=BOUNDS, assert_cube=False)


def rig(B=1):
    """nuScenes-like rig: 6 cameras at 1.5 m height, yaws 0, 60, ..., 300 deg.

    Camera 0 looks along ego +x. Returns rots (B,S,3,3) ego_from_cam, trans
    (B,S,3) and 4x4 intrinsics (B,S,4,4) for an H x W image with 70 deg HFOV.
    """
    rots, trans = [], []
    for s in range(S):
        yaw = math.radians(60 * s)
        z_c = [math.cos(yaw), math.sin(yaw), 0.0]
        x_c = [math.sin(yaw), -math.cos(yaw), 0.0]
        y_c = [0.0, 0.0, -1.0]
        rots.append(torch.tensor([x_c, y_c, z_c]).T)
        trans.append(torch.tensor([0.0, 0.0, 1.5]))
    f = (W / 2) / math.tan(math.radians(35))
    K = torch.eye(4)
    K[0, 0] = K[1, 1] = f
    K[0, 2], K[1, 2] = (W - 1) / 2, (H - 1) / 2
    rots = torch.stack(rots)[None].repeat(B, 1, 1, 1)
    trans = torch.stack(trans)[None].repeat(B, 1, 1)
    return rots, trans, K.view(1, 1, 4, 4).repeat(B, S, 1, 1)


def radar_points(xyz_ego, time=0.0, pdh0=1, rms=10, vel=(0.0, 0.0)):
    """(19, R) ego-frame radar rows."""
    R = len(xyz_ego)
    r = torch.zeros(19, R)
    r[:3] = torch.tensor(xyz_ego, dtype=torch.float32).T
    r[RU.F_QUALITY] = 1
    r[RU.F_AMBIG] = 3
    r[RU.F_PDH0] = pdh0
    for f in (RU.F_X_RMS, RU.F_Y_RMS, RU.F_VX_RMS, RU.F_VY_RMS):
        r[f] = rms
    r[RU.F_VX_COMP], r[RU.F_VZ_COMP] = vel
    r[RU.F_VX], r[RU.F_VZ] = vel
    r[RU.F_TIME] = time
    return r


def make_sample(B=1, R=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    rots, trans, K = rig(B)
    imgs = torch.rand((B, 1, S, 3, H, W), generator=g)
    pts = [[5 + 10 * torch.rand(1, generator=g).item(), 8 * (torch.rand(1, generator=g).item() - .5), 0.5]
           for _ in range(R)]
    radar = torch.stack([radar_points(pts, time=0.0 if i % 2 else 0.05)
                         for i in range(B)])[:, None]
    zeros = lambda *s: torch.zeros(B, 1, *s)
    return (imgs, rots[:, None], trans[:, None], K[:, None], zeros(3, 10), zeros(3, 10), zeros(3, 10),
            zeros(3, 10), zeros(150, 19), zeros(150), zeros(150), zeros(150), zeros(1, Z, X),
            torch.ones(B, 1, 1, Z, X), zeros(1, Z, X), zeros(2, Z, X), radar, torch.eye(4).repeat(B, 1, 1, 1))


class FakeTeachers:
    """Deterministic teachers: DINO = random fixed projection of pixels, depth = 12 m."""

    def __init__(self, depth=12.0):
        g = torch.Generator().manual_seed(1)
        self.proj = torch.randn(TD, 3, generator=g)
        self.const_depth = depth

    def dino(self, imgs):
        x = torch.nn.functional.avg_pool2d(imgs, 14)[..., :4, :8]
        return torch.einsum('dc,nchw->ndhw', self.proj, x)

    def depth(self, imgs):
        return torch.full((imgs.shape[0], 1, 28, 56), self.const_depth)


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return SSLBEVNet(Z, Y, X, make_vox(), latent_dim=32, depth_bins=16, dmin=1.0, dmax=32.0,
                     teacher_dim=TD, encoder_pretrained=False)


# ----------------------------------------------------------------- radar
def test_rms_table_monotone():
    t = torch.tensor(RU.RMS_TABLE)
    assert (t[1:] >= t[:-1]).all() and len(t) == 32
    assert len(RU.PDH0_P_REAL) == 8


def test_radar_to_cam0_and_splat_location():
    """A return 10 m ahead of ego lands 10 m along +z of cam0, in the right cell."""
    vox = make_vox()
    rots, trans, _ = rig()
    velo_T_cams = utils.geom.merge_rtlist(rots, trans)
    cam0_T_ego = utils.geom.safe_inverse(velo_T_cams[:, 0])
    r = radar_points([[10.3, -0.3, 0.5]], vel=(3.0, 0.0))[None]
    rc, valid = RU.radar_ego_to_cam0(r, cam0_T_ego)
    assert valid.all()
    # ego (x fwd, y left, z up) -> cam0 (x right, y down, z fwd); cam at 1.5 m
    assert torch.allclose(rc[0, 0, :3], torch.tensor([0.3, 1.0, 10.3]), atol=1e-4)
    # ego +x velocity becomes cam0 +z velocity
    assert torch.allclose(rc[0, 0, [RU.F_VX, RU.F_VZ]], torch.tensor([0.0, 3.0]), atol=1e-4)
    enc = RU.RadarUncertaintyEncoder(Z, X, vox.XMIN, vox.XMAX, vox.ZMIN, vox.ZMAX, out_channels=8,
                                     motion_compensate=False)
    out = enc(rc, valid)
    zc, xc = np.unravel_index(out['evidence'][0, 0].argmax().item(), (Z, X))
    mem = vox.Ref2Mem(rc[:, :, :3], Z, Y, X)[0, 0]
    assert (xc, zc) == (round(mem[0].item()), round(mem[2].item()))


def test_radar_uncertainty_widens_splat():
    vox = make_vox()
    enc = RU.RadarUncertaintyEncoder(Z, X, vox.XMIN, vox.XMAX, vox.ZMIN, vox.ZMAX, out_channels=8)
    pt = torch.zeros(1, 1, 19)
    pt[0, 0, RU.F_Z] = 10.0
    pt[0, 0, RU.F_AMBIG] = 3
    pt[0, 0, RU.F_QUALITY] = 1
    pt[0, 0, RU.F_PDH0] = 1
    sharp, wide = pt.clone(), pt.clone()
    sharp[..., [RU.F_X_RMS, RU.F_Y_RMS]] = 5      # 0.018 m -> floor 0.25 m
    wide[..., [RU.F_X_RMS, RU.F_Y_RMS]] = 24      # 2.187 m
    v = torch.ones(1, 1, dtype=torch.bool)
    e_sharp = enc(sharp, v)['evidence']
    e_wide = enc(wide, v)['evidence']
    assert (e_wide > 1e-3).sum() > (e_sharp > 1e-3).sum()
    assert e_sharp.max() > e_wide.max()


def test_low_pdh0_lowers_confidence():
    vox = make_vox()
    enc = RU.RadarUncertaintyEncoder(Z, X, vox.XMIN, vox.XMAX, vox.ZMIN, vox.ZMAX, out_channels=8)
    good = radar_points([[0, 0, 0]], pdh0=1).T[None].clone()
    bad = radar_points([[0, 0, 0]], pdh0=6).T[None].clone()
    good[..., RU.F_Z] = bad[..., RU.F_Z] = 10.0
    v = torch.ones(1, 1, dtype=torch.bool)
    assert enc(good, v)['point_conf'] > enc(bad, v)['point_conf']


def test_encoder_empty_input():
    vox = make_vox()
    enc = RU.RadarUncertaintyEncoder(Z, X, vox.XMIN, vox.XMAX, vox.ZMIN, vox.ZMAX, out_channels=8)
    out = enc(torch.zeros(2, 5, 19), torch.zeros(2, 5, dtype=torch.bool))
    assert out['feat'].abs().sum() == 0 and out['conf'].sum() == 0


def test_split_disjoint_and_recent_only():
    radar = torch.zeros(2, 200, 19)
    radar[..., RU.F_TIME] = torch.rand(2, 200) * 0.5
    valid = torch.rand(2, 200) > 0.1
    inp, tgt = T.split_radar(radar, valid, 0.7, 0.1, torch.Generator().manual_seed(0))
    assert not (inp & tgt).any()
    assert ((inp | tgt) == valid).all()
    assert (radar[..., RU.F_TIME][tgt] <= 0.1).all()


def test_flip_radar_mirrors_position_and_velocity():
    r = torch.zeros(1, 1, 19)
    r[0, 0, RU.F_X], r[0, 0, RU.F_VX] = 3.0, 2.0
    out = RU.flip_radar_points(r, torch.ones(1, 1, dtype=torch.bool), torch.tensor([True]),
                               torch.tensor([False]), 0.0, 0.0)
    assert out[0, 0, RU.F_X] == -3.0 and out[0, 0, RU.F_VX] == -2.0


# ----------------------------------------------------------- targets
def _geom(B=1):
    return unpack_batch(make_sample(B), 'cpu')


def test_semantic_target_is_localized():
    """Front-camera patches at 12 m land ~12 m ahead, not along the whole ray."""
    g = _geom()
    grid = T.BEVGrid(make_vox(), Z, X)
    dino = torch.zeros(1, S, TD, 4, 8)
    dino[:, 0, 0] = 1.0                      # front camera only, channel 0
    depth = torch.full((1, S, 1, 28, 56), 12.0)
    tgt, cov = T.semantic_bev_target(dino, depth, g['pix_T_cams'], g['cam0_T_camXs'], (H, W), grid)
    zs = torch.nonzero((cov[0, 0] > 0.3) & (tgt[0, 0] > 0.9))[:, 0]
    z_m = grid.zmin + (zs.float() + 0.5) * grid.cz
    assert len(zs) > 0
    assert (z_m - 12.0).abs().max() < 3.0, z_m


def test_semantic_target_independent_of_radar_input():
    """Shortcut test (a): the target does not depend on which radar the student sees."""
    torch.manual_seed(0)
    sample = make_sample()
    teach = FakeTeachers()
    grid = T.BEVGrid(make_vox(), Z, X)
    net = SSLBEVNet(Z, Y, X, make_vox(), latent_dim=16, depth_bins=16, dmax=32.0, teacher_dim=TD,
                    encoder_pretrained=False)
    cfg = StepConfig(p_drop_camera=0.0, p_drop_radar=0.0)
    _, _, e1 = pretrain_step(net, teach, sample, grid, cfg, 'cpu', torch.Generator().manual_seed(1))
    _, _, e2 = pretrain_step(net, teach, sample, grid, cfg, 'cpu', torch.Generator().manual_seed(2))
    assert not torch.equal(e1['in_valid'], e2['in_valid'])
    assert torch.allclose(e1['sem_t'], e2['sem_t']) and torch.allclose(e1['cov'], e2['cov'])


def test_occupancy_target_free_before_surface():
    g = _geom()
    grid = T.BEVGrid(make_vox(), Z, X)
    depth = torch.full((1, S, 1, 28, 56), 12.0)
    tgt, w, _ = T.occupancy_target(depth, g['pix_T_cams'], g['cam0_T_camXs'], g['ego_T_cam0'],
                                   (H, W), grid, ray_hw=(14, 28), max_depth=19.0)
    col = X // 2
    def cell(zm):
        return int((zm - grid.zmin) / grid.cz)
    assert w[0, 0, cell(6.0), col] > 0 and tgt[0, 0, cell(6.0), col] == 0     # free in front
    assert tgt[0, 0, cell(12.0), col] == 1                                     # surface
    assert w[0, 0, cell(17.0), col] == 0                                       # unknown behind


def test_depth_scale_alignment_recovers_factor():
    g = _geom()
    radar, valid = RU.radar_ego_to_cam0(g['radar_ego'], g['cam0_T_ego'])
    # teacher depth = true depth / 2 everywhere radar hits the front camera
    u, v, z = T.project_to_cameras(radar[..., :3], g['cam0_T_camXs'], g['pix_T_cams'])
    depth = torch.full((1, S, 1, H, W), 5.0)
    m = valid[0] & (u[0, 0] >= 0) & (u[0, 0] <= W - 1) & (v[0, 0] >= 0) & (v[0, 0] <= H - 1) & (z[0, 0] > 2)
    depth[0, 0, 0, v[0, 0, m].round().long(), u[0, 0, m].round().long()] = z[0, 0, m] / 2
    scale, count = T.align_depth_scale(depth, radar, valid, g['cam0_T_camXs'], g['pix_T_cams'], (H, W))
    assert count[0, 0] >= 5
    assert abs(scale[0, 0].item() - 2.0) < 0.05


# ------------------------------------------------------------- model
def test_lifting_localizes_along_ray(model):
    """Shortcut test (c): with a peaked depth distribution, lifted mass sits at
    that depth instead of being copied along the whole ray (Simple-BEV)."""
    g = _geom()
    B = 1
    feats = torch.ones(S, 32, 8, 14)
    logp = torch.full((S, model.bins.n, 8, 14), -30.0)
    k = int(round(model.bins.index(torch.tensor(12.0)).item()))
    logp[:, k] = 0.0
    logp = torch.log_softmax(logp, 1)
    pix = utils.geom.scale_intrinsics(g['pix_T_cams'][0], 14 / W, 8 / H)
    camXs_T_cam0 = utils.geom.safe_inverse(g['cam0_T_camXs'][0])
    vol, mass, vis = model.lift(feats, logp, pix, camXs_T_cam0, B, S)
    col = mass[0, 0, :, X // 2]
    zs = torch.arange(Z).float()
    z_m = make_vox().ZMIN + (zs + 0.5) * (40.0 / Z)
    front = z_m > 1
    peak_z = z_m[front][col[front].argmax()]
    assert abs(peak_z - 12.0) < 2.5
    near = col[(z_m > 2) & (z_m < 6)].sum()
    assert near < 0.05 * col[front].sum()


def test_full_step_backward_and_branch_gradients(model):
    teach = FakeTeachers()
    grid = T.BEVGrid(make_vox(), Z, X)
    cfg = StepConfig(p_drop_camera=0.0, p_drop_radar=0.0)
    model.zero_grad()
    loss, metrics, extras = pretrain_step(model, teach, make_sample(B=2), grid, cfg, 'cpu',
                                          torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    loss.backward()
    for name, mod in [('encoder', model.encoder), ('depth_head', model.depth_head),
                      ('radar_encoder', model.radar_encoder), ('fusion', model.fusion),
                      ('sem_rad', model.sem_heads['rad']), ('vel_head', model.vel_head)]:
        grads = [p.grad.abs().sum() for p in mod.parameters() if p.grad is not None]
        assert grads and sum(grads) > 0, name
    for k in ('image', 'depth', 'sem_cam', 'sem_fused', 'sem_rad', 'occ_cam', 'occ_rad', 'occ_fused'):
        assert metrics[k] > 0, k


def test_modality_dropout_never_both(model):
    teach = FakeTeachers()
    grid = T.BEVGrid(make_vox(), Z, X)
    cfg = StepConfig(p_drop_camera=0.5, p_drop_radar=1.0)
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        _, _, e = pretrain_step(model, teach, make_sample(B=4), grid, cfg, 'cpu', gen)
        assert not (e['drop_cam'] & e['drop_rad']).any()
        assert e['drop_cam'].any() or e['drop_rad'].any()


def test_seg_mode_with_flip_shapes(model):
    g = _geom(B=2)
    radar, valid = RU.radar_ego_to_cam0(g['radar_ego'], g['cam0_T_ego'])
    model.rand_flip = True
    model.train()
    out = model(g['imgs'] - 0.5, g['pix_T_cams'], g['cam0_T_camXs'], radar=radar, radar_valid=valid,
                mode='seg')
    model.rand_flip = False
    assert out[2].shape == (2, 1, Z, X)
    out[2].sum().backward()
