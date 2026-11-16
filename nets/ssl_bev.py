"""Camera + radar BEV network for self-supervised pretraining and fine-tuning.

Differences to Segnet that matter for the SSL objectives:
  * camera lifting is depth-aware (LSS-style density along each ray), so a
    camera BEV cell only receives a pixel's feature where depth mass is;
  * radar goes through ``RadarUncertaintyEncoder`` (uses the ARS408 quality
    fields) instead of a binary occupancy grid;
  * camera and radar are fused with confidence-driven gates;
  * camera-only, radar-only and fused BEV maps are all exposed, each with
    its own SSL heads, so cross-modal losses can target one branch at a time;
  * modality dropout zeroes a whole branch (features and confidence).

``forward(mode='seg')`` returns the Segnet tuple (raw, feat, seg, center,
offset), so ``train_nuscenes.py`` can fine-tune it unchanged.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint

import utils.basic
import utils.geom
from nets.fusion import UncertaintyGatedFusion
from nets.radar_uncertainty import RadarUncertaintyEncoder, flip_radar_points
from nets.segnet import Decoder, UpsamplingConcat


class CameraEncoder(nn.Module):
    """Simple-BEV's ResNet101 encoder (stride 8) with optional ImageNet init."""

    def __init__(self, C, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet101(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3
        self.upsampling_layer = UpsamplingConcat(1536, 512)
        self.depth_layer = nn.Conv2d(512, C, kernel_size=1)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        return self.depth_layer(self.upsampling_layer(x2, x1))


class LayerNorm2d(nn.Module):
    """Per-cell channel LayerNorm: no spatial statistics, so sparse maps
    (radar) are not blown up the way InstanceNorm amplifies a few cells."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def head(cin, cout, hidden=None, sparse=False):
    hidden = hidden or cin
    norm = LayerNorm2d(hidden) if sparse else nn.InstanceNorm2d(hidden)
    return nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1, bias=False),
                         norm, nn.GELU(), nn.Conv2d(hidden, cout, 1))


class DepthBins:
    """Log-spaced depth bins shared by the student head and the targets."""

    def __init__(self, n=64, dmin=1.0, dmax=64.0):
        self.n, self.dmin, self.dmax = n, dmin, dmax
        self.log_min, self.log_max = math.log(dmin), math.log(dmax)
        self.dlog = (self.log_max - self.log_min) / (n - 1)

    def centers(self, device=None):
        return torch.exp(torch.linspace(self.log_min, self.log_max, self.n, device=device))

    def index(self, depth):
        """Continuous bin coordinate (0..n-1) of metric depth."""
        return (torch.log(depth.clamp_min(1e-3)) - self.log_min) / self.dlog

    def bin_width(self, depth):
        return depth * self.dlog


class SSLBEVNet(nn.Module):
    def __init__(self, Z, Y, X, vox_util, latent_dim=128, depth_bins=64, dmin=1.0,
                 dmax=64.0, teacher_dim=384, use_radar=True, rand_flip=False,
                 encoder_pretrained=True):
        super().__init__()
        self.Z, self.Y, self.X = Z, Y, X
        self.latent_dim = latent_dim
        self.use_radar = use_radar
        self.rand_flip = rand_flip
        self.bins = DepthBins(depth_bins, dmin, dmax)
        self.cell_xz = (vox_util.XMAX - vox_util.XMIN) / X
        self.x_center = 0.5 * (vox_util.XMIN + vox_util.XMAX)
        self.z_center = 0.5 * (vox_util.ZMIN + vox_util.ZMAX)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        xyz_mem = utils.basic.gridcloud3d(1, Z, Y, X, norm=False, device='cpu')
        self.register_buffer('xyz_cam0', vox_util.Mem2Ref(xyz_mem, Z, Y, X, assert_cube=False),
                             persistent=False)

        C = latent_dim
        self.encoder = CameraEncoder(C, pretrained=encoder_pretrained)
        self.depth_head = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(C), nn.ReLU(inplace=True),
                                        nn.Conv2d(C, depth_bins, 1))
        self.cam_compressor = nn.Sequential(nn.Conv2d(C * Y, C, 3, padding=1, bias=False),
                                            nn.InstanceNorm2d(C), nn.GELU())
        if use_radar:
            self.radar_encoder = RadarUncertaintyEncoder(
                Z, X, vox_util.XMIN, vox_util.XMAX, vox_util.ZMIN, vox_util.ZMAX, out_channels=C)
            self.radar_context = nn.Sequential(
                nn.Conv2d(C + 1, C, 3, padding=1, bias=False), LayerNorm2d(C), nn.GELU(),
                nn.Conv2d(C, C, 3, padding=1, bias=False), LayerNorm2d(C), nn.GELU())
            self.fusion = UncertaintyGatedFusion(C, cam_conf_channels=2, rad_conf_channels=1)
        else:
            self.fusion = None

        # SSL heads (unused in 'seg' mode; ignored when fine-tuning).
        self.image_head = head(C, teacher_dim)
        branches = ['cam', 'fused'] + (['rad'] if use_radar else [])
        self.sem_heads = nn.ModuleDict({b: head(C, teacher_dim, sparse=b == 'rad') for b in branches})
        self.occ_heads = nn.ModuleDict({b: head(C, 2, sparse=b == 'rad') for b in branches})  # logit, log-var
        self.vel_head = head(C, 2) if use_radar else None

        # Supervised decoder (Simple-BEV) and its loss-balancing weights.
        self.decoder = Decoder(in_channels=C, n_classes=1, predict_future_flow=False)
        self.ce_weight = nn.Parameter(torch.tensor(0.0))
        self.center_weight = nn.Parameter(torch.tensor(0.0))
        self.offset_weight = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------------ camera
    def lift(self, feats, depth_logp, featpix_T_cams, camXs_T_cam0, B, S):
        """Depth-weighted lifting into cam0 voxels.

        feats: (B*S, C, h, w); depth_logp: (B*S, D, h, w). Returns
        volume (B, C, Z, Y, X), surface mass (B, 1, Z, X), visibility (B, 1, Z, X).
        """
        Z, Y, X = self.Z, self.Y, self.X
        N, C, h, w = feats.shape
        xyz = self.xyz_cam0.expand(N, -1, -1)
        xyz_cam = utils.geom.apply_4x4(camXs_T_cam0, xyz)
        pix = utils.geom.apply_4x4(utils.basic.matmul2(featpix_T_cams, camXs_T_cam0), xyz)
        depth = xyz_cam[..., 2]
        u = pix[..., 0] / pix[..., 2].clamp_min(1e-6)
        v = pix[..., 1] / pix[..., 2].clamp_min(1e-6)
        valid = (depth > 0) & (u > -0.5) & (u < w - 0.5) & (v > -0.5) & (v < h - 0.5)
        nx = 2 * u / (w - 1) - 1
        ny = 2 * v / (h - 1) - 1
        nd = 2 * self.bins.index(depth) / (self.bins.n - 1) - 1
        grid3 = torch.stack((nx, ny, nd), -1).view(N, Z, Y, X, 3)
        prob = F.grid_sample(depth_logp.exp().unsqueeze(1), grid3, align_corners=True,
                             padding_mode='zeros')                       # (N,1,Z,Y,X)
        density = (self.cell_xz / self.bins.bin_width(depth.clamp_min(1e-3))).clamp(max=1.0)
        wgt = (prob * density.view(N, 1, Z, Y, X)).clamp(max=1.0) * valid.view(N, 1, Z, Y, X)
        grid2 = torch.stack((nx, ny), -1).view(N, Z, Y * X, 2)
        sampled = F.grid_sample(feats, grid2, align_corners=True).view(N, C, Z, Y, X)
        vol = (sampled * wgt).view(B, S, C, Z, Y, X).sum(1)
        mass = wgt.view(B, S, 1, Z, Y, X).sum((1, 4))
        vis = valid.view(B, S, 1, Z, Y, X).any(1).float().mean(3)
        return vol, mass, vis

    def camera_branch(self, rgb_camXs, pix_T_cams, cam0_T_camXs):
        B, S, _, H, W = rgb_camXs.shape
        rgb = utils.basic.pack_seqdim(rgb_camXs, B)
        rgb = (rgb + 0.5 - self.mean) / self.std
        feats = self.encoder(rgb)
        depth_logp = F.log_softmax(self.depth_head(feats).float(), dim=1)
        _, C, h, w = feats.shape
        pix_T_cams_ = utils.basic.pack_seqdim(pix_T_cams, B)
        featpix_T_cams = utils.geom.scale_intrinsics(pix_T_cams_, w / float(W), h / float(H))
        camXs_T_cam0 = utils.geom.safe_inverse(utils.basic.pack_seqdim(cam0_T_camXs, B))
        if torch.is_grad_enabled():
            # recompute the (large) lifted volume in backward instead of storing it
            vol, mass, vis = checkpoint(self.lift, feats.float(), depth_logp, featpix_T_cams,
                                        camXs_T_cam0, B, S, use_reentrant=False)
        else:
            vol, mass, vis = self.lift(feats.float(), depth_logp, featpix_T_cams, camXs_T_cam0, B, S)
        return feats, depth_logp, vol, mass, vis

    # ------------------------------------------------------------------ forward
    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, radar=None, radar_valid=None,
                drop_camera=None, drop_radar=None, mode='ssl'):
        """mode: 'ssl' -> dict of SSL outputs, 'bev' -> dict of BEV maps,
        'seg' -> Segnet-compatible tuple (raw, feat, seg, center, offset)."""
        B = rgb_camXs.shape[0]
        feats, depth_logp, vol, mass, vis = self.camera_branch(rgb_camXs, pix_T_cams, cam0_T_camXs)

        flip_x = flip_z = None
        if mode == 'seg' and self.rand_flip and self.training:
            flip_x = torch.from_numpy(np.random.choice([0, 1], B).astype(bool)).to(vol.device)
            flip_z = torch.from_numpy(np.random.choice([0, 1], B).astype(bool)).to(vol.device)
            vol = torch.where(flip_x.view(B, 1, 1, 1, 1), vol.flip(-1), vol)
            vol = torch.where(flip_z.view(B, 1, 1, 1, 1), vol.flip(-3), vol)
            mass = torch.where(flip_x.view(B, 1, 1, 1), mass.flip(-1), mass)
            mass = torch.where(flip_z.view(B, 1, 1, 1), mass.flip(-2), mass)
            vis = torch.where(flip_x.view(B, 1, 1, 1), vis.flip(-1), vis)
            vis = torch.where(flip_z.view(B, 1, 1, 1), vis.flip(-2), vis)

        cam_bev = vol.permute(0, 1, 3, 2, 4).reshape(B, self.latent_dim * self.Y, self.Z, self.X)
        f_cam = self.cam_compressor(cam_bev)
        c_cam = torch.cat((vis, 1.0 - torch.exp(-mass)), 1)
        if drop_camera is not None:
            keep = (~drop_camera).float().view(B, 1, 1, 1)
            f_cam, c_cam = f_cam * keep, c_cam * keep

        out = dict(cam=f_cam, cam_conf=c_cam)
        if self.use_radar:
            if radar is None:
                radar = rgb_camXs.new_zeros((B, 1, 19))
                radar_valid = torch.zeros((B, 1), dtype=torch.bool, device=rgb_camXs.device)
            if flip_x is not None:
                radar = flip_radar_points(radar, radar_valid, flip_x, flip_z, self.x_center, self.z_center)
            r = self.radar_encoder(radar, radar_valid)
            f_rad = self.radar_context(torch.cat((r['feat'], r['conf']), 1)) * r['conf'].gt(0).float()
            c_rad = r['conf']
            if drop_radar is not None:
                keep = (~drop_radar).float().view(B, 1, 1, 1)
                f_rad, c_rad = f_rad * keep, c_rad * keep
            fused, gates = self.fusion(f_cam, c_cam, f_rad, c_rad)
            out.update(rad=f_rad, rad_conf=c_rad, fused=fused, gates=gates,
                       point_conf=r['point_conf'])
        else:
            out['fused'] = f_cam

        if mode == 'bev':
            return out
        if mode == 'seg':
            dec = self.decoder(out['fused'], None if flip_x is None else
                               (flip_x.cpu().numpy(), flip_z.cpu().numpy()))
            return (dec['raw_feat'], dec['feat'], dec['segmentation'],
                    dec['instance_center'], dec['instance_offset'])

        out['depth_logp'] = depth_logp
        out['image_sem'] = self.image_head(feats)
        for name in self.sem_heads:
            out['sem_' + name] = self.sem_heads[name](out[name])
            occ = self.occ_heads[name](out[name])
            out['occ_' + name] = occ[:, :1]
            out['occ_logvar_' + name] = occ[:, 1:].clamp(-4, 4)
        if self.vel_head is not None:
            out['vel'] = self.vel_head(out['fused'])
        return out
