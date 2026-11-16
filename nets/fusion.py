import torch
import torch.nn as nn


class UncertaintyGatedFusion(nn.Module):
    """Fuse camera and radar BEV features with confidence-driven gates.

    The gate network sees both feature maps *and* their confidence maps
    (camera: visibility + expected surface mass; radar: evidence), so it can
    learn to trust radar where it has strong returns and camera elsewhere.
    A dropped modality arrives as zeros with zero confidence.
    """

    def __init__(self, channels=128, cam_conf_channels=2, rad_conf_channels=1):
        super().__init__()
        n_conf = cam_conf_channels + rad_conf_channels
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels + n_conf, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.out = nn.Sequential(
            nn.Conv2d(2 * channels + n_conf, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels), nn.GELU(),
        )

    def forward(self, feat_cam, conf_cam, feat_rad, conf_rad):
        x = torch.cat((feat_cam, feat_rad, conf_cam, conf_rad), 1)
        gates = torch.sigmoid(self.gate(x))
        g_cam, g_rad = gates[:, :1], gates[:, 1:]
        fused = self.out(torch.cat((g_cam * feat_cam, g_rad * feat_rad, conf_cam, conf_rad), 1))
        return fused, gates
