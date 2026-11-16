"""Full-grid (200x8x200) integration on CPU: fine-tune step + probe feature path."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_nuscenes  # noqa: E402
import probe_bev  # noqa: E402
from nets.ssl_bev import SSLBEVNet  # noqa: E402
from ssl_pretrain.common import X, Y, Z, make_vox  # noqa: E402
from tests import test_ssl as T  # noqa: E402


def full_sample(B=1):
    s = list(T.make_sample(B))
    s[12] = torch.zeros(B, 1, 1, Z, X)
    s[12][..., 100:104, 100:104] = 1
    s[13] = torch.ones(B, 1, 1, Z, X)
    s[14] = torch.zeros(B, 1, 1, Z, X)
    s[15] = torch.zeros(B, 1, 2, Z, X)
    return tuple(s)


def small_ssl(use_radar=True, rand_flip=True):
    torch.manual_seed(0)
    return SSLBEVNet(Z, Y, X, make_vox(), latent_dim=128, use_radar=use_radar, rand_flip=rand_flip,
                     encoder_pretrained=False)


def test_finetune_run_model_ssl():
    model = torch.nn.DataParallel(small_ssl())
    model.train()
    loss_fn = train_nuscenes.SimpleLoss(2.13)
    loss, metrics = train_nuscenes.run_model(model, loss_fn, full_sample(), device='cpu')
    assert torch.isfinite(loss)
    loss.backward()
    assert model.module.decoder.first_conv.weight.grad.abs().sum() > 0
    assert model.module.radar_encoder.feat_out.weight.grad.abs().sum() > 0


def test_probe_backbone_ssl_branches(monkeypatch):
    monkeypatch.setattr(probe_bev, 'SSLBEVNet',
                        lambda *a, **k: SSLBEVNet(*a, **{**k, 'encoder_pretrained': False}))
    bb = probe_bev.Backbone('ssl', '', 'cpu', branch='fused')
    f, g = bb.feats(full_sample())
    assert f.shape == (1, 128, Z, X)
    for branch in ('cam', 'rad'):
        bb.branch = branch
        assert bb.feats(full_sample())[0].shape == (1, 128, Z, X)
    bb.radar_mode = 'empty'
    bb.branch = 'rad'
    assert bb.feats(full_sample())[0].abs().sum() == 0
    head = probe_bev.LinearProbe()
    assert head(f).shape == (1, 1, Z, X)
    logits, _ = bb.seg(full_sample())
    assert logits.shape == (1, 1, Z, X)
