"""Frozen-feature evaluation of BEV representations (vehicle IoU on nuScenes val).

The same protocol for every model: take the 128-d BEV map at the fusion /
compressor output, freeze the network, train a readout on train, evaluate on
the full val split.

  --readout=linear   BN(no affine) + 1x1 conv  (linear probe)
  --readout=decoder  Simple-BEV ResNet18 decoder trained on frozen features
  --readout=model    no training; evaluate the model's own decoder

  --model_type=ssl     SSLBEVNet (--branch fused|cam|rad)
  --model_type=segnet  Simple-BEV Segnet (official checkpoints; --use_radar,
                       --use_metaradar). Omit --init_dir for ImageNet init.

  --radar_mode=matched|empty|wrong   radar reliance control at eval time.

Examples:
  python probe_bev.py --model_type=segnet --init_dir=checkpoints/8x5_5e-4_rgb12_22:43:46 \
      --readout=linear --out=probe_results/simplebev_rgb_linear.json
  python probe_bev.py --model_type=ssl --init_dir=checkpoints_ssl/<run> --branch=fused \
      --readout=linear --out=probe_results/ssl_fused_linear.json
"""
import json
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
from fire import Fire
from torch.utils.data import DataLoader, Dataset

import utils.basic
import utils.geom
from nets.radar_uncertainty import radar_ego_to_cam0
from nets.segnet import Decoder, Segnet
from nets.ssl_bev import SSLBEVNet
from ssl_pretrain.common import X, Y, Z, make_loaders, make_vox
from ssl_pretrain.step import unpack_batch

THRESHOLDS = np.round(np.arange(0.1, 0.95, 0.05), 2)


def load_ckpt(init_dir, model, device):
    path = pathlib.Path(init_dir)
    if path.is_dir():
        files = sorted(path.glob('model-*.pth'), key=lambda p: int(p.stem.split('-')[1]))
        if not files:
            raise FileNotFoundError(f'no model-*.pth in {init_dir}')
        path = files[-1]
    state = torch.load(path, map_location=device, weights_only=False)['model_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'loaded {path}\n  missing: {missing}\n  unexpected: {unexpected}')
    return str(path), missing, unexpected


class SwapRadar(Dataset):
    """Same frames, radar taken from a frame half the split away (other scene)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = list(self.base[i])
        item[16] = self.base[(i + len(self.base) // 2) % len(self.base)][16]
        return tuple(item)


class LinearProbe(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.norm = nn.BatchNorm2d(c, affine=False)
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(self.norm(x))


class DecoderProbe(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.decoder = Decoder(in_channels=c, n_classes=1, predict_future_flow=False)

    def forward(self, x):
        return self.decoder(x)['segmentation']


class Backbone:
    """Wraps either model type behind feats(sample) / seg(sample)."""

    def __init__(self, model_type, init_dir, device, branch='fused', use_radar=True,
                 use_metaradar=True, radar_mode='matched'):
        self.model_type, self.branch, self.device = model_type, branch, device
        self.radar_mode = radar_mode
        vox = make_vox()
        self.vox = make_vox(device)
        if model_type == 'ssl':
            self.model = SSLBEVNet(Z, Y, X, vox, use_radar=use_radar)
        elif model_type == 'segnet':
            self.model = Segnet(Z, Y, X, vox, use_radar=use_radar, use_metaradar=use_metaradar,
                                do_rgbcompress=True, encoder_type='res101', rand_flip=False)
        else:
            raise ValueError(model_type)
        self.use_radar = use_radar
        self.model = self.model.to(device).eval()
        self.ckpt = load_ckpt(init_dir, self.model, device)[0] if init_dir else None
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _radar(self, g):
        if self.radar_mode == 'empty':
            g['radar_ego'] = torch.zeros_like(g['radar_ego'])
        return g

    def _segnet_inputs(self, g):
        if not self.use_radar:
            return None
        rad = g['radar_ego'].permute(0, 2, 1)                       # B, R, 19
        xyz = utils.geom.apply_4x4(g['cam0_T_ego'], rad[:, :, :3])
        if self.model.use_metaradar:
            return self.vox.voxelize_xyz_and_feats(xyz, rad[:, :, 3:], Z, Y, X, assert_cube=False)
        return self.vox.voxelize_xyz(xyz, Z, Y, X, assert_cube=False)

    def _ssl_kwargs(self, g):
        if not self.use_radar:
            return {}
        radar, valid = radar_ego_to_cam0(g['radar_ego'], g['cam0_T_ego'])
        return dict(radar=radar, radar_valid=valid)

    @torch.no_grad()
    def feats(self, sample):
        g = self._radar(unpack_batch(sample, self.device))
        rgb = g['imgs'] - 0.5
        if self.model_type == 'segnet':
            return self.model(rgb, g['pix_T_cams'], g['cam0_T_camXs'], self.vox,
                              rad_occ_mem0=self._segnet_inputs(g), return_bev=True), g
        out = self.model(rgb, g['pix_T_cams'], g['cam0_T_camXs'], mode='bev', **self._ssl_kwargs(g))
        return out[self.branch], g

    @torch.no_grad()
    def seg(self, sample):
        g = self._radar(unpack_batch(sample, self.device))
        rgb = g['imgs'] - 0.5
        if self.model_type == 'segnet':
            out = self.model(rgb, g['pix_T_cams'], g['cam0_T_camXs'], self.vox,
                             rad_occ_mem0=self._segnet_inputs(g))
        else:
            out = self.model(rgb, g['pix_T_cams'], g['cam0_T_camXs'], mode='seg', **self._ssl_kwargs(g))
        return out[2], g


def bce_loss(logits, target, valid, pos_weight=2.13):
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=logits.new_tensor([pos_weight]), reduction='none')
    return utils.basic.reduce_masked_mean(loss, valid)


def main(model_type='ssl', init_dir='', branch='fused', readout='linear', radar_mode='matched',
         use_radar=True, use_metaradar=True, data_dir='../nuscenes', dset='trainval', res_scale=2,
         nsweeps=5, batch_size=4, nworkers=8, train_iters=5000, lr=None, eval_max=0,
         out='probe_results/result.json', device='cuda:0', seed=125):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rotate = model_type == 'ssl'   # official Segnet checkpoints use unrotated velocities
    train_loader, val_loader = make_loaders(data_dir, dset, res_scale, batch_size, nworkers, nsweeps,
                                            augment=False, shuffle=True, do_shuffle_cams=False,
                                            rotate_radar_velocity=rotate)
    # deterministic, complete val pass (compile_data's val loader shuffles and drops the last batch)
    val_set = SwapRadar(val_loader.dataset) if radar_mode == 'wrong' else val_loader.dataset
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=nworkers)
    bb = Backbone(model_type, init_dir, device, branch, use_radar, use_metaradar, radar_mode)

    head = None
    if readout in ('linear', 'decoder'):
        head = (LinearProbe() if readout == 'linear' else DecoderProbe()).to(device)
        lr = lr or (1e-3 if readout == 'linear' else 5e-4)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_iters)
        head.train()
        it = iter(train_loader)
        for step in range(1, train_iters + 1):
            try:
                sample = next(it)
            except StopIteration:
                it = iter(train_loader)
                sample = next(it)
            f, g = bb.feats(sample)
            loss = bce_loss(head(f.float()), g['seg_bev'], g['valid_bev'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 200 == 0:
                print(f'probe step {step}/{train_iters} loss {loss.item():.4f}', flush=True)
        head.eval()

    tp = np.zeros(len(THRESHOLDS))
    fp = np.zeros(len(THRESHOLDS))
    fn = np.zeros(len(THRESHOLDS))
    n = 0
    with torch.no_grad():
        for sample in val_loader:
            if head is None:
                logits, g = bb.seg(sample)
            else:
                f, g = bb.feats(sample)
                logits = head(f.float())
            prob = torch.sigmoid(logits)
            gt = g['seg_bev'] > 0.5
            valid = g['valid_bev'] > 0.5
            for i, t in enumerate(THRESHOLDS):
                pred = prob > t
                tp[i] += (pred & gt & valid).sum().item()
                fp[i] += (pred & ~gt & valid).sum().item()
                fn[i] += (~pred & gt & valid).sum().item()
            n += prob.shape[0]
            if n % 400 < batch_size:
                print(f'eval {n}', flush=True)
            if eval_max and n >= eval_max:
                break
    iou = tp / np.maximum(tp + fp + fn, 1)
    i05 = int(np.argmin(np.abs(THRESHOLDS - 0.5)))
    result = dict(model_type=model_type, init_dir=init_dir, ckpt=bb.ckpt, branch=branch, readout=readout,
                  radar_mode=radar_mode, use_radar=use_radar, samples=n, train_iters=train_iters,
                  iou_at_0p5=float(iou[i05]), best_iou=float(iou.max()),
                  best_threshold=float(THRESHOLDS[iou.argmax()]),
                  iou_by_threshold={str(t): float(v) for t, v in zip(THRESHOLDS, iou)})
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == '__main__':
    Fire(main)
