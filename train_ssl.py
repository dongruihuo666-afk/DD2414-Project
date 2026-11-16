"""Self-supervised camera+radar BEV pretraining (no labels, no LiDAR).

Example (4 GPUs, effective batch 8 x 5):

    python train_ssl.py --exp_name=ssl_full --data_dir=../nuscenes \
        --dino_hub_dir='' --dav2_source=../Depth-Anything-V2 \
        --dav2_weights=../ckpts/depth_anything_v2_metric_vkitti_vits.pth \
        --batch_size=8 --grad_acc=5 --max_iters=25000 --device_ids=[0,1,2,3]

Ablations: --use_radar=False, --p_drop_camera=0 --p_drop_radar=0,
--weights='{"sem_rad":0,"occ_cam":0}' (no cross-modal), etc.
"""
import datetime
import json
import os
import random
import time

import numpy as np
import torch
from fire import Fire
from tensorboardX import SummaryWriter

import saverloader
from nets.ssl_bev import SSLBEVNet
from ssl_pretrain.common import Teachers, X, Z, make_loaders, make_vox
from ssl_pretrain.step import StepConfig, pretrain_step
from ssl_pretrain.targets import BEVGrid
from ssl_pretrain.teachers import load_depth_anything_metric, load_dino

torch.multiprocessing.set_sharing_strategy('file_system')


def main(exp_name='ssl', data_dir='../nuscenes', dset='trainval', log_dir='logs_ssl',
         ckpt_dir='checkpoints_ssl', max_iters=25000, batch_size=8, grad_acc=5, lr=3e-4,
         weight_decay=1e-7, nworkers=8, res_scale=2, nsweeps=5, use_radar=True,
         encoder_pretrained=True, dino_name='dinov2_vits14_reg', dino_hub_dir='',
         dav2_source='../Depth-Anything-V2', dav2_weights='', dav2_encoder='vits',
         radar_input_frac=0.7, p_drop_camera=0.2, p_drop_radar=0.3, max_depth=60.0,
         weights=None, log_freq=100, save_freq=1000, keep_latest=3, init_dir='',
         device_ids=(0,), seed=125, amp=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_ids = list(device_ids)
    device = 'cuda:%d' % device_ids[0]
    assert batch_size % len(device_ids) == 0
    weights = json.loads(weights) if isinstance(weights, str) else (weights or {})

    name = '%s_%dx%d_%s' % (exp_name, batch_size, grad_acc, datetime.datetime.now().strftime('%m%d_%H%M%S'))
    ckpt_dir = os.path.join(ckpt_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    config = {k: v for k, v in locals().items() if isinstance(v, (int, float, str, bool, list, dict))}
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    writer = SummaryWriter(os.path.join(log_dir, name), max_queue=10, flush_secs=60)

    train_loader, _ = make_loaders(data_dir, dset, res_scale, batch_size, nworkers, nsweeps, augment=True)
    image_hw = (int(224 * res_scale), int(400 * res_scale))

    vox = make_vox()
    grid = BEVGrid(vox, Z, X)
    model = SSLBEVNet(Z, 8, X, vox, use_radar=use_radar, encoder_pretrained=encoder_pretrained).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    teachers = Teachers(load_dino(dino_name, device, dino_hub_dir or None),
                        load_depth_anything_metric(dav2_source, dav2_weights, dav2_encoder, device=device),
                        image_hw)
    cfg = StepConfig(radar_input_frac=radar_input_frac, p_drop_camera=p_drop_camera,
                     p_drop_radar=p_drop_radar, max_depth=max_depth, weights=weights)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, max_iters + 100, pct_start=0.05,
                                                    cycle_momentum=False, anneal_strategy='linear')
    global_step = 0
    if init_dir:
        global_step = saverloader.load(init_dir, model.module, optimizer, scheduler,
                                       device_ids=device_ids)
    gen = torch.Generator().manual_seed(seed)
    it = iter(train_loader)
    model.train()
    while global_step < max_iters:
        global_step += 1
        t0 = time.time()
        sums = {}
        for _ in range(grad_acc):
            try:
                sample = next(it)
            except StopIteration:
                it = iter(train_loader)
                sample = next(it)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                loss, metrics, _ = pretrain_step(model, teachers, sample, grid, cfg, device, gen)
            (loss / grad_acc).backward()
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v / grad_acc
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if global_step % 10 == 0:
            for k, v in sums.items():
                writer.add_scalar('train/' + k, v, global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
        if global_step % log_freq == 0 or global_step == 1:
            print('%s step %06d/%d (%.2fs) ' % (name, global_step, max_iters, time.time() - t0)
                  + ' '.join('%s=%.4f' % kv for kv in sorted(sums.items())), flush=True)
        if global_step % save_freq == 0 or global_step == max_iters:
            saverloader.save(ckpt_dir, optimizer, model.module, global_step, scheduler=scheduler,
                             keep_latest=keep_latest)
    writer.close()


if __name__ == '__main__':
    Fire(main)
