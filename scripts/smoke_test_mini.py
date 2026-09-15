#!/usr/bin/env python3
"""One-batch nuScenes mini smoke test for Simple-BEV."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import nuscenesdataset  # noqa: E402
import saverloader  # noqa: E402
from nets.segnet import Segnet  # noqa: E402
from train_nuscenes import (  # noqa: E402
    SimpleLoss,
    Z,
    Y,
    X,
    bounds,
    run_model,
    scene_centroid,
    scene_centroid_py,
)


BATCH_NAMES = (
    'images', 'rotations', 'translations', 'intrinsics',
    'lidar_current', 'lidar_current_extra', 'lidar_sweeps',
    'lidar_sweeps_extra', 'boxes_lrt', 'visibility', 'track_ids',
    'scores', 'segmentation_bev', 'valid_bev', 'center_bev',
    'offset_bev', 'radar', 'ego_pose',
)


def describe(name, value):
    if torch.is_tensor(value):
        print(
            f'{name:24s} shape={tuple(value.shape)!s:24s} '
            f'dtype={value.dtype} device={value.device}'
        )
    elif isinstance(value, (list, tuple)):
        print(f'{name:24s} type={type(value).__name__} len={len(value)}')
    else:
        print(f'{name:24s} type={type(value).__name__}')


def make_visualization(outputs, output_path):
    rgb = (outputs['rgb_camXs'][0].detach().cpu() + 0.5).clamp(0, 1)
    camera_images = []
    for camera in rgb:
        array = (camera.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        camera_images.append(Image.fromarray(array))

    camera_w, camera_h = camera_images[0].size
    canvas = Image.new('RGB', (camera_w * 3, camera_h * 2 + 240), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, camera in enumerate(camera_images):
        x = (index % 3) * camera_w
        y = (index // 3) * camera_h
        canvas.paste(camera, (x, y))
        draw.text((x + 5, y + 5), f'camera {index}', fill='yellow')

    prediction = torch.sigmoid(outputs['seg_bev_logits'][0, 0]).detach().cpu().numpy()
    ground_truth = outputs['seg_bev_gt'][0, 0].detach().cpu().numpy()
    pred_rgb = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    pred_rgb[..., 0] = (prediction * 255).astype(np.uint8)
    gt_rgb = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    gt_rgb[..., 1] = (ground_truth * 255).astype(np.uint8)
    pred_image = Image.fromarray(pred_rgb).resize((200, 200), Image.Resampling.NEAREST)
    gt_image = Image.fromarray(gt_rgb).resize((200, 200), Image.Resampling.NEAREST)
    panel_y = camera_h * 2 + 30
    canvas.paste(pred_image, (10, panel_y))
    canvas.paste(gt_image, (230, panel_y))
    draw.text((10, panel_y - 20), 'prediction probability', fill='black')
    draw.text((230, panel_y - 20), 'ground truth', fill='black')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--visualization-name')
    parser.add_argument('--dump-npz', type=Path)
    parser.add_argument('--encoder-type', default='effb0', choices=('effb0', 'effb4', 'res50', 'res101'))
    parser.add_argument('--res-scale', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--nsweeps', type=int, default=1)
    parser.add_argument('--use-radar', action='store_true')
    parser.add_argument('--use-metaradar', action='store_true')
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--backward', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    required = ('samples', 'sweeps', 'maps', 'v1.0-mini')
    missing = [name for name in required if not (args.data_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f'nuScenes mini is incomplete at {args.data_root}; missing: {missing}'
        )
    if args.use_metaradar and not args.use_radar:
        raise ValueError('--use-metaradar requires --use-radar')

    # Encoder skip connections require stride-8 features to be exactly twice
    # the stride-16 size. Align reduced smoke-test inputs to 16 pixels.
    final_dim = tuple(
        max(16, int(size * args.res_scale) // 16 * 16)
        for size in (224, 400)
    )
    data_aug_conf = {
        'crop_offset': 0,
        'resize_lim': [1.0, 1.0],
        'final_dim': final_dim,
        'H': 900,
        'W': 1600,
        'cams': [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
        ],
        'ncams': 6,
    }

    print(f'data_root: {args.data_root}')
    print(f'image_resolution: {final_dim}')
    load_start = time.perf_counter()
    train_loader, _ = nuscenesdataset.compile_data(
        'mini', str(args.data_root), data_aug_conf=data_aug_conf,
        centroid=scene_centroid_py, bounds=bounds, res_3d=(Z, Y, X),
        bsz=1, nworkers=args.num_workers, nworkers_val=args.num_workers,
        shuffle=False, nsweeps=args.nsweeps, seqlen=1, refcam_id=1,
        get_tids=True, temporal_aug=False, use_radar_filters=False,
        do_shuffle_cams=False,
    )
    batch = next(iter(train_loader))
    print(f'data_load_seconds: {time.perf_counter() - load_start:.3f}')
    print('batch tensors:')
    for name, value in zip(BATCH_NAMES, batch):
        describe(name, value)

    radar = batch[16]
    valid_radar_points = (radar[:, :, :3].abs().sum(dim=2) > 0).sum().item()
    print(f'valid_radar_points: {valid_radar_points}')
    if args.data_only:
        print('DATA_SMOKE_TEST_OK')
        return

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run this script where WSL GPU access is allowed')
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    vox_util = nuscenesdataset.utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    model = Segnet(
        Z, Y, X, vox_util=vox_util, use_radar=args.use_radar,
        use_metaradar=args.use_metaradar, do_rgbcompress=True,
        encoder_type=args.encoder_type, rand_flip=False,
        pretrained_backbone=args.checkpoint is None,
    ).to(device)
    if args.checkpoint:
        saverloader.load(str(args.checkpoint), model)
    loss_fn = SimpleLoss(2.13).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train(args.backward)

    start = time.perf_counter()
    with torch.set_grad_enabled(args.backward), torch.autocast(
        device_type='cuda', dtype=torch.float16, enabled=args.amp
    ):
        loss, metrics, outputs = run_model(
            model, loss_fn, batch, device=str(device), return_outputs=True
        )
    forward_seconds = time.perf_counter() - start
    print('model forward shapes:')
    for name, shape in model.last_forward_shapes.items():
        print(f'{name:24s} {shape}')
    print(f'loss: {loss.item():.6f}')
    print(f'metrics: {metrics}')
    print(f'forward_seconds: {forward_seconds:.3f}')

    if args.backward:
        backward_start = time.perf_counter()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f'backward_and_step_seconds: {time.perf_counter() - backward_start:.3f}')

        checkpoint_dir = args.output_dir / 'smoke_checkpoint'
        saverloader.save(str(checkpoint_dir), optimizer, model, 1, keep_latest=1)
        first_parameter = next(model.parameters())
        expected = first_parameter.detach().clone()
        with torch.no_grad():
            first_parameter.zero_()
        step = saverloader.load(str(checkpoint_dir), model, optimizer=optimizer)
        if step != 1 or not torch.equal(first_parameter, expected):
            raise RuntimeError('checkpoint reload did not restore model parameters')
        print('checkpoint_reload: OK')

    if args.visualization_name:
        visualization_name = args.visualization_name
    elif args.checkpoint:
        visualization_name = 'mini_camera_pretrained.png'
    elif args.use_radar:
        visualization_name = 'mini_radar_smoke.png'
    else:
        visualization_name = 'mini_camera_smoke.png'
    visualization_path = args.output_dir / visualization_name
    make_visualization(outputs, visualization_path)
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    if args.dump_npz:
        rgb_uint8 = (
            (outputs['rgb_camXs'][0].detach().cpu() + 0.5)
            .clamp(0, 1)
            .mul(255)
            .byte()
            .permute(0, 2, 3, 1)
            .numpy()
        )
        radar_bev = (
            outputs['radar_voxels'][0, 0].detach().float().sum(dim=1)
            .clamp(0, 1).cpu().numpy().astype(np.uint8)
        )
        args.dump_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dump_npz,
            rgb=rgb_uint8,
            ground_truth=outputs['seg_bev_gt'][0, 0].detach().cpu().numpy(),
            prediction=torch.sigmoid(
                outputs['seg_bev_logits'][0, 0]
            ).detach().float().cpu().numpy(),
            radar_bev=radar_bev,
            loss=np.asarray(loss.item(), dtype=np.float32),
            iou=np.asarray(metrics['iou'], dtype=np.float32),
            forward_seconds=np.asarray(forward_seconds, dtype=np.float32),
            peak_cuda_memory_gib=np.asarray(peak_gib, dtype=np.float32),
        )
        print(f'npz_snapshot: {args.dump_npz}')
    print(f'peak_cuda_memory_gib: {peak_gib:.3f}')
    print(f'visualization: {visualization_path}')
    print('MODEL_SMOKE_TEST_OK')


if __name__ == '__main__':
    main()
