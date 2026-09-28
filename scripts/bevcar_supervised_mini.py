#!/usr/bin/env python3
"""Small supervised Simple-BEV + official BEVCar radar-encoder demonstration.

The camera/decoder weights come from the official camera-only Simple-BEV
checkpoint. Only the new fusion convolution and BEVCar radar branch train.
The BEVCar source is imported from a separate checkout; no BEVCar checkpoint
or DINOv2 objective is used. This is a mini-data diagnostic, not a benchmark.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nuscenesdataset  # noqa: E402
from nets.bevcar_radar_bridge import BEVCarRadarBridge  # noqa: E402
from nets.segnet import Segnet  # noqa: E402
from train_nuscenes import (  # noqa: E402
    SimpleLoss, X, Y, Z, bounds, run_model, scene_centroid, scene_centroid_py,
)
import utils.vox  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--camera-checkpoint', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, default=None)
    parser.add_argument('--encoder', choices=('bevcar', 'light'), default='bevcar')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--train-samples', type=int, default=4)
    parser.add_argument('--val-samples', type=int, default=12)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=125)
    return parser.parse_args()


def data_loaders(data_root):
    config = {
        'crop_offset': 0, 'resize_lim': [1.0, 1.0],
        'final_dim': (112, 192), 'H': 900, 'W': 1600,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': 6,
    }
    return nuscenesdataset.compile_data(
        'mini', str(data_root), data_aug_conf=config,
        centroid=scene_centroid_py, bounds=bounds, res_3d=(Z, Y, X),
        bsz=1, nworkers=0, nworkers_val=0, shuffle=False, nsweeps=1,
        seqlen=1, refcam_id=1, get_tids=True, temporal_aug=False,
        use_radar_filters=False, do_shuffle_cams=False,
        rotate_radar_velocity=True,
    )


def checkpoint_state(checkpoint_dir):
    paths = sorted(checkpoint_dir.glob('model-*.pth'))
    if not paths:
        raise FileNotFoundError(f'camera checkpoint missing in {checkpoint_dir}')
    return torch.load(paths[-1], map_location='cpu', weights_only=True)['model_state_dict']


def build_models(vox_util, source_dir, camera_state, device, encoder='bevcar'):
    camera = Segnet(
        Z, Y, X, vox_util=vox_util, use_radar=False, do_rgbcompress=True,
        encoder_type='res101', rand_flip=False, pretrained_backbone=False,
    ).to(device)
    camera.load_state_dict(camera_state, strict=False)
    camera.eval()

    fused = Segnet(
        Z, Y, X, vox_util=vox_util, use_radar_encoder=True,
        radar_encoder_channels=64, do_rgbcompress=True,
        encoder_type='res101', rand_flip=False, pretrained_backbone=False,
    ).to(device)
    if encoder == 'bevcar':
        if source_dir is None:
            raise ValueError('--bevcar-source is required with --encoder bevcar')
        fused.radar_encoder = BEVCarRadarBridge(vox_util, source_dir, 64).to(device)
    own = fused.state_dict()
    copied = {
        key: value for key, value in camera_state.items()
        if (key.startswith('encoder.') or key.startswith('decoder.')
            or key in ('ce_weight', 'center_weight', 'offset_weight'))
        and key in own and own[key].shape == value.shape
    }
    if len(copied) < 100 or not any(key.startswith('decoder.') for key in copied):
        raise RuntimeError('camera encoder/decoder checkpoint transfer failed')
    fused.load_state_dict(copied, strict=False)
    with torch.no_grad():
        source = camera_state['bev_compressor.0.weight']
        target = fused.bev_compressor[0].weight
        if source.shape != target[:, :source.shape[1]].shape:
            raise RuntimeError('camera fusion weight shape is incompatible')
        target[:, :source.shape[1]].copy_(source)
        torch.nn.init.normal_(target[:, source.shape[1]:], std=1e-3)
    for part in (fused.encoder, fused.decoder):
        part.requires_grad_(False)
    for parameter in (fused.ce_weight, fused.center_weight, fused.offset_weight):
        parameter.requires_grad_(False)
    return camera, fused, len(copied)


def mini_batches(train_loader, val_loader, train_count, val_count):
    if train_count > len(train_loader.dataset) or val_count > len(val_loader.dataset):
        raise ValueError('requested more samples than exist in mini split')
    train_iter = iter(train_loader)
    train = [next(train_iter) for _ in range(train_count)]
    indices = np.linspace(0, len(val_loader.dataset)-1, val_count,
                          dtype=int).tolist()
    if len(set(indices)) != val_count:
        raise AssertionError('validation indices are not unique')
    val = [torch.utils.data.default_collate([val_loader.dataset[i]])
           for i in indices]
    train_set = train_loader.dataset
    val_set = val_loader.dataset
    train_scenes = {
        train_set.nusc.get('scene',
            train_set.ixes[int(train_set.indices[i][0])]['scene_token'])['name']
        for i in range(train_count)
    }
    val_metadata = []
    for i in indices:
        record = val_set.ixes[int(val_set.indices[i][0])]
        scene = val_set.nusc.get('scene', record['scene_token'])['name']
        if scene in train_scenes:
            raise AssertionError('train/validation scene overlap')
        val_metadata.append({'index': i, 'scene': scene,
                             'sample_token': record['token']})
    return train, val, val_metadata, sorted(train_scenes)


def measured(model, loss_fn, batch, device):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        loss, metrics, outputs = run_model(
            model, loss_fn, batch, device=str(device), return_outputs=True,
        )
    logits = outputs['seg_bev_logits'][0, 0]
    probability = torch.sigmoid(logits).float().cpu().numpy()
    return {
        'loss': float(loss), 'iou': float(metrics['iou']),
        'segmentation_loss': float(metrics['ce_loss']),
        'prediction': probability,
        'ground_truth': outputs['seg_bev_gt'][0, 0].float().cpu().numpy(),
        'radar': outputs['radar_voxels'][0, 0].float().sum(dim=1)
                 .clamp(0, 1).cpu().numpy(),
    }


def assess(camera, fused, loss_fn, val_batches, metadata, device):
    camera.eval()
    fused.eval()
    results = []
    for index, (batch, detail) in enumerate(zip(val_batches, metadata)):
        empty = list(batch)
        empty[16] = torch.zeros_like(batch[16])
        wrong = list(batch)
        wrong[16] = val_batches[(index + len(val_batches)//2) % len(val_batches)][16]
        camera_result = measured(camera, loss_fn, batch, device)
        correct_result = measured(fused, loss_fn, batch, device)
        empty_result = measured(fused, loss_fn, empty, device)
        wrong_result = measured(fused, loss_fn, wrong, device)
        results.append({
            **detail,
            'camera': camera_result, 'correct': correct_result,
            'empty': empty_result, 'wrong': wrong_result,
            'correct_empty_mean_abs_probability': float(np.mean(np.abs(
                correct_result['prediction'] - empty_result['prediction']))),
        })
    return results


def summary(results):
    return {name: {
        'mean_iou': float(np.mean([item[name]['iou'] for item in results])),
        'mean_segmentation_loss': float(np.mean([
            item[name]['segmentation_loss'] for item in results])),
    } for name in ('camera', 'correct', 'empty', 'wrong')}


def visual(results, before, losses, path, encoder='bevcar'):
    canvas = Image.new('RGB', (1510, 845), '#f7f9fc')
    draw = ImageDraw.Draw(canvas)
    base = '/usr/share/fonts/truetype/dejavu/'
    title = ImageFont.truetype(base + 'DejaVuSans-Bold.ttf', 24)
    body = ImageFont.truetype(base + 'DejaVuSans.ttf', 15)
    draw.text((25, 18), f'Simple-BEV + {encoder} radar encoder: supervised mini diagnostic',
              font=title, fill='#1c2736')
    draw.text((25, 56),
              'Human box-derived BEV labels | frozen camera/decoder checkpoint | new trainable radar + fusion',
              font=body, fill='#455468')
    before_summary, after_summary = summary(before), summary(results)
    draw.text((25, 96),
              f'Validation IoU: camera {after_summary["camera"]["mean_iou"]:.3f}; '
              f'fusion {before_summary["correct"]["mean_iou"]:.3f} -> '
              f'{after_summary["correct"]["mean_iou"]:.3f}; '
              f'empty radar {after_summary["empty"]["mean_iou"]:.3f}; '
              f'wrong radar {after_summary["wrong"]["mean_iou"]:.3f}',
              font=body, fill='#1c2736')
    best = int(np.argmax([item['correct']['iou'] - item['empty']['iou']
                          for item in results]))
    worst = int(np.argmin([item['correct']['iou'] - item['empty']['iou']
                           for item in results]))
    for row, index in enumerate((best, worst)):
        item = results[index]
        y = 185 + 300 * row
        draw.text((25, y - 30),
                  f'{item["scene"]}, validation index {item["index"]}: '
                  f'correct IoU {item["correct"]["iou"]:.3f}, '
                  f'empty IoU {item["empty"]["iou"]:.3f}',
                  font=body, fill='#1c2736')
        panels = (
            ('Human label', item['correct']['ground_truth']),
            ('Radar returns', item['correct']['radar']),
            ('Camera only', item['camera']['prediction']),
            ('Correct radar', item['correct']['prediction']),
            ('No radar', item['empty']['prediction']),
            ('Wrong radar', item['wrong']['prediction']),
        )
        for column, (label, values) in enumerate(panels):
            x = 25 + 245 * column
            draw.text((x, y), label, font=body, fill='#455468')
            rgb = np.zeros((Z, X, 3), dtype=np.uint8)
            if column == 0:
                rgb[values > 0.5] = (0, 190, 70)
            elif column == 1:
                rgb[values > 0.5] = (0, 220, 235)
            else:
                rgb[..., 0] = (np.clip(values, 0, 1) * 255).astype(np.uint8)
            panel = Image.fromarray(rgb).resize((200, 200), Image.Resampling.NEAREST)
            canvas.paste(panel, (x, y + 28))
    draw.text((25, 797),
              'Small mini subset; official camera checkpoint has broader pretraining. '
              'Fusion/camera comparison is not a matched-budget benchmark.',
              font=body, fill='#a14929')
    canvas.save(path)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if args.train_samples < 1 or args.val_samples < 2 or args.steps < 1:
        raise ValueError('need train samples, two validation scenes, and steps')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    train_loader, val_loader = data_loaders(args.data_root)
    train_batches, val_batches, metadata, train_scenes = mini_batches(
        train_loader, val_loader, args.train_samples, args.val_samples
    )
    if len({item['scene'] for item in metadata}) < 2:
        raise AssertionError('selected validation examples need both scenes')
    if any(metadata[i]['scene'] == metadata[(i + len(metadata)//2) % len(metadata)]['scene']
           for i in range(len(metadata))):
        raise AssertionError('wrong-radar examples must come from another scene')
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    if args.encoder == 'bevcar' and args.bevcar_source is None:
        raise ValueError('--bevcar-source is required when --encoder bevcar')
    camera_state = checkpoint_state(args.camera_checkpoint)
    camera, fused, copied = build_models(
        vox_util, args.bevcar_source, camera_state, device, args.encoder
    )
    del camera_state
    loss_fn = SimpleLoss(2.13).to(device)
    before = assess(camera, fused, loss_fn, val_batches, metadata, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in fused.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=1e-7,
    )
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0, growth_interval=1000)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    losses = []
    fused.train()
    fused.encoder.eval()
    fused.decoder.eval()
    for step in range(args.steps):
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss, metrics = run_model(fused, loss_fn, batch, device=str(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite supervised loss at step {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in fused.parameters() if p.requires_grad], 5.0,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(metrics['ce_loss'] + metrics['center_loss']
                            + metrics['offset_loss']))
        if step < 3 or (step + 1) % 10 == 0:
            print(f'step {step + 1}/{args.steps}: task_loss={losses[-1]:.5f}',
                  flush=True)
    torch.cuda.synchronize(device)
    train_seconds = time.monotonic() - started
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    radar_gradient = [p.grad for p in fused.radar_encoder.parameters()
                      if p.grad is not None]
    if not radar_gradient or not all(torch.isfinite(g).all() for g in radar_gradient):
        raise RuntimeError(f'{args.encoder} radar branch did not receive valid gradients')
    point_gradient_norm = volume_gradient_norm = None
    if args.encoder == 'bevcar':
        point_gradient = fused.radar_encoder.voxelnet.svfe.vfe_1.fcn.linear.weight.grad
        volume_gradient = fused.radar_encoder.voxelnet.cml.conv3d_1.conv.weight.grad
        if point_gradient is None or volume_gradient is None:
            raise RuntimeError('BEVCar point and volume layers need gradients')
        point_gradient_norm = float(point_gradient.float().norm())
        volume_gradient_norm = float(volume_gradient.float().norm())
        if not (np.isfinite(point_gradient_norm) and point_gradient_norm > 0
                and np.isfinite(volume_gradient_norm) and volume_gradient_norm > 0):
            raise RuntimeError('BEVCar point/volume gradient norm invalid')
    after = assess(camera, fused, loss_fn, val_batches, metadata, device)
    before_summary, after_summary = summary(before), summary(after)
    report = {
        'scope': '4-frame supervised mini adaptation; 12-frame scene-disjoint validation',
        'train_scenes': train_scenes, 'val_metadata': metadata,
        'camera_checkpoint': str(args.camera_checkpoint),
        'bevcar_source': str(args.bevcar_source) if args.bevcar_source else None,
        'radar_encoder': args.encoder,
        'copied_camera_decoder_checkpoint_tensors': copied,
        'training_samples': args.train_samples, 'steps': args.steps,
        'seed': args.seed,
        'supervised_box_derived_labels_used': True,
        'dinov2_used': False,
        'bevcar_point_gradient_norm': point_gradient_norm,
        'bevcar_volume_gradient_norm': volume_gradient_norm,
        'training_task_losses': losses,
        'training_seconds': train_seconds,
        'peak_allocated_cuda_gib': peak_gib,
        'before_summary': before_summary, 'after_summary': after_summary,
        'samples': [{
            'index': item['index'], 'scene': item['scene'],
            'sample_token': item['sample_token'],
            'camera_iou': item['camera']['iou'],
            'correct_iou_before': before[i]['correct']['iou'],
            'correct_iou_after': item['correct']['iou'],
            'empty_iou_after': item['empty']['iou'],
            'wrong_iou_after': item['wrong']['iou'],
            'correct_empty_mean_abs_probability': item['correct_empty_mean_abs_probability'],
        } for i, item in enumerate(after)],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (f'{args.encoder}_supervised_mini' if args.seed == 125
            else f'{args.encoder}_supervised_mini_seed{args.seed}')
    json_path = args.output_dir / f'{stem}.json'
    figure_path = args.output_dir / f'{stem}.png'
    json_path.write_text(json.dumps(report, indent=2) + '\n')
    visual(after, before, losses, figure_path, args.encoder)
    print(f'camera validation IoU: {after_summary["camera"]["mean_iou"]:.6f}')
    for condition in ('correct', 'empty', 'wrong'):
        print(f'{condition} validation IoU: '
              f'{after_summary[condition]["mean_iou"]:.6f}')
    print(f'train_seconds: {train_seconds:.2f}')
    print(f'peak_allocated_cuda_gib: {peak_gib:.3f}')
    if args.encoder == 'bevcar':
        print(f'BEVCar_SVFE_gradient_norm: {point_gradient_norm:.6g}')
        print(f'BEVCar_CML_gradient_norm: {volume_gradient_norm:.6g}')
    print(f'report: {json_path}')
    print(f'figure: {figure_path}')
    print(f'{args.encoder}_supervised_mini_ok')


if __name__ == '__main__':
    main()
