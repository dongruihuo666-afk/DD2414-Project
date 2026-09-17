#!/usr/bin/env python3
"""Radar-only four-frame learning check against cached frozen-DINOv2 targets.

This is an equal-data/equal-update optimization diagnostic, not an accuracy
benchmark or a comparison of complete camera-radar Simple-BEV models.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from dinov2_bev_demo import (  # noqa: E402
    build_loader, camera_geometry, extract_teacher_features, load_teacher,
    make_radar_bev, radar_anchored_soft_targets,
)
from nets.bevcar_voxel_adapter import prepare_bevcar_voxels  # noqa: E402
from nets.radar_encoder import (  # noqa: E402
    RadarPointEncoder, transform_radar_to_camera_bev,
)
from semantic_distill_smoke import semantic_loss  # noqa: E402
from train_nuscenes import Z, Y, X, bounds, scene_centroid  # noqa: E402
import utils.basic  # noqa: E402
import utils.geom  # noqa: E402
import utils.vox  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--target-cache', type=Path, required=True)
    parser.add_argument('--bevcar-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--heldout-samples', type=int, default=0,
                        help='Spread this many samples across the mini validation split')
    parser.add_argument('--seed-list', default='125',
                        help='Comma-separated model initialization seeds')
    parser.add_argument('--diagnose-radar', action='store_true',
                        help='Evaluate radar-removal and measurement ablations')
    return parser.parse_args()


def official_voxelnet(source):
    path = source / 'nets' / 'voxelnet.py'
    if not path.is_file():
        raise FileNotFoundError(f'official BEVCar source missing: {path}')
    spec = importlib.util.spec_from_file_location('official_bevcar_voxelnet', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VoxelNet


class RadarStudent(nn.Module):
    def __init__(self, branch, official_class=None):
        super().__init__()
        if branch == 'light':
            self.encoder = RadarPointEncoder(
                Z=Z, Y=Y, X=X, bounds=bounds, scene_centroid=(0, 0, 0),
                out_channels=64, use_quality_mask=True,
            )
            self.projection = nn.Identity()
        elif branch == 'bevcar':
            self.encoder = official_class(
                use_col=False, reduced_zx=False, output_dim=128,
                use_radar_occupancy_map=False,
            )
            self.projection = nn.Conv2d(128, 64, 1)
        else:
            raise ValueError(branch)
        self.head = nn.Conv2d(64, 384, 1)

    def forward(self, radar_input):
        feature = (self.encoder(*radar_input) if isinstance(radar_input, tuple)
                   else self.encoder(radar_input))
        return self.head(self.projection(feature))


def ablate_radar(radar, branch, mode):
    """Change only the test-time radar input; keep weights and targets fixed."""
    if branch == 'light':
        if mode == 'empty':
            return torch.zeros_like(radar)
        modified = radar.clone()
        numeric_columns = (5, 8, 9, 18)
        if mode == 'geometry_only':
            modified[..., list(numeric_columns)] = 0
        elif mode == 'permuted_measurements':
            mask = radar[..., :3].abs().sum(dim=-1).gt(0)
            for column in numeric_columns:
                values = radar[0, mask[0], column]
                modified[0, mask[0], column] = values.flip(0)
        else:
            raise ValueError(mode)
        return modified
    features, coords, counts = radar
    if mode == 'empty':
        return (torch.zeros_like(features), torch.zeros_like(coords),
                torch.zeros_like(counts))
    modified = features.clone()
    if mode == 'geometry_only':
        modified[..., 3:6] = 0
    elif mode == 'permuted_measurements':
        valid = features[..., 6].gt(0)
        metadata = modified[..., 3:6]
        metadata[valid] = metadata[valid].flip(0)
    else:
        raise ValueError(mode)
    return modified, coords, counts


def cached_target(path, batch, device):
    with np.load(path, allow_pickle=False) as cache:
        if str(cache['model_name']) != 'dinov2_vits14':
            raise ValueError('expected the frozen dinov2_vits14 target cache')
        for key, batch_index in (
                ('intrinsics', 3), ('rotations', 1), ('translations', 2)):
            actual = batch[batch_index][0, 0].numpy()
            if not np.allclose(actual, cache[key], rtol=1e-4, atol=1e-4):
                raise ValueError(f'target/sample geometry mismatch: {path}, {key}')
        target = torch.from_numpy(cache['semantic_target'].astype(np.float32)).to(device)
        confidence = torch.from_numpy(cache['confidence'].astype(np.float32)).to(device)
    if target.shape != (384, Z, X) or confidence.shape != (Z, X):
        raise ValueError(f'unexpected target geometry in {path}')
    return target[None], confidence[None]


def radar_inputs(batch, device):
    radar_vehicle = batch[16][:, 0].to(device).permute(0, 2, 1)
    rotations = batch[1][:, 0].to(device)
    translations = batch[2][:, 0].to(device)
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    B = radar_vehicle.shape[0]
    camera_from_vehicle = utils.basic.unpack_seqdim(
        utils.geom.safe_inverse(utils.basic.pack_seqdim(vehicle_from_cameras, B)), B
    )[:, 0]
    radar_camera = transform_radar_to_camera_bev(
        radar_vehicle, camera_from_vehicle
    )
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    with torch.no_grad():
        features, coords, counts, diagnostics = prepare_bevcar_voxels(
            radar_camera, vox_util, quality_filter=False,
        )
    return radar_camera, (features, coords, counts), diagnostics


def run_branch(name, inputs, targets, confidences, steps, learning_rate,
               device, official_class, heldout=None, seed=125,
               diagnose_radar=False):
    torch.manual_seed(seed)
    model = RadarStudent(name, official_class).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-5)
    branch_inputs = [entry[0 if name == 'light' else 1] for entry in inputs]
    def evaluate(source_inputs, source_targets, source_confidences):
        model.eval()
        with torch.inference_mode():
            values = []
            for radar, target, confidence in zip(
                    source_inputs, source_targets, source_confidences):
                loss, _ = semantic_loss(model(radar), target, confidence)
                values.append(float(loss))
        return values

    initial = evaluate(branch_inputs, targets, confidences)
    if heldout is not None:
        val_inputs, val_targets, val_confidences = heldout
        val_branch_inputs = [entry[0 if name == 'light' else 1]
                             for entry in val_inputs]
        val_initial = evaluate(val_branch_inputs, val_targets, val_confidences)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    history = []
    radar_gradients = []
    for step in range(steps):
        index = step % len(inputs)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(branch_inputs[index])
        loss, _ = semantic_loss(prediction, targets[index], confidences[index])
        if not torch.isfinite(loss):
            raise RuntimeError(f'{name}: nonfinite loss')
        loss.backward()
        gradients = [p.grad.float().norm() for p in model.encoder.parameters()
                     if p.grad is not None]
        if not gradients:
            raise RuntimeError(f'{name}: no encoder gradients')
        grad_norm = torch.stack(gradients).norm().item()
        if not np.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError(f'{name}: invalid encoder gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(float(loss.detach()))
        radar_gradients.append(grad_norm)
        print(f'{name} step {step + 1}/{steps}: loss={history[-1]:.6f} '
              f'encoder_grad={grad_norm:.5f}', flush=True)
    final = evaluate(branch_inputs, targets, confidences)
    if heldout is not None:
        val_final = evaluate(val_branch_inputs, val_targets, val_confidences)
        # Keep targets fixed but feed each frame a distant validation frame's
        # radar. With our balanced, scene-sorted selection this swaps scenes.
        shift = len(val_branch_inputs) // 2
        shifted_inputs = val_branch_inputs[shift:] + val_branch_inputs[:shift]
        val_shifted = evaluate(shifted_inputs, val_targets, val_confidences)
        if diagnose_radar:
            ablations = {}
            for mode in ('empty', 'geometry_only', 'permuted_measurements'):
                changed = [ablate_radar(radar, name, mode)
                           for radar in val_branch_inputs]
                ablations[mode] = evaluate(
                    changed, val_targets, val_confidences
                )
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    result = {
        'trainable_parameters': sum(p.numel() for p in model.parameters()
                                    if p.requires_grad),
        'before_losses': initial,
        'after_losses': final,
        'step_losses': history,
        'encoder_gradient_norms': radar_gradients,
        'elapsed_seconds': elapsed,
        'peak_allocated_cuda_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    if heldout is not None:
        result['heldout_before_losses'] = val_initial
        result['heldout_after_losses'] = val_final
        result['heldout_shifted_radar_losses'] = val_shifted
        if diagnose_radar:
            result['radar_ablation_losses'] = ablations
    return result


def validation_examples(data_root, device, count, train_records, train_scenes):
    loader = build_loader(data_root, num_workers=0, nsweeps=1,
                          rotate_radar_velocity=True, split='val')
    dataset = loader.dataset
    if count > len(dataset):
        raise ValueError(f'only {len(dataset)} mini validation frames are available')
    chosen = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
    if len(set(chosen)) != count:
        raise AssertionError('validation selection contains duplicate indices')
    teacher = load_teacher('dinov2_vits14', device)
    if any(p.requires_grad for p in teacher.parameters()):
        raise AssertionError('DINOv2 teacher must remain frozen')
    val_inputs, val_targets, val_confidences, details = [], [], [], []
    with torch.inference_mode():
        for index in chosen:
            record = dataset.ixes[int(dataset.indices[index][0])]
            if record['token'] in train_records:
                raise AssertionError('training/validation token overlap')
            batch = torch.utils.data.default_collate([dataset[index]])
            images = batch[0][:, 0]
            features = extract_teacher_features(teacher, images, device)
            _, pixel_from_camera, _, cameras_from_vehicle = camera_geometry(
                batch, features.shape[-2], features.shape[-1], device
            )
            vox_util = utils.vox.Vox_util(
                Z, Y, X, scene_centroid=scene_centroid.to(device),
                bounds=bounds, assert_cube=False,
            )
            _, radar_vehicle, radar_camera = make_radar_bev(
                batch, cameras_from_vehicle, vox_util, device
            )
            target, confidence, view_count = radar_anchored_soft_targets(
                features, pixel_from_camera, cameras_from_vehicle,
                radar_vehicle, radar_camera, vox_util,
            )
            soft_cells = int((confidence > 0.05).sum())
            if soft_cells == 0:
                raise RuntimeError(f'validation frame {index} has no target cells')
            light_input, bevcar_input, counts = radar_inputs(batch, device)
            val_inputs.append((light_input, bevcar_input))
            val_targets.append(target[None].half())
            val_confidences.append(confidence[None].half())
            scene = dataset.nusc.get('scene', record['scene_token'])['name']
            if scene in train_scenes:
                raise AssertionError('training/validation scene overlap')
            details.append({
                'split_index': index, 'sample_token': record['token'],
                'scene': scene, 'soft_target_cells': soft_cells,
                'visible_radar_points': int((view_count > 0).sum()),
                'radar_points_in_roi': int(counts['retained_points'][0]),
            })
            print(f'held-out index={index} scene={scene} soft_cells={soft_cells}',
                  flush=True)
            del features, target, confidence, vox_util
    del teacher
    torch.cuda.empty_cache()
    if len({item['scene'] for item in details}) < 2 and count >= 2:
        raise AssertionError('held-out examples must cover both mini validation scenes')
    if count >= 2 and any(
            details[index]['scene'] == details[(index + count // 2) % count]['scene']
            for index in range(count)):
        raise AssertionError('radar-shift control must pair different scenes')
    return (val_inputs, val_targets, val_confidences), details


def plot_report(report, path):
    """Render an intentionally labelled optimization diagnostic."""
    image = Image.new('RGB', (1080, 650), '#f6f8fb')
    draw = ImageDraw.Draw(image)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    body = ImageFont.truetype(font_path, 17)
    heading = ImageFont.truetype(bold_path, 25)
    small = ImageFont.truetype(font_path, 14)
    draw.text((30, 22), 'Radar-only DINOv2 target: four-frame overfit',
              font=heading, fill='#1c2736')
    draw.text((30, 62),
              f'Same frozen-teacher targets, {report["steps_per_branch"]} updates each; random radar encoders',
              font=body, fill='#455468')
    plot = (70, 120, 1020, 455)
    draw.rectangle(plot, fill='white', outline='#b8c1ce', width=2)
    draw.text((30, 265), 'Loss', font=small, fill='#455468')
    colors = {'light': '#3678cc', 'bevcar': '#d06431'}
    for name in ('light', 'bevcar'):
        values = report['branches'][name]['step_losses']
        points = [
            (plot[0] + 20 + index * (plot[2] - plot[0] - 40) / max(len(values)-1, 1),
             plot[3] - 20 - (value - 0.25) / 0.85 * (plot[3] - plot[1] - 40))
            for index, value in enumerate(values)
        ]
        draw.line(points, fill=colors[name], width=4)
    draw.text((70, 467), 'Update 1', font=small, fill='#455468')
    draw.text((943, 467), f'Update {report["steps_per_branch"]}',
              font=small, fill='#455468')
    for index, name in enumerate(('light', 'bevcar')):
        result = report['branches'][name]
        y = 508 + index * 43
        draw.rectangle((70, y + 4, 90, y + 24), fill=colors[name])
        draw.text((102, y),
                  f'{name}: same-frame eval {np.mean(result["before_losses"]):.3f} '
                  f'-> {np.mean(result["after_losses"]):.3f}; '
                  f'peak allocated CUDA {result["peak_allocated_cuda_gib"]:.2f} GiB',
                  font=body, fill='#1c2736')
    draw.text((30, 609),
              'Training-sample loss only. Different capacity/filtering; no held-out or camera-fusion result.',
              font=small, fill='#a14929')
    image.save(path)


def plot_heldout(report, path):
    image = Image.new('RGB', (1120, 715), '#f6f8fb')
    draw = ImageDraw.Draw(image)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    body = ImageFont.truetype(font_path, 17)
    title = ImageFont.truetype(bold_path, 25)
    small = ImageFont.truetype(font_path, 14)
    draw.text((30, 22), 'Radar-only DINOv2: mini validation frames',
              font=title, fill='#1c2736')
    draw.text((30, 62),
              f'4 training frames / {report["heldout_samples"]} unseen validation frames; '
              f'{report["steps_per_branch"]} updates per branch',
              font=body, fill='#455468')
    colors = {'light': '#3678cc', 'bevcar': '#d06431'}
    left, top, right, bottom = 80, 142, 1060, 474
    draw.rectangle((left, top, right, bottom), fill='white',
                   outline='#b8c1ce', width=2)
    for value in (0.4, 0.6, 0.8, 1.0):
        y = bottom - 25 - (value - 0.2) / 1.0 * (bottom - top - 50)
        draw.line((left + 1, y, right - 1, y), fill='#e3e8ef', width=1)
        draw.text((35, y - 10), f'{value:.1f}', font=small, fill='#455468')
    n = report['heldout_samples']
    for name in ('light', 'bevcar'):
        all_values = np.asarray([
            report['seed_results'][str(seed)][name]['heldout_after_losses']
            for seed in report['seeds']
        ])
        values = all_values.mean(axis=0)
        points = [(left + 25 + index * (right - left - 50) / max(n - 1, 1),
                   bottom - 25 - (value - 0.2) / 1.0 * (bottom - top - 50))
                  for index, value in enumerate(values)]
        draw.line(points, fill=colors[name], width=4)
        for index, (x, y) in enumerate(points):
            high = all_values[:, index].max()
            low = all_values[:, index].min()
            y_high = bottom - 25 - (high - 0.2) / 1.0 * (bottom - top - 50)
            y_low = bottom - 25 - (low - 0.2) / 1.0 * (bottom - top - 50)
            draw.line((x, y_high, x, y_low), fill=colors[name], width=2)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[name])
    draw.text((80, 482), 'Held-out frame index (spread over both mini validation scenes)',
              font=small, fill='#455468')
    for index, name in enumerate(('light', 'bevcar')):
        runs = [report['seed_results'][str(seed)][name]
                for seed in report['seeds']]
        before = np.mean([np.mean(r['heldout_before_losses']) for r in runs])
        after = np.mean([np.mean(r['heldout_after_losses']) for r in runs])
        after_range = [np.mean(r['heldout_after_losses']) for r in runs]
        shifted = np.mean([np.mean(r['heldout_shifted_radar_losses'])
                           for r in runs])
        y = 520 + index * 55
        draw.rectangle((80, y + 3, 101, y + 24), fill=colors[name])
        draw.text((113, y),
                  f'{name}: held-out mean {before:.3f} -> {after:.3f}; '
                  f'seed range {min(after_range):.3f}-{max(after_range):.3f}; '
                  f'shifted radar {shifted:.3f}',
                  font=body, fill='#1c2736')
    draw.text((30, 660),
              'Unseen scenes; points show seed mean, whiskers show seed range. Not segmentation accuracy.',
              font=small, fill='#a14929')
    image.save(path)


def plot_diagnostic(report, path):
    """Compact meeting figure: lower cosine loss is better in every cell."""
    image = Image.new('RGB', (1120, 625), '#f6f8fb')
    draw = ImageDraw.Draw(image)
    base = '/usr/share/fonts/truetype/dejavu/'
    body = ImageFont.truetype(base + 'DejaVuSans.ttf', 17)
    bold = ImageFont.truetype(base + 'DejaVuSans-Bold.ttf', 24)
    small = ImageFont.truetype(base + 'DejaVuSans.ttf', 14)
    draw.text((30, 25), 'Does the trained radar model use the right radar?',
              font=bold, fill='#1c2736')
    draw.text((30, 66), '12 unseen frames / 2 scenes / 3 seeds; same weights and DINOv2 targets in every column',
              font=body, fill='#455468')
    names = (
        ('aligned', 'Correct radar'),
        ('cross_scene', 'Other scene'),
        ('geometry_only', 'Positions only'),
        ('permuted_measurements', 'Mixed values'),
        ('empty', 'No radar'),
    )
    column_x = (225, 400, 575, 750, 925)
    draw.rectangle((30, 130, 1090, 455), fill='white', outline='#b8c1ce', width=2)
    for (_, label), x in zip(names, column_x):
        draw.text((x, 155), label, font=small, fill='#455468')
    for row, branch in enumerate(('light', 'bevcar')):
        y = 235 + row * 125
        draw.text((55, y + 18), branch, font=body,
                  fill='#3678cc' if branch == 'light' else '#d06431')
        runs = [report['seed_results'][str(seed)][branch]
                for seed in report['seeds']]
        values = {
            'aligned': [np.mean(r['heldout_after_losses']) for r in runs],
            'cross_scene': [np.mean(r['heldout_shifted_radar_losses']) for r in runs],
        }
        for key in ('geometry_only', 'permuted_measurements', 'empty'):
            values[key] = [np.mean(r['radar_ablation_losses'][key]) for r in runs]
        for (key, _), x in zip(names, column_x):
            mean = np.mean(values[key])
            draw.text((x, y), f'{mean:.3f}', font=bold, fill='#1c2736')
            draw.text((x, y + 37),
                      f'{min(values[key]):.3f}-{max(values[key]):.3f}',
                      font=small, fill='#68778a')
    draw.text((40, 480), 'Large loss changes after removing an input indicate dependence on it.',
              font=body, fill='#1c2736')
    draw.text((40, 516), 'Small differences can arise from noise; these are test-time interventions, not retraining.',
              font=body, fill='#1c2736')
    draw.text((40, 574), 'Frozen-teacher cosine loss is not segmentation or detection accuracy.',
              font=small, fill='#a14929')
    image.save(path)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this small comparison')
    if not 1 <= args.samples <= 4 or args.steps < 1 or args.heldout_samples < 0:
        raise ValueError('use 1-4 cached samples, >=1 update, >=0 held-out samples')
    if args.diagnose_radar and args.heldout_samples < 2:
        raise ValueError('radar diagnosis requires at least two held-out frames')
    seeds = [int(part.strip()) for part in args.seed_list.split(',')]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seed-list must have distinct integer seeds')
    torch.set_num_threads(2)
    device = torch.device('cuda')
    loader = build_loader(args.data_root, num_workers=0, nsweeps=1,
                          rotate_radar_velocity=True)
    batches = [next(iter(loader))] if args.samples == 1 else []
    if args.samples > 1:
        iterator = iter(loader)
        batches = [next(iterator) for _ in range(args.samples)]
    targets, confidences, inputs, counts = [], [], [], []
    for index, batch in enumerate(batches):
        target, confidence = cached_target(
            args.target_cache / f'sample_{index:03d}.npz', batch, device
        )
        light_input, bevcar_input, info = radar_inputs(batch, device)
        targets.append(target)
        confidences.append(confidence)
        inputs.append((light_input, bevcar_input))
        counts.append({key: int(value[0]) for key, value in info.items()})
    train_dataset = loader.dataset
    train_records = {
        train_dataset.ixes[int(train_dataset.indices[index][0])]['token']
        for index in range(args.samples)
    }
    train_scenes = {
        train_dataset.nusc.get('scene',
            train_dataset.ixes[int(train_dataset.indices[index][0])]['scene_token'])['name']
        for index in range(args.samples)
    }
    heldout, heldout_details = (None, [])
    if args.heldout_samples:
        heldout, heldout_details = validation_examples(
            args.data_root, device, args.heldout_samples,
            train_records, train_scenes,
        )
    official_class = official_voxelnet(args.bevcar_source)
    seed_results = {}
    for seed in seeds:
        seed_results[str(seed)] = {}
        for branch in ('light', 'bevcar'):
            seed_results[str(seed)][branch] = run_branch(
                branch, inputs, targets, confidences, args.steps,
                args.learning_rate, device, official_class,
                heldout=heldout, seed=seed,
                diagnose_radar=args.diagnose_radar,
            )
            torch.cuda.empty_cache()
    results = seed_results[str(seeds[0])]
    report = {
        'scope': 'radar-only, same frozen-DINOv2 cached targets, random weights',
        'samples': args.samples, 'steps_per_branch': args.steps,
        'learning_rate': args.learning_rate,
        'target_model': 'dinov2_vits14 pretrained frozen teacher',
        'supervised_bev_checkpoint_loaded': False,
        'box_loss_used': False,
        'input_counts': counts,
        'train_sample_tokens': sorted(train_records),
        'train_scenes': sorted(train_scenes),
        'heldout_samples': args.heldout_samples,
        'heldout_frames': heldout_details,
        'diagnose_radar': args.diagnose_radar,
        'seeds': seeds,
        'seed_results': seed_results,
        'branches': results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = ('radar_dino_ablation' if args.diagnose_radar else
            'radar_dino_heldout_comparison' if heldout is not None else
            'radar_dino_tiny_comparison')
    path = args.output_dir / f'{stem}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    figure = args.output_dir / f'{stem}.png'
    if args.diagnose_radar:
        plot_diagnostic(report, figure)
    elif heldout is None:
        plot_report(report, figure)
    else:
        plot_heldout(report, figure)
    print(f'report: {path}')
    print(f'figure: {figure}')
    for seed in seeds:
        for branch, result in seed_results[str(seed)].items():
            print(f'seed {seed} {branch}: '
                  f'train={np.mean(result["before_losses"]):.6f}'
                  f'->{np.mean(result["after_losses"]):.6f} '
                  f'peak_cuda_gib={result["peak_allocated_cuda_gib"]:.3f}')
            if heldout is not None:
                print(f'seed {seed} {branch} heldout: '
                      f'{np.mean(result["heldout_before_losses"]):.6f}'
                      f'->{np.mean(result["heldout_after_losses"]):.6f} '
                      f'shifted_radar='
                      f'{np.mean(result["heldout_shifted_radar_losses"]):.6f}')
                if args.diagnose_radar:
                    print('  ablations: ' + ' '.join(
                        f'{mode}={np.mean(values):.6f}'
                        for mode, values in result['radar_ablation_losses'].items()
                    ))
    print('RADAR_DINO_TINY_COMPARISON_OK')


if __name__ == '__main__':
    main()
