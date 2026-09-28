#!/usr/bin/env python3
"""Four-sample DINOv2 BEV distillation without a supervised BEV checkpoint."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import saverloader  # noqa: E402
import utils.vox  # noqa: E402
from dinov2_bev_demo import (  # noqa: E402
    build_loader,
    camera_geometry,
    extract_teacher_features,
    load_teacher,
    make_radar_bev,
    radar_anchored_soft_targets,
)
from nets.segnet import Segnet  # noqa: E402
from semantic_distill_smoke import (  # noqa: E402
    SemanticDistillationModel,
    prepare_inputs,
    semantic_loss,
    similarity_image,
)
from train_nuscenes import Z, Y, X, bounds, scene_centroid  # noqa: E402


BACKGROUND = (245, 247, 250)
TEXT = (25, 30, 40)


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--model-name', default='dinov2_vits14')
    return parser.parse_args()


def create_targets(batches, teacher, device, cache_dir, model_name):
    targets = []
    confidences = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(batches):
            images = batch[0][:, 0]
            features = extract_teacher_features(teacher, images, device)
            _, pixel_from_camera, cameras_from_camera0, cameras_from_vehicle = camera_geometry(
                batch, features.shape[-2], features.shape[-1], device
            )
            vox_util = utils.vox.Vox_util(
                Z, Y, X, scene_centroid=scene_centroid.to(device),
                bounds=bounds, assert_cube=False,
            )
            _, radar_vehicle, radar_camera0 = make_radar_bev(
                batch, cameras_from_vehicle, vox_util, device
            )
            target, confidence, view_count = radar_anchored_soft_targets(
                features, pixel_from_camera, cameras_from_vehicle,
                radar_vehicle, radar_camera0, vox_util,
            )
            target = target.half()
            confidence = confidence.half()
            targets.append(target)
            confidences.append(confidence)
            np.savez_compressed(
                cache_dir / f'sample_{index:03d}.npz',
                semantic_target=target.cpu().numpy(),
                confidence=confidence.cpu().numpy(),
                radar_view_count=view_count.byte().cpu().numpy(),
                intrinsics=batch[3][0, 0].numpy(),
                rotations=batch[1][0, 0].numpy(),
                translations=batch[2][0, 0].numpy(),
                model_name=np.asarray(model_name),
            )
            print(
                f'target sample {index}: visible_radar={(view_count > 0).sum().item()} '
                f'soft_cells={(confidence > 0.05).sum().item()}'
            )
            del features, target, confidence, vox_util
    torch.cuda.synchronize()
    return targets, confidences, time.perf_counter() - start


def evaluate(model, inputs, targets, confidences):
    model.eval()
    losses = []
    similarities = []
    with torch.inference_mode():
        for model_input, target, confidence in zip(inputs, targets, confidences):
            # A completely random ResNet can overflow fp16 in eval mode before
            # BatchNorm running statistics have ever been calibrated. Use
            # float32 for the small before/after comparison; training stays AMP.
            prediction, *_ = model(*model_input)
            loss, similarity = semantic_loss(
                prediction, target[None], confidence[None]
            )
            losses.append(float(loss))
            similarities.append(similarity[0].float().cpu().numpy())
    return np.asarray(losses, dtype=np.float32), similarities


def draw_curve(draw, values, box):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill='white', outline=(180, 185, 195), width=2)
    values = np.asarray(values, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    span = max(high - low, 1e-6)
    points = []
    for index, value in enumerate(values):
        x = x0 + 14 + index * (x1 - x0 - 28) / max(len(values) - 1, 1)
        y = y1 - 14 - (float(value) - low) / span * (y1 - y0 - 28)
        points.append((x, y))
    draw.line(points, fill=(218, 60, 50), width=3)
    draw.text((x0 + 8, y0 + 7), f'max {high:.3f}', font=font(12), fill=(80, 85, 95))
    draw.text((x0 + 8, y1 - 24), f'min {low:.3f}', font=font(12), fill=(80, 85, 95))


def confidence_image(confidence):
    confidence = confidence.astype(np.float32)
    confidence /= max(float(confidence.max()), 1e-6)
    rgb = np.zeros((*confidence.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * confidence).astype(np.uint8)
    rgb[..., 1] = (190 * np.sqrt(confidence)).astype(np.uint8)
    return rgb


def render(losses, before_losses, after_losses, before_similarity,
           after_similarity, confidences, target_seconds, train_seconds,
           peak_gib, output_path):
    reductions = before_losses - after_losses
    selected = (
        ('Largest loss decrease', int(np.argmax(reductions))),
        ('Smallest loss decrease', int(np.argmin(reductions))),
    )
    width, height = 1170, 1050
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 18), 'DINOv2 BEV distillation: four samples, no supervised BEV checkpoint',
              font=font(27, True), fill=TEXT)
    draw.text((25, 60),
              'Random Simple-BEV student | frozen DINOv2 teacher | radar-anchored soft targets | no box loss',
              font=font(15), fill=(70, 78, 92))
    draw.text((25, 105), 'DINO cosine loss over 60 updates',
              font=font(19, True), fill=TEXT)
    draw_curve(draw, losses, (25, 140, 555, 390))
    draw.text((25, 410),
              f'First 4-step mean {np.mean(losses[:4]):.3f} -> last 4-step mean {np.mean(losses[-4:]):.3f}',
              font=font(16, True), fill=TEXT)

    draw.text((610, 105), 'Per-sample DINO loss', font=font(20, True), fill=TEXT)
    for index, (before, after) in enumerate(zip(before_losses, after_losses)):
        draw.text((610, 150 + index * 42),
                  f'sample {index}:  {before:.3f} -> {after:.3f}',
                  font=font(18), fill=TEXT)
    draw.text((610, 330),
              f'target generation {target_seconds:.1f}s | training {train_seconds:.1f}s',
              font=font(15), fill=TEXT)
    draw.text((610, 362), f'peak VRAM {peak_gib:.2f} GiB',
              font=font(15), fill=TEXT)
    draw.text((610, 410), 'No segmentation, center, offset, or 3D-box loss.',
              font=font(15, True), fill=(145, 65, 35))

    panel_size = 190
    labels = ('Soft target confidence', 'Before training', 'After training')
    for row, (description, index) in enumerate(selected):
        y = 510 + row * 250
        draw.text((25, y - 34),
                  f'{description} | sample {index} | loss {before_losses[index]:.3f} -> {after_losses[index]:.3f}',
                  font=font(18, True), fill=TEXT)
        confidence = confidences[index].float().cpu().numpy()
        arrays = (
            confidence_image(confidence),
            similarity_image(before_similarity[index], confidence),
            similarity_image(after_similarity[index], confidence),
        )
        for col, (label, array) in enumerate(zip(labels, arrays)):
            x = 25 + col * 300
            draw.text((x, y), label, font=font(16, True), fill=TEXT)
            panel = Image.fromarray(array).resize(
                (panel_size, panel_size), Image.Resampling.NEAREST
            )
            canvas.paste(panel, (x, y + 27))

    draw.text((25, 1014),
              'Four-sample overfit verifies the label-free loss path; it is not downstream accuracy or generalization.',
              font=font(14), fill=(145, 65, 35))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if args.num_samples != 4:
        raise ValueError('this presentation prototype is intentionally fixed to 4 samples')

    torch.manual_seed(125)
    np.random.seed(125)
    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    loader = build_loader(args.data_root, num_workers=0, nsweeps=1)
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(args.num_samples)]

    teacher = load_teacher(args.model_name, device)
    for parameter in teacher.parameters():
        if parameter.requires_grad:
            raise RuntimeError('DINOv2 teacher must remain frozen')
    cache_dir = args.output_dir / 'dinov2_mini4_targets'
    targets, confidences, target_seconds = create_targets(
        batches, teacher, device, cache_dir, args.model_name
    )
    del teacher
    torch.cuda.empty_cache()

    model_inputs = [prepare_inputs(batch, device) for batch in batches]
    torch.manual_seed(125)
    student = Segnet(
        Z, Y, X, vox_util=model_inputs[0][3],
        use_radar=True, use_metaradar=True, do_rgbcompress=True,
        encoder_type='res101', rand_flip=False, pretrained_backbone=False,
    ).to(device)
    model = SemanticDistillationModel(student).to(device)
    for parameter in model.student.decoder.parameters():
        parameter.requires_grad_(False)
    model.student.ce_weight.requires_grad_(False)
    model.student.center_weight.requires_grad_(False)
    model.student.offset_weight.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-5
    )
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0, growth_interval=1000)

    before_losses, before_similarity = evaluate(
        model, model_inputs, targets, confidences
    )
    if not np.isfinite(before_losses).all():
        raise RuntimeError('random-student float32 evaluation is non-finite')
    losses = []
    gradient_norms = []
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model.train()
    model.student.decoder.eval()
    for step in range(args.steps):
        index = step % args.num_samples
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            prediction, *_ = model(*model_inputs[index])
            loss, _ = semantic_loss(
                prediction, targets[index][None], confidences[index][None]
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, 5.0, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        gradient_norms.append(float(grad_norm))
        if step < 4 or (step + 1) % 10 == 0:
            print(
                f'step {step + 1:02d}/{args.steps}: sample={index} '
                f'loss={losses[-1]:.6f} grad_norm={gradient_norms[-1]:.4f}'
            )

    after_losses, after_similarity = evaluate(
        model, model_inputs, targets, confidences
    )
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    if not np.isfinite(losses).all() or not np.isfinite(after_losses).all():
        raise RuntimeError('non-finite DINO distillation result')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / 'dinov2_mini4_checkpoint'
    saverloader.save(
        str(checkpoint_dir), optimizer, model, args.steps, keep_latest=1
    )
    output_path = args.output_dir / 'dinov2_mini4_distillation.png'
    render(
        losses, before_losses, after_losses, before_similarity,
        after_similarity, confidences, target_seconds, train_seconds,
        peak_gib, output_path,
    )
    metrics_path = args.output_dir / 'dinov2_mini4_metrics.npz'
    np.savez_compressed(
        metrics_path,
        losses=np.asarray(losses, dtype=np.float32),
        gradient_norms=np.asarray(gradient_norms, dtype=np.float32),
        before_losses=before_losses,
        after_losses=after_losses,
        target_seconds=np.asarray(target_seconds, dtype=np.float32),
        train_seconds=np.asarray(train_seconds, dtype=np.float32),
        peak_cuda_memory_gib=np.asarray(peak_gib, dtype=np.float32),
        supervised_bev_checkpoint_loaded=np.asarray(False),
        box_loss_used=np.asarray(False),
    )
    print(f'first_cycle_mean_loss: {np.mean(losses[:4]):.6f}')
    print(f'last_cycle_mean_loss: {np.mean(losses[-4:]):.6f}')
    print(f'before_mean_loss: {before_losses.mean():.6f}')
    print(f'after_mean_loss: {after_losses.mean():.6f}')
    print(f'target_generation_seconds: {target_seconds:.3f}')
    print(f'train_and_eval_seconds: {train_seconds:.3f}')
    print(f'peak_cuda_memory_gib: {peak_gib:.3f}')
    print('supervised_bev_checkpoint_loaded: False')
    print('box_loss_used: False')
    print(f'target_cache: {cache_dir}')
    print(f'metrics: {metrics_path}')
    print(f'visualization: {output_path}')
    print('DINOV2_MINI4_EXPERIMENT_OK')


if __name__ == '__main__':
    main()
