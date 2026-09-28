#!/usr/bin/env python3
"""Overfit one nuScenes mini sample with radar-anchored DINOv2 targets."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import nuscenesdataset  # noqa: E402
import saverloader  # noqa: E402
import utils.basic  # noqa: E402
import utils.geom  # noqa: E402
import utils.vox  # noqa: E402
from nets.segnet import Segnet  # noqa: E402
from train_nuscenes import Z, Y, X, bounds, scene_centroid, scene_centroid_py  # noqa: E402


BACKGROUND = (245, 247, 250)
TEXT = (25, 30, 40)


class SemanticDistillationModel(nn.Module):
    def __init__(self, student, teacher_dim=384, camera_distill=False):
        super().__init__()
        self.student = student
        self.camera_distill = camera_distill
        self.semantic_head = nn.Sequential(
            nn.Conv2d(student.latent_dim, student.latent_dim, 3, padding=1, bias=False),
            nn.InstanceNorm2d(student.latent_dim),
            nn.GELU(),
            nn.Conv2d(student.latent_dim, teacher_dim, 1),
        )
        if camera_distill:
            self.camera_head = nn.Sequential(
                nn.Conv2d(student.feat2d_dim * student.Y, student.feat2d_dim,
                          3, padding=1, bias=False),
                nn.InstanceNorm2d(student.feat2d_dim),
                nn.GELU(),
                nn.Conv2d(student.feat2d_dim, teacher_dim, 1),
            )
        else:
            self.camera_head = None

    def forward(self, rgb, pixel_from_cameras, camera0_from_cameras,
                vox_util, radar_voxels):
        shared_bev, camera_bev = self.student(
            rgb_camXs=rgb,
            pix_T_cams=pixel_from_cameras,
            cam0_T_camXs=camera0_from_cameras,
            vox_util=vox_util,
            rad_occ_mem0=radar_voxels,
            return_shared_bev=True,
        )
        fusion_prediction = self.semantic_head(shared_bev)
        camera_prediction = (
            self.camera_head(camera_bev) if self.camera_head is not None else None
        )
        return fusion_prediction, camera_prediction, shared_bev, camera_bev


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--target-cache', type=Path, required=True)
    parser.add_argument('--teacher-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--camera-distill', action='store_true',
                        help='add a dense, camera-visibility DINOv2 term on the camera BEV')
    parser.add_argument('--camera-weight', type=float, default=1.0,
                        help='weight of the camera-side dense term relative to the fusion term')
    parser.add_argument('--train-encoder', action='store_true',
                        help='unfreeze the camera encoder so the camera-side term can shape it')
    return parser.parse_args()


def build_batch(data_root):
    config = {
        'crop_offset': 0,
        'resize_lim': [1.0, 1.0],
        'final_dim': (112, 192),
        'H': 900,
        'W': 1600,
        'cams': [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
        ],
        'ncams': 6,
    }
    loader, _ = nuscenesdataset.compile_data(
        'mini', str(data_root), data_aug_conf=config,
        centroid=scene_centroid_py, bounds=bounds, res_3d=(Z, Y, X),
        bsz=1, nworkers=0, nworkers_val=0, shuffle=False, nsweeps=1,
        seqlen=1, refcam_id=1, get_tids=True, temporal_aug=False,
        use_radar_filters=False, do_shuffle_cams=False,
    )
    return next(iter(loader))


def prepare_inputs(batch, device):
    images = batch[0][:, 0].float().to(device)
    rotations = batch[1][:, 0].to(device)
    translations = batch[2][:, 0].to(device)
    intrinsics = batch[3][:, 0].to(device)
    radar = batch[16][:, 0].to(device).permute(0, 2, 1)
    batch_size = images.shape[0]
    pack = lambda value: utils.basic.pack_seqdim(value, batch_size)
    unpack = lambda value: utils.basic.unpack_seqdim(value, batch_size)

    pixel_from_cameras = unpack(utils.geom.merge_intrinsics(
        *utils.geom.split_intrinsics(pack(intrinsics))
    ))
    vehicle_from_cameras = utils.geom.merge_rtlist(rotations, translations)
    cameras_from_vehicle = unpack(utils.geom.safe_inverse(pack(vehicle_from_cameras)))
    camera0_from_cameras = utils.geom.get_camM_T_camXs(
        vehicle_from_cameras, ind=0
    )

    radar_xyz = radar[:, :, :3]
    radar_meta = radar[:, :, 3:]
    radar_camera0 = utils.geom.apply_4x4(cameras_from_vehicle[:, 0], radar_xyz)
    vox_util = utils.vox.Vox_util(
        Z, Y, X, scene_centroid=scene_centroid.to(device),
        bounds=bounds, assert_cube=False,
    )
    radar_voxels = vox_util.voxelize_xyz_and_feats(
        radar_camera0, radar_meta, Z, Y, X, assert_cube=False
    )
    return (
        images - 0.5, pixel_from_cameras, camera0_from_cameras,
        vox_util, radar_voxels,
    )


def semantic_loss(prediction, target, confidence):
    prediction = F.normalize(prediction.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    cosine = (prediction * target).sum(dim=1)
    weights = confidence.float()
    weights = torch.where(weights > 0.05, weights, torch.zeros_like(weights))
    weights = weights / weights.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    loss = ((1.0 - cosine) * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, cosine


def heatmap(values):
    values = np.clip(values, 0, 1)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * values).astype(np.uint8)
    rgb[..., 1] = (210 * np.sqrt(values)).astype(np.uint8)
    rgb[..., 2] = (40 * (1.0 - values)).astype(np.uint8)
    return rgb


def similarity_image(similarity, confidence):
    similarity = np.clip((similarity + 1.0) / 2.0, 0, 1)
    rgb = heatmap(similarity)
    rgb[confidence <= 0.05] = 0
    return rgb


def render_curve(draw, losses, box):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill='white', outline=(180, 185, 195), width=2)
    high, low = max(losses), min(losses)
    span = max(high - low, 1e-6)
    points = []
    for index, value in enumerate(losses):
        x = x0 + 16 + index * (x1 - x0 - 32) / max(len(losses) - 1, 1)
        y = y1 - 16 - (value - low) / span * (y1 - y0 - 32)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(220, 55, 45), width=4)
    for point in points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(220, 55, 45))


def render_result(losses, initial_similarity, final_similarity, confidence,
                  ground_truth, gradient_norms, peak_gib, elapsed, output_path):
    width, height = 1180, 720
    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), 'Self-supervised semantic distillation: 1-sample overfit',
              font=font(30, True), fill=TEXT)
    draw.text((28, 62),
              'Student shared BEV -> projection head -> radar-anchored frozen DINOv2 target',
              font=font(16), fill=(70, 78, 92))

    draw.text((28, 110), 'Cosine loss curve', font=font(20, True), fill=TEXT)
    render_curve(draw, losses, (28, 145, 545, 445))
    draw.text((45, 462), f'start {losses[0]:.4f}', font=font(16), fill=TEXT)
    draw.text((210, 462), f'end {losses[-1]:.4f}', font=font(16), fill=TEXT)
    reduction = (1.0 - losses[-1] / max(losses[0], 1e-8)) * 100
    draw.text((365, 462), f'down {reduction:.1f}%', font=font(16, True), fill=(180, 45, 35))

    panel_size = 220
    panels = (
        ('Before training', similarity_image(initial_similarity, confidence)),
        ('After training', similarity_image(final_similarity, confidence)),
        ('Teacher confidence', heatmap(confidence / max(float(confidence.max()), 1e-6))),
    )
    for index, (label, array) in enumerate(panels):
        x = 585 + (index % 2) * 270
        y = 110 + (index // 2) * 290
        draw.text((x, y), label, font=font(20, True), fill=TEXT)
        panel = Image.fromarray(array).resize((panel_size, panel_size), Image.Resampling.NEAREST)
        canvas.paste(panel, (x, y + 35))

    gt = np.zeros((Z, X, 3), dtype=np.uint8)
    gt[ground_truth > 0.5] = (0, 255, 80)
    draw.text((855, 400), 'GT (view only)', font=font(20, True), fill=TEXT)
    canvas.paste(Image.fromarray(gt).resize((panel_size, panel_size), Image.Resampling.NEAREST), (855, 435))

    draw.text((28, 520),
              f'Finite gradient norm: {gradient_norms[0]:.3f} -> {gradient_norms[-1]:.3f}',
              font=font(16), fill=TEXT)
    draw.text((28, 550), f'Elapsed {elapsed:.1f}s | peak VRAM {peak_gib:.2f} GiB',
              font=font(16), fill=TEXT)
    draw.text((28, 580), 'No segmentation or 3D-box loss used',
              font=font(16, True), fill=TEXT)
    draw.text((28, 625), 'Bright = higher student/teacher similarity.',
              font=font(14), fill=(70, 78, 92))
    draw.text((28, 650), 'Black = outside the soft target mask.',
              font=font(14), fill=(70, 78, 92))
    draw.text((28, 680), 'Debugging overfit, not validation accuracy.',
              font=font(14), fill=(145, 65, 35))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    for path in (args.target_cache, args.teacher_cache):
        if not path.is_file():
            raise FileNotFoundError(f'cache not found: {path}')

    torch.manual_seed(125)
    np.random.seed(125)
    torch.cuda.init()
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch = build_batch(args.data_root)

    teacher_metadata = np.load(args.teacher_cache)
    for name, batch_value, cache_value in (
        ('intrinsics', batch[3][0, 0], teacher_metadata['intrinsics']),
        ('rotations', batch[1][0, 0], teacher_metadata['rotations']),
        ('translations', batch[2][0, 0], teacher_metadata['translations']),
    ):
        if not np.allclose(batch_value.numpy(), cache_value, atol=1e-5):
            raise RuntimeError(f'{name} do not match cached teacher sample')

    cached = np.load(args.target_cache)
    target = torch.from_numpy(cached['semantic_target'].astype(np.float32))[None].to(device)
    confidence = torch.from_numpy(cached['confidence'].astype(np.float32))[None].to(device)
    if args.camera_distill:
        if 'dense_semantic_target' not in cached or 'camera_coverage' not in cached:
            raise RuntimeError(
                'target cache lacks dense camera keys; re-run run_dinov2_bev_demo.sh'
            )
        dense_target = torch.from_numpy(
            cached['dense_semantic_target'].astype(np.float32)
        )[None].to(device)
        dense_confidence = torch.from_numpy(
            cached['camera_coverage'].astype(np.float32)
        )[None].gt(0).float().to(device)
    inputs = prepare_inputs(batch, device)
    student = Segnet(
        Z, Y, X, vox_util=inputs[3], use_radar=True, use_metaradar=True,
        do_rgbcompress=True, encoder_type='res101', rand_flip=False,
        pretrained_backbone=False,
    ).to(device)
    saverloader.load(str(args.checkpoint), student)
    model = SemanticDistillationModel(
        student, camera_distill=args.camera_distill
    ).to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.student.bev_compressor.parameters():
        parameter.requires_grad_(True)
    for parameter in model.semantic_head.parameters():
        parameter.requires_grad_(True)
    if args.camera_distill:
        for parameter in model.camera_head.parameters():
            parameter.requires_grad_(True)
    if args.train_encoder:
        for parameter in model.student.encoder.parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    losses = []
    camera_losses = []
    gradient_norms = []
    initial_similarity = None
    start = time.perf_counter()
    model.train()
    if args.train_encoder:
        model.student.encoder.train()
    else:
        model.student.encoder.eval()
    model.student.decoder.eval()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            prediction, camera_prediction, _, _ = model(*inputs)
            loss, similarity = semantic_loss(prediction, target, confidence)
            if args.camera_distill:
                camera_loss, _ = semantic_loss(
                    camera_prediction, dense_target, dense_confidence
                )
                loss = loss + args.camera_weight * camera_loss
        if initial_similarity is None:
            initial_similarity = similarity.detach().float().cpu().numpy()[0]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        if args.camera_distill:
            camera_losses.append(float(camera_loss.detach()))
        gradient_norms.append(float(grad_norm))
        suffix = f' camera_loss={camera_losses[-1]:.6f}' if args.camera_distill else ''
        print(f'step {step + 1:02d}/{args.steps}: loss={losses[-1]:.6f}{suffix} grad_norm={gradient_norms[-1]:.4f}')

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        final_prediction, final_camera_prediction, shared_bev, camera_bev = model(*inputs)
        final_loss, final_similarity = semantic_loss(final_prediction, target, confidence)
        final_total_loss = final_loss
        if args.camera_distill:
            final_camera_loss, _ = semantic_loss(
                final_camera_prediction, dense_target, dense_confidence
            )
            final_total_loss = final_loss + args.camera_weight * final_camera_loss
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    final_similarity = final_similarity.float().cpu().numpy()[0]
    confidence_np = confidence[0].float().cpu().numpy()
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / 'semantic_distill_smoke_checkpoint'
    saverloader.save(str(checkpoint_dir), optimizer, model, args.steps, keep_latest=1)
    first_head_parameter = next(model.semantic_head.parameters())
    expected = first_head_parameter.detach().clone()
    with torch.no_grad():
        first_head_parameter.zero_()
    restored_step = saverloader.load(str(checkpoint_dir), model, optimizer=optimizer)
    if restored_step != args.steps or not torch.equal(first_head_parameter, expected):
        raise RuntimeError('semantic checkpoint reload failed')

    output_path = args.output_dir / 'semantic_distillation_overfit.png'
    render_result(
        losses, initial_similarity, final_similarity, confidence_np,
        batch[12][0, 0, 0].numpy(), gradient_norms, peak_gib, elapsed,
        output_path,
    )
    np.savez_compressed(
        args.output_dir / 'semantic_distillation_overfit_metrics.npz',
        losses=np.asarray(losses, dtype=np.float32),
        camera_losses=np.asarray(camera_losses, dtype=np.float32),
        gradient_norms=np.asarray(gradient_norms, dtype=np.float32),
        final_loss=np.asarray(float(final_loss), dtype=np.float32),
        final_total_loss=np.asarray(float(final_total_loss), dtype=np.float32),
        peak_cuda_memory_gib=np.asarray(peak_gib, dtype=np.float32),
        elapsed_seconds=np.asarray(elapsed, dtype=np.float32),
    )

    compressor_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.student.bev_compressor.parameters()
    )
    print(f'initial_loss: {losses[0]:.6f}')
    print(f'last_train_loss: {losses[-1]:.6f}')
    print(f'final_eval_loss: {float(final_loss):.6f}')
    if args.camera_distill:
        print(f'final_camera_loss: {float(final_camera_loss):.6f}')
    print(f'final_total_loss: {float(final_total_loss):.6f}')
    print(f'loss_reduction_percent: {(1 - float(final_total_loss) / losses[0]) * 100:.2f}')
    print(f'shared_bev_shape: {tuple(shared_bev.shape)}')
    print(f'camera_bev_shape: {tuple(camera_bev.shape)}')
    print(f'prediction_shape: {tuple(final_prediction.shape)}')
    print(f'compressor_has_finite_gradient: {compressor_grad}')
    print(f'peak_cuda_memory_gib: {peak_gib:.3f}')
    print(f'elapsed_seconds: {elapsed:.3f}')
    print(f'visualization: {output_path}')
    print('SEMANTIC_DISTILL_SMOKE_OK')


if __name__ == '__main__':
    main()
