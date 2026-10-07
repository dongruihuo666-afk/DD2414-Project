"""Batch-safe helpers and progress records for the epoch-based full-size run."""

import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from compare_motion_target_bevcar_tiny import motion_loss, motion_target
from dinov2_bev_demo import (
    PATCH_SIZE,
    TEACHER_HEIGHT,
    TEACHER_WIDTH,
    camera_geometry,
    make_radar_bev,
    radar_anchored_soft_targets,
)
from semantic_distill_smoke import semantic_loss


class AcknowledgedShuffleSampler(Sampler):
    """Deterministic shuffling whose resume cursor ignores worker prefetch.

    DataLoader workers may request samples ahead of the last optimizer update.
    Iteration therefore does not mutate the saved cursor; the trainer advances
    it only after a batch has completed successfully.
    """

    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.position = 0

    def _order(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.data_source), generator=generator).tolist()

    def __iter__(self):
        return iter(self._order()[self.position:])

    def __len__(self):
        return len(self.data_source) - self.position

    def acknowledge(self, count):
        count = int(count)
        if count < 1 or self.position + count > len(self.data_source):
            raise ValueError('invalid acknowledged sample count')
        self.position += count
        if self.position == len(self.data_source):
            self.epoch += 1
            self.position = 0

    def state_dict(self):
        return {
            'seed': self.seed,
            'dataset_size': len(self.data_source),
            'epoch': self.epoch,
            'position': self.position,
        }

    def load_state_dict(self, state):
        if int(state['seed']) != self.seed:
            raise ValueError('checkpoint sampler seed does not match current run')
        if int(state['dataset_size']) != len(self.data_source):
            raise ValueError('checkpoint sampler dataset size does not match current run')
        epoch = int(state['epoch'])
        position = int(state['position'])
        if epoch < 0 or not 0 <= position < len(self.data_source):
            raise ValueError('checkpoint sampler position is invalid')
        self.epoch = epoch
        self.position = position


def atomic_json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def slice_batch(batch, index):
    """Keep a one-sample batch dimension for every collated tensor."""
    output = []
    for value in batch:
        if not torch.is_tensor(value) or value.ndim == 0:
            raise TypeError('full-size training expects a tuple of batched tensors')
        output.append(value[index:index + 1])
    return tuple(output)


def extract_teacher_features_batched(teacher, images, device):
    """Extract all B x S camera maps in one frozen DINO forward."""
    if images.ndim != 5:
        raise ValueError('images must have shape (B, S, 3, H, W)')
    batch, cameras, channels, height, width = images.shape
    flat = images.reshape(batch * cameras, channels, height, width).to(
        device, non_blocking=True,
    )
    flat = F.interpolate(
        flat, size=(TEACHER_HEIGHT, TEACHER_WIDTH), mode='bicubic',
        align_corners=False, antialias=True,
    )
    mean = flat.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = flat.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    output = teacher.forward_features((flat - mean) / std)
    tokens = output['x_norm_patchtokens']
    features = tokens.transpose(1, 2).reshape(
        batch, cameras, tokens.shape[-1],
        TEACHER_HEIGHT // PATCH_SIZE, TEACHER_WIDTH // PATCH_SIZE,
    )
    return features


def build_targets_batched(batch, teacher, device):
    """Build one radar-anchored target per sample while batching DINO."""
    images = batch[0][:, 0]
    with torch.inference_mode():
        features = extract_teacher_features_batched(teacher, images, device)
        targets = []
        confidences = []
        for index in range(images.shape[0]):
            single = slice_batch(batch, index)
            sample_features = features[index]
            _, pixel_from_camera, _, cameras_from_vehicle = camera_geometry(
                single, sample_features.shape[-2], sample_features.shape[-1], device,
            )
            # make_radar_bev constructs the same Vox_util used by the historical path.
            from train_nuscenes import Z, Y, X, bounds, scene_centroid
            import utils.vox
            vox_util = utils.vox.Vox_util(
                Z, Y, X, scene_centroid=scene_centroid.to(device),
                bounds=bounds, assert_cube=False,
            )
            _, radar_vehicle, radar_camera0 = make_radar_bev(
                single, cameras_from_vehicle, vox_util, device,
            )
            target, confidence, _ = radar_anchored_soft_targets(
                sample_features, pixel_from_camera, cameras_from_vehicle,
                radar_vehicle, radar_camera0, vox_util,
            )
            targets.append(target.half())
            confidences.append(confidence.half())
        return torch.stack(targets), torch.stack(confidences)


def build_motion_targets_batched(batch, device):
    fields = []
    coverages = []
    infos = []
    for index in range(batch[0].shape[0]):
        field, coverage, info = motion_target(slice_batch(batch, index), device)
        fields.append(field)
        coverages.append(coverage)
        infos.append(info)
    return torch.stack(fields), torch.stack(coverages), infos


def semantic_loss_batched(prediction, target, confidence):
    """Average historical batch-one semantic losses across samples."""
    values = [
        semantic_loss(
            prediction[index:index + 1], target[index:index + 1],
            confidence[index:index + 1],
        )[0]
        for index in range(prediction.shape[0])
    ]
    return torch.stack(values).mean()


def motion_loss_batched(prediction, target, coverage):
    """Average historical batch-one motion losses across samples."""
    values = []
    covered_cells = 0
    for index in range(prediction.shape[0]):
        value, cells = motion_loss(
            prediction[index:index + 1], target[index], coverage[index],
        )
        values.append(value)
        covered_cells += cells
    return torch.stack(values).mean(), covered_cells


def learning_rate_at_step(base_lr, step, total_steps, warmup_steps,
                          minimum_ratio=0.1):
    """Warm up, then cosine-decay as a function of epoch-derived steps."""
    if base_lr <= 0 or total_steps < 1 or warmup_steps < 0:
        raise ValueError('invalid learning-rate schedule')
    if not 0 < minimum_ratio <= 1:
        raise ValueError('minimum_ratio must be in (0, 1]')
    step = min(max(int(step), 0), total_steps - 1)
    if warmup_steps and step < warmup_steps:
        # Start from the same nonzero floor used by the cosine tail. Beginning
        # at nearly zero made a quarter-epoch warmup unnecessarily inert on
        # the 4,096-sample control.
        progress = (step + 1) / warmup_steps
        factor = minimum_ratio + (1.0 - minimum_ratio) * progress
    else:
        denominator = max(total_steps - warmup_steps - 1, 1)
        progress = (step - warmup_steps) / denominator
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = minimum_ratio + (1.0 - minimum_ratio) * cosine
    return base_lr * factor


def query_nvidia_smi():
    """Return lightweight live GPU telemetry, or None when unavailable."""
    command = [
        'nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits',
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=3,
        )
        fields = [value.strip() for value in result.stdout.splitlines()[0].split(',')]
        return {
            'utilization_percent': float(fields[0]),
            'memory_used_mib': float(fields[1]),
            'memory_total_mib': float(fields[2]),
            'temperature_c': float(fields[3]),
            'power_w': float(fields[4]),
        }
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def progress_status(*, state, scale, epochs, dataset_size, batch_size,
                    optimizer_step, total_steps, samples_seen, elapsed_seconds,
                    latest, ema, learning_rate, gpu=None, message=None):
    total_samples = epochs * dataset_size
    fraction = min(samples_seen / max(total_samples, 1), 1.0)
    rate = samples_seen / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = max(total_samples - samples_seen, 0)
    status = {
        'state': state,
        'scale': scale,
        'epochs': epochs,
        'dataset_size': dataset_size,
        'batch_size': batch_size,
        'epoch_progress': samples_seen / dataset_size,
        'optimizer_step': optimizer_step,
        'total_optimizer_steps': total_steps,
        'samples_seen': samples_seen,
        'total_sample_exposures': total_samples,
        'percent_complete': 100.0 * fraction,
        'elapsed_seconds': elapsed_seconds,
        'samples_per_second': rate,
        'eta_seconds': remaining / rate if rate > 0 else None,
        'learning_rate': learning_rate,
        'latest': latest,
        'ema': ema,
        'gpu': gpu,
        'message': message,
        'updated_unix_time': time.time(),
    }
    return status
