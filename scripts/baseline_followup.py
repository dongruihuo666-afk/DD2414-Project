"""Opt-in downstream utilities for the historical full-data checkpoints."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from compare_motion_target_bevcar_tiny import (
    build_model, official_voxelnet, prepare_bevcar_inputs,
)
from dinov2_bev_demo import build_loaders
from fullsize_training import atomic_json_write
from radar_evaluation import apply_radar_velocity_mode
from radar_scaling_data import IndexedSubset, dataset_records
from train_fullsize_linear_probe import balanced_binary_loss, metrics_from_counts
from train_nuscenes import SimpleLoss, balanced_mse_loss
import utils.basic


ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def code_commit():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
    ).strip()


def load_datasets(args, manifest, sweeps, resolution=(112, 192), rotate=True):
    raw = build_loaders(args.data_root, 0, sweeps,
                        rotate_radar_velocity=rotate, dset='trainval')
    datasets = [loader.dataset for loader in raw]
    # This is the actual historical resolution (not the generic 224x400
    # supervised profile). Change only the opt-in dataset instance.
    for dataset in datasets:
        dataset.data_aug_conf['final_dim'] = tuple(resolution)
    for dataset, expected in zip(datasets, (manifest['scales']['full'],
                                             manifest['val'])):
        rows = dataset_records(dataset)
        tokens = [rows[index]['token'] for index in expected['dataset_indices']]
        if tokens != expected['tokens']:
            raise ValueError('live dataset token order differs from manifest')
    return datasets


def labels_from_batch(batch, device):
    return {key: batch[index][:, 0].to(device, non_blocking=True)
            for key, index in [('segmentation', 12), ('valid', 13),
                               ('center', 14), ('offset', 15)]}


def official_counts(logits, target, valid):
    prediction = logits.sigmoid().round().bool()
    truth, mask = target.gt(0.5), valid.gt(0.5)
    return dict(tp=int((prediction & truth & mask).sum()),
                fp=int((prediction & ~truth & mask).sum()),
                fn=int((~prediction & truth & mask).sum()),
                tn=int((~prediction & ~truth & mask).sum()))


def supervised_loss(student, outputs, labels, segmentation_loss):
    _, _, segmentation, center, offset = outputs
    ce = 10 * segmentation_loss(segmentation, labels['segmentation'],
                               labels['valid']) / student.ce_weight.exp()
    ctr = balanced_mse_loss(center, labels['center']) / (
        2 * student.center_weight.exp())
    off = utils.basic.reduce_masked_mean(
        (offset - labels['offset']).abs().sum(1, keepdim=True),
        labels['segmentation'] * labels['valid'],
    ) / (2 * student.offset_weight.exp())
    uncertainty = .5 * (student.ce_weight + student.center_weight
                        + student.offset_weight)
    parts = dict(segmentation=ce, center=ctr, offset=off,
                 uncertainty=uncertainty)
    return sum(parts.values()), parts, segmentation


class LegacyDownstream(nn.Module):
    def __init__(self, backbone, mode):
        super().__init__()
        self.backbone, self.mode = backbone, mode
        self.probe = (nn.Conv2d(backbone.student.latent_dim, 1, 1)
                      if mode != 'finetune' else None)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        if mode == 'finetune':
            for parameter in backbone.student.parameters():
                parameter.requires_grad_(True)
        elif mode == 'unfrozen-probe':
            for module in (backbone.student.encoder,
                           backbone.student.bevcar_encoder,
                           backbone.student.bev_compressor):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'frozen-probe':
            self.backbone.eval()
        elif self.mode == 'unfrozen-probe':
            self.backbone.student.decoder.eval()
        return self

    def forward(self, batch, device, velocity_mode):
        images, pixels, cameras, vox, radar = prepare_bevcar_inputs(batch, device)
        radar = apply_radar_velocity_mode(radar, velocity_mode)
        kwargs = dict(rgb_camXs=images, pix_T_cams=pixels,
                      cam0_T_camXs=cameras, vox_util=vox, bevcar_voxels=radar)
        if self.mode == 'finetune':
            return self.backbone.student(**kwargs)
        if self.mode == 'frozen-probe':
            with torch.no_grad():
                features = self.backbone.student(**kwargs,
                                                  return_shared_bev=True)[0]
        else:
            features = self.backbone.student(**kwargs, return_shared_bev=True)[0]
        return self.probe(features.float())


def load_legacy(args, device):
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = payload['run_config']
    if config['experiment'] != 'fullsize_epoch_baseline_v1':
        raise ValueError('expected a historical epoch-baseline checkpoint')
    if payload['sampler']['epoch'] != config['epochs']:
        raise ValueError('pretraining checkpoint is incomplete')
    if config['seed'] != args.seed:
        raise ValueError('checkpoint seed mismatch')
    cls = official_voxelnet(args.bevcar_source)
    source = args.bevcar_source / 'nets/voxelnet.py'
    if digest_file(source) != config['bevcar_voxelnet_sha256']:
        raise ValueError('BEVCar source hash mismatch')
    backbone = build_model(device, args.seed, cls, zero_camera=False)
    backbone.load_state_dict(payload['model'], strict=True)
    model = LegacyDownstream(backbone, args.mode).to(device)
    return model, config


def forward_loss(model, batch, device, velocity, criterion):
    labels = labels_from_batch(batch, device)
    outputs = model(batch, device, velocity)
    if model.mode == 'finetune':
        return supervised_loss(model.backbone.student, outputs, labels, criterion)
    loss = balanced_binary_loss(outputs, labels['segmentation'], labels['valid'])
    return loss, {'balanced_bce': loss}, outputs


def save_mask_panel(logits, target, valid, token, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    probability = logits.detach().float().sigmoid().cpu().numpy().squeeze()
    truth = target.cpu().numpy().squeeze() > .5
    mask = valid.cpu().numpy().squeeze() > .5
    prediction = probability > .5
    errors = np.zeros((*truth.shape, 3))
    errors[prediction & truth & mask] = (0, .8, .2)
    errors[prediction & ~truth & mask] = (1, .15, .1)
    errors[~prediction & truth & mask] = (.1, .3, 1)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for axis, values, title in zip(axes, [truth*mask, probability, prediction*mask,
                                         errors], ['GT mask', 'Probability',
                                         'Prediction mask', 'TP green / FP red / FN blue']):
        axis.imshow(values, origin='lower', vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis('off')
    fig.suptitle(f'{output.parent.name} | token {token}', fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def evaluate_predictions(predict, dataset, indices, args, manifest, output_dir,
                         status_callback=None):
    """Incremental full-val confusion counts and fixed-token qualitative panels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = dict(tp=0, fp=0, fn=0, tn=0)
    selected = json.loads((ROOT / 'artifacts/fullsize_meeting_visuals.json').read_text())
    selected = {row['sample_token'] for row in selected['samples']}
    rows = dataset_records(dataset)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(IndexedSubset(dataset, indices), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True,
                        generator=generator)
    samples, iou_sum = 0, 0.
    with (output_dir / 'per_frame.jsonl').open('w') as handle:
        for positions, batch in loader:
            logits = predict(batch)
            target, valid = (value.to(logits.device) for value in
                             (batch[12][:, 0], batch[13][:, 0]))
            if not torch.isfinite(logits).all():
                raise RuntimeError('non-finite validation prediction')
            for index, position in enumerate(positions.tolist()):
                token = rows[indices[position]]['token']
                counts = official_counts(logits[index], target[index], valid[index])
                for key in totals:
                    totals[key] += counts[key]
                record = {**rows[indices[position]], **metrics_from_counts(counts)}
                iou_sum += record['vehicle_iou'] or 0.
                handle.write(json.dumps(record) + '\n')
                if token in selected:
                    save_mask_panel(logits[index], target[index], valid[index],
                                    token, output_dir / f'mask_{token}.png')
            samples += len(positions)
            if samples % (args.batch_size * 100) == 0 or samples == len(indices):
                handle.flush()
                if status_callback:
                    status_callback(samples)
                print(f'validation={samples}/{len(indices)}', flush=True)
    result = {**metrics_from_counts(totals), 'samples': samples,
              'mean_per_frame_iou_auxiliary': iou_sum / samples,
              'predicted_positive_fraction': (totals['tp']+totals['fp']) / sum(totals.values()),
              'metric_definition': 'global valid-cell intersection / union; sigmoid.round threshold',
              'manifest_sha256': manifest['manifest_sha256']}
    atomic_json_write(result, output_dir / 'summary.json')
    return result
