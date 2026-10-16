#!/usr/bin/env python3
"""Evaluate official Simple-BEV weights using the full validation manifest."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from nets.segnet import Segnet
from baseline_followup import (code_commit, digest_file, evaluate_predictions,
                               load_datasets)
from fullsize_training import atomic_json_write
from radar_scaling_data import validate_scaling_manifest
from train_nuscenes import SimpleLoss, Z, Y, X, run_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--use-radar', action='store_true')
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--val-samples', type=int, default=6019)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=125)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    manifest = json.loads(args.manifest.read_text())
    validate_scaling_manifest(manifest)
    if not 1 <= args.val_samples <= manifest['val']['count']:
        raise ValueError('invalid val-samples')
    _, dataset = load_datasets(args, manifest, 5 if args.use_radar else 1,
                              (args.height, args.width), rotate=False)
    device = torch.device('cuda:0')
    model = Segnet(Z, Y, X, use_radar=args.use_radar,
        use_metaradar=args.use_radar, encoder_type='res101',
        do_rgbcompress=True, rand_flip=False, pretrained_backbone=False).to(device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    del payload
    model.eval()
    criterion = SimpleLoss(2.13).to(device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(experiment='official_pretrained_reference',
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest_file(args.checkpoint),
        source_code_commit=code_commit(), manifest_sha256=manifest['manifest_sha256'],
        use_radar=args.use_radar, nsweeps=5 if args.use_radar else 1,
        image_resolution=[args.height, args.width], batch_size=args.batch_size,
        rotate_radar_velocity=False, precision='float32', camera_reference='CAM_FRONT',
        supervised_training_in_this_run=False)
    atomic_json_write(config, args.run_dir / 'config.json')
    start = time.monotonic()
    def status(samples):
        atomic_json_write(dict(state='validating', samples=samples,
            total=args.val_samples, updated_unix_time=time.time(),
            elapsed_seconds=time.monotonic()-start), args.run_dir / 'status.json')

    def predict(batch):
        # The unchanged official-compatible helper supplies geometry and the
        # metaradar voxelization. Counts are accumulated globally below.
        _, _, outputs = run_model(model, criterion, batch, str(device),
                                   return_outputs=True)
        return outputs['seg_bev_logits']

    with torch.inference_mode():
        result = evaluate_predictions(predict, dataset,
            manifest['val']['dataset_indices'][:args.val_samples], args, manifest,
            args.run_dir / 'validation', status)
    summary = dict(status='complete', config=config, final_validation=result,
                    elapsed_seconds=time.monotonic()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    atomic_json_write(summary, args.run_dir / 'summary.json')
    atomic_json_write(dict(state='complete', **summary), args.run_dir / 'status.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
