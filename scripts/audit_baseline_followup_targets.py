#!/usr/bin/env python3
"""Training-split depth illustrations and independent object-velocity coverage."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from nuscenes.nuscenes import NuScenes
from fullsize_training import atomic_json_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    tokens = set(manifest['scales']['full']['tokens'])
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    counts = dict(vehicles=0, finite_velocity=0, unavailable_velocity=0,
                  moving=0, stationary=0)
    for annotation in nusc.sample_annotation:
        if annotation['sample_token'] not in tokens or not annotation['category_name'].startswith('vehicle.'):
            continue
        counts['vehicles'] += 1
        velocity = nusc.box_velocity(annotation['token'])[:2]
        if np.isfinite(velocity).all():
            counts['finite_velocity'] += 1
            counts['moving' if np.linalg.norm(velocity)>1 else 'stationary'] += 1
        else:
            counts['unavailable_velocity'] += 1
    atomic_json_write(counts, args.run_dir/'object_velocity_coverage.json')
    source = ROOT/'.cache/Depth-Anything-V2'
    weights = ROOT/'.cache/depth_anything_v2_vits.pth'
    depth = dict(status='unavailable', reason='optional source/weights not cached')
    if (source/'depth_anything_v2/dpt.py').is_file() and weights.is_file():
        try:
            import cv2
            import torch
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            sys.path.insert(0, str(source))
            from depth_anything_v2.dpt import DepthAnythingV2
            model = DepthAnythingV2(encoder='vits', features=64,
                                    out_channels=[48,96,192,384])
            model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
            model.eval().cuda()
            from baseline_followup import digest_file
            paths = []
            # Training-only deterministic examples. No downstream GT is used.
            for token in manifest['scales']['full']['tokens'][:4]:
                sample = nusc.get('sample', token)
                record = nusc.get('sample_data', sample['data']['CAM_FRONT'])
                path = args.data_root/record['filename']
                image = cv2.imread(str(path))
                with torch.inference_mode():
                    prediction = model.infer_image(image)
                fig, axes = plt.subplots(1,2,figsize=(12,4))
                axes[0].imshow(image[...,::-1])
                axes[0].set_title('Training camera image')
                axes[1].imshow(prediction, cmap='inferno')
                axes[1].set_title('Depth Anything V2 relative depth (not meters)')
                for axis in axes:
                    axis.axis('off')
                fig.tight_layout()
                output = args.run_dir/f'relative_depth_{token}.png'
                fig.savefig(output, dpi=130)
                plt.close(fig)
                paths.append(str(output))
            depth = dict(status='complete', examples=paths,
                source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                weight_sha256=digest_file(weights), units='relative depth; not metric')
        except Exception as error:
            depth = dict(status='failed', reason=f'{type(error).__name__}: {error}')
    summary = dict(status='complete', object_velocity_coverage=counts, depth=depth,
        limitations=['Coverage audit only; no motion readout was trained or evaluated.',
                     'Depth illustrations are not BEV or metric-depth validation.',
                     'Radar-depth alignment and object-motion pooling remain future work.'])
    atomic_json_write(summary, args.run_dir/'summary.json')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
