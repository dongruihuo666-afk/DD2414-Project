"""Bounded, reproducible sample selection for GPU-cached held-out probes."""

import numpy as np
from torch.utils.data import DataLoader, Subset

from dinov2_bev_demo import build_loaders


def add_dataset_args(parser):
    parser.add_argument('--dset', choices=('mini', 'trainval'), default='mini')
    parser.add_argument('--sample-selection', choices=('head', 'uniform'),
                        default='head',
                        help='uniform spreads a bounded subset across the sorted split')


def validate_probe_args(args):
    for name in ('train_samples', 'val_samples', 'steps', 'nsweeps'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    # A dense FP16 BEV target alone needs ~29.3 MiB per frame. Inputs and
    # training activations need additional memory. These are cached probes,
    # not a streaming full-dataset trainer.
    if args.train_samples + args.val_samples > 160:
        raise ValueError('GPU-cached probes support at most 160 total samples. '
                         'Use a bounded subset (e.g. 64 train / 16 val); '
                         'full-split distillation requires a streaming trainer.')


def select_indices(total, count, selection):
    if not 1 <= count <= total:
        raise ValueError(f'requested {count} samples from a split with {total}')
    if selection == 'head':
        return list(range(count))
    if selection == 'uniform':
        return np.linspace(0, total - 1, count, dtype=int).tolist()
    raise ValueError(f'unknown sample selection: {selection}')


def load_probe_batches(args, rotate_radar_velocity=False):
    """Share one devkit instance and record the scene-disjoint selected frames."""
    validate_probe_args(args)
    loaders = build_loaders(args.data_root, 0, args.nsweeps,
                            rotate_radar_velocity=rotate_radar_velocity,
                            dset=args.dset)
    datasets = [loader.dataset for loader in loaders]
    split_scenes = [{row['scene_token'] for row in ds.ixes} for ds in datasets]
    if split_scenes[0] & split_scenes[1]:
        raise RuntimeError('training and validation scenes overlap')
    selected, manifest = [], {}
    for split, ds, count in zip(('train', 'val'), datasets,
                               (args.train_samples, args.val_samples)):
        indices = select_indices(len(ds), count, args.sample_selection)
        rows = [ds.ixes[int(ds.indices[i][0])] for i in indices]
        manifest[split] = {
            'available_samples': len(ds),
            'available_scenes': len({row['scene_token'] for row in ds.ixes}),
            'indices': indices,
            'tokens': [row['token'] for row in rows],
            'scene_tokens': [row['scene_token'] for row in rows],
        }
        print(f'{args.dset} {split}: selecting {count}/{len(ds)} samples '
              f'from {len(set(manifest[split]["scene_tokens"]))} scenes '
              f'({args.sample_selection})', flush=True)
        # VizData creates CUDA geometry while fetching a sample, so avoid
        # forked DataLoader workers until that behavior is refactored.
        selected.append(list(DataLoader(Subset(ds, indices), batch_size=1,
                                        shuffle=False, num_workers=0)))
    return selected[0], selected[1], manifest
