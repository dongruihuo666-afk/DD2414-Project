"""Explicit, lazy CLI. Help and normal imports use only the standard library."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['train', 'evaluate'])
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return
    command = argv.pop(0)
    if command == 'train':
        from dd2414.cli.train_args import parse_args
        # Parse here first: --help needs neither Torch, assets nor a checkout.
        parse_args(argv)
        from dd2414.paths import repository_root
        root = repository_root(required=True)
        from dd2414.runtime.provenance import record_package_implementation
        record_package_implementation({}, root)
        path = root / 'scripts/train_multisweep_teacher.py'
        sys.path.insert(0, str(path.parent))  # Legacy CLI dispatch only.
        sys.path.insert(0, str(root))
        sys.argv = [str(path), *argv]
        runpy.run_path(str(path), run_name='__main__')
        return
    if command == 'evaluate':
        p = argparse.ArgumentParser(description='Compute official binary Vehicle IoU from saved logits/GT; no model inference.')
        p.add_argument('--arrays', type=Path, required=True,
                       help='NPZ with same-shaped logits, target and valid arrays [N,1,Z,X]. No pickle.')
        args = p.parse_args(argv)
        import numpy as np
        import torch
        from dd2414.evaluation.segmentation import official_counts
        from dd2414.evaluation.probe import metrics_from_counts
        with np.load(args.arrays, allow_pickle=False) as arrays:
            values = [torch.from_numpy(arrays[k]) for k in ('logits', 'target', 'valid')]
        if any(v.shape != values[0].shape for v in values) or values[0].ndim != 4 or values[0].shape[1] != 1:
            raise ValueError('Expected equal [N,1,Z,X] arrays')
        if not all(bool(torch.isfinite(v).all()) for v in values):
            raise ValueError('Evaluation arrays must be finite')
        result = metrics_from_counts(official_counts(*values))
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    parser.error('unknown command: ' + command)
