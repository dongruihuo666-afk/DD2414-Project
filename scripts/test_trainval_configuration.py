#!/usr/bin/env python3
"""CPU-only configuration checks: no project model or dataset is executed."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class FakeDataset:
    def __init__(self, prefix, count=10):
        self.ixes = [{'token': f'{prefix}-{i}', 'scene_token': f'{prefix}-scene'}
                     for i in range(count)]
        self.indices = np.arange(count).reshape(-1, 1)

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, index):
        return index


class TrainvalConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(dset='trainval', data_root=Path('/unused'),
                                    nsweeps=5, train_samples=4, val_samples=3,
                                    steps=10, sample_selection='uniform')
        self.data = definitions('scripts/heldout_data.py',
                                {'select_indices', 'validate_probe_args', 'load_probe_batches'},
                                {'np': np, 'DataLoader': DataLoader, 'Subset': Subset})

    def test_selection_and_count_errors(self):
        select = self.data['select_indices']
        self.assertEqual(select(10, 4, 'head'), [0, 1, 2, 3])
        self.assertEqual(select(10, 4, 'uniform'), [0, 3, 6, 9])
        for count in (0, -1, 11):
            with self.assertRaises(ValueError):
                select(10, count, 'uniform')

    def test_cached_probe_budget(self):
        validate = self.data['validate_probe_args']
        validate(self.args)
        self.args.train_samples = 28130
        with self.assertRaisesRegex(ValueError, 'streaming'):
            validate(self.args)
        self.args.train_samples = 4
        self.args.nsweeps = 0
        with self.assertRaisesRegex(ValueError, 'nsweeps'):
            validate(self.args)

    def test_manifest_and_split_disjointness(self):
        seen = []
        def loaders(*args, **kwargs):
            seen.append(kwargs)
            return [SimpleNamespace(dataset=FakeDataset(prefix)) for prefix in ('train', 'val')]
        self.data['build_loaders'] = loaders
        train, val, manifest = self.data['load_probe_batches'](self.args, True)
        self.assertEqual((len(train), len(val)), (4, 3))
        self.assertEqual(manifest['train']['tokens'], ['train-0', 'train-3', 'train-6', 'train-9'])
        self.assertEqual(seen, [{'rotate_radar_velocity': True, 'dset': 'trainval'}])
        shared = SimpleNamespace(dataset=FakeDataset('same'))
        self.data['build_loaders'] = lambda *args, **kwargs: (shared, shared)
        with self.assertRaisesRegex(RuntimeError, 'overlap'):
            self.data['load_probe_batches'](self.args)

    def test_shared_loader_routing_preserves_mini(self):
        seen = []
        def compile_data(*args, **kwargs):
            seen.append((args, kwargs))
            return 'train-loader', 'val-loader'
        scope = definitions('scripts/dinov2_bev_demo.py', {'build_loaders', 'build_loader'},
                            {'nuscenesdataset': SimpleNamespace(compile_data=compile_data),
                             'scene_centroid_py': [0, 1, 0], 'bounds': (), 'Z': 200, 'Y': 8, 'X': 200})
        self.assertEqual(scope['build_loader']('/unused', 0, 1), 'train-loader')
        self.assertEqual(seen[-1][0][0], 'mini')
        self.assertEqual(scope['build_loader']('/unused', 0, 10, split='val', dset='trainval'), 'val-loader')
        self.assertEqual(seen[-1][0], ('trainval', '/unused'))
        self.assertEqual(seen[-1][1]['nsweeps'], 10)
        self.assertEqual(seen[-1][1]['nworkers_val'], 0)
        with self.assertRaises(ValueError):
            scope['build_loader']('/unused', 0, 1, dset='test')


if __name__ == '__main__':
    unittest.main()
