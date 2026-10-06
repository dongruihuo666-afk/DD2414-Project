#!/usr/bin/env python3
"""CPU-only tests for deterministic radar-scaling manifests and streaming."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from torch.utils.data import Dataset

from radar_scaling_data import (
    build_scaling_manifest,
    make_streaming_loader,
    scale_indices,
    validate_scaling_manifest,
    write_manifest,
)


class FakeDataset(Dataset):
    def __init__(self, split, scenes, samples_per_scene):
        self.nusc = SimpleNamespace(version='v1.0-trainval')
        self.ixes = []
        for scene_index in range(scenes):
            scene_token = f'{split}-scene-{scene_index:03d}'
            for sample_index in range(samples_per_scene):
                self.ixes.append({
                    'token': f'{scene_token}-sample-{sample_index:03d}',
                    'scene_token': scene_token,
                })
        self.indices = np.arange(len(self.ixes)).reshape(-1, 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return int(index)


class RadarScalingDataTests(unittest.TestCase):
    def setUp(self):
        self.train = FakeDataset('train', scenes=12, samples_per_scene=3)
        self.val = FakeDataset('val', scenes=4, samples_per_scene=2)

    def test_manifest_is_reproducible_nested_and_scene_spread(self):
        first = build_scaling_manifest(self.train, self.val, seed=125,
                                       scale_counts=(4, 8, 16))
        second = build_scaling_manifest(self.train, self.val, seed=125,
                                        scale_counts=(4, 8, 16))
        self.assertEqual(first, second)
        self.assertEqual(first['scales']['4']['unique_scenes'], 4)
        self.assertEqual(first['scales']['8']['unique_scenes'], 8)
        self.assertEqual(first['scales']['16']['unique_scenes'], 12)
        self.assertEqual(
            first['scales']['16']['dataset_indices'][:8],
            first['scales']['8']['dataset_indices'],
        )
        self.assertEqual(len(first['scales']['full']['dataset_indices']), 36)
        self.assertEqual(len(set(first['scales']['full']['dataset_indices'])), 36)
        self.assertTrue(validate_scaling_manifest(first))

    def test_seed_changes_order_but_not_membership(self):
        first = build_scaling_manifest(self.train, self.val, seed=125,
                                       scale_counts=(4, 8, 16))
        second = build_scaling_manifest(self.train, self.val, seed=42,
                                        scale_counts=(4, 8, 16))
        self.assertNotEqual(scale_indices(first, 4), scale_indices(second, 4))
        self.assertEqual(
            set(scale_indices(first, 'full')),
            set(scale_indices(second, 'full')),
        )

    def test_overlap_and_invalid_scale_are_rejected(self):
        shared_val = FakeDataset('train', scenes=2, samples_per_scene=2)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            build_scaling_manifest(self.train, shared_val, scale_counts=(4, 8))
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            build_scaling_manifest(self.train, self.val, scale_counts=(8, 4))
        with self.assertRaisesRegex(ValueError, 'smaller than full'):
            build_scaling_manifest(self.train, self.val, scale_counts=(36,))

    def test_atomic_write_and_streaming_loader(self):
        manifest = build_scaling_manifest(self.train, self.val, seed=125,
                                          scale_counts=(4, 8, 16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            write_manifest(manifest, path)
            self.assertEqual(json.loads(path.read_text()), manifest)
            self.assertEqual(list(path.parent.glob('.*.tmp-*')), [])
        batches = list(make_streaming_loader(
            self.train, scale_indices(manifest, 8), batch_size=2,
        ))
        self.assertEqual(len(batches), 4)
        observed = [int(value) for batch in batches for value in batch]
        self.assertEqual(observed, scale_indices(manifest, 8))


if __name__ == '__main__':
    unittest.main()
