#!/usr/bin/env python3
"""CPU tests for wrong-scene pairing, radar modes and incremental output."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from radar_evaluation import (
    apply_radar_velocity_mode,
    IncrementalEvaluationWriter,
    RADAR_MODES,
    build_wrong_scene_map,
    radar_for_mode,
    spread_positions,
    validate_wrong_scene_map,
    wrong_scene_dataset_indices,
)


class RadarEvaluationTests(unittest.TestCase):
    def test_train_time_zero_velocity_preserves_nonvelocity_inputs(self):
        features = torch.arange(42, dtype=torch.float32).reshape(1, 1, 6, 7)
        coords = torch.tensor([[[1, 2, 3]]])
        counts = torch.tensor([1])
        voxels = features, coords, counts
        self.assertIs(apply_radar_velocity_mode(voxels, 'full'), voxels)
        zero = apply_radar_velocity_mode(voxels, 'zero')
        self.assertTrue(torch.equal(zero[0][..., :4], features[..., :4]))
        self.assertTrue(torch.equal(zero[0][..., 4:6], torch.zeros_like(features[..., 4:6])))
        self.assertTrue(torch.equal(zero[0][..., 6:], features[..., 6:]))
        self.assertIs(zero[1], coords)
        self.assertIs(zero[2], counts)
        self.assertTrue(torch.equal(features, voxels[0]))
        with self.assertRaisesRegex(ValueError, 'unknown radar velocity mode'):
            apply_radar_velocity_mode(voxels, 'invalid')

    def setUp(self):
        self.tokens = [f'token-{index}' for index in range(9)]
        self.scenes = ['scene-a'] * 4 + ['scene-b'] * 3 + ['scene-c'] * 2

    def test_wrong_scene_mapping_is_reproducible_bijective_and_disjoint(self):
        first = build_wrong_scene_map(self.tokens, self.scenes, seed=125)
        second = build_wrong_scene_map(self.tokens, self.scenes, seed=125)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(self.tokens))
        self.assertEqual(set(first.values()), set(self.tokens))
        scene = dict(zip(self.tokens, self.scenes))
        self.assertTrue(all(scene[target] != scene[source]
                            for target, source in first.items()))
        self.assertTrue(validate_wrong_scene_map(first, self.tokens, self.scenes))

    def test_manifest_indices_follow_target_order(self):
        manifest = {'val': {
            'tokens': self.tokens,
            'scene_tokens': self.scenes,
            'dataset_indices': list(range(100, 109)),
        }}
        source_indices, mapping = wrong_scene_dataset_indices(manifest, seed=125)
        index_by_token = dict(zip(self.tokens, range(100, 109)))
        self.assertEqual(
            source_indices,
            [index_by_token[mapping[token]] for token in self.tokens],
        )

    def test_radar_modes_change_only_the_intended_input(self):
        features = torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(1, 2, 3, 7)
        coords = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
        counts = torch.tensor([2])
        matched = features, coords, counts
        wrong = features + 100, coords + 10, counts + 1
        self.assertIs(radar_for_mode(matched, 'matched'), matched)
        self.assertIs(radar_for_mode(matched, 'wrong_scene', wrong), wrong)
        zero_velocity = radar_for_mode(matched, 'zero_velocity')
        self.assertTrue(torch.equal(zero_velocity[0][..., :4], features[..., :4]))
        self.assertTrue(torch.equal(zero_velocity[0][..., 4:6], torch.zeros_like(features[..., 4:6])))
        self.assertTrue(torch.equal(zero_velocity[0][..., 6:], features[..., 6:]))
        empty = radar_for_mode(matched, 'empty')
        self.assertEqual(sum(int(value.count_nonzero()) for value in empty), 0)

    def test_incremental_writer_finalizes_only_on_success(self):
        record = {
            'token': 'target',
            'scene_token': 'scene-a',
            'wrong_radar_token': 'source',
            'wrong_radar_scene_token': 'scene-b',
            'modes': {mode: {'motion_loss': float(index)}
                      for index, mode in enumerate(RADAR_MODES)},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            with IncrementalEvaluationWriter(path) as writer:
                writer.write(record)
            self.assertEqual(json.loads(path.read_text().strip()), record)
            self.assertFalse((path.parent / f'.{path.name}.partial').exists())
            with self.assertRaisesRegex(ValueError, 'different scene'):
                with IncrementalEvaluationWriter(path) as writer:
                    writer.write({**record, 'wrong_radar_scene_token': 'scene-a'})
            self.assertTrue((path.parent / f'.{path.name}.partial').exists())
            nonfinite = {**record, 'token': 'nan'}
            nonfinite['modes'] = dict(record['modes'])
            nonfinite['modes']['matched'] = {'motion_loss': float('nan')}
            with self.assertRaisesRegex(ValueError, 'JSON compliant'):
                with IncrementalEvaluationWriter(path) as writer:
                    writer.write(nonfinite)

    def test_impossible_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least two scenes'):
            build_wrong_scene_map(['a', 'b'], ['same', 'same'])

    def test_spread_positions_include_both_ends(self):
        self.assertEqual(spread_positions(10, 4), [0, 3, 6, 9])
        self.assertEqual(spread_positions(10, 1), [0])


if __name__ == '__main__':
    unittest.main()
