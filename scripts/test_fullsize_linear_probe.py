#!/usr/bin/env python3
"""CPU tests for the one-layer vehicle probe metrics and mode contract."""

import unittest

import torch

from train_fullsize_linear_probe import (
    balanced_binary_loss,
    binary_segmentation_counts,
    metrics_from_counts,
    resolve_velocity_mode,
)


class FullsizeLinearProbeTests(unittest.TestCase):
    def test_balanced_loss_gives_classes_equal_weight(self):
        logits = torch.tensor([[[[2.0, -2.0], [0.0, 0.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        valid = torch.ones_like(target)
        loss = balanced_binary_loss(logits, target, valid)
        positive = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[target.bool()], target[target.bool()], reduction='mean',
        )
        negative = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[~target.bool()], target[~target.bool()], reduction='mean',
        )
        self.assertTrue(torch.allclose(loss, (positive + negative) / 2))

    def test_counts_and_metrics(self):
        logits = torch.tensor([[[[4.0, 4.0], [-4.0, -4.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        valid = torch.ones_like(target)
        counts = binary_segmentation_counts(logits, target, valid)
        self.assertEqual(counts, {'tp': 1, 'fp': 1, 'fn': 1, 'tn': 1})
        metrics = metrics_from_counts(counts)
        self.assertAlmostEqual(metrics['vehicle_iou'], 1 / 3)
        self.assertAlmostEqual(metrics['precision'], 0.5)
        self.assertAlmostEqual(metrics['recall'], 0.5)
        self.assertAlmostEqual(metrics['f1'], 0.5)

    def test_checkpoint_velocity_mode_defaults_to_historical_full(self):
        legacy = {'run_config': {}}
        zero = {'run_config': {'radar_velocity_mode': 'zero'}}
        self.assertEqual(resolve_velocity_mode('auto', legacy), 'full')
        self.assertEqual(resolve_velocity_mode('auto', zero), 'zero')
        with self.assertRaisesRegex(ValueError, 'does not match checkpoint'):
            resolve_velocity_mode('full', zero)


if __name__ == '__main__':
    unittest.main()
