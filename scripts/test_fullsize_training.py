#!/usr/bin/env python3
"""CPU tests for epoch scheduling, batch losses, and monitor status."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from fullsize_training import (
    AcknowledgedShuffleSampler,
    atomic_json_write,
    learning_rate_at_step,
    motion_loss_batched,
    progress_status,
    semantic_loss_batched,
    slice_batch,
)
from render_fullsize_progress import read_metrics, read_status, terminal_summary


class FullsizeTrainingTests(unittest.TestCase):
    def test_acknowledged_sampler_does_not_save_prefetched_positions(self):
        dataset = list(range(10))
        sampler = AcknowledgedShuffleSampler(dataset, seed=125)
        iterator = iter(sampler)
        requested = [next(iterator) for _ in range(6)]
        sampler.acknowledge(4)
        state = sampler.state_dict()
        resumed = AcknowledgedShuffleSampler(dataset, seed=125)
        resumed.load_state_dict(state)
        remaining = list(resumed)
        self.assertEqual(remaining[0], requested[4])
        self.assertEqual(len(remaining), 6)
        resumed.acknowledge(6)
        self.assertEqual(resumed.epoch, 1)
        self.assertEqual(resumed.position, 0)

    def test_slice_batch_preserves_batch_dimension(self):
        batch = (torch.arange(12).reshape(3, 4), torch.ones(3, 2, 5))
        sliced = slice_batch(batch, 1)
        self.assertEqual(sliced[0].shape, (1, 4))
        self.assertEqual(sliced[1].shape, (1, 2, 5))
        self.assertTrue(torch.equal(sliced[0], batch[0][1:2]))

    def test_semantic_batch_loss_is_mean_of_batch_one_losses(self):
        torch.manual_seed(4)
        prediction = torch.randn(3, 5, 4, 4)
        target = torch.randn(3, 5, 4, 4)
        confidence = torch.rand(3, 4, 4)
        combined = semantic_loss_batched(prediction, target, confidence)
        individual = torch.stack([
            semantic_loss_batched(
                prediction[index:index + 1], target[index:index + 1],
                confidence[index:index + 1],
            )
            for index in range(3)
        ]).mean()
        self.assertTrue(torch.allclose(combined, individual))

    def test_motion_batch_loss_is_mean_of_batch_one_losses(self):
        prediction = torch.zeros(2, 2, 3, 3)
        target = torch.zeros_like(prediction)
        coverage = torch.zeros(2, 3, 3)
        target[0, :, 1, 1] = 2.0
        coverage[0, 1, 1] = 1.0
        combined, cells = motion_loss_batched(prediction, target, coverage)
        first, _ = motion_loss_batched(
            prediction[:1], target[:1], coverage[:1],
        )
        second, _ = motion_loss_batched(
            prediction[1:], target[1:], coverage[1:],
        )
        self.assertEqual(cells, 1)
        self.assertTrue(torch.allclose(combined, (first + second) / 2))

    def test_epoch_schedule_warms_up_and_reaches_minimum(self):
        values = [
            learning_rate_at_step(4e-4, step, 100, 10, 0.1)
            for step in range(100)
        ]
        self.assertAlmostEqual(values[0], 4e-4 * 0.19)
        self.assertAlmostEqual(values[9], 4e-4)
        self.assertAlmostEqual(values[-1], 4e-5)
        self.assertLess(values[-1], values[50])

    def test_atomic_status_and_monitor_summary(self):
        status = progress_status(
            state='running', scale='full', epochs=8, dataset_size=28130,
            batch_size=4, optimizer_step=25, total_steps=56264,
            samples_seen=100, elapsed_seconds=10.0,
            latest={'loss': 1.0, 'semantic_loss': 0.5, 'motion_loss': 1.0},
            ema={'loss': 0.9, 'semantic_loss': 0.45, 'motion_loss': 0.9},
            learning_rate=4e-4,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            atomic_json_write(status, run_dir / 'status.json')
            (run_dir / 'metrics.jsonl').write_text(
                json.dumps({'optimizer_step': 25}) + '\n'
            )
            loaded = read_status(run_dir)
            self.assertEqual(loaded['optimizer_step'], 25)
            self.assertEqual(read_metrics(run_dir), [{'optimizer_step': 25}])
            summary = terminal_summary(loaded)
            self.assertIn('epoch=', summary)
            self.assertIn('ETA=', summary)


if __name__ == '__main__':
    unittest.main()
