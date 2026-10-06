#!/usr/bin/env python3
"""CPU tests for atomic checkpointing, RNG restoration and sampler resume."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from training_checkpoint import (
    StatefulShuffleSampler,
    load_training_checkpoint,
    save_periodic_checkpoint,
    save_training_checkpoint,
)


def make_components(dataset, seed=125):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    sampler = StatefulShuffleSampler(dataset, seed=seed)
    return model, optimizer, scaler, sampler


class TrainingCheckpointTests(unittest.TestCase):
    def setUp(self):
        inputs = torch.arange(30, dtype=torch.float32).reshape(10, 3) / 10
        targets = inputs.sum(dim=1, keepdim=True)
        self.dataset = TensorDataset(inputs, targets)
        self.config = {'seed': 125, 'scale': 64, 'objective': 'joint'}

    def test_sampler_continues_at_exact_next_index(self):
        first = StatefulShuffleSampler(self.dataset, seed=125)
        iterator = iter(first)
        prefix = [next(iterator) for _ in range(4)]
        state = first.state_dict()
        suffix = list(iterator)
        resumed = StatefulShuffleSampler(self.dataset, seed=125)
        resumed.load_state_dict(state)
        self.assertEqual(list(resumed), suffix)
        self.assertEqual(len(prefix + suffix), len(self.dataset))
        self.assertEqual(len(set(prefix + suffix)), len(self.dataset))

    def test_atomic_round_trip_restores_all_rng_states(self):
        model, optimizer, scaler, sampler = make_components(self.dataset)
        iterator = iter(sampler)
        next(iterator)
        expected_python = random.Random(99)
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            save_training_checkpoint(
                path, model=model, optimizer=optimizer, scaler=scaler,
                sampler=sampler, progress={'epoch': 0, 'update': 1},
                run_config=self.config,
            )
            expected = (
                expected_python.random(),
                np.random.RandomState(99).random_sample(),
                float(torch.rand(1, generator=torch.Generator().manual_seed(99))),
            )
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            loaded = load_training_checkpoint(
                path, model=model, optimizer=optimizer, scaler=scaler,
                sampler=sampler, expected_run_config=self.config,
                map_location='cpu',
            )
            observed = (random.random(), np.random.random(), float(torch.rand(1)))
            self.assertEqual(loaded, {'epoch': 0, 'update': 1})
            self.assertEqual(observed, expected)
            self.assertEqual(list(path.parent.glob('.*.tmp-*')), [])

    def test_configuration_and_sampler_mismatch_are_rejected(self):
        model, optimizer, scaler, sampler = make_components(self.dataset)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            save_training_checkpoint(
                path, model=model, optimizer=optimizer, scaler=scaler,
                sampler=sampler, progress={'epoch': 0, 'update': 0},
                run_config=self.config,
            )
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_training_checkpoint(
                    path, model=model, optimizer=optimizer, scaler=scaler,
                    sampler=sampler,
                    expected_run_config={**self.config, 'scale': 256},
                    map_location='cpu',
                )
            wrong_sampler = StatefulShuffleSampler(TensorDataset(
                torch.zeros(9, 3), torch.zeros(9, 1)), seed=125,
            )
            with self.assertRaisesRegex(ValueError, 'dataset size'):
                load_training_checkpoint(
                    path, model=model, optimizer=optimizer, scaler=scaler,
                    sampler=wrong_sampler, expected_run_config=self.config,
                    map_location='cpu',
                )

    def test_periodic_checkpoint_interval_and_force(self):
        model, optimizer, scaler, sampler = make_components(self.dataset)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            skipped = save_periodic_checkpoint(
                path, every_updates=10, model=model, optimizer=optimizer,
                scaler=scaler, sampler=sampler,
                progress={'epoch': 0, 'update': 9}, run_config=self.config,
            )
            self.assertIsNone(skipped)
            self.assertFalse(path.exists())
            saved = save_periodic_checkpoint(
                path, every_updates=10, model=model, optimizer=optimizer,
                scaler=scaler, sampler=sampler,
                progress={'epoch': 0, 'update': 10}, run_config=self.config,
            )
            self.assertIsNotNone(saved)
            self.assertTrue(path.exists())
            forced = save_periodic_checkpoint(
                path, every_updates=10, force=True, model=model,
                optimizer=optimizer, scaler=scaler, sampler=sampler,
                progress={'epoch': 0, 'update': 11}, run_config=self.config,
            )
            self.assertEqual(forced, saved)


if __name__ == '__main__':
    unittest.main()
