"""Tiny CPU checkpoint fixtures for real-model admission verification."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch

from depth_pilot import inspect_checkpoint, verify_transition


class DepthPilotTests(unittest.TestCase):
    def checkpoint(self, update):
        return dict(checkpoint_kind='ddp_full_model', world_size=2, run_config_sha256='same',
            run_config=dict(train_samples=28130, val_samples=6019, epochs=5),
            progress=dict(update=update, samples_seen=2 * update, epoch=0),
            model=dict(weight=torch.tensor([float(update)])), optimizer=dict(state={0: {}}),
            distributed_state=[dict(rank=rank, sampler=dict(rank=rank, position=update, epoch=0),
                rng=dict(torch_cuda=torch.tensor([1], dtype=torch.uint8)), buffers={}) for rank in range(2)])

    def test_full_manifest_rank_local_checkpoint_state_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(self.checkpoint(2), path)
            inspect_checkpoint(path, 2)
            bad = self.checkpoint(2)
            bad['distributed_state'][1]['sampler']['position'] = 1
            torch.save(bad, path)
            with self.assertRaises(ValueError):
                inspect_checkpoint(path, 2)

    def test_restart_must_not_replay_updates_or_leave_model_unchanged(self):
        before, after = self.checkpoint(2), self.checkpoint(3)
        metrics = [dict(update=n, loss=.5, gradient_norm=1.) for n in [1, 2, 3]]
        self.assertEqual(verify_transition(before, after, metrics), 1)
        with self.assertRaises(ValueError):
            verify_transition(before, after, metrics + [metrics[-1]])
        bad = copy.deepcopy(after)
        bad['model'] = before['model']
        with self.assertRaises(ValueError):
            verify_transition(before, bad, metrics)


if __name__ == '__main__':
    unittest.main()
