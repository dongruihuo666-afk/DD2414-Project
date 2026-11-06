"""Stdlib matrix/launch tests: never allocate resources or start training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from depth_study import load_config, require_allocation, run_pretrain, treatment_plan

HERE = Path(__file__).resolve().parent


class DepthStudyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(HERE / 'depth_study.example.json')

    def test_three_equal_budget_pretrainers_and_twelve_dependent_readouts(self):
        plans = [treatment_plan(self.config, mode) for mode in ['none', 'radar', 'radar-dav2']]
        self.assertEqual(len({p['pretrain']['run_dir'] for p in plans}), 3)
        for plan in plans:
            task = plan['pretrain']
            self.assertEqual((task['gpus'], task['epochs'], task['global_batch_size'],
                              task['updates_per_epoch']), (2, 5, 2, 14065))
            argv = task['argv']
            self.assertIn('--nproc_per_node=2', argv)
            self.assertIn('--distributed', argv)
            self.assertEqual(argv[argv.index('--pretrain-initial-sha256') + 1], self.config['parent_sha256'])
            self.assertEqual(len(plan['downstream']), 4)
            for readout in plan['downstream']:
                self.assertEqual(readout['gpus'], 1)
                self.assertNotIn('--distributed', readout['argv'])
                source = readout['argv'][readout['argv'].index('--checkpoint') + 1]
                self.assertEqual(source, str(Path(task['run_dir']) / 'checkpoints/latest.pt'))
        self.assertEqual(sum(len(plan['downstream']) for plan in plans), 12)

    def test_changed_parent_or_node_or_relative_paths_are_rejected(self):
        for key, value in [('parent_sha256', 'other'), ('node', 'moria'),
                           ('run_root', 'relative'), ('workers_per_rank', 8)]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(dict(self.config, **{key: value})))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_login_node_and_missing_allocation_are_rejected(self):
        valid = dict(SLURM_JOB_ID='123', SLURM_JOB_GPUS='0,1', SLURM_JOB_NUM_NODES='1',
                     SLURM_CPUS_PER_TASK='8')
        require_allocation(self.config, valid, 'arwen')
        for environment, host in [({}, 'arwen'), (valid, 'moria'),
                (dict(valid, SLURM_JOB_NUM_NODES='2'), 'arwen'),
                (dict(valid, SLURM_CPUS_PER_TASK='4'), 'arwen'),
                (dict(valid, SLURM_JOB_GPUS=''), 'arwen')]:
            with self.assertRaises(RuntimeError):
                require_allocation(self.config, environment, host)

    def test_early_stop_flag_is_preserved_and_partial_exit_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            task = dict(run_dir=directory, argv=['must-not-execute'])
            summary = Path(directory) / 'summary.json'
            summary.write_text(json.dumps(dict(status='partial')))
            def fake_popen(*a, **k):
                # Simulate an early USR1 arriving AFTER child creation but BEFORE
                # rank zero startup. Worker must not clear this request.
                import signal
                signal.getsignal(signal.SIGUSR1)(signal.SIGUSR1, None)
                class Child:
                    def wait(self):
                        return 0
                return Child()
            with patch('depth_study.require_allocation'), patch('depth_study.subprocess.Popen', fake_popen):
                self.assertEqual(run_pretrain(self.config, task), 75)
            self.assertTrue((Path(directory) / 'REQUEST_CHECKPOINT').exists())
            summary.write_text(json.dumps(dict(status='complete')))
            with patch('depth_study.require_allocation'), patch('depth_study.subprocess.Popen') as start:
                start.return_value.wait.return_value = 0
                self.assertEqual(run_pretrain(self.config, task, resume=True, stop_after_updates=3), 0)
                self.assertIn('--resume', start.call_args[0][0])
                self.assertFalse((Path(directory) / 'REQUEST_CHECKPOINT').exists())


if __name__ == '__main__':
    unittest.main()
