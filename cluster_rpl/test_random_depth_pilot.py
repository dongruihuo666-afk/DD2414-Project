"""CPU contracts for the formal CUDA admission gate; never allocate a GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from random_depth_pilot import inspect_checkpoint
import random_depth_pilot as pilot
from random_depth_study import ROOT, TREATMENTS, require_pilot, load_config, treatment_plan


def checkpoint():
    return dict(checkpoint_kind='ddp_full_model', world_size=5,
        progress=dict(update=3, samples_seen=15, epoch=0),
        run_config=dict(train_samples=28130, val_samples=6019, epochs=50,
            training_profile='random-metric-bev', treatment='dino-only',
            random_student_initialization=dict(kind='random'),
            official_camera_initialization=None, pretraining_continuation=None),
        distributed_state=[dict(rank=rank, sampler=dict(rank=rank, world_size=5,
            position=3, epoch=0), rng=dict(torch_cuda=torch.ones(1)), buffers=dict(bn=torch.zeros(1)))
            for rank in range(5)], model=dict(weight=torch.ones(1)),
        optimizer=dict(state={0:dict(step=torch.tensor(3.), exp_avg=torch.ones(1))}))


class RandomDepthPilotTests(unittest.TestCase):
    def test_real_batch_entry_is_used_twice_with_explicit_new_process_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT/'cluster_rpl/random_depth_study.example.json')
            config['run_root'] = directory
            run = Path(treatment_plan(config, 'dino-only')['pretrain']['run_dir'])
            run.mkdir(parents=True)
            metrics = [dict(update=i, loss=1., gradient_norm=1.) for i in (1, 2, 3)]
            (run/'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in metrics))
            before = dict(run_config_sha256='config', model=dict(weight=torch.ones(1)))
            after = dict(run_config_sha256='config', model=dict(weight=torch.zeros(1)), run_config={
                'random_student_initialization': {'student_state_sha256': 'identical'}})
            with patch.object(pilot.subprocess, 'run', return_value=SimpleNamespace(returncode=75)) as child, \
                    patch.object(pilot, 'inspect_checkpoint', side_effect=[before, after]), \
                    patch.object(pilot, 'read', return_value=dict(peak_allocated_gib=7., latest={})), \
                    patch.object(pilot, 'digest', return_value='checkpoint'):
                result = pilot.run_treatment(config, Path(directory)/'config.json', 'dino-only')
            self.assertEqual(result['samples_seen'], 15)
            commands = [call.args[0] for call in child.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][1], str(ROOT/'cluster_rpl/random_depth_job.sh'))
            self.assertEqual(commands[0][-2:], ['--stop-after-updates', '2'])
            self.assertEqual(commands[1][-3:], ['--resume', '--stop-after-updates', '3'])

    def test_unexpected_child_exit_cannot_publish_a_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT/'cluster_rpl/random_depth_study.example.json')
            config['run_root'] = directory
            with patch.object(pilot.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(pilot, 'write') as write, self.assertRaises(RuntimeError):
                pilot.run_treatment(config, Path(directory)/'config.json', 'dino-only')
            write.assert_not_called()

    def test_combined_report_is_absent_if_second_treatment_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(run_root=directory, gpus_per_experiment=5)
            with patch('sys.argv', ['pilot', '--config', str(Path(directory)/'config.json')]), \
                    patch.object(pilot, 'load_config', return_value=config), \
                    patch.object(pilot, 'require_allocation'), \
                    patch.object(pilot, 'git_commit', return_value='commit'), \
                    patch.object(pilot, 'read', return_value=dict(status='passed', settings=config, code_commit='commit')), \
                    patch.object(pilot, 'run_treatment', side_effect=[{'status': 'passed'}, RuntimeError('failure')]), \
                    self.assertRaises(RuntimeError):
                pilot.main()
            self.assertFalse((Path(directory)/'random_depth_pilot_passed.json').exists())

    def test_actual_checkpoint_loading_accepts_full_random_five_rank_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            torch.save(checkpoint(), path)
            self.assertEqual(inspect_checkpoint(path, 3, 5, 'dino-only')['progress']['samples_seen'], 15)

    def test_wrong_budget_provenance_cursor_or_nan_cannot_pass(self):
        for name in ('epoch-budget', 'parent', 'missing-rank', 'cursor', 'model-nan', 'optimizer-nan'):
            value = checkpoint()
            if name == 'epoch-budget':
                value['run_config']['epochs'] = 5
            elif name == 'parent':
                value['run_config']['pretraining_continuation'] = {'parent': True}
            elif name == 'missing-rank':
                value['distributed_state'].pop()
            elif name == 'cursor':
                value['distributed_state'][4]['sampler']['position'] = 2
            elif name == 'model-nan':
                value['model']['weight'][0] = float('nan')
            else:
                value['optimizer']['state'][0]['exp_avg'][0] = float('nan')
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'checkpoint.pt'
                torch.save(value, path)
                with self.assertRaises(ValueError):
                    inspect_checkpoint(path, 3, 5, 'dino-only')

    def test_formal_gate_requires_both_pilots_same_commit_config_and_world(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(run_root=directory, gpus_per_experiment=5)
            report = dict(status='passed', settings=config, code_commit='commit', treatments={
                treatment:dict(status='passed', resumed_update=3, world_size=5) for treatment in TREATMENTS})
            path = Path(directory)/'random_depth_pilot_passed.json'
            path.write_text(json.dumps(report))
            require_pilot(config, 'commit')
            for changed in ('missing', 'world', 'commit', 'settings', 'partial'):
                value = copy.deepcopy(report)
                if changed == 'missing':
                    value['treatments'].pop(TREATMENTS[1])
                elif changed == 'world':
                    value['treatments'][TREATMENTS[1]]['world_size'] = 2
                elif changed == 'commit':
                    value['code_commit'] = 'old'
                elif changed == 'settings':
                    value['settings']['gpus_per_experiment'] = 2
                else:
                    value['treatments'][TREATMENTS[0]]['status'] = 'partial'
                path.write_text(json.dumps(value))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    require_pilot(config, 'commit')


if __name__ == '__main__':
    unittest.main()
