"""CPU tests for node admission, owned path boundaries and cleanup prerequisites."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cleanup_random_depth import backup_results, completed_experiments, verified_target
from random_depth_study import ROOT, TREATMENTS, load_config
from stage_random_depth import check_owned_directory
import random_depth_readout as readout
from random_depth_study import treatment_plan


class MigrationTests(unittest.TestCase):
    def test_readout_partial_exit_blocks_success_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(ROOT/'cluster_rpl/random_depth_study.example.json')
            config['run_root'] = directory
            task = treatment_plan(config, 'dino-only')['downstream'][0]
            path = Path(task['argv'][task['argv'].index('--run-dir')+1])/'summary.json'
            path.parent.mkdir(parents=True)
            for state, expected in [('complete', 0), ('partial', 75)]:
                path.write_text(json.dumps(dict(status=state)))
                with patch('sys.argv', ['readout', '--config', '/config', '--treatment', 'dino-only',
                                        '--readout', 'frozen_camera_bev_probe']), \
                        patch.object(readout, 'load_config', return_value=config), \
                        patch.object(readout.socket, 'gethostname', return_value='eowyn'), \
                        patch.object(readout, 'require_pilot'), \
                        patch.object(readout.subprocess, 'check_output', return_value='commit'), \
                        patch.object(readout.subprocess, 'Popen', return_value=SimpleNamespace(wait=lambda: 0)), \
                        patch.dict(os.environ, SLURM_JOB_ID='test', SLURM_CPUS_PER_TASK='4'):
                    self.assertEqual(readout.main(), expected)

    def test_only_supported_staged_nodes_are_admitted(self):
        config = json.loads((ROOT/'cluster_rpl/random_depth_study.example.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for node in ('arwen', 'eowyn', 'galadriel'):
                config['node'] = node
                path.write_text(json.dumps(config))
                self.assertEqual(load_config(path)['node'], node)
            config['node'] = 'moria'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_owned_path_guard_rejects_symlinks_and_wrong_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'data'
            root.mkdir()
            check_owned_directory(root, root)
            link = Path(directory)/'link'
            link.symlink_to(root, target_is_directory=True)
            for actual, expected in [(link, link), (root, link)]:
                with self.assertRaises(ValueError):
                    check_owned_directory(actual, expected)

    def test_source_retirement_needs_matching_complete_target_audit(self):
        config = dict(node='eowyn')
        audit = dict(status='passed', settings=config, code_commit='commit', node='eowyn',
            sensor_files=dict(referenced=2631083, missing=0, empty=0, unsafe=0))
        transfer = dict(status='copied', settings=config, source_node='arwen', target_node='eowyn')
        verified_target(config, audit, transfer, 'commit')
        for key, value in [('status', 'failed'), ('node', 'arwen'), ('code_commit', 'old')]:
            with self.assertRaises(ValueError):
                verified_target(config, dict(audit, **{key:value}), transfer, 'commit')
        bad = dict(audit, sensor_files=dict(referenced=2631083, missing=1, empty=0, unsafe=0))
        with self.assertRaises(ValueError):
            verified_target(config, bad, transfer, 'commit')

    def test_target_retirement_waits_for_all_eight_complete_full_validation_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                completed_experiments(root)
            paths = []
            for treatment in TREATMENTS:
                for name, epochs in [('pretrain', 50), ('frozen_camera_bev_probe', 10),
                        ('frozen_fused_bev_probe', 10), ('end_to_end_segmentation', 5)]:
                    path = root/treatment/name/'summary.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(dict(status='complete', config=dict(epochs=epochs,
                        training_profile='random-metric-bev', treatment=treatment), final_validation=dict(samples=6019))))
                    paths.append(path)
            completed_experiments(root)
            last = json.loads(paths[-1].read_text())
            last['status'] = 'partial'
            paths[-1].write_text(json.dumps(last))
            with self.assertRaises(ValueError):
                completed_experiments(root)

    def test_backup_hashes_actual_bytes_and_never_overwrites_or_follows_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root, backup = Path(directory)/'run', Path(directory)/'backup'
            (root/'checkpoints').mkdir(parents=True)
            (root/'checkpoints/latest.pt').write_bytes(b'checkpoint-evidence')
            backup_results(root, backup)
            report = json.loads((backup/'backup_manifest.json').read_text())
            self.assertEqual(report['status'], 'verified')
            self.assertIn('checkpoints/latest.pt', report['sha256'])
            self.assertEqual((backup/'checkpoints/latest.pt').read_bytes(), b'checkpoint-evidence')
            with self.assertRaises(FileExistsError):
                backup_results(root, backup)
            (root/'linked').symlink_to(root/'checkpoints/latest.pt')
            with self.assertRaises(ValueError):
                backup_results(root, Path(directory)/'other-backup')


if __name__ == '__main__':
    unittest.main()
