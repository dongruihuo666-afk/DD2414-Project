"""Stdlib launch regression, including actual execution from a Slurm-like spool."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from random_depth_study import load_config, require_allocation, treatment_plan

ROOT = Path(__file__).resolve().parents[1]


class RandomDepthLaunchTests(unittest.TestCase):
    def test_two_independent_fifty_epoch_treatments_and_no_student_parent(self):
        config = load_config(ROOT/'cluster_rpl/random_depth_study.example.json')
        a, b = [treatment_plan(config, t) for t in ('dino-only', 'dino-metric-depth-bev')]
        self.assertNotEqual(a['pretrain']['run_dir'], b['pretrain']['run_dir'])
        for plan in (a, b):
            self.assertEqual(plan['pretrain']['gpus'], 5)
            self.assertNotIn('--pretrain-initial-checkpoint', plan['pretrain']['argv'])
            self.assertNotIn('--official-camera-checkpoint', plan['pretrain']['argv'])
            self.assertEqual(len(plan['downstream']), 3)
        self.assertNotIn('--metric-teacher-weights', a['pretrain']['argv'])
        self.assertIn('--metric-teacher-weights', b['pretrain']['argv'])

    def test_no_login_node_execution_or_insufficient_cpu_allocation(self):
        config = load_config(ROOT/'cluster_rpl/random_depth_study.example.json')
        environment = dict(SLURM_JOB_ID='test', SLURM_JOB_NUM_NODES='1',
                           SLURM_JOB_GPUS='0,1,2,3,4', SLURM_CPUS_PER_TASK='20')
        require_allocation(config, environment, 'arwen')
        for changes, host in [({}, 'moria'), ({'SLURM_JOB_ID': ''}, 'arwen'),
                               ({'SLURM_CPUS_PER_TASK': '8'}, 'arwen')]:
            with self.assertRaises(RuntimeError):
                require_allocation(config, dict(environment, **changes), host)

    def test_both_batch_wrappers_work_from_relocated_spool_copy(self):
        for wrapper, entry in [('depth_ddp_job.sh', 'depth_study.py'),
                               ('random_depth_job.sh', 'random_depth_study.py')]:
            with self.subTest(wrapper=wrapper), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                project, spool = base/'project with spaces', base/'slurmd/job123'
                (project/'cluster_rpl').mkdir(parents=True)
                spool.mkdir(parents=True)
                # A real Python process confirms the resolved executable path,
                # cwd, config and flags; no Slurm service or model is mocked in.
                (project/'cluster_rpl'/entry).write_text('import os,sys,json; print(json.dumps(dict(cwd=os.getcwd(),argv=sys.argv)))')
                env_prefix = base/'env'
                (env_prefix/'bin').mkdir(parents=True)
                (env_prefix/'bin/python').symlink_to(sys.executable)
                config = project/'config.json'
                config.write_text(json.dumps(dict(env_prefix=str(env_prefix))))
                script = spool/'slurm_script'
                shutil.copyfile(ROOT/'cluster_rpl'/wrapper, script)
                env = dict(os.environ, SLURM_JOB_ID='123', SLURM_SUBMIT_DIR=str(project),
                           BOOTSTRAP_PYTHON=sys.executable)
                env.pop('DD2414_PROJECT_ROOT', None)
                result = subprocess.run(['bash', str(script), str(config), 'dino-only', '--resume'],
                    cwd=base, env=env, text=True, capture_output=True, check=True)
                observed = json.loads(result.stdout)
                self.assertEqual(observed['cwd'], str(project))
                self.assertEqual(observed['argv'][0], str(project/'cluster_rpl'/entry))
                self.assertIn('--resume', observed['argv'])
                self.assertIn('--execute', observed['argv'])
                env['SLURM_SUBMIT_DIR'] = str(base)
                rejected = subprocess.run(['bash', str(script), str(config), 'dino-only'],
                    cwd=base, env=env, text=True, capture_output=True)
                self.assertEqual(rejected.returncode, 2)
                self.assertIn('Explicit project root required', rejected.stderr)


if __name__ == '__main__':
    unittest.main()
