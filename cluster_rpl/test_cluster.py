"""CPU-only contracts; never connect, install, allocate GPUs, or submit jobs."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from control import (ROOT, completed, inventory, parent_checkpoint, read, settings,
                     submit_command, tasks, validate_finished_config, write,
                     execute, clean_checkout, run_child)
from check_assets import sensor_files, tree_digest
from source_provenance import check_camera_source_provenance


def finished_config(task):
    command = task['command']
    cfg = dict(image_resolution=[448,800], teacher_resolution=[448,798],
        depth_mode='none', seed=125, train_samples=28130, val_samples=6019,
        image_weight=.2, bev_weight=.8, motion_weight=.5, batch_size=1,
        camera_lr=1e-5, probe_lr=1e-3, learning_rate=1e-4 if task['mode']=='finetune' else 2e-4,
        experiment='baseline_multisweep_double_distillation',
        manifest_sha256=read(ROOT/'configs/full_data_official_protocol.json')['manifest_sha256'],
        official_camera_initialization=dict(sha256=read(ROOT/'configs/full_data_official_protocol.json')['camera_checkpoint_sha256']))
    for key in ['mode','velocity_mode','probe_location','precision','epochs','nsweeps']:
        value = command[command.index('--'+key.replace('_','-'))+1]
        cfg[key] = int(value) if key in ['epochs','nsweeps'] else value
    return cfg


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = settings(HERE/'config.example.json')
        self.cfg['node'] = 'arwen'

    def test_reverse_priority_respects_pretraining_dependencies(self):
        rows = tasks(self.cfg)
        self.assertEqual(len(rows),11)
        self.assertEqual(len({t['name'] for t in rows}),11)
        self.assertIn('velocity_zero_pretrain',rows[0]['name'])
        self.assertEqual(rows[1]['mode'],'finetune')
        self.assertEqual(rows[2]['mode'],'object-motion')
        self.assertEqual(rows[3]['command'][rows[3]['command'].index('--probe-location')+1],'fused_bev')
        seen = set()
        for task in rows:
            self.assertTrue(task['requires'] is None or task['requires'] in seen)
            seen.add(task['name'])

    def test_environment_avoids_old_conda_graph_and_pins_cuda_wheels(self):
        import yaml
        spec = yaml.safe_load((HERE/'environment.yml').read_text())
        self.assertEqual(spec['channels'],['conda-forge','nodefaults'])
        packages = spec['dependencies']
        self.assertIn('python=3.10.14',packages)
        self.assertIn('libgl',packages)
        self.assertNotIn('pillow=9.4.0',packages)
        wheels = next(p['pip'] for p in packages if isinstance(p,dict))
        for requirement in ['torch==2.5.1+cu121','torchvision==0.20.1+cu121',
                            'Pillow==9.4.0','numpy==1.23.5','scipy==1.15.3']:
            self.assertIn(requirement,wheels)
        self.assertIn('opencv-python==4.6.0.66',wheels)
        self.assertFalse(any(p.startswith('opencv-python-headless') for p in wheels))

    def test_scientific_configuration_is_unchanged(self):
        for task in tasks(self.cfg):
            command = task['command']
            self.assertNotIn('--train-samples',command)
            self.assertNotIn('--val-samples',command)
            self.assertEqual(command[command.index('--nsweeps')+1],'10')
            start = command.index('--image-resolution')
            self.assertEqual(command[start+1:start+3],['448','800'])
            self.assertEqual(command[command.index('--depth-mode')+1],'none')
            self.assertEqual('--allow-source-relocation' in command, bool(task['requires']))

    def test_submission_is_one_pinned_single_gpu_job(self):
        command = submit_command(self.cfg,HERE/'config.example.json',tasks(self.cfg)[0])
        self.assertIn('--gres=gpu:1',command)
        self.assertIn('--nodelist=arwen',command)
        self.assertIn('--signal=B:USR1@300',command)
        self.assertNotIn('--requeue',command)
        self.assertNotIn('ssh',command)

    def test_placeholder_node_cannot_submit(self):
        cfg = settings(HERE/'config.example.json')
        with self.assertRaisesRegex(ValueError,'Select a real node'):
            submit_command(cfg,HERE/'config.example.json',tasks(cfg)[0])

    def test_pilot_is_bounded_and_test_only_is_explicit(self):
        command = submit_command(self.cfg,HERE/'config.example.json',tasks(self.cfg)[0],pilot=True,test_only=True)
        self.assertIn('--time=00:30:00',command)
        self.assertIn('--test-only',command)
        self.assertIn('--pilot',command)

    def test_settings_reject_injection_and_invalid_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'config.json'
            for key,value in [('node','arwen\n--gres=gpu:9'),('precision','auto'),
                              ('workers',9),('time','49:00:00'),('memory','all'),
                              ('data_root','relative/path')]:
                cfg = copy.deepcopy(self.cfg)
                cfg[key] = value
                write(path,cfg)
                with self.assertRaises(ValueError):
                    settings(path)

    def test_completion_must_match_protocol_and_full_validation(self):
        task = tasks(self.cfg)[0]
        cfg = finished_config(task)
        validate_finished_config(cfg,task)
        for key,value in [('precision','fp16'),('epochs',1),('train_samples',4096),
                          ('learning_rate',.1),('manifest_sha256','bad')]:
            changed = dict(cfg)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_finished_config(changed,task)

    def test_completed_index_skips_without_copying_a_running_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(self.cfg,run_root=temporary,completed_index=str(Path(temporary)/'completed.json'))
            task = tasks(cfg)[0]
            record = dict(config=finished_config(task),summary=dict(status='complete',final_validation=dict(samples=6019)))
            write(cfg['completed_index'],dict(completed={task['name']:record}))
            self.assertTrue(completed(cfg,task))
            self.assertEqual(inventory(cfg)[0]['status'],'complete')
            record['summary']['final_validation']['samples'] = 16
            write(cfg['completed_index'],dict(completed={task['name']:record}))
            with self.assertRaises(ValueError):
                completed(cfg,task)

    def test_downstream_requires_transferred_parent_not_just_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(self.cfg,run_root=temporary,completed_index=str(Path(temporary)/'completed.json'))
            parent,child = tasks(cfg)[:2]
            record = dict(config=finished_config(parent),summary=dict(status='complete',final_validation=dict(samples=6019)),checkpoint_sha256='bad')
            write(cfg['completed_index'],dict(completed={parent['name']:record}))
            with self.assertRaisesRegex(ValueError,'has not been transferred'):
                parent_checkpoint(cfg,child)
            checkpoint = Path(parent['directory'])/'checkpoints/latest.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'not a matching checkpoint')
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                parent_checkpoint(cfg,child)

    def test_snapshot_active_task_is_not_reported_as_not_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(self.cfg,run_root=temporary,completed_index=str(Path(temporary)/'completed.json'))
            task = tasks(cfg)[0]
            write(cfg['completed_index'],dict(completed={},in_progress=[task['name']]))
            self.assertEqual(inventory(cfg)[0]['status'],'active_on_workstation_snapshot')

    def test_allocated_telemetry_selects_uuid_not_first_physical_gpu(self):
        from fullsize_training import query_nvidia_smi
        with patch.dict('os.environ',{'DD2414_TELEMETRY_GPU_UUID':'GPU-allocated'}), \
             patch('fullsize_training.subprocess.run',return_value=type('Result',(),{'stdout':'75, 4000, 10000, 60, 200\n'})()) as run:
            result = query_nvidia_smi()
            self.assertIn('--id=GPU-allocated',run.call_args.args[0])
            self.assertEqual(result['memory_used_mib'],4000.)

    def test_sensor_file_audit_rejects_missing_empty_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'ok').write_bytes(b'x')
            (root/'empty').touch()
            result = sensor_files(root,[dict(filename=n) for n in ['ok','empty','missing','../bad','/bad']])
            self.assertEqual(result['referenced'],5)
            self.assertEqual(result['missing'],1)
            self.assertEqual(result['empty'],1)
            self.assertEqual(result['unsafe'],2)

    def test_teacher_tree_hash_detects_code_changes_not_pyc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'a.py').write_text('a=1')
            initial = tree_digest(root)
            (root/'a.pyc').write_bytes(b'cache')
            self.assertEqual(tree_digest(root),initial)
            (root/'a.py').write_text('a=2')
            self.assertNotEqual(tree_digest(root),initial)

    def test_default_cli_plan_never_executes_subprocess(self):
        import control
        with patch.object(sys,'argv',['control.py','--config',str(HERE/'config.example.json'),'plan']), \
             patch('control.subprocess.run') as run, patch('control.subprocess.check_output') as output, \
             patch('builtins.print'):
            control.main()
            run.assert_not_called()
            output.assert_not_called()

    def test_dirty_checkout_is_rejected_without_resetting_anything(self):
        with patch('control.subprocess.check_output',return_value=' M scripts/example.py\n'):
            with self.assertRaisesRegex(ValueError,'Tracked project changes'):
                clean_checkout()

    def test_execution_outside_slurm_is_rejected(self):
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'never on moria'):
                execute(self.cfg,tasks(self.cfg)[0],pilot=True)

    def test_pilot_demands_two_processes_finite_updates_and_writes_admission(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            cfg = dict(self.cfg,run_root=temporary)
            calls = []
            def child(command,directory):
                calls.append(command)
                update = int(command[command.index('--stop-after-updates')+1])
                write(directory/'summary.json',dict(status='partial',progress=dict(
                    update=update,latest=dict(loss=.4,gradient_norm=.1))))
                checkpoint = directory/'checkpoints/latest.pt'
                checkpoint.parent.mkdir(exist_ok=True)
                checkpoint.write_bytes(b'fake CPU-test checkpoint')
                return False
            with patch.dict('os.environ',{'SLURM_JOB_ID':'123','SLURMD_NODENAME':'arwen'}), \
                 patch('control.clean_checkout'), patch('control.validate_asset_report'), \
                 patch('control.git_commit',return_value='commit'), \
                 patch.object(torch.cuda,'device_count',return_value=1), \
                 patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(uuid='GPU-test')), \
                 patch.object(torch.cuda,'get_device_capability',return_value=(8,6)), \
                 patch.object(torch.cuda,'is_bf16_supported',return_value=True), \
                 patch('control.run_child',side_effect=child):
                execute(cfg,tasks(cfg)[0],pilot=True)
            self.assertEqual(len(calls),2)
            self.assertNotIn('--resume',calls[0])
            self.assertIn('--resume',calls[1])
            report = read(Path(temporary)/'engineering_pretrain_resume/passed.json')
            self.assertEqual(report['code_commit'],'commit')
            self.assertIn('not a scientific subset',report['scope'])

    def test_checkpoint_signal_is_forwarded_to_only_this_child(self):
        handlers = {}
        process = SimpleNamespace(poll=lambda:None)
        from unittest.mock import Mock
        process.send_signal = Mock()
        def wait():
            import signal
            handlers[signal.SIGUSR1](signal.SIGUSR1,None)
            return 0
        process.wait = wait
        with tempfile.TemporaryDirectory() as temporary, \
             patch('control.signal.signal',side_effect=lambda sig,handler:handlers.update({sig:handler})), \
             patch('control.subprocess.Popen',return_value=process):
            self.assertTrue(run_child(['python','trainer.py'],Path(temporary)))
            process.send_signal.assert_called_once()

    def test_submit_without_execute_never_calls_slurm(self):
        import control
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'config.json'
            write(path,self.cfg)
            task = tasks(self.cfg)[0]
            with patch.object(sys,'argv',['control.py','--config',str(path),'submit','--task',task['name']]), \
                 patch('control.subprocess.run') as run, patch('control.subprocess.check_output') as output, \
                 patch('builtins.print'):
                control.main()
                run.assert_not_called()
                output.assert_not_called()

    def test_batch_wrapper_execs_environment_python_for_signal_delivery(self):
        source = (HERE/'job.sh').read_text()
        self.assertIn('exec "${ENV_PREFIX}/bin/python"',source)
        self.assertNotIn('mamba run',source)
        subprocess.run(['bash','-n',str(HERE/'job.sh')],check=True)
        subprocess.run(['bash','-n',str(HERE/'setup_environment.sh')],check=True)


class SourceRelocationTests(unittest.TestCase):
    def setUp(self):
        self.path = Path('/new/official.pth')
        self.stored = dict(path='/old/official.pth',sha256='abc',tensors=568,scope='encoder only')
        self.cfg = dict(official_camera_checkpoint='/old/official.pth',official_camera_initialization=self.stored)
        self.current = dict(self.stored,path=str(self.path))

    def test_default_remains_path_strict(self):
        with self.assertRaises(ValueError):
            check_camera_source_provenance(self.cfg,self.path,self.current)

    def test_explicit_content_verified_relocation(self):
        check_camera_source_provenance(self.cfg,self.path,self.current,True)

    def test_hash_shape_count_and_scope_cannot_be_bypassed(self):
        for key,value in [('sha256','wrong'),('tensors',1),('scope','full model')]:
            changed = dict(self.current)
            changed[key] = value
            with self.assertRaises(ValueError):
                check_camera_source_provenance(self.cfg,self.path,changed,True)

    def test_missing_initialization_cannot_be_relocated(self):
        with self.assertRaises(ValueError):
            check_camera_source_provenance({},self.path,None,True)


if __name__ == '__main__':
    unittest.main()
