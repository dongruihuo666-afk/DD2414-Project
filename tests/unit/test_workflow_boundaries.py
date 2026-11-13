import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowBoundaryTests(unittest.TestCase):
    def test_training_queues_do_not_execute_git_publishers(self):
        for name in ('run_official_full_data', 'run_multisweep_followup', 'run_baseline_followup_day'):
            tree = ast.parse((ROOT / 'scripts' / (name + '.py')).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('run', 'Popen', 'check_call', 'check_output'):
                    self.assertNotIn('publish_', ast.unparse(node))
                    self.assertNotIn("'git', 'push'", ast.unparse(node))

    def test_portable_plan_respects_paths_budget_and_no_implicit_resume(self):
        launcher = module_at('launch_training', ROOT / 'tools/launch_training.py')
        cfg = json.loads((ROOT / 'configs/deployment/portable.example.json').read_text())
        for name in ('random-metric-bev-dino-only', 'random-metric-bev-depth'):
            experiment = json.loads((ROOT / 'configs/experiments' / (name + '.json')).read_text())
            argv, _, _, _ = launcher.command(cfg, experiment)
            self.assertIn('--nproc_per_node=2', argv)
            self.assertEqual(argv[argv.index('--epochs') + 1], '50')
            self.assertNotIn('--resume', argv)
            self.assertNotIn('--official-camera-checkpoint', argv)
            self.assertEqual(argv[argv.index('--velocity-mode') + 1], 'full')
        cfg['local_ranks'] = 3
        with self.assertRaises(ValueError):
            launcher.command(cfg, experiment)

    def test_hook_is_advisory_until_explicitly_enabled(self):
        hook = module_at('handoff_guard', ROOT / '.codex/hooks/handoff_guard.py')
        responses = [SimpleNamespace(returncode=0, stdout=str(ROOT)),
                     SimpleNamespace(returncode=0, stdout=' M teammate.txt'),
                     SimpleNamespace(returncode=0, stdout='0')]
        for enforce in ('0', '1'):
            output = io.StringIO()
            with patch.object(hook.subprocess, 'run', side_effect=responses), patch.object(hook.json, 'load', return_value=dict(hook_event_name='Stop', cwd=str(ROOT))), patch.dict(os.environ, DD2414_ENFORCE_HANDOFF=enforce), contextlib.redirect_stdout(output):
                hook.main()
            value = json.loads(output.getvalue())
            if enforce == '0':
                self.assertIn('systemMessage', value)
                self.assertNotIn('decision', value)
            else:
                self.assertEqual(value['decision'], 'block')
