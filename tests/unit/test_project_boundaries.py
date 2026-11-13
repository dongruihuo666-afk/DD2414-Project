"""Checks for data views, source provenance and publication boundaries."""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from dd2414.cli.main import ENTRIES
from dd2414.data.batch import as_batch
from dd2414.runtime.provenance import record_package_implementation

ROOT = Path(__file__).resolve().parents[2]


class BoundaryTests(unittest.TestCase):
    def test_batch_view_preserves_storage_and_policy(self):
        values = [torch.randn(1) for _ in range(17)]
        view = as_batch(values)
        self.assertIs(view.radar, values[16])
        self.assertIs(as_batch(view), view)
        self.assertIs(view.images, values[0])
        with self.assertRaises(ValueError):
            view.validate()

    def test_full_contract_checks_camera_count_and_homogeneous_intrinsics(self):
        values = [None] * 18
        values[:4] = [
            torch.zeros(2, 1, 6, 3, 4, 8),
            torch.zeros(2, 1, 6, 3, 3),
            torch.zeros(2, 1, 6, 3),
            torch.zeros(2, 1, 6, 4, 4),
        ]
        values[12:17] = [
            torch.zeros(2, 1, channels, 5, 5) for channels in (1, 1, 1, 2)
        ] + [torch.zeros(2, 1, 19, 9)]
        self.assertIs(as_batch(values).validate().radar, values[16])
        values[3] = torch.zeros(2, 1, 6, 3, 3)
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            as_batch(values).validate()

    def test_all_help_commands_are_stdlib_only(self):
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        for command in ENTRIES:
            code = (
                "import sys; sys.modules['torch'] = None; "
                "sys.modules['nuscenes'] = None; "
                "from dd2414.cli.main import main; "
                f"main([{command!r}, '--help'])"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], text=True, capture_output=True,
                env=environment, cwd="/tmp",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--data-root", result.stdout)

    def test_actual_source_hash_binding_rejects_modified_source(self):
        config = {
            "original": "kept",
            "model_data_implementation_sha256": {"existing": "kept"},
        }
        record_package_implementation(config, ROOT)
        self.assertEqual(config["model_data_implementation_sha256"]["existing"], "kept")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            shutil.copytree(ROOT / "src", copied / "src")
            path = copied / "src/dd2414/losses/semantic.py"
            path.write_text(path.read_text() + "# changed\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                record_package_implementation({}, copied)

    def test_legacy_model_data_and_manifest_bytes_are_unchanged(self):
        pins = json.loads(
            (ROOT / "configs/reproducibility/source_boundary.json").read_text()
        )
        for name, expected in pins["unchanged_files_sha256"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_training_queues_do_not_call_publishers(self):
        subprocess_calls = ("run", "Popen", "check_call", "check_output")
        for path in (ROOT / "scripts").glob("run_*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in subprocess_calls
                ):
                    self.assertNotIn("publish_", ast.unparse(node))

    def test_explicit_publisher_plan_cannot_touch_git_or_select_artifacts(self):
        spec = importlib.util.spec_from_file_location(
            "explicit_publisher", ROOT / "tools/publish_results.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (
            patch.object(sys, "argv", ["publish_results", "--path", "results/benchmarks.json"]),
            patch.object(module.subprocess, "run", side_effect=AssertionError("unexpected Git write")),
            patch.object(module.subprocess, "check_output", side_effect=AssertionError("unexpected Git query")),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            module.main()
        with (
            patch.object(sys, "argv", ["publish_results", "--path", "artifacts/result.json"]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                module.main()
        self.assertEqual(error.exception.code, 2)

    def test_artifacts_are_ignored_and_absent_from_git_index(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "artifacts"], cwd=ROOT, text=True,
        )
        self.assertFalse(tracked.strip())
        result = subprocess.run(
            ["git", "check-ignore", "artifacts/new.png"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)

    def test_qualitative_tokens_match_recovered_selection(self):
        path = ROOT / "configs/visualization/qualitative_samples.json"
        if not path.exists():
            self.skipTest("This historical hybrid branch has no followup renderer")
        rows = json.loads(path.read_text())["samples"]
        self.assertEqual(
            [row["sample_token"] for row in rows],
            [
                "9ce5d3ae2cce4a0aaea1626e752f65d9",
                "3608204ac02a47afa1165e06f47d761d",
                "6d1c615ee4694f68996b1b40d2302aaf",
                "ad830c774eb7484a87fd2c776c570c2f",
            ],
        )
