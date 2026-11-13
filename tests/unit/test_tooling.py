"""Regression checks for command dispatch and documentation tooling."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from dd2414.cli.main import _run_script

ROOT = Path(__file__).resolve().parents[2]


def load_link_checker():
    spec = importlib.util.spec_from_file_location(
        "documentation_links", ROOT / "tools/quality/check_links.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToolingTests(unittest.TestCase):
    def check_dispatch_restores_state(self, error=None):
        original_cwd = os.getcwd()
        original_argv = sys.argv
        original_path = sys.path
        path_contents = sys.path.copy()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()

            def legacy_driver(path, run_name):
                self.assertEqual(os.getcwd(), str(root))
                self.assertEqual(path, str(root / "scripts/driver.py"))
                self.assertEqual(run_name, "__main__")
                self.assertEqual(sys.argv[1:], ["--example"])
                self.assertEqual(sys.path[0], str(root / "scripts"))
                # Some legacy drivers replace these globals rather than mutating them.
                sys.path = ["driver path"]
                sys.argv = ["driver argv"]
                os.chdir(root / "scripts")
                if error is not None:
                    raise error

            with patch("dd2414.cli.main.runpy.run_path", side_effect=legacy_driver):
                if error is None:
                    _run_script(root, "driver.py", ["--example"])
                else:
                    with self.assertRaises(type(error)):
                        _run_script(root, "driver.py", ["--example"])

        self.assertEqual(os.getcwd(), original_cwd)
        self.assertIs(sys.argv, original_argv)
        self.assertIs(sys.path, original_path)
        self.assertEqual(sys.path, path_contents)

    def test_cli_restores_state_after_return(self):
        self.check_dispatch_restores_state()

    def test_cli_restores_state_after_failure_or_exit(self):
        for error in (RuntimeError("driver failed"), SystemExit(2)):
            with self.subTest(error=type(error).__name__):
                self.check_dispatch_restores_state(error)

    def test_link_checker_import_does_not_exit_or_print(self):
        output = io.StringIO()
        with patch.object(sys, "exit", side_effect=AssertionError("unexpected exit")):
            with contextlib.redirect_stdout(output):
                load_link_checker()
        self.assertEqual(output.getvalue(), "")

    def test_link_checker_reports_only_missing_local_targets(self):
        checker = load_link_checker()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "results").mkdir()
            (root / "PROJECT_HANDOFF.md").write_text("")
            (root / "results/README.md").write_text("")
            (root / "docs/present.md").write_text("")
            (root / "README.md").write_text(
                "[present](docs/present.md)\n"
                "[missing](docs/missing.md)\n"
                "[external](https://example.com/page)\n"
                "[section](#section)\n"
                "```\n[example](docs/example-only.md)\n```\n"
            )
            errors = checker.broken_links(root)
            self.assertEqual(errors, [str(root / "README.md") + ": docs/missing.md"])
