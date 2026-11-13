"""Run this experiment's training commands."""

import argparse
import importlib
import os
import runpy
import sys

from dd2414.paths import repository_root

ENTRIES = {
    'downstream': ('train_hybrid_teacher_downstream.py', 'args_train_hybrid_teacher_downstream'),
    'pretrain': ('train_hybrid_teacher_pretrain.py', 'args_train_hybrid_teacher_pretrain'),
}


def _run_script(root, filename, arguments):
    """Run a legacy entrypoint and restore the caller's process settings."""
    original_cwd = os.getcwd()
    original_argv = sys.argv
    original_path = sys.path
    path_contents = sys.path.copy()
    script = root / "scripts" / filename
    try:
        sys.path[:0] = [str(root / "scripts"), str(root)]
        sys.argv = [str(script), *arguments]
        os.chdir(root)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        sys.path = original_path
        sys.path[:] = path_contents


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=sorted(ENTRIES))
    if not arguments or arguments[0] in ("--help", "-h"):
        parser.print_help()
        return

    command = arguments.pop(0)
    if command not in ENTRIES:
        parser.error("unknown command: " + command)
    filename, argument_module = ENTRIES[command]
    # Parse help before importing the training dependencies.
    module = importlib.import_module("dd2414.cli." + argument_module)
    module.parse_args(arguments)
    root = repository_root(required=True)

    from dd2414.runtime.provenance import record_package_implementation

    record_package_implementation({}, root)
    _run_script(root, filename, arguments)
