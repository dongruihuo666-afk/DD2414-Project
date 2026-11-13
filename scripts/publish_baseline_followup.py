#!/usr/bin/env python3
"""Publish only generated study summaries, preserving unrelated staged work."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fullsize_training import atomic_json_write


def git(*args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0',
               GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env,
                                   stderr=subprocess.STDOUT, text=True, timeout=90).strip()


def publish(study):
    raise RuntimeError('Generated-artifact publication is retired. Use tools/publish_results.py for an explicitly reviewed results summary; original behavior is recoverable from the snapshot tag.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-dir', type=Path, required=True)
    publish(parser.parse_args().study_dir.resolve())
