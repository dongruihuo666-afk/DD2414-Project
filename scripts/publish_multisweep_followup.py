#!/usr/bin/env python3
"""Commit/push compact stage evidence only; never checkpoints, logs or PDF."""

import argparse
import json
from pathlib import Path

from publish_baseline_followup import ROOT,git
from fullsize_training import atomic_json_write


def publish(study):
    raise RuntimeError('Generated-artifact publication is retired. Use tools/publish_results.py for an explicitly reviewed results summary; original behavior is recoverable from the snapshot tag.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study-dir',type=Path,required=True)
    publish(p.parse_args().study_dir.resolve())
