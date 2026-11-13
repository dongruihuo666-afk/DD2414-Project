#!/usr/bin/env python3
"""Publish compact official-split results without staging unrelated files."""
import argparse
import json
from pathlib import Path

from publish_baseline_followup import ROOT, git


def publish(study):
    raise RuntimeError('Generated-artifact publication is retired. Use tools/publish_results.py for an explicitly reviewed results summary; original behavior is recoverable from the snapshot tag.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-dir', type=Path, required=True)
    publish(parser.parse_args().study_dir.resolve())
