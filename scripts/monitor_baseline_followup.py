#!/usr/bin/env python3
"""Read the persistent study and current child status without requiring CUDA."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--study-dir', type=Path, required=True)
args = parser.parse_args()
path = args.study_dir/'pipeline.json'
if not path.is_file():
    raise SystemExit(f'Queue has not written status yet: {path}')
state = json.loads(path.read_text())
print(f"Queue: {state.get('status','running')} | Stage: {state.get('stage')}")
print('Locked paired configuration:', json.dumps(state.get('paired_configuration',{})))
print('Current child:', json.dumps(state.get('child_status',{}), indent=2))
for name, stage in state.get('stages',{}).items():
    print(f"{stage.get('status','unknown'):10s} {name}")
print('Results:', args.study_dir/'comparison/REPORT.md')
