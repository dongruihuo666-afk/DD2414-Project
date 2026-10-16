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
child = state.get('child_status', {})
if state.get('stage') == 'throughput_pilot':
    pilot_status = args.study_dir/'pilot/status.json'
    if pilot_status.is_file():
        child = json.loads(pilot_status.read_text())
print(f"Queue: {state.get('status','running')} | Stage: {state.get('current_stage',state.get('stage'))}")
if 'paired_configuration' in state:
    print('Locked paired configuration:', json.dumps(state['paired_configuration']))
elif 'plan' in state:
    print(f'Planned stages: {len(state["plan"])}; no idle or wall-time kill rule')
print('Current child:', json.dumps(child, indent=2))
stages = ([(task['name'],state.get('stages',{}).get(task['name'],{'status':'queued'}))
           for task in state['plan']] if 'plan' in state else state.get('stages',{}).items())
for name, stage in stages:
    print(f"{stage.get('status','unknown'):10s} {name}")
print('Results:', args.study_dir/('REPORT.md' if 'plan' in state else 'comparison/REPORT.md'))
