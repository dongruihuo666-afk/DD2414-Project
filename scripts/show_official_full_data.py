#!/usr/bin/env python3
"""Small read-only terminal view of the official full-data queue."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study-dir', type=Path,
        default=ROOT/'fullsize_baseline/runs/official_rgb_full_10sweep_20261103')
    args = p.parse_args()
    try:
        state = json.loads((args.study_dir/'pipeline.json').read_text())
    except FileNotFoundError:
        print('Queue has not written its first status yet.')
        return
    stages = state.get('stages', {})
    print('Protocol: official 700-scene train / 150-scene validation; full data only')
    print('Queue:', state.get('status'))
    print('Completed tasks:', sum(t.get('status')=='complete' for t in stages.values()), '/ 11')
    print('Current:', state.get('current_stage', 'waiting for an idle GPU'))
    child = state.get('child_status', {})
    print('Last child:', child.get('state', '-'), 'epoch:', child.get('epoch_progress', '-'),
          '/', child.get('epochs', '-'), 'update:', child.get('update', '-'))
    print('Losses:', child.get('latest', {}))
    print('GPU admission:', state.get('gpu', {}))
    if state.get('message'):
        print('Note:', state['message'])
    print('Directory:', args.study_dir)


if __name__ == '__main__':
    main()
