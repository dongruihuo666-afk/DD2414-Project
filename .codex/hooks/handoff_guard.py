#!/usr/bin/env python3
"""Give Codex one chance to finish a handoff before a changed turn ends.

This hook never edits files, stages changes, commits, or pushes. A human/agent
must review the change scope and decide whether a GitHub write is authorized.
"""

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    event = json.load(sys.stdin)
    if event.get('hook_event_name') != 'Stop' or event.get('stop_hook_active'):
        print('{}')
        return

    cwd = Path(event.get('cwd') or '.').resolve()
    root_result = subprocess.run(
        ['git', '-C', str(cwd), 'rev-parse', '--show-toplevel'],
        capture_output=True, text=True, check=False, timeout=2,
    )
    if root_result.returncode != 0:
        print('{}')
        return
    root = Path(root_result.stdout.strip()).resolve()
    if root != Path(__file__).resolve().parents[2]:
        print('{}')
        return

    status = subprocess.run(
        ['git', '-C', str(root), 'status', '--porcelain',
         '--untracked-files=normal'],
        capture_output=True, text=True, check=False, timeout=2,
    )
    if status.returncode != 0 or not status.stdout.strip():
        print('{}')
        return

    print(json.dumps({
        'decision': 'block',
        'reason': (
            'This repository still has local changes. Before finishing, review '
            'git status and the current task scope. For your own project edits, '
            'update PROJECT_HANDOFF.md with the change, reproduction steps, '
            'tests, limitations, and next step; then commit and push only if '
            'authorized. Do not stage unrelated user or teammate changes. If '
            'the changes are intentionally left uncommitted, explain that and '
            'finish without forcing a commit or push.'
        ),
    }))


if __name__ == '__main__':
    main()
