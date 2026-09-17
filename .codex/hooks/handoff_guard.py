#!/usr/bin/env python3
"""Give Codex one chance to finish a handoff or push before a turn ends.

This hook never edits files, stages changes, commits, or pushes. A human/agent
must review the change scope before a GitHub write.
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
    if status.returncode != 0:
        print('{}')
        return

    ahead = subprocess.run(
        ['git', '-C', str(root), 'rev-list', '--count', '@{upstream}..HEAD'],
        capture_output=True, text=True, check=False, timeout=2,
    )
    unpushed_commits = (
        ahead.returncode == 0 and ahead.stdout.strip().isdigit()
        and int(ahead.stdout.strip()) > 0
    )
    if not status.stdout.strip() and not unpushed_commits:
        print('{}')
        return

    print(json.dumps({
        'decision': 'block',
        'reason': (
            'This feature branch has local changes or commits ahead of its '
            'upstream. Before finishing, review git status and the task scope. '
            'For your own project edits, update PROJECT_HANDOFF.md with the '
            'change, reproduction steps, tests, limitations, and next step. '
            'Commit and push scoped work to the feature branch, then verify '
            'the remote ref. The project owner has authorized routine pushes '
            'to the feature branch; follow any environment-level approval. '
            'Do not stage unrelated user/teammate changes, push to main, '
            'force-push, or merge as a routine step. If unfinished work is '
            'intentionally left local, explain that and finish.'
        ),
    }))


if __name__ == '__main__':
    main()
