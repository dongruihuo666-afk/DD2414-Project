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
    branch = 'dd2414-mini-baseline'
    if git('branch', '--show-current') != branch:
        raise RuntimeError('publication is restricted to dd2414-mini-baseline')
    generated = ['artifacts/meeting_followup_day_results.json',
                 'artifacts/meeting_followup_day_report.md']
    handoff = 'PROJECT_HANDOFF.md'
    # Respect manual work on these particular files. Unrelated staged files,
    # including the project PDF, never enter the explicit-path commit.
    if git('status', '--porcelain', '--', handoff, *generated):
        raise RuntimeError('summary/handoff has manual local changes; publication deferred')
    state = json.loads((study/'pipeline.json').read_text())
    comparison = study/'comparison'
    metrics = json.loads((comparison/'metrics.json').read_text())
    atomic_json_write(dict(pipeline=state, results=metrics), ROOT/generated[0])
    (ROOT/generated[1]).write_text((comparison/'REPORT.md').read_text())
    start, end = '<!-- baseline-followup-runtime:start -->', '<!-- baseline-followup-runtime:end -->'
    lines = [start, '', '### 2026-10-16 — Unattended baseline follow-up runtime', '',
        f'- Stage: `{state.get("stage")}`; queue state: `{state.get("status", "running")}`.',
        '- Generated files: `artifacts/meeting_followup_day_results.json` and',
        '  `artifacts/meeting_followup_day_report.md`, plus this runtime handoff block.',
        '- Reproduce: `python scripts/run_baseline_followup_day.py --study-dir',
        f'  {study.relative_to(ROOT)}` (requires the documented local CUDA environment).',
        '- Tests/evidence: completed stages require exit code zero and a complete',
        '  summary; exact metrics, frame counts, configurations, and failures are in the JSON.',
        '- Limitations: one seed; different official/legacy input resolutions;',
        '  CUDA backward is not guaranteed bitwise deterministic when unfreezing camera projection.',
        '- Next: let the queue continue, then review full-val masks and the matched-epoch comparison.',
        '', end]
    text = (ROOT/handoff).read_text()
    block = '\n'.join(lines)
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before+block+after
    else:
        text = text.rstrip()+'\n\n'+block+'\n'
    (ROOT/handoff).write_text(text)
    paths = [handoff, *generated]
    git('add', '-f', '--', *paths)
    if git('diff', '--cached', '--name-only', '--', *paths):
        git('commit', '--only', '-m', f'Record baseline follow-up: {state.get("stage")}', '--', *paths)
    git('push', 'origin', branch)
    remote = git('ls-remote', '--heads', 'origin', branch).split()[0]
    if remote != git('rev-parse', 'HEAD'):
        raise RuntimeError('remote commit verification failed')
    print(f'PUBLISHED {remote}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-dir', type=Path, required=True)
    publish(parser.parse_args().study_dir.resolve())
