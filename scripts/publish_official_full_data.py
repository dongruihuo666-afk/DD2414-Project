#!/usr/bin/env python3
"""Publish compact official-split results without staging unrelated files."""
import argparse
import json
from pathlib import Path

from publish_baseline_followup import ROOT, git


def publish(study):
    branch = 'dd2414-mini-baseline'
    if git('branch', '--show-current') != branch:
        raise RuntimeError('Publication is restricted to the feature branch')
    paths = ['PROJECT_HANDOFF.md', 'artifacts/official_full_data_results.json']
    if git('status', '--porcelain', '--', *paths):
        raise RuntimeError('Manual handoff/result changes exist; publication deferred')
    state = json.loads((study/'pipeline.json').read_text())
    results = []
    for task in state['plan']:
        summary = Path(task['directory'])/'summary.json'
        value = json.loads(summary.read_text()) if summary.exists() else {}
        results.append(dict(name=task['name'], command=task['command'],
            status=value.get('status', 'pending'), config=value.get('config'),
            final_validation=value.get('final_validation'),
            progress={k: value.get('progress', {}).get(k) for k in
                      ['epoch', 'update', 'samples_seen', 'training_seconds']}))
    (ROOT/paths[1]).write_text(json.dumps(dict(protocol=state['protocol'],
        queue_status=state['status'], engineering_checks=state['engineering_checks'],
        results=results), indent=2)+'\n')
    start, end = '<!-- official-full-runtime:start -->', '<!-- official-full-runtime:end -->'
    block = '\n'.join([start, '', '### Official-split full-data runtime', '',
        '- Changed: compact `artifacts/official_full_data_results.json` and this runtime block.',
        f'- Queue state: `{state["status"]}`; last stage: `{state.get("current_stage")}`.',
        '- Reproduce/resume: `bash scripts/start_official_full_data.sh`.',
        '- Evidence: completed summaries require a successful worker exit and full validation;',
        '  the JSON records exact commands, initialization, epochs and actual metrics.',
        '- Limitations: one seed, finite epoch budgets, no unseen local test set,',
        '  pre/post compression differs, GT boxes supplied to motion readout.',
        '- Next: continue pending tasks or inspect the recorded failure before restarting.', '', end])
    text = (ROOT/paths[0]).read_text()
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before+block+after
    else:
        text = text.rstrip()+'\n\n'+block+'\n'
    (ROOT/paths[0]).write_text(text)
    git('add', '-f', '--', *paths)
    if git('diff', '--cached', '--name-only', '--', *paths):
        git('commit', '--only', '-m', f'Record official full-data stage: {state.get("current_stage")}', '--', *paths)
    git('push', 'origin', branch)
    if git('ls-remote', '--heads', 'origin', branch).split()[0] != git('rev-parse', 'HEAD'):
        raise RuntimeError('Remote ref verification failed')
    print('Published official full-data stage', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-dir', type=Path, required=True)
    publish(parser.parse_args().study_dir.resolve())
