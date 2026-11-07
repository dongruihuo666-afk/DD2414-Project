#!/usr/bin/env python3
"""Commit/push compact stage evidence only; never checkpoints, logs or PDF."""

import argparse
import json
from pathlib import Path

from publish_baseline_followup import ROOT,git
from fullsize_training import atomic_json_write


def publish(study):
    branch = 'dd2414-mini-baseline'
    if git('branch','--show-current') != branch:
        raise RuntimeError('publication is restricted to baseline feature branch')
    paths = ['PROJECT_HANDOFF.md','artifacts/multisweep_teacher_results.json',
             'artifacts/multisweep_teacher_report.md']
    if git('status','--porcelain','--',*paths):
        raise RuntimeError('handoff/summary has manual changes; publication deferred')
    state = json.loads((study/'pipeline.json').read_text())
    results = json.loads((study/'results.json').read_text())
    # Commands/configuration are in the queue state; per-object/per-frame files
    # stay local. Report queued/partial/failed states, not imagined successes.
    atomic_json_write(dict(pipeline=state,results=results),ROOT/paths[1])
    (ROOT/paths[2]).write_text((study/'REPORT.md').read_text())
    start,end = '<!-- multisweep-runtime:start -->','<!-- multisweep-runtime:end -->'
    block = '\n'.join([start,'','### Multi-sweep double-distillation runtime','',
        f'- Queue: `{state.get("status")}`; last stage: `{state.get("current_stage")}`.',
        '- Changed runtime files: the two compact `artifacts/multisweep_teacher_*` reports and this block.',
        '- Reproduce/continue: `bash scripts/start_multisweep_followup.sh` on the baseline branch.',
        '- Evidence: a stage is complete only after a zero exit code and its complete summary;',
        '  actual full-split metrics and missing/failed stages are listed in the JSON/Markdown.',
        '- Limitations: one seed, short epoch budget, legacy/new objective differences,',
        '  low image resolution, radar-derived depth anchors and oracle GT vehicle locations.',
        '- Next: review full-val masks, paired 5/10 sweeps, frozen/unfrozen probes and object velocity errors;',
        '  extend epoch-matched training before claiming convergence or causal root causes.','',end])
    text = (ROOT/paths[0]).read_text()
    if start in text:
        before,rest = text.split(start,1)
        _,after = rest.split(end,1)
        text = before+block+after
    else:
        text = text.rstrip()+'\n\n'+block+'\n'
    (ROOT/paths[0]).write_text(text)
    git('add','-f','--',*paths)
    if git('diff','--cached','--name-only','--',*paths):
        git('commit','--only','-m',f'Record multi-sweep stage: {state.get("current_stage")}','--',*paths)
    git('push','origin',branch)
    remote = git('ls-remote','--heads','origin',branch).split()[0]
    if remote != git('rev-parse','HEAD'):
        raise RuntimeError('remote ref verification failed')
    print('PUBLISHED '+remote,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study-dir',type=Path,required=True)
    publish(p.parse_args().study_dir.resolve())
