"""Explicit reviewed summary publication; defaults to printing, never force-adds artifacts."""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ALLOWED=('results/benchmarks.json','PROJECT_HANDOFF.md')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path',action='append',choices=ALLOWED,required=True)
    p.add_argument('--message',default='Record reviewed experiment summary')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    commands=[['git','add','--',*a.path],['git','commit','--only','-m',a.message,'--',*a.path]]
    for command in commands:print(shlex.join(command))
    print('git push origin <current-experiment-branch>')
    if not a.execute:return
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch not in ('dd2414-mini-baseline','experiment/hybrid-teacher-10sweep-cluster','experiment/rpl-full-data-double-distill','experiment/rpl-depth-10sweep-velocity-study'):
        raise RuntimeError('Select an independent experiment branch; main/detached publication refused')
    for path in a.path:
        value=ROOT/path
        if not value.is_file() or value.is_symlink():raise ValueError('Expected a reviewed regular summary file: '+path)
    for command in commands:subprocess.run(command,cwd=ROOT,check=True)
    subprocess.run(['git','push','origin',branch],cwd=ROOT,check=True)
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]
    if local!=remote:raise RuntimeError('Remote publication ref mismatch')

if __name__=='__main__':main()
