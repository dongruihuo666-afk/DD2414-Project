"""Explicit branch-local command planner; no SSH, Slurm or Git mutation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();c=json.loads(a.config.read_text())
    command=[c['python'],*c['launcher'],'-m','dd2414',c['entry'],*c['arguments']]
    print(shlex.join(command),flush=True)
    if a.execute:
        root=Path(c['project_root']).resolve()
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
        if branch!=c['branch']:raise ValueError('Wrong experiment checkout')
        subprocess.run(command,cwd=root,env=dict(os.environ,DD2414_PROJECT_ROOT=str(root)),check=True)

if __name__=='__main__':main()
