#!/usr/bin/env python3
"""Explicit Git publication, separate from training. Review the publisher scope first."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, choices=['official', 'multisweep', 'baseline'])
    parser.add_argument('--study-dir', required=True, type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicitly authorize the selected legacy Git publisher; default prints the command only.')
    args = parser.parse_args()
    scripts = dict(official='publish_official_full_data.py',
                   multisweep='publish_multisweep_followup.py', baseline='publish_baseline_followup.py')
    command = [sys.executable, str(ROOT / 'scripts' / scripts[args.study]),
               '--study-dir', str(args.study_dir.resolve())]
    import shlex
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
