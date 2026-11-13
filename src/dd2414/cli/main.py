"""Branch-local explicit entrypoints. Help uses only the standard library."""
import argparse
import importlib
import os
import runpy
import sys
from dd2414.paths import repository_root
ENTRIES={'train': ('train_multisweep_teacher.py', 'args_train_multisweep_teacher')}
def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=sorted(ENTRIES))
    if not argv or argv[0] in ('--help','-h'):
        parser.print_help();return
    command=argv.pop(0)
    if command not in ENTRIES:parser.error('unknown command: '+command)
    filename,module=ENTRIES[command]
    importlib.import_module('dd2414.cli.'+module).parse_args(argv)
    root=repository_root(required=True)
    from dd2414.runtime.provenance import record_package_implementation
    record_package_implementation({},root)
    sys.path[:0]=[str(root/'scripts'),str(root)]
    sys.argv=[str(root/'scripts'/filename),*argv]
    os.chdir(root)
    runpy.run_path(str(root/'scripts'/filename),run_name='__main__')
