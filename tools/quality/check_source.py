#!/usr/bin/env python3
"""Fast stdlib-only syntax/import-boundary checks; never loads models or data."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def violations(root=ROOT):
    problems = []
    for folder in ['src', 'scripts', 'tools', 'tests', 'cluster_rpl']:
        for path in sorted((root / folder).rglob('*.py')):
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as error:
                problems.append(f'{path.relative_to(root)}: {error}')
                continue
            if folder != 'src':
                continue
            for node in ast.walk(tree):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                         [n.name for n in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    if (name.startswith(('scripts.', 'cluster_rpl.', 'train_')) or
                            any(part in name for part in ('_tiny', '_demo', '_smoke'))):
                        problems.append(f'{path.relative_to(root)}:{node.lineno}: experiment dependency {name}')
    return problems


if __name__ == '__main__':
    errors = violations()
    print('\n'.join(errors) if errors else 'PASS: Python syntax and formal-package import boundary')
    sys.exit(bool(errors))
