#!/usr/bin/env python3
"""Check local Markdown file targets, including moved archives; no network calls."""
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def main():
    files = [*ROOT.glob('*.md'), *(ROOT / 'docs').rglob('*.md'),
             ROOT / 'tests/README.md', ROOT / 'experiments/README.md',
             ROOT / 'research/README.md', ROOT / 'cluster_rpl/README.md',
             ROOT / 'configs/environments/README.md']
    errors = []
    for path in files:
        content = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        for match in re.finditer(r'\]\(([^)]+)\)', content):
            target = match[1].strip('<>')
            url = urlsplit(target)
            if url.scheme or target.startswith(('#', '/')):
                continue
            if not (path.parent / unquote(url.path)).exists():
                errors.append(f'{path.relative_to(ROOT)}: {target}')
    print('\n'.join(errors) if errors else f'PASS: local Markdown links in {len(files)} documents (including archives)')
    sys.exit(bool(errors))


if __name__ == '__main__':
    main()
