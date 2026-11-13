#!/usr/bin/env python3
"""Read-only file hash check against reviewed asset pins. Never downloads."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / 'configs/assets.json').read_text())['files']
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', required=True, choices=sorted(pins))
    p.add_argument('--path', required=True, type=Path)
    args = p.parse_args()
    digest = hashlib.sha256()
    with args.path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != pins[args.asset]['sha256']:
        raise ValueError(f'{args.asset} SHA-256 mismatch: {actual}')
    print(f'PASS {args.asset} {actual}')


if __name__ == '__main__':
    main()
