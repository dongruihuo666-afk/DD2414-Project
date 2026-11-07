#!/usr/bin/env python3
"""Retire ALL hongen-owned arwen-local files after a verified durable backup.

Default is inspection only. Execute in a CPU-only, single-node arwen Slurm
allocation. Shared Home, eowyn data/jobs and other users are outside this scope.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import stat
import subprocess

ROOT = Path('/local_storage/users/hongen')
PROJECT = Path('/Midgard/home/hongen/projects/DD2414-Project')
BACKUPS = PROJECT/'fullsize_baseline/runs/arwen_retirement'


def inventory(root):
    """Do not follow links, cross mounts, or include another account's files."""
    root = Path(root)
    if root.is_symlink() or root.resolve() != root or not root.is_dir() or root.is_mount():
        raise ValueError('Expected a real, non-mounted owner directory: '+str(root))
    device = root.stat().st_dev
    records = {}

    def visit(path):
        info = path.lstat()
        if info.st_uid != os.getuid():
            raise ValueError('Another owner inside retirement scope: '+str(path))
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            records[relative] = dict(kind='symlink', target=os.readlink(path))
        elif info.st_dev != device or path.is_mount():
            raise ValueError('Nested mount or filesystem boundary: '+str(path))
        elif stat.S_ISDIR(info.st_mode):
            records[relative] = dict(kind='directory')
            for child in sorted(path.iterdir()):
                visit(child)
        elif stat.S_ISREG(info.st_mode):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024), b''):
                    digest.update(block)
            records[relative] = dict(kind='file', size=info.st_size, sha256=digest.hexdigest())
        else:
            raise ValueError('Special file needs inspection before retirement: '+str(path))

    visit(root)
    return records


def require_context():
    job = os.environ.get('SLURM_JOB_ID', '')
    if (pwd.getpwuid(os.getuid()).pw_name != 'hongen'
            or socket.gethostname().split('.')[0] != 'arwen'
            or not job.isdigit() or os.environ.get('SLURM_JOB_NUM_NODES') != '1'
            or os.environ.get('SLURM_JOB_GPUS')):
        raise ValueError('Use the hongen account in a CPU-only single-node arwen Slurm job')
    return job


def require_idle(job):
    active = subprocess.check_output(['squeue', '-u', 'hongen', '-w', 'arwen',
        '-h', '-o', '%i'], text=True).split()
    if set(active)-{job}:
        raise RuntimeError('Other own queued/running arwen jobs must be inspected first: '+str(active))


def retire(root, backup, check_idle, execute=False):
    root, backup = Path(root), Path(backup)
    check_idle()
    if not root.exists() and not root.is_symlink():
        return dict(status='already_absent', source=str(root), deletion_performed=False)
    # The 397-GiB dataset has already been retired. A surprise replacement
    # must not be copied into the 100-GiB shared Home quota without inspection.
    dataset = root/'datasets/nuscenes'
    if dataset.exists() and not dataset.is_symlink():
        raise RuntimeError('nuScenes unexpectedly remains; inspect migration before backing up this root')
    before = inventory(root)
    size = sum(row.get('size', 0) for row in before.values())
    report = dict(status='inspected', source=str(root), backup=str(backup),
        entries=len(before), logical_file_bytes=size, deletion_performed=False)
    if not execute:
        return report
    if backup.exists() or backup.is_symlink():
        raise FileExistsError('Never overwrite a retirement backup: '+str(backup))
    backup.mkdir(parents=True, mode=0o700)
    copied = backup/'files'
    # Symlink targets are recorded/copied as links, NEVER read or deleted.
    shutil.copytree(root, copied, symlinks=True)
    if inventory(copied) != before or inventory(root) != before:
        raise RuntimeError('Backup mismatch or source changed; source deletion is blocked')
    (backup/'manifest.json').write_text(json.dumps(dict(status='verified', source=str(root),
        sha256_and_entries=before), indent=2)+'\n')
    check_idle()
    if inventory(root) != before:
        raise RuntimeError('Source changed after backup; source deletion is blocked')
    free_before = shutil.disk_usage(root.parent).free
    shutil.rmtree(root)
    if root.exists() or root.is_symlink():
        raise RuntimeError('Owner directory still exists; retirement is incomplete')
    report.update(status='complete', deletion_performed=True, source_absent=True,
        observed_free_delta_bytes=shutil.disk_usage(root.parent).free-free_before,
        preservation='Verified shared backup; shared Home and eowyn untouched')
    (backup/'retirement.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Back up, verify and remove the exact owner root')
    args = parser.parse_args()
    job = require_context()
    backup = BACKUPS/('job-'+job)
    # This entry point has no configurable deletion root or arbitrary backup path.
    result = retire(ROOT, backup, lambda:require_idle(job), args.execute)
    result.update(node='arwen', account='hongen', slurm_job_id=job)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
