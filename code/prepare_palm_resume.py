#!/usr/bin/env python3
"""Repair pinned PALM Git metadata and preserve an incomplete smoke attempt.

The pinned palm_help.m assumes .git/HEAD names a readable loose branch ref.
Do not edit PALM source or relax the existing analysis/input contract.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from run_dmn_heteroscedasticity import PALM_COMMIT


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def ensure_readable_head(root, expected=PALM_COMMIT):
    root = Path(root).resolve()
    if git(root, 'rev-parse', 'HEAD') != expected:
        raise ValueError('PALM is not at the pinned commit; refusing to change it')
    if git(root, 'status', '--porcelain'):
        raise ValueError('PALM checkout is dirty; refusing to change it')
    gitdir = root / '.git'
    if not gitdir.is_dir():
        raise ValueError('PALM requires a standalone clone, not a linked Git worktree')
    head = (gitdir / 'HEAD').read_text().strip()
    if head.startswith('ref: refs/heads/'):
        ref = gitdir / head[5:]
        if ref.is_file() and ref.read_text().strip() == expected:
            return dict(commit=expected, branch=head[16:], changed=False)

    # A new branch creates a loose ref even when existing refs are packed.
    # Never move, delete, or overwrite another branch.
    base = 'srndna-palm-pinned-' + expected[:7]
    branch = base
    count = 0
    existing = set(git(root, 'for-each-ref', '--format=%(refname:short)',
                       'refs/heads/').splitlines())
    while branch in existing:
        count += 1
        branch = f'{base}-{count}'
    subprocess.run(['git', '-C', str(root), 'switch', '-c', branch, expected], check=True)
    ref = gitdir / 'refs' / 'heads' / branch
    if not ref.is_file() or ref.read_text().strip() != expected:
        raise ValueError('PALM loose branch ref was not created correctly')
    if git(root, 'rev-parse', 'HEAD') != expected or git(root, 'status', '--porcelain'):
        raise ValueError('PALM source changed unexpectedly')
    return dict(commit=expected, branch=branch, changed=True)


def preserve_smoke(work, report):
    smoke = work / 'smoke'
    if not smoke.exists() or (smoke / 'complete.json').exists():
        return None
    archive = work / 'smoke-attempts'
    archive.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    attempt = Path(tempfile.mkdtemp(prefix=stamp + '-', dir=archive))
    # Rename only the incomplete smoke directory; inputs/blocks/jobs stay put.
    smoke.rename(attempt / 'smoke')
    log = attempt / 'smoke' / 'run.txt'
    if log.is_file():
        destination = report / 'smoke-history'
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, destination / (attempt.name + '.txt'))
    return str(attempt / 'smoke')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--palm-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--report-root', type=Path, required=True)
    args = parser.parse_args()
    work = args.work_root.resolve()
    report = args.report_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = ensure_readable_head(args.palm_root)
        result['preserved_smoke'] = preserve_smoke(work, report)
        print(json.dumps(result, indent=2))
        print('PASS: pinned PALM source unchanged; retained analyses preserved')


if __name__ == '__main__':
    main()
