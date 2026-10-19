#!/usr/bin/env python3
"""Snapshot PALM failure evidence without changing jobs, inputs, or receipts."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/ZPOOL/data/scratch/srndna-ultimatum-dmn-followup-v1/palm')
REPORT = ROOT / 'results/dmn_followup/v1/palm'
TAIL_BYTES = 128 * 1024


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(work, report):
    """Caller holds batch.lock. Original failed attempts stay in scratch."""
    history = report / 'failure-history'
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(tempfile.mkdtemp(prefix=stamp + '-', dir=history))
    for name in ('status.json', 'summary.tsv', 'contract.json', 'inventory.tsv'):
        if (report / name).is_file():
            shutil.copy2(report / name, output / name)
    records = []
    jobs = work / 'jobs'
    for log in sorted(jobs.glob('*/*/attempt-*/run.txt')):
        if not log.resolve().is_relative_to(jobs.resolve()):
            raise ValueError(f'log escapes PALM jobs directory: {log}')
        receipt = log.parent.parent / 'complete.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if Path(saved['attempt']).resolve() == log.parent.resolve():
                continue
        relative = log.relative_to(jobs)
        destination = output / 'logs' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = log.stat().st_size
        with log.open('rb') as stream:
            stream.seek(max(0, size - TAIL_BYTES))
            tail = stream.read()
        destination.write_bytes(tail)
        command = log.parent / 'command.json'
        if command.is_file():
            if command.stat().st_size > TAIL_BYTES:
                raise ValueError(f'unexpected oversized command metadata: {command}')
            shutil.copy2(command, destination.parent / 'command.json')
        records.append(dict(source=str(log),source_sha256=digest(log),
            source_bytes=size,excerpt=str(destination.relative_to(output)),
            excerpt_sha256=digest(destination),truncated=size > TAIL_BYTES))
    # Memory/limits are useful context, not proof of the earlier failure cause.
    for name in ('meminfo', 'self/limits'):
        source = Path('/proc') / name
        if source.is_file():
            (output / name.replace('/', '-')).write_text(source.read_text())
    metadata = dict(logs=records,failed_or_incomplete_attempts=len(records),
        policy='last 128 KiB per unfinished/failed log; full original files preserved in scratch')
    (output / 'collection.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'COLLECTED: {len(records)} failed/incomplete attempt logs -> {output}',flush=True)
    return output


def collect(work, report):
    if not work.is_dir():
        raise FileNotFoundError(f'PALM scratch directory missing: {work}')
    with (work / 'batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('PALM batch is active; do not collect or start another batch') from None
        return snapshot(work,report)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,default=WORK)
    parser.add_argument('--report-root',type=Path,default=REPORT)
    args=parser.parse_args()
    collect(args.work_root.resolve(),args.report_root.resolve())
