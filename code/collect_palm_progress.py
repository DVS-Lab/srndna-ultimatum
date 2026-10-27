#!/usr/bin/env python3
"""Collect verified PALM completions and log excerpts without launching jobs.

Safe during a running batch: writes only a new dated collection directory.
Completion receipts and their recorded hashes determine what is complete.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import tempfile

from collect_palm_failure_logs import digest, TAIL_BYTES, WORK, REPORT


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_rows(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def batch_active(work):
    lock_path = work / 'batch.lock'
    if not lock_path.exists():
        return False
    with lock_path.open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def collect(work, report, metadata_only=False):
    contract_path = work / 'contract.json'
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    history = report / 'collections'
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(tempfile.mkdtemp(prefix=stamp + '-', dir=history))
    active = batch_active(work)
    write_json(output / 'collection_status.json', dict(collection_complete=False))
    (output / 'contract.json').write_bytes(contract_bytes)
    complete, pending = [], []
    # Capture reports as observations; a running batch may update them later.
    for name in ('status.json', 'summary.tsv', 'recovery_preflight.json'):
        source = report / name
        if source.is_file():
            shutil.copyfile(source, output / ('last_batch_' + name))
    for model in contract['models']:
        for method in contract['methods']:
            folder = work / 'jobs' / model / method
            receipt = folder / 'complete.json'
            if not receipt.is_file():
                pending.append(dict(model=model, method=method, status='no_completion_receipt_at_inspection'))
                continue
            receipt_bytes = receipt.read_bytes()
            saved = json.loads(receipt_bytes)
            attempt = Path(saved['attempt']).resolve()
            if not attempt.is_relative_to(folder.resolve()) or not saved['outputs']:
                raise ValueError(f'invalid completed attempt: {receipt}')
            destination = output / 'completed' / model / method
            destination.mkdir(parents=True)
            for source_name, expected in saved['outputs'].items():
                source = Path(source_name).resolve()
                if source.parent != attempt or source.stat().st_size > 20_000_000:
                    raise ValueError(f'unexpected completed artifact: {source}')
                if digest(source) != expected:
                    raise ValueError(f'completed output changed: {source}')
                if not metadata_only:
                    target = destination / source.name
                    shutil.copyfile(source, target)
                    if digest(target) != expected:
                        raise ValueError(f'collected output checksum differs: {source}')
            if receipt.read_bytes() != receipt_bytes:
                raise ValueError(f'completion receipt changed during collection: {receipt}')
            (destination / 'complete.json').write_bytes(receipt_bytes)
            for name in (() if metadata_only else ('design.mat', 'design.con', 'variance_groups.csv', 'participants.tsv', 'mask.nii.gz')):
                source = report / 'designs' / model / name
                if source.is_file():
                    target = output / 'designs' / model / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            complete.append(dict(model=model, method=method, attempt=str(attempt),
                                 verified_files=len(saved['outputs'])))
    # A log without a receipt may still be running. Never label it as failed.
    for row in pending:
        folder = work / 'jobs' / row['model'] / row['method']
        for source in sorted(folder.glob('attempt-*/run.txt')):
            target = output / 'unfinished_logs' / row['model'] / row['method'] / source.parent.name / 'run.txt'
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as stream:
                stream.seek(max(0, source.stat().st_size - TAIL_BYTES))
                target.write_bytes(stream.read(TAIL_BYTES))
    for pattern in ('recovery-*/run.txt', 'recovery-*/exit_status.txt', 'resume-*/run.txt',
                    'resume-*/exit_status.txt', 'headless-*/run.txt', 'headless-*/exit_status.txt'):
        for source in sorted(report.parent.glob(pattern)):
            target = output / 'batch_logs' / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as stream:
                stream.seek(max(0, source.stat().st_size - TAIL_BYTES))
                target.write_bytes(stream.read(TAIL_BYTES))
    if contract_path.read_bytes() != contract_bytes:
        raise ValueError('PALM contract changed during collection')
    write_rows(output / 'completed_jobs.tsv', complete, ['model', 'method', 'attempt', 'verified_files'])
    write_rows(output / 'unfinished_jobs.tsv', pending, ['model', 'method', 'status'])
    state = dict(collection_complete=True, all_jobs_complete=not pending,
                 metadata_only=metadata_only,
                 verified_completed_jobs=len(complete), unfinished_jobs=len(pending),
                 expected_jobs=len(complete) + len(pending), batch_lock_held_at_start=active,
                 finished_utc=datetime.now(timezone.utc).isoformat(),
                 collector_sha256=digest(Path(__file__)),
                 interpretation='Receipt-based collection; unfinished can include running jobs. '
                 'last_batch_summary.tsv may omit jobs completed since that summary was written. '
                 'Metadata-only collections retain verified receipts rather than duplicate maps. '
                 'No jobs launched, stopped, or overwritten.')
    write_json(output / 'collection_status.json', state)
    files = [dict(path=str(p.relative_to(output)), sha256=digest(p), bytes=p.stat().st_size)
             for p in sorted(output.rglob('*')) if p.is_file()]
    write_rows(output / 'inventory.tsv', files, ['path', 'sha256', 'bytes'])
    print(f'PALM: {len(complete)}/{len(complete)+len(pending)} verified completed jobs; {len(pending)} unfinished')
    print(f'Batch lock held at start: {active}')
    print(f'Collected results and logs: {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=WORK)
    parser.add_argument('--report-root', type=Path, default=REPORT)
    parser.add_argument('--metadata-only', action='store_true',
                        help='Verify completed outputs and collect receipts/logs without duplicating maps.')
    args = parser.parse_args()
    collect(args.work_root.resolve(), args.report_root.resolve(), args.metadata_only)
