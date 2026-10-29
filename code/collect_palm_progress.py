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
import re
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


def progress_record(folder, model, method, completed):
    row=dict(model=model,method=method,status='verified_complete' if completed else 'unfinished',
        overall_percent=100 if completed else '',contrast_current='',contrasts_total='',
        shuffle_current='',shuffles_per_contrast='',log_age_seconds='',attempt='')
    if completed:
        attempt=Path(json.loads((folder/'complete.json').read_text())['attempt'])
    else:
        attempts=[p for p in folder.glob('attempt-*') if p.is_dir()]
        if not attempts:
            row['status']='not_started';return row
        def activity(path):
            return max([f.stat().st_mtime for f in (path/'run.txt',path/'matlab_console.txt') if f.is_file()]+[path.stat().st_mtime])
        attempt=max(attempts,key=activity)
    row['attempt']=str(attempt)
    logs=[p for p in (attempt/'run.txt',attempt/'matlab_console.txt') if p.is_file()]
    if logs:
        log=max(logs,key=lambda p:p.stat().st_mtime)
        row['log_age_seconds']=max(0,round(datetime.now(timezone.utc).timestamp()-log.stat().st_mtime))
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-TAIL_BYTES))
            text=stream.read(TAIL_BYTES).decode(errors='replace')
        matches=list(re.finditer(r'([\d.]+)%\s*\[Design [^\]]*?Contrast (\d+)/(\d+), Shuffling (\d+)/(\d+),',text))
        if matches:
            values=matches[-1].groups()
            row.update(overall_percent=100 if completed else float(values[0]),
                contrast_current=int(values[1]),contrasts_total=int(values[2]),
                shuffle_current=int(values[3]),shuffles_per_contrast=int(values[4]))
    exit_path=attempt/'exit_status.json'
    if not completed and exit_path.is_file():
        state=json.loads(exit_path.read_text())
        failed=state.get('returncode')!=0 or state.get('statement_reached') is False
        row['status']='latest_attempt_failed' if failed else 'awaiting_completion_validation'
    return row


def interim_results(work, output, contract, completed, config_path):
    """Recompute summaries ONLY for hash-verified, fully completed jobs."""
    import run_dmn_heteroscedasticity as palm
    if digest(Path(palm.__file__))!=contract['code'] or digest(Path(palm.pa.__file__))!=contract['helper']:
        raise ValueError('PALM statistical/helper code differs from the saved contract')
    if digest(config_path)!=contract['source']:
        raise ValueError('source model configuration differs from the saved PALM contract')
    models=json.loads(config_path.read_text())['models']
    if [m['name'] for m in models]!=contract['models']:
        raise ValueError('source model names/order differ from PALM contract')
    by_name={m['name']:m for m in models};rows=[]
    for job in completed:
        model=by_name[job['model']];attempt=Path(job['attempt'])
        saved=json.loads((output/'completed'/job['model']/job['method']/'complete.json').read_text())
        hashes={str(Path(p).resolve()):h for p,h in saved['outputs'].items()}
        mask_path=(work/'designs'/model['name']/'mask.nii.gz').resolve()
        block=json.loads((work/'blocks'/model['block']/'complete.json').read_text())
        block_hashes={str(Path(p).resolve()):h for p,h in block['outputs'].items()}
        if str(mask_path) not in block_hashes or digest(mask_path)!=block_hashes[str(mask_path)]:
            raise ValueError(f'analysis mask is not the verified original: {mask_path}')
        mask=palm.nib.load(mask_path)
        for path,stats in palm.probability_maps(attempt,job['method'],len(model['c']),mask):
            if str(path.resolve()) not in hashes or digest(path)!=hashes[str(path.resolve())]:
                raise ValueError(f'probability map is not in the verified completion receipt: {path}')
            rows.append(dict(model=job['model'],method=job['method'],
                contrast_name=model['labels'][stats['contrast']-1],**stats))
    write_json(output/'interim_source.json',dict(config=str(config_path),config_sha256=digest(config_path),
        interpretation='Completed full-shuffle jobs only; spatial FWE per signed contrast, not across models/methods. '
        'No partial-permutation results or new fits.'))
    return rows


def collect(work, report, metadata_only=False, summarize=False, source_config=None):
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
    complete, pending, progress = [], [], []
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
                progress.append(progress_record(folder,model,method,False))
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
            progress.append(progress_record(folder,model,method,True))
    # A log without a receipt may still be running; only an explicit exit status
    # establishes a failed attempt. Collect logs, never partial statistical maps.
    for row in pending:
        folder = work / 'jobs' / row['model'] / row['method']
        for name in ('run.txt', 'matlab_console.txt', 'exit_status.json'):
            for source in sorted(folder.glob('attempt-*/' + name)):
                target = output / 'unfinished_logs' / row['model'] / row['method'] / source.parent.name / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open('rb') as stream:
                    stream.seek(max(0, source.stat().st_size - TAIL_BYTES))
                    target.write_bytes(stream.read(TAIL_BYTES))
    for pattern in ('recovery-*/run.txt', 'recovery-*/exit_status.txt', 'resume-*/run.txt',
                    'resume-*/exit_status.txt', 'headless-*/run.txt', 'headless-*/exit_status.txt',
                    'scale-*/run.txt', 'scale-*/exit_status.txt'):
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
    write_rows(output/'progress.tsv',progress,list(progress[0]) if progress else ['model','method','status'])
    interim=[]
    if summarize:
        if source_config is None:
            source_config=Path(__file__).resolve().parents[1]/'results/dmn_covariate_standardized/v1/randomise/config.json'
        interim=interim_results(work,output,contract,complete,source_config)
        write_rows(output/'interim_results.tsv',interim,
                   ['model','method','contrast_name','contrast','minimum_spatial_fwe_p','significant_voxels'])
    state = dict(collection_complete=True, all_jobs_complete=not pending,
                 metadata_only=metadata_only,
                 interim_summary_requested=summarize,interim_summary_rows=len(interim),
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
    if summarize:
        print(f'Completed-job interim results: {output}/interim_results.tsv ({len(interim)} rows)')
    print(f'Per-job progress: {output}/progress.tsv')
    recent=sorted([r for r in progress if r['status']!='verified_complete' and r['log_age_seconds']!=''],
                  key=lambda r:r['log_age_seconds'])[:8]
    for row in recent:
        percent=row['overall_percent'] if row['overall_percent']!='' else 'unknown'
        print(f'OBSERVED: {row["model"]} {row["method"]}: {row["status"]}, '
              f'{percent}% overall, log age {row["log_age_seconds"]}s')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=WORK)
    parser.add_argument('--report-root', type=Path, default=REPORT)
    parser.add_argument('--metadata-only', action='store_true',
                        help='Verify completed outputs and collect receipts/logs without duplicating maps.')
    parser.add_argument('--summarize',action='store_true',
                        help='Summarize verified, fully completed FWE maps in a new interim results table.')
    parser.add_argument('--source-config',type=Path,
                        help='Source randomise model config; must match the original PALM contract hash.')
    args = parser.parse_args()
    collect(args.work_root.resolve(),args.report_root.resolve(),args.metadata_only,args.summarize,args.source_config)
