#!/usr/bin/env python3
"""Resume locked PALM jobs at low concurrency; stop scheduling on first failure.

The original statistical runner and its input/software contract are unchanged.
Completed jobs are validated/reused. Failed attempts are never overwritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import collect_palm_failure_logs as logs
import run_dmn_heteroscedasticity as palm


def bounded_run(tasks, worker, jobs):
    """No queue of 45 MATLAB processes; drain running jobs after first failure."""
    if not 1 <= jobs <= 2:
        raise ValueError('conservative recovery supports one or two concurrent MATLAB jobs')
    remaining=list(tasks); summaries=[]; failures=[]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        active={}
        def fill():
            while remaining and len(active)<jobs and not failures:
                item=remaining.pop(0)
                active[pool.submit(worker,item)]=item
        fill()
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                model,method=active.pop(future)
                try:
                    summaries.extend(future.result())
                except Exception as exc:
                    failures.append(dict(model=model['name'],method=method,error=str(exc)))
                    print(f'STOP_SCHEDULING: {model["name"]} {method}: {exc}',flush=True)
            fill()
    return summaries,failures,[dict(model=m['name'],method=method) for m,method in remaining]


def validate_participants(models, report, sample):
    """Public tables contain only existing pseudonymous IDs and variance groups."""
    for model in models:
        path=report/'designs'/model['name']/'participants.tsv'
        rows=palm.pa.read_rows(path)
        if not rows or set(rows[0])!={'subject','variance_group'}:
            raise ValueError(f'unexpected participant-table fields: {path}')
        expected=[dict(subject=s,variance_group=str(int(g))) for s,g in
            zip(model['subjects'],palm.variance_groups(model,sample))]
        if rows!=expected:
            raise ValueError(f'participant ordering/variance groups differ: {path}')
    print(f'PASS: {len(models)} pseudonymous participant-order tables verified',flush=True)


def recover(work, report, source, palm_root, jobs, execute):
    logs.collect(work,report)  # Snapshot old status BEFORE preparation overwrites it.
    subprocess.run([sys.executable,str(palm.ROOT/'code/prepare_palm_resume.py'),
        '--palm-root',str(palm_root),'--work-root',str(work),'--report-root',str(report)],check=True)
    # Reuse every original software, input, sample, design, and contract check.
    subprocess.run([sys.executable,str(Path(palm.__file__).resolve()),
        '--source',str(source),'--work-root',str(work),'--report-root',str(report),
        '--palm-root',str(palm_root),'--jobs',str(jobs)],check=True)
    contract=(work/'contract.json').read_text()
    with (work/'batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (work/'contract.json').read_text()!=contract:
            raise ValueError('PALM contract changed after preflight')
        frozen=json.loads(contract)
        models,sample,_=palm.pa.standardized_dmn_catalog(palm.ROOT,source/'imaging')
        if [m['name'] for m in models]!=frozen['models']:
            raise ValueError('model set changed after preflight')
        smoke=work/'smoke/complete.json'
        if not smoke.is_file():
            raise ValueError('completed PALM engine smoke required; do not launch recovery')
        palm.pa.verify_files(json.loads(smoke.read_text())['outputs'])
        validate_participants(models,report,sample)
        tasks=[(m,method) for m in models for method in palm.METHODS]
        def worker(item):
            model,method=item
            return palm.job(model,method,work/'designs'/model['name'],work,report,
                frozen['engine'],palm_root,frozen['shuffles'])
        summaries=[];pending=[]
        # Validate ALL existing completed jobs before starting ANY new MATLAB job.
        for model,method in tasks:
            if (work/'jobs'/model['name']/method/'complete.json').is_file():
                summaries.extend(worker((model,method)))
            else:
                pending.append((model,method))
        palm.pa.write_json(report/'recovery_preflight.json',dict(
            complete_jobs=len(tasks)-len(pending),pending_jobs=len(pending),
            max_concurrent_matlab=jobs,original_contract_sha256=palm.pa.digest(work/'contract.json'),
            recovery_code_sha256=palm.pa.digest(Path(__file__)),
            collector_code_sha256=palm.pa.digest(Path(logs.__file__))))
        print(f'READY: {len(tasks)-len(pending)} verified complete; {len(pending)} pending; concurrency={jobs}',flush=True)
        if not execute:
            palm.pa.write_json(report/'status.json',dict(complete=False,stage='recovery-preflight',
                jobs=len(tasks),completed_jobs=len(tasks)-len(pending),pending_jobs=len(pending)))
            print('DRY RUN: no MATLAB jobs launched',flush=True)
            return
        palm.pa.write_json(report/'status.json',dict(complete=False,stage='conservative-recovery-running'))
        new,failures,not_attempted=bounded_run(pending,worker,jobs)
        summaries.extend(new)
        if summaries:
            palm.pa.write_rows(report/'summary.tsv',summaries)
        complete=not failures and not not_attempted and len(summaries)==288
        palm.pa.write_json(report/'status.json',dict(complete=complete,jobs=len(tasks),
            summary_rows=len(summaries),failures=failures,not_attempted=not_attempted,
            max_concurrent_matlab=jobs))
        logs.snapshot(work,report)
        palm.pa.write_rows(report/'inventory.tsv',[
            dict(path=str(f.relative_to(report)),sha256=palm.pa.digest(f),bytes=f.stat().st_size)
            for f in sorted(report.rglob('*')) if f.is_file() and f.name!='inventory.tsv'])
        if not complete:
            raise RuntimeError(f'PALM recovery stopped: {len(failures)} failures, {len(not_attempted)} not launched; all completed jobs preserved')
        print('PASS: all 45 PALM jobs complete; 288 signed contrast/method summaries',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,default=logs.WORK)
    parser.add_argument('--report-root',type=Path,default=logs.REPORT)
    parser.add_argument('--source',type=Path,default=palm.ROOT/'results/dmn_covariate_standardized/v1')
    parser.add_argument('--palm-root',type=Path,required=True)
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--execute',action='store_true')
    a=parser.parse_args()
    recover(a.work_root.resolve(),a.report_root.resolve(),a.source.resolve(),a.palm_root.resolve(),a.jobs,a.execute)
