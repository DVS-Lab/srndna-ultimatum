#!/usr/bin/env python3
"""Drain the old Python scheduler, then resume validated PALM jobs in parallel.

SIGINT is sent only to a verified same-user Python driver, never MATLAB or a
process group. Its existing ThreadPoolExecutor waits for active jobs to save
their receipts before exiting. Numerical runtime and statistical code stay fixed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import resume_palm_headless as headless

palm = headless.palm
recovery = headless.recovery
GIB = 1024**3
ROOT = palm.ROOT


def inspect_process(pid, proc=Path('/proc')):
    path=proc/str(pid)
    try:
        status={line.split(':',1)[0]:line.split(':',1)[1].strip()
                for line in (path/'status').read_text().splitlines() if ':' in line}
        stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,uid=int(status['Uid'].split()[0]),ppid=int(status['PPid']),
            start=stat[19],state=stat[0],caught=int(status.get('SigCgt','0'),16),
            ignored=int(status.get('SigIgn','0'),16),blocked=int(status.get('SigBlk','0'),16),
            peak_kb=int(status.get('VmHWM','0 kB').split()[0]),
            argv=[s.decode(errors='replace') for s in (path/'cmdline').read_bytes().split(b'\0') if s],
            cwd=(path/'cwd').resolve())
    except (FileNotFoundError,ProcessLookupError,PermissionError,KeyError,IndexError):
        return None


def is_driver(info,repo,work=None):
    allowed={repo/'code/resume_palm_headless.py',repo/'code/scale_palm_recovery.py'}
    if (not info or info['uid']!=os.getuid() or info['pid']==os.getpid() or info['cwd']!=repo
        or not info['argv'] or not Path(info['argv'][0]).name.startswith('python')):
        return False
    recorded=headless.logs.WORK
    for index,arg in enumerate(info['argv']):
        if arg=='--work-root':
            if index+1>=len(info['argv']):
                return False
            recorded=Path(info['argv'][index+1])
        elif arg.startswith('--work-root='):
            recorded=Path(arg.split('=',1)[1])
    if not recorded.is_absolute():
        recorded=info['cwd']/recorded
    if recorded.resolve()!=(work or headless.logs.WORK).resolve():
        return False
    return any(Path(arg).name in {p.name for p in allowed} and
               (Path(arg) if Path(arg).is_absolute() else info['cwd']/arg).resolve() in allowed
               for arg in info['argv'][1:])


def drivers(repo,work=None):
    found=[]
    for path in Path('/proc').glob('[0-9]*'):
        info=inspect_process(int(path.name))
        if is_driver(info,repo,work):
            found.append(info)
    return found


def descendant_peak(driver):
    processes=[inspect_process(int(p.name)) for p in Path('/proc').glob('[0-9]*')]
    descendants={driver['pid']}; changed=True
    while changed:
        old=len(descendants)
        descendants.update(p['pid'] for p in processes if p and p['ppid'] in descendants)
        changed=len(descendants)!=old
    return max([p['peak_kb']*1024 for p in processes if p and p['pid'] in descendants]+[0])


def drain_driver(info, evidence, poll_seconds=5):
    current=inspect_process(info['pid'])
    if not current or current['start']!=info['start']:
        return
    bit=1 << (signal.SIGINT-1)
    if current['uid']!=os.getuid() or not current['caught'] & bit or (current['ignored']|current['blocked']) & bit:
        raise RuntimeError('Python driver does not expose the expected SIGINT handler; no signal sent')
    if (current['argv'],current['cwd'])!=(info['argv'],info['cwd']):
        raise RuntimeError('driver identity changed; no signal sent')
    palm.pa.write_json(evidence,dict(pid=info['pid'],process_start=info['start'],
        action='SIGINT to Python driver only; executor drains active workers',
        requested_utc=datetime.now(timezone.utc).isoformat()))
    print(f'DRAIN: asking Python PID {info["pid"]} to stop scheduling; current MATLAB jobs finish normally',flush=True)
    try:
        os.kill(info['pid'],signal.SIGINT)
    except ProcessLookupError:
        return
    last=time.monotonic()
    while True:
        current=inspect_process(info['pid'])
        if not current or current['start']!=info['start'] or current['state']=='Z':
            break
        if time.monotonic()-last>=60:
            print('DRAIN: waiting for the existing MATLAB jobs and completion receipts',flush=True)
            last=time.monotonic()
        time.sleep(poll_seconds)
    print('PASS: old Python scheduler drained and exited; no MATLAB process was signalled',flush=True)


def memory_budget():
    values={line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:','MemAvailable:'))}
    total,available=values['MemTotal'],values['MemAvailable']
    # Honor every applicable cgroup-v2 ancestor, including shared slice limits.
    records=[]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            base=Path('/sys/fs/cgroup'); leaf=base/line[3:].lstrip('/')
            for folder in (leaf,*leaf.parents):
                if folder!=base and base not in folder.parents:
                    continue
                limit,current=folder/'memory.max',folder/'memory.current'
                if limit.is_file() and current.is_file() and limit.read_text().strip()!='max':
                    cap=int(limit.read_text()); used=int(current.read_text())
                    total=min(total,cap); available=min(available,max(0,cap-used))
                    records.append(dict(path=str(folder),limit=cap,current=used))
    return dict(total=total,available=available,cgroups=records)


def choose_jobs(requested, budget, observed_peak, cpus):
    if not 1<=requested<=40:
        raise ValueError('request 1..40 jobs')
    per_job=max(4*GIB,int(observed_peak*1.5))
    reserve=max(2*GIB,int(budget['total']*.15))
    slots=max(0,(budget['available']-reserve)//per_job)
    selected=min(requested,max(1,cpus-2),slots)
    if selected<1:
        raise RuntimeError('insufficient memory headroom for a new MATLAB job; existing outputs preserved')
    return dict(requested_jobs=requested,selected_jobs=int(selected),
        available_bytes=budget['available'],reserved_bytes=reserve,
        estimated_bytes_per_job=per_job,observed_peak_bytes=observed_peak,
        policy='max(4 GiB, 1.5 x observed descendant peak) per slot; reserve 15% RAM or 2 GiB')


def validated_runtime(work):
    for profile in sorted((work/'runtime-headless').glob('*')):
        path=profile/'contract.json'
        if not path.is_file():
            continue
        contract=json.loads(path.read_text())
        if (contract.get('runtime_code_sha256')!=palm.pa.digest(Path(headless.__file__)) or
            contract.get('scheduler_sha256')!=palm.pa.digest(Path(recovery.__file__)) or
            contract.get('original_contract_sha256')!=palm.pa.digest(work/'contract.json')):
            continue
        names=('startup_selection.json','smoke_complete.json','canary_complete.json')
        if not all((profile/name).is_file() for name in names):
            continue
        for name in names:
            palm.pa.verify_files(json.loads((profile/name).read_text())['outputs'])
        return profile
    raise RuntimeError('no validated full MATLAB canary found; leave the existing batch running')


def parallel_jobs(tasks,worker,jobs):
    remaining=list(tasks); rows=[]; failures=[]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        active={}
        def fill():
            while remaining and len(active)<jobs and not failures:
                item=remaining.pop(0); active[pool.submit(worker,item)]=item
        fill()
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                model,method=active.pop(future)
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    failures.append(dict(model=model['name'],method=method,error=str(exc)))
                    print(f'STOP_SCHEDULING: {model["name"]} {method}: {exc}',flush=True)
            fill()
    return rows,failures,[dict(model=m['name'],method=method) for m,method in remaining]


def resume(work,report,source,palm_root,jobs):
    # Original low-concurrency runner performs only its full preflight. Its
    # source stays unchanged, so the successful runtime gates remain reusable.
    recovery.recover(work,report,source,palm_root,min(jobs,2),False)
    contract=(work/'contract.json').read_bytes()
    with (work/'batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (work/'contract.json').read_bytes()!=contract:
            raise ValueError('PALM contract changed after preflight')
        frozen=json.loads(contract)
        models,sample,_=palm.pa.standardized_dmn_catalog(ROOT,source/'imaging')
        if [m['name'] for m in models]!=frozen['models']:
            raise ValueError('retained model set changed')
        recovery.validate_participants(models,report,sample)
        runtime=headless.HeadlessRecovery(work,report,palm_root);runtime.models=models
        def worker(item):
            model,method=item
            return runtime.job(model,method,work/'designs'/model['name'],work,report,
                               frozen['engine'],palm_root,frozen['shuffles'])
        rows=[];pending=[]
        for model in models:
            for method in palm.METHODS:
                if (work/'jobs'/model['name']/method/'complete.json').is_file():
                    rows.extend(worker((model,method)))
                else:
                    pending.append((model,method))
        try:
            if pending:
                runtime.validate(pending,frozen)
            print(f'PARALLEL: {len(pending)} pending jobs; at most {jobs} MATLAB processes',flush=True)
            palm.pa.write_json(report/'status.json',dict(complete=False,stage='scaled-recovery-running',max_concurrent_matlab=jobs))
            new,failures,not_attempted=parallel_jobs(pending,worker,jobs)
            rows.extend(new)
            palm.pa.write_rows(report/'summary.tsv',rows)
            complete=not failures and not not_attempted and len(rows)==288
            palm.pa.write_json(report/'status.json',dict(complete=complete,jobs=45,summary_rows=len(rows),
                failures=failures,not_attempted=not_attempted,max_concurrent_matlab=jobs))
            if not complete:
                raise RuntimeError('PALM parallel recovery stopped; push diagnostic logs before another attempt')
            print('PASS: all 45 PALM jobs complete; 288 signed contrast/method summaries',flush=True)
        finally:
            headless.logs.snapshot(work,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,default=headless.logs.WORK)
    parser.add_argument('--report-root',type=Path,default=headless.logs.REPORT)
    parser.add_argument('--source',type=Path,default=ROOT/'results/dmn_covariate_standardized/v1')
    parser.add_argument('--palm-root',type=Path,required=True)
    parser.add_argument('--jobs',type=int,choices=range(1,41),default=8)
    a=parser.parse_args();work=a.work_root.resolve();report=a.report_root.resolve()
    if not sys.platform.startswith('linux'):
        raise RuntimeError('run this handoff on Linux1')
    validated=validated_runtime(work)
    report.mkdir(parents=True,exist_ok=True)
    # Serialize handoff requests separately from the active batch lock.
    with (work/'handoff.lock').open('a') as handoff:
        fcntl.flock(handoff,fcntl.LOCK_EX|fcntl.LOCK_NB)
        found=drivers(ROOT,work)
        if len(found)>1:
            raise RuntimeError('multiple PALM Python schedulers found; no signal sent')
        evidence=Path(tempfile_directory(report))
        peak=descendant_peak(found[0]) if found else 0
        statuses={p:p.read_bytes() for p in (work/'jobs').glob('*/*/attempt-*/exit_status.json')}
        if found:
            drain_driver(found[0],evidence/'drain_request.json')
        else:
            state=json.loads((report/'status.json').read_text()) if (report/'status.json').is_file() else {}
            if state.get('failures') or state.get('stage')=='runtime-validation-failed':
                raise RuntimeError('previous batch failed; review its logs before increasing concurrency')
        for path in (work/'jobs').glob('*/*/attempt-*/exit_status.json'):
            if statuses.get(path)!=path.read_bytes() and json.loads(path.read_text()).get('returncode')!=0:
                raise RuntimeError(f'job failed while draining; preserve logs and review before resuming: {path}')
        budget=memory_budget()
        cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
        chosen=choose_jobs(a.jobs,budget,peak,cpus)
        palm.pa.write_json(evidence/'resources.json',dict(**chosen,memory=budget,
            validated_runtime=str(validated),code_sha256=palm.pa.digest(Path(__file__))))
        print(f'CAPACITY: requested {a.jobs}; selected {chosen["selected_jobs"]}; '
              f'available RAM {budget["available"]/GIB:.1f} GiB',flush=True)
        resume(work,report,a.source.resolve(),a.palm_root.resolve(),chosen['selected_jobs'])


def tempfile_directory(report):
    import tempfile
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return tempfile.mkdtemp(prefix='capacity-'+stamp+'-',dir=report)


if __name__=='__main__':
    main()
