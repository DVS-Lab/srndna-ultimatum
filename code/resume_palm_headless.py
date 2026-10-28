#!/usr/bin/env python3
"""Recover PALM with headless MATLAB and an isolated child environment.

Reuse the original statistical/input contract and completed jobs. Validate the
runtime against saved smoke maps, then finish one full pending job serially
before releasing the remaining jobs. No automatic change of engine or method.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np

import collect_palm_failure_logs as logs
import resume_palm_conservative as recovery

palm = recovery.palm
FLAGS = ['-nodisplay', '-singleCompThread']
STARTED = 'SRNDNA_MATLAB_STARTED'
STARTUP_MODES = ('clean', 'inherited')
REMOVED_ENV = ('DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY', 'LD_PRELOAD',
               'LD_LIBRARY_PATH', 'MATLABPATH', 'MATLAB_JAVA', 'PYTHONHOME',
               'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH')


def child_environment(attempt, original=None, mode='clean'):
    """Keep HOME/licensing settings; change only the MATLAB child environment."""
    env = dict(os.environ if original is None else original)
    if mode not in STARTUP_MODES:
        raise ValueError(f'unknown MATLAB startup mode: {mode}')
    for name in (REMOVED_ENV if mode == 'clean' else ('DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY')):
        env.pop(name, None)
    prefixes = [Path(value) for key, value in env.items()
                if (key == 'CONDA_PREFIX' or key.startswith('CONDA_PREFIX_')) and value]
    def conda_path(value):
        path = Path(value)
        return any(path == prefix or prefix in path.parents for prefix in prefixes) or any(
            part.lower() in ('anaconda3', 'miniconda3', 'miniforge3', 'condabin')
            for part in path.parts)
    if mode == 'clean':
        env['PATH'] = os.pathsep.join(value for value in env.get('PATH', os.defpath).split(os.pathsep)
                                      if value and not conda_path(value))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    # Existing preferences can participate in sign-in/licensing initialization.
    # Batch mode already disables preference changes; do not redirect them.
    if mode == 'clean':
        target = attempt / 'tmp'
        target.mkdir(exist_ok=True)
        env['TMPDIR'] = str(target)
    return env


def command(engine, palm_root, args, logfile=None, startup_only=False):
    if 'octave' in Path(engine).name.lower():
        raise ValueError('this recovery requires the original MATLAB engine; no engine substitution')
    expression = f"fprintf('{STARTED}\\n');disp(version);"
    if not startup_only:
        expression += ('restoredefaultpath;addpath(' + palm.quote(palm_root) + ');' +
                       'palm(' + ','.join(palm.quote(value) for value in args) + ');')
    log_args = ['-logfile',str(logfile)] if logfile is not None else []
    return [str(engine), *FLAGS, *log_args, '-batch', expression]


def launch(engine, palm_root, args, attempt, mode='clean', startup_only=False):
    cmd = command(engine, palm_root, args, attempt/'matlab_console.txt',startup_only)
    env = child_environment(attempt,mode=mode)
    palm.pa.write_json(attempt / 'command.json', cmd)
    # Record policy and system context, never the complete environment or tokens.
    palm.pa.write_json(attempt / 'runtime.json', dict(flags=FLAGS,
        startup_mode=mode,startup_only=startup_only,
        removed_environment_variables=list(REMOVED_ENV if mode == 'clean' else ('DISPLAY','WAYLAND_DISPLAY','XAUTHORITY')),
        conda_path_entries_removed=mode == 'clean',
        private_preferences=False,private_tmp=mode == 'clean',
        platform='-'.join((platform.system(),platform.release(),platform.machine())),
        cpu_count=os.cpu_count(),runtime_code_sha256=palm.pa.digest(Path(__file__))))
    for name in ('meminfo', 'self/limits', 'self/cgroup'):
        source = Path('/proc') / name
        if source.is_file():
            (attempt / name.replace('/', '-')).write_text(source.read_text())
    for name in ('memory.max', 'memory.current', 'memory.events'):
        source = Path('/sys/fs/cgroup') / name
        if source.is_file():
            (attempt / ('cgroup-' + name)).write_text(source.read_text())
    try:
        with (attempt / 'run.txt').open('w') as stream:
            result = subprocess.run(cmd, cwd=attempt, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT,timeout=120 if startup_only else None)
    except subprocess.TimeoutExpired:
        palm.pa.write_json(attempt/'exit_status.json',dict(returncode=None,timeout_seconds=120))
        raise RuntimeError(f'MATLAB startup timed out; inspect {attempt}') from None
    text = '\n'.join(p.read_text(errors='replace') for p in
                     (attempt/'run.txt',attempt/'matlab_console.txt') if p.is_file())
    reached = STARTED in text
    palm.pa.write_json(attempt/'exit_status.json',dict(returncode=result.returncode,statement_reached=reached))
    if result.returncode or not reached:
        print(f'MATLAB mode={mode}, exit={result.returncode}, statement reached={reached}',flush=True)
        print(text[-4000:] if text.strip() else 'Both MATLAB output logs are empty; failure precedes the startup marker.',flush=True)
        raise RuntimeError(f'headless MATLAB failed ({result.returncode}); inspect {attempt}/run.txt and matlab_console.txt')


def snapshot_report(attempt, target):
    """Copy small files into a fresh report folder; preserve scratch originals."""
    target.mkdir(parents=True, exist_ok=True)
    for source in attempt.iterdir():
        if source.is_file() and not source.is_symlink():
            if source.stat().st_size > 20_000_000:
                raise ValueError(f'oversized compact runtime artifact: {source}')
            shutil.copy2(source, target / source.name)


def compare_smoke(reference, candidate, count):
    """Check raw statistics AND corrected probabilities on the same 64 voxels."""
    rows = []
    mask = nib.load(reference / 'mask.nii.gz')
    keep = mask.get_fdata() > 0
    for method in ('tfce', 'cluster-z3.1'):
        palm.probability_maps(candidate, method, count, mask)
        unit = 'tfce' if method == 'tfce' else 'clustere'
        for contrast in range(1, count + 1):
            for tag in ('', '_fwep'):
                pattern = f'palm_{unit}_*{tag}_c{contrast}.nii*'
                # Empty tag must select the raw map rather than the p-value map.
                candidates = [p for p in reference.glob(pattern)
                              if (('_fwep_' in p.name) == bool(tag)) and '_uncp_' not in p.name]
                if len(candidates) != 1:
                    raise ValueError(f'missing/ambiguous reference smoke map: {pattern}')
                source = candidates[0]
                matches = list(candidate.glob(source.name))
                if len(matches) != 1:
                    raise ValueError(f'missing headless smoke map: {source.name}')
                a, b = nib.load(source), nib.load(matches[0])
                if not palm.pa.compatible(a, b) or not palm.pa.compatible(a, mask):
                    raise ValueError('runtime smoke geometry changed')
                av, bv = a.get_fdata()[keep], b.get_fdata()[keep]
                if not np.isfinite(av).all() or not np.isfinite(bv).all():
                    raise ValueError('nonfinite runtime smoke values')
                if not np.allclose(av, bv, rtol=1e-6, atol=1e-6):
                    raise ValueError(f'runtime smoke differs from original: {source.name}')
                rows.append(dict(map=source.name,max_abs_difference=float(np.max(np.abs(av-bv)))))
    return rows


class HeadlessRecovery:
    def __init__(self, work, report, palm_root):
        self.work, self.report, self.palm_root = work, report, palm_root
        self.profile = None
        self.mode = None

    def validate(self, pending, frozen):
        profile = dict(original_contract_sha256=palm.pa.digest(self.work / 'contract.json'),
            runtime_code_sha256=palm.pa.digest(Path(__file__)),
            scheduler_sha256=palm.pa.digest(Path(recovery.__file__)),flags=FLAGS,
            removed_env=list(REMOVED_ENV),startup_modes=list(STARTUP_MODES),
            preserve_preferences=True,smoke_rtol=1e-6,smoke_atol=1e-6,
            engine=frozen['engine'],engine_sha256=frozen['engine_sha256'])
        if palm.pa.digest(Path(frozen['engine'])) != frozen['engine_sha256']:
            raise ValueError('original MATLAB executable changed')
        identity = palm.pa.identity(profile)
        self.profile = self.work / 'runtime-headless' / identity
        self.profile.mkdir(parents=True, exist_ok=True)
        contract = self.profile / 'contract.json'
        if contract.exists() and json.loads(contract.read_text()) != profile:
            raise ValueError('runtime profile changed')
        palm.pa.write_json(contract, profile)
        report = self.report / 'runtime-headless' / identity
        report.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contract, report / 'contract.json')
        receipt = self.profile / 'smoke_complete.json'
        selection = self.profile / 'startup_selection.json'
        if selection.is_file():
            saved = json.loads(selection.read_text())
            palm.pa.verify_files(saved['outputs'])
            self.mode = saved['mode']
            if self.mode not in STARTUP_MODES:
                raise ValueError('invalid saved MATLAB startup mode')
        else:
            failures=[]
            # Select from trivial startup success, never from scientific results.
            for mode in STARTUP_MODES:
                probe=Path(tempfile.mkdtemp(prefix='startup-'+mode+'-',dir=self.profile))
                try:
                    launch(frozen['engine'],self.palm_root,[],probe,mode=mode,startup_only=True)
                    self.mode=mode
                    palm.pa.write_json(selection,dict(mode=mode,
                        outputs={str(p):palm.pa.digest(p) for p in probe.iterdir() if p.is_file()}))
                    print(f'PASS: MATLAB startup with Java and existing preferences; mode={mode}',flush=True)
                    break
                except RuntimeError as exc:
                    failures.append(str(exc))
                    print(f'STARTUP_PROBE_FAILED: {mode}: {exc}',flush=True)
                finally:
                    snapshot_report(probe,report/probe.name)
            if self.mode is None:
                raise RuntimeError('MATLAB cannot start in either headless environment; no PALM jobs launched. '+ '; '.join(failures))
        shutil.copy2(selection,report/selection.name)
        if receipt.is_file():
            palm.pa.verify_files(json.loads(receipt.read_text())['outputs'])
            print('SKIP_COMPLETE: headless MATLAB equivalence smoke', flush=True)
        else:
            attempt = Path(tempfile.mkdtemp(prefix='smoke-', dir=self.profile))
            reference = self.work / 'smoke'
            for name in ('copes.nii.gz', 'mask.nii.gz', 'design.mat', 'design.con', 'variance_groups.csv'):
                (attempt / name).symlink_to((reference / name).resolve())
            args = palm.arguments(attempt, 'tfce', attempt / 'palm', 20, 42) + ['-C', '3.1', '-Cstat', 'extent']
            try:
                launch(frozen['engine'], self.palm_root, args, attempt,mode=self.mode)
                count = len(next(m for m in self.models if m['name'] == frozen['models'][0])['c'])
                rows = compare_smoke(reference, attempt, count)
                palm.pa.write_rows(attempt / 'equivalence.tsv', rows)
                outputs = {str(p):palm.pa.digest(p) for p in attempt.iterdir() if p.is_file() and not p.is_symlink()}
                palm.pa.write_json(receipt, dict(outputs=outputs,attempt=str(attempt)))
                print('PASS: headless smoke matches original raw and corrected maps', flush=True)
            finally:
                snapshot_report(attempt, report / attempt.name)
        # Full 10,000-shuffle canary is itself a scientific job, then reused.
        model, method = pending[0]
        canary = self.profile / 'canary_complete.json'
        if canary.is_file():
            saved = json.loads(canary.read_text())
            palm.pa.verify_files(saved['outputs'])
            print('SKIP_COMPLETE: full headless canary', flush=True)
        else:
            print(f'CANARY: serial full job {model["name"]} {method}; all other jobs wait', flush=True)
            self.job(model, method, self.work/'designs'/model['name'], self.work,
                     self.report, frozen['engine'], self.palm_root, frozen['shuffles'])
            job_receipt = self.work/'jobs'/model['name']/method/'complete.json'
            palm.pa.write_json(canary,dict(model=model['name'],method=method,
                outputs={str(job_receipt):palm.pa.digest(job_receipt)}))
            print('PASS: full 10,000-shuffle headless canary complete; releasing pending jobs', flush=True)
        shutil.copy2(receipt, report / receipt.name)
        shutil.copy2(canary, report / canary.name)

    def job(self, model, method, inputs, work, report, engine, palm_root, n):
        folder = work/'jobs'/model['name']/method
        if (folder/'complete.json').is_file():
            return palm.job(model,method,inputs,work,report,engine,palm_root,n)
        if self.profile is None or self.mode not in STARTUP_MODES:
            raise ValueError('headless runtime not validated')
        folder.mkdir(parents=True,exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix='attempt-headless-',dir=folder))
        args = palm.arguments(inputs,method,attempt/'palm',n,20261018)
        print(f'RUN_HEADLESS: {model["name"]} {method}',flush=True)
        try:
            launch(engine,palm_root,args,attempt,mode=self.mode)
            palm.probability_maps(attempt,method,len(model['c']),nib.load(inputs/'mask.nii.gz'))
            palm.pa.write_json(folder/'complete.json',dict(attempt=str(attempt),
                runtime_contract_sha256=palm.pa.digest(self.profile/'contract.json'),
                outputs={str(p):palm.pa.digest(p) for p in attempt.iterdir() if p.is_file()}))
        except Exception:
            snapshot_report(attempt, report/'runtime-headless'/self.profile.name/'failed-attempts'/model['name']/method/attempt.name)
            raise
        print(f'COMPLETED_HEADLESS: {model["name"]} {method}',flush=True)
        return palm.job(model,method,inputs,work,report,engine,palm_root,n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,default=logs.WORK)
    parser.add_argument('--report-root',type=Path,default=logs.REPORT)
    parser.add_argument('--source',type=Path,default=palm.ROOT/'results/dmn_covariate_standardized/v1')
    parser.add_argument('--palm-root',type=Path,required=True)
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    work, report, source = args.work_root.resolve(), args.report_root.resolve(), args.source.resolve()
    runtime = HeadlessRecovery(work,report,args.palm_root.resolve())
    runtime.models, _, _ = palm.pa.standardized_dmn_catalog(palm.ROOT,source/'imaging')
    recovery.recover(work,report,source,args.palm_root.resolve(),args.jobs,args.execute,
                     pending_job=runtime.job,before_pending=runtime.validate)


if __name__ == '__main__':
    main()
