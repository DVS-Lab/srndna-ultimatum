#!/usr/bin/env python3
"""Stable-arithmetic rerun of the unchanged bias model; diagnostics always exported.

Completed fits are hash-verified and skipped. Interrupted fits start a fresh
attempt without deleting prior output. Exit zero means evidence was exported,
not that convergence or individual-parameter recovery has been established.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import sys
import traceback

import run_gu_choice_extensions as ext
import run_gu_bias_pilot as pilot
import validate_gu_stable as validation

ROOT = ext.ROOT
SOURCES = ['code/run_gu_bias_stable.py', 'code/run_gu_bias_stable.sh',
    'code/validate_gu_stable.py', 'code/stan/gu_choice_stable.stan',
    'code/stan/gu_stable_functions.stan', 'code/stan/gu_stable_kernel_check.stan',
    'code/run_gu_choice_extensions.py', 'code/run_gu_bias_pilot.py',
    'code/export_gu_stan_diagnostics.py', 'code/audit_gu_divergences.py']


def job_config(baseline, sources):
    config = json.loads(json.dumps(baseline))
    config.update(warmup=4000, samples=6000, chains=4, threads_per_chain=5,
                  adapt_delta=.995, max_treedepth=12,
                  likelihood_implementation='log_scale_stable_v1',
                  reference_config_hash=ext.digest(baseline))
    config['sources'].update(sources)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path,
        default=Path('/ZPOOL/data/scratch/srndna-gu-bias-pilot-v1'))
    parser.add_argument('--work-root', type=Path,
        default=Path('/ZPOOL/data/scratch/srndna-gu-bias-stable-v1'))
    parser.add_argument('--output-dir', type=Path,
        default=ROOT/'results/norm_learning/stan-bias-stable-v1')
    parser.add_argument('--cmdstan', type=Path,
        default=Path('/ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0'))
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    work, baseline, output = a.work_root.resolve(), a.baseline_root.resolve(), a.output_dir.resolve()
    pilot.validate_paths(work, baseline, output)
    import cmdstanpy
    cmdstanpy.set_cmdstan_path(str(a.cmdstan.resolve()))
    version = next(line.split(':=', 1)[1].strip()
        for line in (a.cmdstan/'makefile').read_text().splitlines()
        if line.startswith('CMDSTAN_VERSION :='))
    if version != ext.CMDSTAN_VERSION or cmdstanpy.__version__ != '1.3.0':
        raise ValueError('Requires CmdStan 2.40.0 and CmdStanPy 1.3.0, as in the baseline')
    work.mkdir(parents=True, exist_ok=True)
    with (work/'.stable.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        units, _, _ = ext.load_trials(ROOT)
        reference = pilot.baseline_records(baseline, units)
        sources = {s: ext.sha(ROOT/s) for s in SOURCES}
        jobs = []
        for old, record in reference:
            config = job_config(old, sources)
            config['software'] = dict(cmdstan=version, cmdstanpy=cmdstanpy.__version__)
            jobs.append((work/'fits'/record['job'], ext.build_data(units, pilot.MODEL, old['stage']), config, None))
        for directory, _, config, _ in jobs:
            print(f'PLAN: {directory.name}; {config["warmup"]} warmup + {config["samples"]} retained per chain', flush=True)
        print('CPU budget: 2 fits x 4 chains x 5 threads = 40 maximum; no changes to data or priors.', flush=True)
        guard = dict(baseline=str(baseline), output=str(output), sources=sources,
                     configs=[ext.digest(j[2]) for j in jobs])
        if not a.execute:
            # Fail before launch if an existing directory cannot be resumed safely.
            for directory, _, config, _ in jobs:
                if (directory/'configuration.json').exists():
                    if json.loads((directory/'configuration.json').read_text()) != config:
                        raise ValueError(f'changed configuration: use a new work root ({directory})')
                    ext.completed(directory, ext.digest(config))
            print('PASS: baseline hashes, data/priors, software and paths. Add --execute for compiled validation and fitting.')
            return 0
        output.mkdir(parents=True, exist_ok=True)
        try:
            ext.config_guard(work/'workflow', guard)
            for directory, _, config, _ in jobs:
                ext.config_guard(directory, config)
            ext.atomic_json(output/'workflow_configuration.json', guard)
            stable = validation.compile_model('gu_choice_stable', work)
            original = validation.compile_model('gu_choice_extensions', work)
            harness = validation.compile_model('gu_stable_kernel_check', work)
            audit = validation.validate(stable, original, harness, units, work)
            ext.atomic_json(output/'compiled_equivalence.json', audit)
            options = argparse.Namespace(chains=4, threads_per_chain=5, warmup=4000,
                samples=6000, adapt_delta=.995, max_treedepth=12)
            records, errors = [], []
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(ext.fit_job,j,stable.exe_file,options,units):j[0].name for j in jobs}
                for future in as_completed(futures):
                    try:
                        records.append(future.result())
                    except Exception as exc:
                        errors.append(dict(job=futures[future], error=str(exc)))
                        print(f'FAILED: {futures[future]}: {exc}', flush=True)
            records.sort(key=lambda r:r['job'])
            ext.atomic_json(output/'fit_run_status.json', dict(completed=len(records), failed=errors))
            if records:
                ext.collect(records, work, output)
            if errors:
                raise RuntimeError('Some fits failed; partial summaries exported. Rerun same command to resume.')
            for script, folder in [('export_gu_stan_diagnostics.py','diagnostics-v1'),
                                   ('audit_gu_divergences.py','divergence-audit-v1')]:
                pilot.run_logged([sys.executable,'-u',str(ROOT/'code'/script),
                    '--work-root',str(work),'--output-dir',str(output/folder),
                    '--expected-jobs','2'], output/'export.txt')
            rows = pilot.comparison(reference, work)
            for row in rows:
                row['batch'] = 'stable' if row['batch']=='pilot' else 'previous_pilot'
            ext.write_table(output/'baseline_stable_comparison.tsv', rows)
            flagged = [r['stage'] for r in rows if r['batch']=='stable' and not r['inference_eligible']]
            ext.atomic_json(output/'workflow_status.json', dict(workflow_complete=True,
                diagnostic_review_required=flagged, imaging_covariates_released=False,
                parameter_recovery_validated=False))
            print(f'WORKFLOW_COMPLETE: {output}; diagnostic review required: {flagged}', flush=True)
            return 0
        except Exception as exc:
            ext.atomic_json(output/'workflow_status.json', dict(workflow_complete=False, error=str(exc)))
            traceback.print_exc()
            print('Stopped. Preserve/push partial logs; no successful-validation claim is made.', flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
