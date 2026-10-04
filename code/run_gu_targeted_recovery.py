#!/usr/bin/env python3
"""Posterior-conditioned parameter recovery; separate scratch, no imaging release.

Sixteen synthetic datasets: both default-prior empirical fits, all four chains,
one seeded joint draw from each half-chain. No selection by parameter values or
diagnostic pass. This conditional experiment is not independent validation/SBC.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.special import expit

from audit_gu_trial_signals import read_job, source_jobs
from export_gu_stan_diagnostics import sha
from run_gu_hierarchical import (ROOT, STAN, CMDSTAN_VERSION, MODELS, atomic_json,
    build_data, collect, config_guard, cpu_plan, digest, fit_job, load_trials, logits)


def truth_draws(chains, seed):
    """Balance source chains/halves, retain the whole joint participant-partner draw."""
    rng = np.random.default_rng(seed)
    for chain, values in sorted(chains):
        for half, indices in enumerate(np.array_split(np.arange(len(values)), 2), 1):
            if not len(indices):
                raise ValueError('need at least two posterior draws per chain')
            draw = int(rng.choice(indices))
            yield chain, half, draw, values[draw].copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch-base', type=Path, default=Path('/ZPOOL/data/scratch'))
    p.add_argument('--collect-to', type=Path, required=True)
    p.add_argument('--cmdstan', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=40)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    a.chains, a.threads_per_chain = 4, 2
    a.warmup, a.samples, a.adapt_delta, a.max_treedepth = 3000, 6000, .99, 12
    if not 8 <= a.jobs <= 40:
        p.error('total CPU budget must be 8..40')
    workers = cpu_plan(a.jobs, a.chains, a.threads_per_chain)
    scratch = a.scratch_base.resolve()
    work = scratch/'srndna-gu-stan-targeted-v1'
    export = a.collect_to.resolve()
    if ROOT == work or ROOT in work.parents or work in ROOT.parents:
        p.error('fits must remain outside the code repository')
    if export == work or work in export.parents or export in work.parents:
        p.error('export and fitting directories must be separate')
    if ROOT in export.parents and ROOT/'results/norm_learning' not in export.parents:
        p.error('repository exports belong under results/norm_learning')
    units, _, event_hashes = load_trials(ROOT)
    source_paths = [item for item in source_jobs(scratch) if item[0] in ('baseline', 'repeat_prior1')]
    # A dry run checks configuration, data and diagnostic provenance without reading giant chains.
    if not a.execute:
        for label, path in source_paths:
            item = read_job(path, units, draws=False)
            if item['config']['chains'] != 4:
                raise ValueError('targeted plan requires four chains in each empirical fit')
            print(f'SOURCE: {label}; diagnostics={item["record"]["diagnostics"]["passed"]}; {path}')
        print(f'PLAN: 16 posterior-conditioned recovery fits; {workers} concurrent fits x 4 chains x 2 threads')
        print(f'New scratch: {work}; exports: {export}')
        print('DRY RUN: no compilation or sampling; add --execute. No imaging inputs released.')
        return 0
    import cmdstanpy
    cmdstanpy.set_cmdstan_path(str(a.cmdstan.resolve()))
    version = next(line.split(':=', 1)[1].strip() for line in
                   (a.cmdstan/'makefile').read_text().splitlines() if line.startswith('CMDSTAN_VERSION :='))
    if version != CMDSTAN_VERSION or cmdstanpy.__version__ != '1.3.0':
        raise ValueError(f'expected CmdStan {CMDSTAN_VERSION}/CmdStanPy 1.3.0; got {version}/{cmdstanpy.__version__}')
    source_hashes = dict(event_hashes)
    for name in ('run_gu_targeted_recovery.py', 'audit_gu_trial_signals.py', 'export_gu_stan_diagnostics.py',
                 'run_gu_hierarchical.py', 'run_gu_norm_learning.py', 'gu_norm_model.py',
                 'build_event_corrected_trials.py', 'stan/gu_hierarchical.stan'):
        source_hashes[f'code/{name}'] = sha(ROOT/'code'/name)
    work.mkdir(parents=True, exist_ok=True)
    with (work/'.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = []
        for source_index, (label, path) in enumerate(source_paths):
            item = read_job(path, units)
            if len(item['chains']) != 4:
                raise ValueError('targeted plan requires four chains per empirical fit')
            for chain, half, draw, truth in truth_draws(item['chains'], 20261004+source_index):
                rep = len(jobs)+1
                name = f'recovery-rw_free-full-prior1-targeted-rep{rep:03d}'
                seed = 20261004 + rep*1009
                data = build_data(units, 'rw_free')
                data['choice'] = np.random.default_rng(seed).binomial(
                    1, expit(logits(truth, data['offers'], 'rw_free'))).tolist()
                config = dict(model='rw_free', stage='full', phase='recovery', prior_scale=1.,
                    replicate=rep, seed=seed, sources=source_hashes, data_hash=digest(data),
                    warmup=a.warmup, samples=a.samples, chains=a.chains,
                    threads_per_chain=a.threads_per_chain, adapt_delta=a.adapt_delta,
                    max_treedepth=a.max_treedepth, software=dict(cmdstan=version, cmdstanpy=cmdstanpy.__version__),
                    subjects=list(dict.fromkeys(u['subject'] for u in units)), parameters=list(MODELS['rw_free'][1]),
                    age_mode='none', age_in_parameter_hierarchy=False, imaging_covariates_released=False,
                    recovery_kind='posterior-conditioned joint draw; NOT prior-generated recovery or SBC',
                    truth_hash=digest(truth.tolist()),
                    truth_source=dict(label=label, chain=chain, half=half, draw_zero_based=draw,
                        diagnostics_passed=item['record']['diagnostics']['passed'],
                        source_hashes=item['source_hashes']))
                directory = work/'fits'/name
                config_guard(directory, config)
                truth_path = directory/'simulation_truth.json'
                if truth_path.exists() and digest(json.loads(truth_path.read_text())) != config['truth_hash']:
                    raise ValueError(f'existing simulation truth changed: {truth_path}')
                atomic_json(truth_path, truth.tolist())
                jobs.append((directory, data, config, truth))
            del item
        build = work/'build'/sha(STAN)[:16]
        build.mkdir(parents=True, exist_ok=True)
        copied = build/STAN.name
        if not copied.exists():
            shutil.copyfile(STAN, copied)
        if sha(copied) != sha(STAN):
            raise ValueError('scratch Stan source mismatch')
        compiled = cmdstanpy.CmdStanModel(stan_file=str(copied), cpp_options={'STAN_THREADS': 'true'})
        records, errors = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fit_job, job, compiled.exe_file, a, units): job[0].name for job in jobs}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as exc:
                    errors.append(dict(job=futures[future], error=str(exc)))
                    print(f'FAILED: {futures[future]}: {exc}', flush=True)
        records.sort(key=lambda r: r['job'])
        atomic_json(work/'recovery_status.json', dict(completed=len(records), errors=errors))
        if records:
            collect(records, work, export)
            provenance = json.loads((export/'provenance.json').read_text())
            provenance['recovery'] = ('Posterior-conditioned joint draws from BOTH empirical default-prior fits, '
                'all chains and half-chains; flagged source fit retained. Conditional recovery, NOT SBC.')
            atomic_json(export/'provenance.json', provenance)
        if errors:
            return 1
        flagged = sum(not r['diagnostics']['passed'] for r in records)
        print(f'COMPLETED: {len(records)} targeted fits; {flagged} diagnostic flags; no imaging release', flush=True)
        return 2 if flagged else 0


if __name__ == '__main__':
    raise SystemExit(main())
