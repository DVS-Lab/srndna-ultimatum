#!/usr/bin/env python3
"""Audit posterior-derived norm signals and posterior-conditioned recovery.

Read-only toward completed fits. No imaging EVs or covariates are released.
Trial correlations are BEFORE convolution/filtering, not FEAT estimability tests.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from export_gu_stan_diagnostics import checked_file, read_chain, sha
from run_gu_hierarchical import (ROOT, PARTNERS, MODELS, atomic_json, build_data,
                                 digest, difference_draws, load_trials, write_table)

SIGNALS = ('expectation_before_offer', 'signed_prediction_error', 'updated_norm')


def read_job(job, units, *, recovery=False, draws=True):
    """Verify provenance, alignment, complete chains, and natural parameter bounds."""
    job = Path(job).resolve()
    cp, rp = job/'configuration.json', job/'completed.json'
    config, record = json.loads(cp.read_text()), json.loads(rp.read_text())
    if record['job'] != job.name or record['config_hash'] != digest(config):
        raise ValueError(f'configuration fingerprint mismatch: {job}')
    if (config['model'] != 'rw_free' or config['stage'] != 'full' or
            config.get('age_mode', 'none') != 'none' or
            config['parameters'] != list(MODELS['rw_free'][1])):
        raise ValueError(f'expected full-data age-blind rw_free fit: {job}')
    if config['phase'] != ('recovery' if recovery else 'fit'):
        raise ValueError(f'wrong fit phase: {job}')
    prefix = record['attempt']+'/'
    dp = checked_file(job, record, prefix+'data.json')
    diagnostic = checked_file(job, record, prefix+'diagnostics.json')
    data = json.loads(dp.read_text())
    if digest(data) != config['data_hash'] or json.loads(diagnostic.read_text()) != record['diagnostics']:
        raise ValueError(f'data/diagnostic fingerprint mismatch: {job}')
    expected = build_data(units, 'rw_free', scale=config['prior_scale'])
    keys = ('S', 'U', 'T', 'K', 'subject', 'partner', 'unit_ids', 'offers',
            'observed', 'use_choice', 'run', 'A') + (() if recovery else ('choice',))
    if any(data[k] != expected[k] for k in keys) or config['subjects'] != list(dict.fromkeys(u['subject'] for u in units)):
        raise ValueError(f'current trial identity/order differs from completed fit: {job}')
    variables = [f'theta[{u+1},{k+1}]' for u in range(data['U']) for k in range(4)]
    chains = []
    source_hashes = {str(path): sha(path) for path in (cp, rp, dp, diagnostic)}
    if draws:
        for relative in sorted(record['files']):
            if relative.startswith(prefix) and relative.endswith('.csv'):
                path = checked_file(job, record, relative)
                cid, array = read_chain(path, variables, config['samples'])
                array = array.reshape(config['samples'], data['U'], 4)
                validate_theta(array)
                chains.append((cid, array))
                source_hashes[str(path)] = record['files'][relative]
        if len(chains) != config['chains'] or len(set(c for c, _ in chains)) != len(chains):
            raise ValueError(f'wrong number of unique completed chains: {job}')
    return dict(job=str(job), config=config, record=record, data=data,
                chains=sorted(chains), source_hashes=source_hashes)


def validate_theta(theta):
    upper = np.array([1., 1., 20., 1.])
    if not np.isfinite(theta).all() or np.any(theta < 0) or np.any(theta > upper):
        raise ValueError('invalid natural-scale theta; expected alpha,gamma,f0,epsilon')


def trajectories(theta, offers):
    """Compute each joint draw's trajectory; include missed offers and carry runs."""
    theta, offers = np.asarray(theta), np.asarray(offers)
    validate_theta(theta)
    norm = theta[..., 2].copy()
    before, pe, after = [], [], []
    for offer in offers:
        before.append(norm.copy())
        error = offer - norm
        pe.append(error)
        norm = norm + theta[..., 3]*error
        after.append(norm.copy())
    return {name: np.stack(value, axis=-1)
            for name, value in zip(SIGNALS, (before, pe, after))}


def correlation(a, b, *, spearman=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if spearman:
        a, b = rankdata(a), rankdata(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def audit_signals(theta, units, label, passed):
    trial_rows, metrics = [], []
    for u, unit in enumerate(units):
        signals = trajectories(theta[:, u], unit['offers'])
        summaries = {s: (v.mean(0), v.std(0), np.quantile(v, [.025, .975], axis=0))
                     for s, v in signals.items()}
        for t, trial in enumerate(unit['trials']):
            row = dict(fit=label, diagnostics_passed=passed, subject=unit['subject'],
                       partner=unit['partner'], run=trial['run'], trial_in_run=trial['trial_in_run'],
                       offer=unit['offers'][t], responded=int(not trial['missed']))
            for signal, (mean, sd, ci) in summaries.items():
                row.update({signal+'_mean': mean[t], signal+'_sd': sd[t],
                            signal+'_q025': ci[0, t], signal+'_q975': ci[1, t]})
            trial_rows.append(row)
        # Imaging candidates exclude missed choices, but their offers updated the norm above.
        for run in (1, 2):
            mask = (np.asarray(unit['runs']) == run) & np.isfinite(unit['choices'])
            offer = np.asarray(unit['offers'])[mask]
            for signal, (mean, sd, ci) in summaries.items():
                temporal_sd = np.std(mean[mask]) if mask.any() else 0.
                metrics.append(dict(fit=label, diagnostics_passed=passed,
                    subject=unit['subject'], partner=unit['partner'], run=run, signal=signal,
                    n_responded=int(mask.sum()), offer_correlation=correlation(offer, mean[mask]),
                    posterior_mean_temporal_sd=temporal_sd,
                    mean_posterior_sd=float(sd[mask].mean()) if mask.any() else None,
                    uncertainty_to_temporal_sd=float(sd[mask].mean()/temporal_sd) if temporal_sd > 1e-12 else None,
                    mean_95_interval_width=float((ci[1, mask]-ci[0, mask]).mean()) if mask.any() else None))
    return trial_rows, metrics


def compare_signals(rows):
    frame = pd.DataFrame(rows)
    keys = ['subject', 'partner', 'run', 'trial_in_run']
    if frame.duplicated(['fit']+keys).any():
        raise ValueError('duplicate trial identity in signal summaries')
    out = []
    for left, right in combinations(frame.fit.unique(), 2):
        a = frame[frame.fit == left].set_index(keys).sort_index()
        b = frame[frame.fit == right].set_index(keys).sort_index()
        if not a.index.equals(b.index) or not np.array_equal(a[['offer', 'responded']], b[['offer', 'responded']]):
            raise ValueError('trial alignment differs between signal fits')
        for (subject, partner, run), group in a.groupby(level=keys[:3]):
            idx = group[group.responded == 1].index
            for signal in SIGNALS:
                x, y = a.loc[idx, signal+'_mean'], b.loc[idx, signal+'_mean']
                out.append(dict(fit_a=left, fit_b=right, subject=subject, partner=partner,
                    run=run, signal=signal, n_responded=len(idx),
                    both_diagnostics_passed=bool(group.diagnostics_passed.all() and b.loc[group.index].diagnostics_passed.all()),
                    correlation=correlation(x, y), rmse=float(np.sqrt(np.mean((x-y)**2))) if len(idx) else None))
    return out


def recovery_metrics(theta, truth, data):
    index = {(s, p): i for i, (s, p) in enumerate(zip(data['subject'], data['partner']))}
    averages = lambda x: np.stack([x[..., [index[s, p] for p in (1, 2, 3)], :].mean(-2)
                                  for s in range(1, data['S']+1)], axis=-2)
    targets = {'participant_average': (averages(theta), averages(truth)),
               'pooled_participant_partner': (theta, truth)}
    targets.update({name: (value, difference_draws(truth, data)[name])
                    for name, value in difference_draws(theta, data).items()})
    rows = []
    for target, (draws, true) in targets.items():
        for k, parameter in enumerate(MODELS['rw_free'][1]):
            mean = draws[..., k].mean(0)
            lo, hi = np.quantile(draws[..., k], [.025, .975], axis=0)
            rows.append(dict(target=target, parameter=parameter, n=len(mean),
                spearman_r=correlation(mean, true[..., k], spearman=True),
                rmse=float(np.sqrt(np.mean((mean-true[..., k])**2))),
                coverage_95=float(np.mean((lo <= true[..., k]) & (true[..., k] <= hi))),
                mean_interval_width=float(np.mean(hi-lo))))
    return rows


def source_jobs(scratch):
    scratch = Path(scratch)
    return [('baseline', scratch/'srndna-gu-stan-long-v2/fits/fit-rw_free-full-prior1'),
            *[(f'repeat_prior{scale}', scratch/f'srndna-gu-stan-priors-long-v1/fits/fit-rw_free-full-prior{scale}')
              for scale in ('0.5', '1', '2')]]


def execute_audit(mode, scratch, output):
    units, _, event_hashes = load_trials(ROOT)
    if mode == 'signals':
        jobs = source_jobs(scratch)
    else:
        jobs = [(p.parent.name, p.parent) for p in sorted(
            (scratch/'srndna-gu-stan-targeted-v1/fits').glob('*/configuration.json'))]
        if len(jobs) != 16:
            raise ValueError(f'expected all 16 targeted recovery jobs, found {len(jobs)}')
    # Refuse unidentified output directories; reruns overwrite only this audit's known products.
    names = {'provenance.json', 'trial_signals.tsv', 'signal_metrics.tsv', 'signal_comparison.tsv',
             'recovery_metrics.tsv', 'signal_recovery.tsv', 'fit_status.tsv'}
    if output.exists() and any(p.name not in names for p in output.iterdir()):
        raise ValueError(f'non-audit files in output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output/'provenance.json'
    dependencies = {name: sha(ROOT/'code'/name) for name in
        ('run_gu_hierarchical.py', 'run_gu_norm_learning.py', 'export_gu_stan_diagnostics.py',
         'gu_norm_model.py', 'build_event_corrected_trials.py')}
    expected = dict(mode=mode, audit_sha256=sha(__file__), dependencies=dependencies,
                    numpy=np.__version__, pandas=pd.__version__, event_hashes=event_hashes,
                    jobs={label: sha(path/'completed.json') for label, path in jobs})
    if provenance_path.exists() and json.loads(provenance_path.read_text())['fingerprint'] != digest(expected):
        raise ValueError('audit inputs changed; use a new output directory')
    trials, metrics, recoveries, signal_recoveries, statuses, sources = [], [], [], [], [], {}
    for label, path in jobs:
        item = read_job(path, units, recovery=mode == 'targeted')
        theta = np.concatenate([a for _, a in item['chains']])
        passed = item['record']['diagnostics']['passed']
        statuses.append(dict(fit=label, **item['record']['diagnostics']))
        sources.update(item['source_hashes'])
        if mode == 'signals':
            tr, met = audit_signals(theta, units, label, passed)
            trials.extend(tr); metrics.extend(met)
        else:
            truth_path = path/'simulation_truth.json'
            truth = np.asarray(json.loads(truth_path.read_text()))
            if digest(truth.tolist()) != item['config']['truth_hash']:
                raise ValueError(f'truth fingerprint mismatch: {path}')
            sources[str(truth_path)] = sha(truth_path)
            tags = dict(fit=label, diagnostics_passed=passed,
                        source_fit=item['config']['truth_source']['label'],
                        source_diagnostics_passed=item['config']['truth_source']['diagnostics_passed'])
            recoveries.extend(dict(**tags, **r) for r in recovery_metrics(theta, truth, item['data']))
            for u, unit in enumerate(units):
                draws = trajectories(theta[:, u], unit['offers'])
                true = trajectories(truth[u], unit['offers'])
                for run in (1, 2):
                    mask = (np.asarray(unit['runs']) == run) & np.isfinite(unit['choices'])
                    for signal in SIGNALS:
                        mean = draws[signal].mean(0)[mask]
                        lo, hi = np.quantile(draws[signal][:, mask], [.025, .975], axis=0)
                        target = true[signal][mask]
                        signal_recoveries.append(dict(**tags, subject=unit['subject'], partner=unit['partner'],
                            run=run, signal=signal, correlation=correlation(mean, target),
                            rmse=float(np.sqrt(np.mean((mean-target)**2))) if mask.any() else None,
                            coverage_95=float(np.mean((lo <= target) & (target <= hi))) if mask.any() else None))
        print(f'AUDITED: {label}; diagnostics={"PASS" if passed else "FLAGGED (retained)"}', flush=True)
        del theta, item
    tables = {'fit_status': statuses, 'trial_signals': trials, 'signal_metrics': metrics,
              'signal_comparison': compare_signals(trials) if trials else [],
              'recovery_metrics': recoveries, 'signal_recovery': signal_recoveries}
    for name, rows in tables.items():
        if rows:
            write_table(output/f'{name}.tsv', rows)
    atomic_json(provenance_path, dict(fingerprint=digest(expected), inputs=expected, source_hashes=sources,
        posterior_draws='all saved draws from all completed chains; flagged fits retained and labeled',
        inference='conditional recovery and pre-convolution signal diagnostics; not independent validation or SBC',
        imaging_covariates_released=False, outputs={p.name: sha(p) for p in output.glob('*.tsv')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['signals', 'targeted'])
    parser.add_argument('--scratch-base', type=Path, default=Path('/ZPOOL/data/scratch'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    # Never write into existing fit roots or arbitrary locations in the code repository.
    if ROOT in output.parents and ROOT/'results/norm_learning' not in output.parents:
        parser.error('repository outputs belong under results/norm_learning')
    if args.scratch_base.resolve() == output or args.scratch_base.resolve() in output.parents:
        parser.error('exports must be outside the scratch tree')
    execute_audit(args.mode, args.scratch_base.resolve(), output)
    print(f'Saved {args.mode} audit: {output}; no imaging inputs released', flush=True)


if __name__ == '__main__':
    main()
