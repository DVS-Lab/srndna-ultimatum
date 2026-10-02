#!/usr/bin/env python3
"""Fit and audit Gu-style norm learning on corrected SRNDNA choice histories.

Each participant/partner has a separate norm history across runs. Missed
choices have no likelihood term but their observed offers update the norm.
Outputs are exploratory behavioral analyses, not approved imaging covariates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import spearmanr

from build_event_corrected_trials import event_trials, TRIAL_PATTERN, read_rows
from gu_norm_model import SPECS, fit, fit_family, predictions

ROOT = Path(__file__).resolve().parents[1]
PARTNERS = ('computer', 'similar', 'dissimilar')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(path, records):
    if not records:
        raise ValueError(f'empty result table: {path}')
    fields = list(dict.fromkeys(key for row in records for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def load_trials(repo):
    sample = repo / 'behavioral_analyses/data/participant_L3_47.csv'
    _, people = read_rows(sample)
    subjects = [p['subjID'] for p in people]
    if len(subjects) != 47 or len(set(subjects)) != 47:
        raise ValueError('expected 47 unique sample participants')
    provenance, trials, units = {str(sample): sha(sample)}, [], []
    for person in people:
        subject = person['subjID']
        age = 'younger' if float(person['younger']) == 1 else 'older'
        participant_trials = []
        for run in (1, 2):
            path = repo / f'source_data/bids/{subject}/func/{subject}_task-ultimatum_run-{run:02d}_events.tsv'
            provenance[str(path)] = sha(path)
            _, events = read_rows(path, '\t')
            onsets = [float(r['onset']) for r in events if TRIAL_PATTERN.match(r['trial_type'])]
            if len(onsets) != 72 or np.any(np.diff(onsets) <= 0):
                raise ValueError(f'trial onsets are duplicate or unordered: {path}')
            substantive = event_trials(path, response_delay=1.)
            for index, (row, onset) in enumerate(zip(substantive, onsets), 1):
                partner = 'computer' if row['human'] == '0' else 'similar' if row['ingroup'] == '1' else 'dissimilar'
                offer = float(row['offer'])
                if not np.isfinite(offer) or offer < 0 or offer > 10:
                    raise ValueError(f'invalid/missing observed offer: {path} trial {index}')
                participant_trials.append(dict(subject=subject, age_group=age, run=run,
                    trial_in_run=index, onset=onset, partner=partner, offer_dollars=offer,
                    accept=None if row['missed'] == '1' else int(row['accept']),
                    missed=int(row['missed']), fair_block=int(row['block_fair'])))
        trials.extend(participant_trials)
        for partner in PARTNERS:
            selected = [r for r in participant_trials if r['partner'] == partner]
            if len(selected) != 48 or any(sum(r['run'] == k for r in selected) == 0 for k in (1, 2)):
                raise ValueError(f'expected 48 partner trials spanning both runs: {subject} {partner}')
            units.append(dict(subject=subject, age_group=age, partner=partner,
                              offers=[r['offer_dollars'] for r in selected],
                              choices=[r['accept'] if r['accept'] is not None else np.nan for r in selected],
                              runs=[r['run'] for r in selected], trials=selected))
    # Explicit guard against recurrence of the sub-143/sub-144 duplication.
    sequences = {s: [(r['run'], r['partner'], r['offer_dollars'], r['accept']) for r in trials if r['subject'] == s]
                 for s in ('sub-143', 'sub-144')}
    if sequences['sub-143'] == sequences['sub-144']:
        raise ValueError('sub-143 and sub-144 have identical complete trial histories')
    return units, trials, provenance


def unit_seed(seed, subject, partner):
    return (seed + int(hashlib.sha256(f'{subject}/{partner}'.encode()).hexdigest()[:8], 16)) % 2**32


def analyze_unit(unit, starts, recovery_reps, seed):
    offers, choices, runs = np.array(unit['offers']), np.array(unit['choices']), np.array(unit['runs'])
    identity = {k: unit[k] for k in ('subject', 'age_group', 'partner')}
    seed = unit_seed(seed, unit['subject'], unit['partner'])
    full = fit_family(offers, choices, seed, starts)
    # Fit run 1 only. Future observed offers do not affect any earlier prediction.
    train = fit_family(offers, choices, seed + 100, starts, likelihood_mask=(runs == 1))
    records, trial_predictions = [], []
    for model in SPECS:
        record, theta, prob, norms = full[model]
        train_record, train_theta, _, _ = train[model]
        z, _, _ = predictions(train_theta, offers, model)
        test = (runs == 2) & np.isfinite(choices)
        test_nll = float(np.sum(np.logaddexp(0, z[test]) - choices[test] * z[test]))
        records.append(dict(**identity, **record, heldout_n=int(test.sum()), heldout_nll=test_nll,
                            heldout_mean_nll=test_nll/int(test.sum()),
                            train_converged=train_record['best_converged'],
                            train_boundary_parameters=train_record['boundary_parameters']))
        for i, row in enumerate(unit['trials']):
            trial_predictions.append(dict(**row, model=model, probability_accept=float(prob[i]),
                norm_before=None if model == 'logistic' else float(norms[i, 0]),
                norm_at_choice=None if model == 'logistic' else float(norms[i, 1]),
                norm_after=None if model == 'logistic' else float(norms[i, 2])))
    recoveries = []
    rng = np.random.default_rng(seed + 200)
    for rep in range(recovery_reps):
        # Independent known parameters, not the noisy empirical MLEs. The actual
        # 48-offer sequence and observed missed-response pattern are retained.
        truth = np.array([rng.uniform(.15, .95), rng.uniform(.15, .95),
                          rng.uniform(.02, .8), rng.uniform(6., 18.)])
        z, _, _ = predictions(truth, offers, 'rw_free')
        simulated = rng.binomial(1, expit(z)).astype(float)
        simulated[~np.isfinite(choices)] = np.nan
        recovered = fit_family(offers, simulated, seed + 1000 + rep * 20, starts)
        estimates = recovered['rw_free']
        record, theta, _, _ = estimates
        row = dict(**identity, replicate=rep + 1,
                   fit_converged=record['best_converged'], boundary_parameters=record['boundary_parameters'],
                   generating_model='rw_free',
                   selected_bic_model=min(recovered, key=lambda m: recovered[m][0]['bic']))
        for (parameter, _, _), true, fitted in zip(SPECS['rw_free'], truth, theta):
            row[parameter + '_true'] = float(true)
            row[parameter + '_estimated'] = float(fitted)
        recoveries.append(row)
    return dict(identity=identity, fits=records, predictions=trial_predictions, recovery=recoveries)


def summarize(fits, recovery):
    output = []
    for partner in PARTNERS:
        rows = [r for r in fits if r['partner'] == partner]
        for model in SPECS:
            selected = [r for r in rows if r['model'] == model]
            deltas = []
            wins = 0
            for r in selected:
                candidates = [s for s in rows if s['subject'] == r['subject']]
                best = min(s['bic'] for s in candidates)
                deltas.append(r['bic'] - best)
                wins += r['model'] == min(candidates, key=lambda s: s['bic'])['model']
            output.append(dict(partner=partner, model=model, participants=len(selected),
                total_nll=sum(r['nll'] for r in selected), total_bic=sum(r['bic'] for r in selected),
                median_delta_bic=float(np.median(deltas)), bic_winners=wins,
                total_heldout_nll=sum(r['heldout_nll'] for r in selected),
                heldout_choices=sum(r['heldout_n'] for r in selected),
                boundary_fits=sum(bool(r['boundary_parameters']) for r in selected),
                deficient_fisher_rank=sum(r['fisher_rank'] < r['n_parameters'] for r in selected),
                failed_best_convergence=sum(not r['best_converged'] for r in selected)))
    recovery_summary = []
    for partner in ('all', *PARTNERS):
        rows = [r for r in recovery if partner == 'all' or r['partner'] == partner]
        if not rows:
            continue
        for parameter, _, _ in SPECS['rw_free']:
            true = np.array([r[parameter + '_true'] for r in rows])
            est = np.array([r[parameter + '_estimated'] for r in rows])
            rho = float(spearmanr(true, est).statistic) if np.ptp(est) else None
            recovery_summary.append(dict(partner=partner, parameter=parameter, simulations=len(rows),
                spearman_r=rho, rmse=float(np.sqrt(np.mean((est - true)**2))),
                mean_bias=float(np.mean(est - true)),
                boundary_fits=sum(bool(r['boundary_parameters']) for r in rows),
                failed_convergence=sum(not r['fit_converged'] for r in rows),
                generating_model_selected=sum(r['selected_bic_model'] == 'rw_free' for r in rows)))
    return output, recovery_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--starts', type=int, default=24)
    p.add_argument('--recovery-reps', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--subjects', nargs='+', help='explicit smoke-test subset; full analysis uses all 47')
    a = p.parse_args()
    if not 1 <= a.jobs <= 45 or a.starts < 4 or a.recovery_reps < 0:
        p.error('jobs must be 1..45, starts >=4, recovery-reps >=0')
    units, trials, source_hashes = load_trials(ROOT)
    if a.subjects:
        available = {u['subject'] for u in units}
        if not set(a.subjects) <= available:
            p.error('unknown subject requested')
        units = [u for u in units if u['subject'] in a.subjects]
        trials = [r for r in trials if r['subject'] in a.subjects]
    for path in (Path(__file__), ROOT / 'code/gu_norm_model.py', ROOT / 'code/build_event_corrected_trials.py'):
        source_hashes[str(path)] = sha(path)
    config = dict(sources=source_hashes, starts=a.starts, recovery_reps=a.recovery_reps, seed=a.seed,
                  subjects=sorted({u['subject'] for u in units}), python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__,
                  monetary_unit='dollars; $20 stake; fixed initial norm $10',
                  norm_history='separate per partner; carry across blocks and runs; observe missed-trial offers',
                  gu_update='f_i=f_(i-1)+epsilon*(s_i-f_(i-1)); V_i uses f_i',
                  sensitivity_update='rw_free_prior uses f_(i-1) for choice then updates',
                  paper='https://doi.org/10.1523/JNEUROSCI.2906-14.2015',
                  comparison='standard BIC and AICc, NOT the unspecified modified BIC of Gu 2015',
                  imaging_covariates_released=False)
    out = a.output_root.resolve()
    for protected in (ROOT / 'source_data', ROOT / 'behavioral_analyses', ROOT / 'code'):
        if out == protected or protected in out.parents or out in protected.parents:
            p.error('output overlaps protected source/code directory')
    config_path = out / 'configuration.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError('source/code/options changed; choose a new output root')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError('nonempty output directory without matching configuration')
    else:
        out.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    checkpoint = out / 'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    results, pending = [], []
    for unit in units:
        path = checkpoint / f'{unit["subject"]}_{unit["partner"]}.json'
        if path.is_file():
            result = json.loads(path.read_text())
            if result['identity'] != {k: unit[k] for k in ('subject', 'age_group', 'partner')}:
                raise ValueError(f'checkpoint identity mismatch: {path}')
            results.append(result)
        else:
            pending.append((unit, path))
    print(f'Units: {len(units)}; retained checkpoints: {len(results)}; to fit: {len(pending)}', flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(analyze_unit, u, a.starts, a.recovery_reps, a.seed): path for u, path in pending}
        for future in as_completed(futures):
            result = future.result()
            path = futures[future]
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(result, allow_nan=False) + '\n')
            temp.replace(path)
            results.append(result)
            print(f'Completed {len(results)}/{len(units)}: {path.stem}', flush=True)
    results.sort(key=lambda r: (r['identity']['subject'], r['identity']['partner']))
    fits = [row for r in results for row in r['fits']]
    recovery = [row for r in results for row in r['recovery']]
    prediction_rows = [row for r in results for row in r['predictions']]
    summary, recovery_summary = summarize(fits, recovery)
    write_table(out / 'trial_history.tsv', trials)
    write_table(out / 'fits.tsv', fits)
    write_table(out / 'trial_predictions.tsv', prediction_rows)
    write_table(out / 'model_comparison.tsv', summary)
    if recovery:
        write_table(out / 'parameter_recovery.tsv', recovery)
        write_table(out / 'parameter_recovery_summary.tsv', recovery_summary)
    completion = dict(status='completed', participants=len(config['subjects']), units=len(units),
                      trial_rows=len(trials), models=len(SPECS), fitted_models=len(fits),
                      simulations=len(recovery), imaging_covariates_released=False,
                      interpretation='Exploratory estimates; inspect fit, prediction and recovery before L3 use.')
    (out / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
