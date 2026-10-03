#!/usr/bin/env python3
"""Hierarchical norm learning: guarded preparation, Stan fitting, and recovery.

Heavy outputs belong in scratch, not Git. No imaging covariates are released.
Use --execute explicitly; repeating a command resumes completed *fits*, not
interrupted HMC chains. Incomplete attempts are retained in separate directories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_var] = '1'

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import spearmanr
from run_gu_norm_learning import ROOT, PARTNERS, load_trials, sha, write_table

STAN = ROOT / 'code/stan/gu_hierarchical.stan'
CMDSTAN_VERSION = '2.40.0'
MODELS = {'rw_free': (1, ('alpha', 'gamma', 'f0', 'epsilon')),
          'fs_free': (2, ('alpha', 'gamma', 'f0')),
          'logistic': (3, ('intercept_at_5', 'offer_slope')),
          'rw_free_prior': (4, ('alpha', 'gamma', 'f0', 'epsilon'))}
CONTRAST = np.array([[-2/3, 0], [1/3, .5], [1/3, -.5]])


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prior_scales(model, scale):
    k = len(MODELS[model][1])
    if model == 'logistic':
        values = ([2., 1.], [1., .5], [1., .5], [1., .5])
    else:
        values = ([1.5]*k, [.75]*k, [.75]*k, [.5]*k)
    return {name: (np.array(value)*scale).tolist() for name, value in
            zip(('mu_sd', 'effect_sd', 'subject_sd', 'contrast_sd'), values)}


def age_design(units, mode):
    """One row per participant; repeated partner rows never set the centering."""
    if mode not in ('none', 'group'):
        raise ValueError('age mode must be none or group')
    subjects = list(dict.fromkeys(u['subject'] for u in units))
    groups = {}
    for unit in units:
        group = unit['age_group']
        if group not in ('younger', 'older') or groups.get(unit['subject'], group) != group:
            raise ValueError('missing or inconsistent participant age group')
        groups[unit['subject']] = group
    indicator = np.array([groups[s] == 'older' for s in subjects], float)
    if mode == 'group' and len(set(indicator)) != 2:
        raise ValueError('age-group model requires both age groups')
    return (indicator-indicator.mean())[:, None] if mode == 'group' else np.empty((len(subjects), 0))


def build_data(units, model, stage='full', scale=1., age_mode='none', age_prior_sd=.5):
    subjects = list(dict.fromkeys(u['subject'] for u in units))
    if len(units) != len(subjects)*3:
        raise ValueError('each subject must have all three partners')
    offers = np.array([u['offers'] for u in units], float)
    y = np.array([u['choices'] for u in units], float)
    runs = np.array([u['runs'] for u in units], int)
    if offers.shape != (len(units), 48) or not np.isfinite(offers).all():
        raise ValueError('expected 48 finite offers per unit')
    if not np.array_equal(offers, np.round(offers)) or ((offers < 0) | (offers > 10)).any():
        raise ValueError('PPC bins require integer dollar offers in [0,10]')
    observed = np.isfinite(y)
    if not np.isin(y[observed], [0, 1]).all():
        raise ValueError('choices must be binary or missing')
    if not np.isin(runs, [1, 2]).all() or np.any(np.diff(runs, axis=1) < 0):
        raise ValueError('trial histories must be chronological across runs 1 and 2')
    use = observed & ((runs == 1) if stage == 'run1' else True)
    return dict(S=len(subjects), U=len(units), T=48, K=len(MODELS[model][1]),
        model_id=MODELS[model][0], subject=[subjects.index(u['subject'])+1 for u in units],
        partner=[PARTNERS.index(u['partner'])+1 for u in units],
        unit_ids=list(range(1, len(units)+1)), offers=offers.tolist(),
        choice=np.where(observed, y, 0).astype(int).tolist(), observed=observed.astype(int).tolist(),
        use_choice=use.astype(int).tolist(), run=runs.tolist(), contrast=CONTRAST.tolist(),
        A=int(age_mode != 'none'), age_design=age_design(units, age_mode).tolist(),
        age_sd=[age_prior_sd*scale]*len(MODELS[model][1]), **prior_scales(model, scale))


def latent_to_theta(latent, model):
    if model == 'logistic':
        return np.array([30., 10.])*np.tanh(latent/np.array([30., 10.]))
    theta = expit(latent)
    theta[..., 2] *= 20
    return theta


def draw_prior(data, model, rng):
    """Exact draws from the non-centered hierarchy used in Stan, for PPC/recovery."""
    k, s = data['K'], data['S']
    mu = rng.normal(0, data['mu_sd'])
    effect = rng.normal(0, data['effect_sd'], (2, k))
    sd_s = np.abs(rng.normal(0, data['subject_sd']))
    sd_c = np.abs(rng.normal(0, data['contrast_sd'], (2, k)))
    z_s, z_c = rng.normal(size=(s, k)), rng.normal(size=(2, s, k))
    beta = rng.normal(0, data['age_sd'], (data.get('A', 0), k))
    latent = []
    for subject, partner in zip(data['subject'], data['partner']):
        age = np.array(data['age_design'])[subject-1] @ beta if data.get('A', 0) else 0
        latent.append(mu + age + sd_s*z_s[subject-1] +
                      (CONTRAST[partner-1, :, None]*(effect + sd_c*z_c[:,subject-1])).sum(0))
    return latent_to_theta(np.array(latent), model)


def logits(theta, offers, model):
    """Vectorized over arbitrary leading draw/unit dimensions; time remains serial."""
    offers = np.asarray(offers)
    if model == 'logistic':
        return theta[..., 0, None] + theta[..., 1, None]*(offers-5)
    norm = theta[..., 2].copy()
    result = []
    for t in range(offers.shape[-1]):
        at_choice = norm.copy()
        if model != 'fs_free':
            norm = norm + theta[..., 3]*(offers[..., t]-norm)
        if model != 'rw_free_prior':
            at_choice = norm
        result.append(theta[..., 1]*(offers[..., t] - theta[..., 0]*np.maximum(at_choice-offers[..., t], 0)))
    return np.stack(result, axis=-1)


def describe(draws):
    return dict(mean=float(np.mean(draws)), sd=float(np.std(draws)),
                q025=float(np.quantile(draws, .025)), median=float(np.median(draws)),
                q975=float(np.quantile(draws, .975)))


def difference_draws(theta, data):
    # Explicit subject/partner map, never assume incidental row order.
    index = {(s, p): i for i, (s, p) in enumerate(zip(data['subject'], data['partner']))}
    similar = theta[..., [index[s, 2] for s in range(1, data['S']+1)], :]
    dissimilar = theta[..., [index[s, 3] for s in range(1, data['S']+1)], :]
    computer = theta[..., [index[s, 1] for s in range(1, data['S']+1)], :]
    return {'similar_minus_dissimilar': similar-dissimilar,
            'human_minus_computer': .5*(similar+dissimilar)-computer}


def prior_predictive(data, model, path, seed, draws=400):
    rng = np.random.default_rng(seed)
    rates = np.zeros((draws, 3, 2, 11))
    n = np.zeros((3, 2, 11), int)
    obs = np.array(data['observed'], bool)
    for u, partner in enumerate(data['partner']):
        for t, offer in enumerate(data['offers'][u]):
            if obs[u,t]:
                n[partner-1, data['run'][u][t]-1, int(offer)] += 1
    for d in range(draws):
        y = rng.binomial(1, expit(logits(draw_prior(data, model, rng), data['offers'], model)))
        for u, partner in enumerate(data['partner']):
            for t, offer in enumerate(data['offers'][u]):
                if obs[u,t]:
                    rates[d, partner-1, data['run'][u][t]-1, int(offer)] += y[u,t]
    rows = []
    for p, r, b in zip(*np.where(n > 0)):
        rows.append(dict(partner=PARTNERS[p], run=r+1, offer=b, n=int(n[p,r,b]),
                         **describe(rates[:,p,r,b]/n[p,r,b])))
    write_table(path, rows)


def summarize_fit(fit, data, model, units, directory, max_depth, truth=None):
    summary = fit.summary()
    summary.to_csv(directory/'stan_summary.tsv', sep='\t', index_label='variable')
    # Exclude generated quantities (including constant held-out likelihoods).
    core = summary.loc[[i for i in summary.index if i.startswith(
        ('mu[', 'effect[', 'age_beta[', 'sigma_', 'z_subject[', 'z_contrast[', 'theta['))]]
    methods = fit.method_variables()
    energy = methods['energy__']
    bfmi = np.mean(np.diff(energy, axis=0)**2, axis=0)/np.var(energy, axis=0)
    rhat, essb, esst = (core[c].to_numpy() for c in ('R_hat', 'ESS_bulk', 'ESS_tail'))
    finite = np.isfinite(np.r_[rhat, essb, esst, bfmi]).all()
    diagnostics = dict(divergences=int(methods['divergent__'].sum()),
        max_depth_hits=int((methods['treedepth__'] >= max_depth).sum()),
        max_rhat=float(np.nanmax(rhat)) if np.isfinite(rhat).any() else None,
        min_bulk_ess=float(np.nanmin(essb)) if np.isfinite(essb).any() else None,
        min_tail_ess=float(np.nanmin(esst)) if np.isfinite(esst).any() else None,
        min_ebfmi=float(np.nanmin(bfmi)) if np.isfinite(bfmi).any() else None,
        finite_diagnostics=bool(finite))
    diagnostics['passed'] = bool(finite and diagnostics['divergences'] == 0 and
        diagnostics['max_depth_hits'] == 0 and max(rhat) <= 1.01 and min(essb) >= 400 and
        min(esst) >= 400 and min(bfmi) >= .3)
    atomic_json(directory/'diagnostics.json', diagnostics)
    theta = fit.stan_variable('theta')
    parameters = MODELS[model][1]
    if data.get('A', 0):
        beta = fit.stan_variable('age_beta')[:, 0, :]
        age_rows = []
        for k, parameter in enumerate(parameters):
            stats = summary.loc[f'age_beta[1,{k+1}]']
            age_rows.append(dict(parameter=parameter, contrast='older_minus_younger',
                scale='latent_before_parameter_transform', **describe(beta[:,k]),
                R_hat=float(stats.R_hat), ESS_bulk=float(stats.ESS_bulk),
                ESS_tail=float(stats.ESS_tail), MCSE=float(stats.MCSE)))
        write_table(directory/'age_effects.tsv', age_rows)
    rows = []
    for u, unit in enumerate(units):
        for k, parameter in enumerate(parameters):
            row = dict(subject=unit['subject'], partner=unit['partner'], parameter=parameter,
                       **describe(theta[:,u,k]))
            if truth is not None:
                row['truth'] = float(truth[u,k])
                row['covered_95'] = int(row['q025'] <= truth[u,k] <= row['q975'])
            rows.append(row)
    write_table(directory/'parameters.tsv', rows)
    contrasts, recovery = [], []
    subjects = list(dict.fromkeys(u['subject'] for u in units))
    for name, diffs in difference_draws(theta, data).items():
        true_diffs = difference_draws(truth, data)[name] if truth is not None else None
        for k, parameter in enumerate(parameters):
            for s, subject in enumerate(subjects):
                row = dict(contrast=name, subject=subject, parameter=parameter,
                           **describe(diffs[:,s,k]))
                if truth is not None:
                    row['truth'] = float(true_diffs[s,k])
                    row['covered_95'] = int(row['q025'] <= row['truth'] <= row['q975'])
                contrasts.append(row)
            contrasts.append(dict(contrast=name, subject='sample_mean', parameter=parameter,
                **describe(diffs[:,:,k].mean(1))))
        if truth is not None:
            for k, parameter in enumerate(parameters):
                est, true = diffs[:,:,k].mean(0), true_diffs[:,k]
                rho = spearmanr(true, est).statistic
                recovery.append(dict(target=name, parameter=parameter,
                    spearman_r=float(rho) if np.isfinite(rho) else None,
                    rmse=float(np.sqrt(np.mean((est-true)**2)))))
    write_table(directory/'partner_contrasts.tsv', contrasts)
    if truth is not None:
        for k, parameter in enumerate(parameters):
            rho = spearmanr(truth[:,k], theta[:,:,k].mean(0)).statistic
            recovery.append(dict(target='individual_parameter', parameter=parameter,
                spearman_r=float(rho) if np.isfinite(rho) else None,
                rmse=float(np.sqrt(np.mean((theta[:,:,k].mean(0)-truth[:,k])**2)))))
        write_table(directory/'recovery_summary.tsv', recovery)
    replicated = fit.stan_variable('replicated_accept')
    ppc = []
    for p in range(3):
        for r in (1, 2):
            for b in range(11):
                selected = (np.array(data['partner'])[:,None] == p+1) & (np.array(data['run']) == r) & \
                           (np.array(data['offers']) == b) & np.array(data['observed'], bool)
                n = int(selected.sum())
                if n:
                    ppc.append(dict(partner=PARTNERS[p], run=r, offer=b, n=n,
                        observed_rate=float(np.array(data['choice'])[selected].mean()),
                        **describe(replicated[:,p,r-1,b]/n)))
    write_table(directory/'posterior_predictive.tsv', ppc)
    heldout = np.array(data['observed'], bool) & ~np.array(data['use_choice'], bool)
    scores = []
    if heldout.any():
        joint = fit.stan_variable('heldout_log_lik')
        for u, unit in enumerate(units):
            mask = heldout[u]
            if not mask.any():
                continue
            z = logits(theta[:,u,:], data['offers'][u], model)[:,mask]
            y = np.array(data['choice'][u])[mask]
            ll = y*z-np.logaddexp(0, z)
            scores.append(dict(subject=unit['subject'], partner=unit['partner'], n=int(mask.sum()),
                trialwise_lpd=float((logsumexp(ll, axis=0)-np.log(len(theta))).sum()),
                joint_run_lpd=float(logsumexp(joint[:,u])-np.log(len(theta)))))
        write_table(directory/'heldout_prediction.tsv', scores)
    return diagnostics


def cpu_plan(jobs, chains, threads):
    if jobs < chains*threads:
        raise ValueError('--jobs must cover chains * threads-per-chain for one fit')
    return jobs//(chains*threads)


def config_guard(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'configuration.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f'changed code/data/options: use a NEW work root ({directory})')
    elif any(directory.iterdir()):
        raise FileExistsError(f'nonempty directory without provenance: {directory}')
    else:
        atomic_json(path, config)


def completed(directory, config_hash):
    path = directory/'completed.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record['config_hash'] != config_hash:
        raise ValueError(f'completion fingerprint mismatch: {directory}')
    for relative, expected in record['files'].items():
        candidate = directory/relative
        if not candidate.is_file() or sha(candidate) != expected:
            raise ValueError(f'completed output changed/missing: {candidate}; use a new root')
    return record


def fit_job(job, exe, options, units):
    import cmdstanpy
    directory, data, config, truth = job
    fingerprint = digest(config)
    previous = completed(directory, fingerprint)
    if previous is not None:
        print(f'SKIP_COMPLETE: {directory.name}', flush=True)
        return previous
    attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=directory))
    atomic_json(attempt/'data.json', data)
    print(f'RUN: {directory.name} -> {attempt}', flush=True)
    model = cmdstanpy.CmdStanModel(exe_file=str(exe))
    start = time.monotonic()
    fit = model.sample(data=str(attempt/'data.json'), chains=options.chains,
        parallel_chains=options.chains, threads_per_chain=options.threads_per_chain,
        force_one_process_per_chain=True, seed=config['seed'],
        iter_warmup=options.warmup, iter_sampling=options.samples,
        adapt_delta=options.adapt_delta, max_treedepth=options.max_treedepth,
        output_dir=str(attempt), show_progress=False, show_console=False, refresh=100,
        sig_figs=12)
    diagnostics = summarize_fit(fit, data, config['model'], units, attempt,
                                options.max_treedepth, truth)
    files = {str(p.relative_to(directory)): sha(p) for p in attempt.iterdir() if p.is_file()}
    record = dict(job=directory.name, config_hash=fingerprint, files=files,
                  diagnostics=diagnostics, elapsed_seconds=time.monotonic()-start,
                  attempt=attempt.name, inference_eligible=config['phase'] != 'smoke' and diagnostics['passed'])
    atomic_json(directory/'completed.json', record)
    print(f'COMPLETED: {directory.name}; diagnostics={"PASS" if diagnostics["passed"] else "NEEDS_REVIEW"}', flush=True)
    return record


def collect(records, work, export):
    export.mkdir(parents=True, exist_ok=True)
    aggregate = {}
    rows = []
    for record in records:
        directory = work/'fits'/record['job']
        config = json.loads((directory/'configuration.json').read_text())
        rows.append(dict(job=record['job'], model=config['model'], stage=config['stage'],
            age_mode=config.get('age_mode', 'none'),
            prior_scale=config['prior_scale'], **record['diagnostics'],
            inference_eligible=record['inference_eligible']))
        for name in ('parameters', 'partner_contrasts', 'posterior_predictive', 'heldout_prediction', 'recovery_summary', 'age_effects'):
            path = directory/record['attempt']/f'{name}.tsv'
            if path.exists():
                with path.open() as stream:
                    for row in csv.DictReader(stream, delimiter='\t'):
                        aggregate.setdefault(name, []).append(dict(job=record['job'],
                            model=config['model'], stage=config['stage'], prior_scale=config['prior_scale'],
                            age_mode=config.get('age_mode', 'none'),
                            inference_eligible=record['inference_eligible'], **row))
    write_table(export/'fit_status.tsv', rows)
    for name, values in aggregate.items():
        write_table(export/f'{name}.tsv', values)
    atomic_json(export/'provenance.json', dict(jobs=[json.loads(
        (work/'fits'/r['job']/'configuration.json').read_text()) for r in records],
        imaging_covariates_released=False,
        interval='95% posterior credible interval, not a frequentist confidence interval',
        recovery='hierarchical prior-generated recovery; not a completed SBC validation'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke', 'fit', 'recovery'], default='fit')
    p.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    p.add_argument('--stages', nargs='+', choices=['full', 'run1'], default=['full', 'run1'])
    p.add_argument('--prior-scales', type=float, nargs='+', default=[1.])
    p.add_argument('--jobs', type=int, default=40, help='total CPU budget, NOT concurrent fits')
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--threads-per-chain', type=int, default=2)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--adapt-delta', type=float, default=.95)
    p.add_argument('--max-treedepth', type=int, default=12)
    p.add_argument('--recovery-reps', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--age-mode', choices=['none', 'group'], default='none',
                   help='Optional centered older-group effect on every latent parameter; no age interactions')
    p.add_argument('--age-prior-sd', type=float, default=.5,
                   help='Normal prior SD for older-minus-younger latent effects (also multiplied by prior-scales)')
    p.add_argument('--cmdstan', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--collect-to', type=Path)
    a = p.parse_args()
    if not np.isfinite(a.age_prior_sd) or a.age_prior_sd <= 0:
        p.error('age-prior-sd must be finite and positive')
    if a.age_mode != 'none' and a.phase == 'recovery':
        p.error('age-effect recovery is not implemented; do not label baseline recovery as age validation')
    if a.chains < 4 or a.threads_per_chain < 1 or not 1 <= a.jobs <= 40:
        p.error('use >=4 chains, positive threads, and a total CPU budget of 1..40')
    if min(a.warmup, a.samples) < 20 or not .8 <= a.adapt_delta < 1 or a.max_treedepth < 8:
        p.error('warmup/samples >=20, adapt-delta in [.8,1), max-treedepth >=8')
    if any(not np.isfinite(s) or s <= 0 for s in a.prior_scales) or a.recovery_reps < 1:
        p.error('finite positive prior scales and positive recovery count required')
    if any(len(v) != len(set(v)) for v in (a.models, a.stages, a.prior_scales)):
        p.error('duplicate models/stages/prior-scales would race on output paths')
    workers = cpu_plan(a.jobs, a.chains, a.threads_per_chain)
    if a.phase != 'smoke' and min(a.warmup, a.samples) < 1000:
        p.error('production/recovery requires >=1000 warmup and sampling iterations')
    work = a.work_root.resolve()
    if work == ROOT or ROOT in work.parents or work in ROOT.parents:
        p.error('Stan work root must be scratch OUTSIDE the repository')
    work.mkdir(parents=True, exist_ok=True)
    with (work/'.runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another runner holds this work root; do not launch a duplicate')
        units, _, sources = load_trials(ROOT)
        if a.phase == 'smoke':
            # Includes both historically affected subjects, plus an older adult.
            keep = {units[0]['subject'], 'sub-143', 'sub-144',
                    next(u['subject'] for u in units if u['age_group'] == 'older')}
            units = [u for u in units if u['subject'] in keep]
        source_hashes = {str(Path(path).relative_to(ROOT)): value for path, value in sources.items()}
        for path in (Path(__file__), STAN, ROOT/'code/run_gu_norm_learning.py',
                     ROOT/'code/gu_norm_model.py', ROOT/'code/build_event_corrected_trials.py'):
            source_hashes[str(path.relative_to(ROOT))] = sha(path)
        software = {}
        if a.execute:
            import cmdstanpy
            if a.cmdstan:
                cmdstanpy.set_cmdstan_path(str(a.cmdstan.resolve()))
            # CmdStanPy reports major/minor only. Read the patch from makefile.
            makefile = Path(cmdstanpy.cmdstan_path())/'makefile'
            version = next(line.split(':=', 1)[1].strip() for line in makefile.read_text().splitlines()
                           if line.startswith('CMDSTAN_VERSION :='))
            if version != CMDSTAN_VERSION or cmdstanpy.__version__ != '1.3.0':
                raise ValueError(f'expected CmdStan {CMDSTAN_VERSION}, CmdStanPy 1.3.0; got {version}, {cmdstanpy.__version__}')
            software = dict(cmdstan=version, cmdstanpy=cmdstanpy.__version__)
        jobs = []
        for model in a.models:
            for scale in a.prior_scales:
                stages = ['full'] if a.phase == 'recovery' else a.stages
                reps = range(1, a.recovery_reps+1) if a.phase == 'recovery' else [0]
                for stage in stages:
                    for rep in reps:
                        name = f'{a.phase}-{model}-{stage}-prior{scale:g}' + (f'-rep{rep:03d}' if rep else '')
                        if a.age_mode != 'none':
                            name += f'-age-{a.age_mode}'
                        directory = work/'fits'/name
                        data = build_data(units, model, stage, scale, a.age_mode, a.age_prior_sd)
                        seed = (a.seed + int(hashlib.sha256(name.encode()).hexdigest()[:7], 16)) % 2147483647
                        truth = None
                        if rep:
                            rng = np.random.default_rng(seed)
                            truth = draw_prior(data, model, rng)
                            data['choice'] = rng.binomial(1, expit(logits(truth, data['offers'], model))).tolist()
                        config = dict(model=model, stage=stage, phase=a.phase, prior_scale=scale,
                            replicate=rep, seed=seed, sources=source_hashes, data_hash=digest(data),
                            warmup=a.warmup, samples=a.samples, chains=a.chains,
                            threads_per_chain=a.threads_per_chain, adapt_delta=a.adapt_delta,
                            max_treedepth=a.max_treedepth, software=software,
                            subjects=list(dict.fromkeys(u['subject'] for u in units)),
                            parameters=MODELS[model][1],
                            learning='separate partner history; carry across runs; missed offers update',
                            age_mode=a.age_mode, age_prior_sd=a.age_prior_sd,
                            age_in_parameter_hierarchy=a.age_mode != 'none',
                            age_coding='older indicator centered across unique participants; no interactions',
                            age_groups={u['subject']: u['age_group'] for u in units},
                            parameter_estimand='age-conditional, not age-residualized' if a.age_mode != 'none' else 'age-blind',
                            imaging_covariates_released=False)
                        # JSON roundtrip makes tuples/lists consistent on resume.
                        config = json.loads(json.dumps(config))
                        if a.execute:
                            config_guard(directory, config)
                            if truth is not None:
                                atomic_json(directory/'simulation_truth.json', truth.tolist())
                        jobs.append((directory, data, config, truth))
        print(f'Participant-partners: {len(units)}; planned fits: {len(jobs)}', flush=True)
        print(f'Age mode: {a.age_mode}; no imaging covariates or trial EVs will be released', flush=True)
        print(f'CPU budget: {a.jobs}; up to {workers} fits x {a.chains} chains x {a.threads_per_chain} threads = {workers*a.chains*a.threads_per_chain}', flush=True)
        for directory, data, _, _ in jobs:
            print(f'PLAN: {directory.name}; likelihood choices={int(np.sum(data["use_choice"]))}', flush=True)
        if not a.execute:
            print('DRY RUN: no compilation or sampling. Add --execute to run.', flush=True)
            return 0
        build = work/'build'/sha(STAN)[:16]
        build.mkdir(parents=True, exist_ok=True)
        copied = build/STAN.name
        if not copied.exists():
            shutil.copyfile(STAN, copied)
        compiled = cmdstanpy.CmdStanModel(stan_file=str(copied), cpp_options={'STAN_THREADS': 'true'})
        priors = work/'prior_predictive'
        priors.mkdir(exist_ok=True)
        for model in a.models:
            for scale in a.prior_scales:
                prior_data = build_data(units, model, 'full', scale, a.age_mode, a.age_prior_sd)
                path = priors/f'{model}-prior{scale:g}-{digest(prior_data)[:12]}.tsv'
                if not path.exists():
                    prior_predictive(prior_data, model, path, a.seed)
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
        atomic_json(work/f'{a.phase}_status.json', dict(completed=len(records), failed=errors,
            diagnostics_flagged=[r['job'] for r in records if not r['diagnostics']['passed']]))
        if records:
            export = a.collect_to.resolve() if a.collect_to else work/f'{a.phase}_summary'
            collect(records, work, export)
            for path in priors.glob('*.tsv'):
                shutil.copyfile(path, export/('prior_predictive-' + path.name))
        if errors:
            return 1
        flagged = sum(not r['diagnostics']['passed'] for r in records)
        print(f'COMPLETED {len(records)} fits; {flagged} need diagnostic review. No imaging covariates released.', flush=True)
        # Smoke is a plumbing check, explicitly not inferential validation.
        return 2 if flagged and a.phase != 'smoke' else 0


if __name__ == '__main__':
    raise SystemExit(main())
