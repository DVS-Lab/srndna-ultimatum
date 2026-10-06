#!/usr/bin/env python3
"""One bounded, resumable arithmetic-audit and bias-model sampling pilot.

Unchanged scientific model: no new parameters, priors, exclusions, or imaging EVs.
Exit zero means the workflow exported its evidence, NOT that inference is valid.
"""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import traceback

import run_gu_choice_extensions as ext
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'rw_positive_bias'


def reference_logits(theta, offers):
    """Independent scalar, extended-precision evaluation; no Stan changes."""
    alpha, gamma, norm, epsilon, bias = map(np.longdouble, theta)
    out = []
    for value in offers:
        offer = np.longdouble(value)
        norm += epsilon * (offer - norm)
        penalty = alpha * (norm - offer) if norm > offer else np.longdouble(0)
        out.append(bias + gamma * (offer - penalty))
    return np.array(out, dtype=np.longdouble)


def arithmetic_audit():
    rng = np.random.default_rng(20261006)
    max_error = 0.
    for _ in range(1000):
        theta = [np.exp(rng.uniform(-8, 8)), np.exp(rng.uniform(-8, 8)),
                 rng.uniform(0, 20), rng.uniform(0, 1), rng.uniform(-20, 20)]
        offers = rng.integers(0, 11, 48)
        actual = ext.logits(theta, offers, MODEL)
        expected = np.asarray(reference_logits(theta, offers), dtype=float)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-7)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    # The initial norm equals each offer: penalty must be zero, with either bias sign.
    for bias in (-30., 30.):
        np.testing.assert_allclose(ext.logits([1e100, 2., 5., .3, bias], [5.]*48, MODEL),
                                   [bias + 10.]*48)
    # Boundary stress is deliberately outside realistic fitted values. It diagnoses
    # floating-point risk, not the origin/timing of the recorded console warnings.
    stress = []
    for label, theta in (
        ('very_large_finite', [1e150, 1e150, 20., .001, 0.]),
        ('overflow_product', [1e300, 1e300, 20., .001, 0.]),
        ('overflow_transform_zero_penalty', [float('inf'), 1., 0., .2, 0.]),
    ):
        with np.errstate(over='ignore', invalid='ignore'):
            z = ext.logits(theta, [1., 10., 1.], MODEL)
        stress.append(dict(case=label, nan_logits=int(np.isnan(z).sum()),
                           infinite_logits=int(np.isinf(z).sum())))
    return dict(random_histories_checked=1000, trials_per_history=48,
        max_absolute_logit_difference=max_error, equivalence_passed=True,
        both_bias_signs_checked=True, boundary_stress=stress,
        note='Stress failures are hypothetical numerical risks, not attribution of observed warnings. '
             'The parameter-dependent max() kink is unchanged. No likelihood modification is made.')


def baseline_records(work, units):
    records = []
    for stage in ('full', 'run1'):
        job = work/'fits'/f'fit-{MODEL}-{stage}-prior1'
        config = json.loads((job/'configuration.json').read_text())
        if config['model'] != MODEL or config['stage'] != stage or config['phase'] != 'fit':
            raise ValueError(f'baseline model/stage mismatch: {job}')
        if config['data_hash'] != ext.digest(ext.build_data(units, MODEL, stage)):
            raise ValueError(f'baseline data/priors differ: {job}')
        if config['sources']['code/stan/gu_choice_extensions.stan'] != ext.sha(ext.STAN):
            raise ValueError('baseline Stan model differs; this pilot requires unchanged equations')
        record = ext.completed(job, ext.digest(config))
        if record is None:
            raise ValueError(f'baseline incomplete: {job}')
        records.append((config, record))
    return records


def sampling_command(work, output, cmdstan, execute):
    command = [sys.executable, '-u', str(ROOT/'code/run_gu_choice_extensions.py'),
        '--work-root', str(work), '--collect-to', str(output), '--cmdstan', str(cmdstan),
        '--phase', 'fit', '--models', MODEL, '--stages', 'full', 'run1',
        '--jobs', '40', '--chains', '4', '--threads-per-chain', '5',
        '--warmup', '4000', '--samples', '2000', '--adapt-delta', '.995',
        '--max-treedepth', '12', '--seed', '20261006']
    return command + (['--execute'] if execute else [])


def run_logged(command, path, accepted=(0,)):
    print('RUN: ' + ' '.join(command), flush=True)
    with path.open('a') as log:
        log.write('\nCOMMAND: ' + ' '.join(command) + '\n'); log.flush()
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            status = proc.wait()
        log.write(f'EXIT_STATUS: {status}\n')
    if status not in accepted:
        raise RuntimeError(f'exit {status}; see {path}')
    return status


def comparison(records, work):
    rows = []
    for old_config, old_record in records:
        job = work/'fits'/old_record['job']
        config = json.loads((job/'configuration.json').read_text())
        record = ext.completed(job, ext.digest(config))
        if record is None or config['data_hash'] != old_config['data_hash']:
            raise ValueError('pilot completion missing or data changed')
        for label, c, r in [('baseline', old_config, old_record), ('pilot', config, record)]:
            d = r['diagnostics']; n = c['chains'] * c['samples']
            rows.append(dict(batch=label, stage=c['stage'], retained_draws=n,
                warmup=c['warmup'], adapt_delta=c['adapt_delta'], seed=c['seed'],
                divergences=d['divergences'], divergence_rate=d['divergences']/n,
                max_depth_hits=d['max_depth_hits'], depth_hit_rate=d['max_depth_hits']/n,
                max_rhat=d['max_rhat'], min_bulk_ess=d['min_bulk_ess'],
                min_tail_ess=d['min_tail_ess'], min_ebfmi=d['min_ebfmi'],
                inference_eligible=r['inference_eligible']))
    return rows


def validate_paths(work, baseline, output):
    def overlaps(a, b):
        return a == b or a in b.parents or b in a.parents
    if (overlaps(work, ROOT) or overlaps(work, baseline) or overlaps(work, output)
            or overlaps(baseline, output)):
        raise ValueError('pilot scratch must be separate from repository, baseline, and exports')
    if ROOT/'results/norm_learning' not in output.parents:
        raise ValueError('export under repository results/norm_learning')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path,
        default=Path('/ZPOOL/data/scratch/srndna-gu-stan-positive-bias-v2'))
    parser.add_argument('--work-root', type=Path,
        default=Path('/ZPOOL/data/scratch/srndna-gu-bias-pilot-v1'))
    parser.add_argument('--output-dir', type=Path,
        default=ROOT/'results/norm_learning/stan-bias-pilot-v1')
    parser.add_argument('--cmdstan', type=Path,
        default=Path('/ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0'))
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    work, baseline, output = a.work_root.resolve(), a.baseline_root.resolve(), a.output_dir.resolve()
    validate_paths(work, baseline, output)
    work.mkdir(parents=True, exist_ok=True)
    with (work/'.pilot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('CHECKING: baseline completion hashes, unchanged data/priors and model source', flush=True)
        units, _, _ = ext.load_trials(ROOT)
        records = baseline_records(baseline, units)
        audit = arithmetic_audit()
        command = sampling_command(work, output, a.cmdstan.resolve(), a.execute)
        if not a.execute:
            print(json.dumps(audit, indent=2))
            print('PLAN: ' + ' '.join(command))
            print('Two fits; 2 x 4 chains x 5 threads = 40 CPU threads maximum. No sampling.')
            return 0
        sources = [Path(__file__), ROOT/'code/run_gu_bias_pilot.sh', ROOT/'code/run_gu_choice_extensions.py', ext.STAN,
                   ROOT/'code/export_gu_stan_diagnostics.py', ROOT/'code/audit_gu_divergences.py']
        guard = dict(command=command, baseline=str(baseline),
                     sources={str(p.relative_to(ROOT)):ext.sha(p) for p in sources},
                     baseline_configs=[ext.digest(c) for c,_ in records])
        ext.config_guard(work/'workflow', guard)
        output.mkdir(parents=True, exist_ok=True)
        ext.atomic_json(output/'arithmetic_audit.json', audit)
        ext.atomic_json(output/'workflow_configuration.json', guard)
        try:
            fit_status = run_logged(command, output/'run.txt', accepted=(0, 2))
            for script, folder in [('export_gu_stan_diagnostics.py','diagnostics-v1'),
                                   ('audit_gu_divergences.py','divergence-audit-v1')]:
                run_logged([sys.executable, '-u', str(ROOT/'code'/script),
                    '--work-root', str(work), '--output-dir', str(output/folder),
                    '--expected-jobs', '2'], output/'export.txt')
            rows = comparison(records, work)
            ext.write_table(output/'baseline_pilot_comparison.tsv', rows)
            flagged = [r['stage'] for r in rows if r['batch']=='pilot' and not r['inference_eligible']]
            ext.atomic_json(output/'workflow_status.json', dict(workflow_complete=True,
                sampling_exit_status=fit_status, diagnostic_review_required=flagged,
                imaging_covariates_released=False, parameter_recovery_validated=False))
            (output/'README.txt').write_text(
                'Bounded adaptation pilot, not a new scientific model.\n'
                'All participants, priors, likelihoods and partner histories unchanged.\n'
                '4 chains per fit: 4000 warmup + 2000 samples, adapt_delta .995, depth 12.\n'
                'Seed and thread count also differ from baseline; this is not a controlled attribution of improvement.\n'
                'Compare divergence RATES because sample counts differ.\n'
                'Workflow completion is not inference eligibility or parameter recovery.\n'
                'No imaging covariates, exclusions, automatic recovery fits, or smoothing of the max() term.\n'
                'Completed fits are verified/skipped; interrupted fits restart in new attempts.\n')
            print(f'WORKFLOW_COMPLETE: {output}; diagnostic review required: {flagged}', flush=True)
            return 0
        except Exception as exc:
            ext.atomic_json(output/'workflow_status.json', dict(workflow_complete=False, error=str(exc)))
            traceback.print_exc()
            print('Stopped. Push the partial logs for diagnosis; no clean-fit claim is made.', flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
