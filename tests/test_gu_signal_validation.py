"""Scientific identities, provenance guards, and Linux launcher contracts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from audit_gu_trial_signals import (audit_signals, compare_signals, correlation, execute_audit,
    read_job, recovery_metrics, source_jobs, trajectories)
from export_gu_stan_diagnostics import sha
from run_gu_hierarchical import build_data, digest, logits, PARTNERS
from run_gu_targeted_recovery import truth_draws


def units_fixture():
    units = []
    for s in range(3):
        for partner in PARTNERS:
            offers = [float((t+s) % 11) for t in range(48)]
            runs = [1]*24+[2]*24
            choices = [t % 2 for t in range(48)]
            choices[0] = np.nan
            trials = [dict(run=runs[t], trial_in_run=t % 24+1, missed=int(t == 0)) for t in range(48)]
            units.append(dict(subject=f'sub-{s}', age_group='younger' if s < 2 else 'older',
                partner=partner, offers=offers, runs=runs, choices=choices, trials=trials))
    return units


class SignalsTests(unittest.TestCase):
    def test_drawwise_trajectory_and_update_convention(self):
        theta = np.array([[.9, .8, 8., .1], [.7, .6, 4., .8]])
        offers = np.array([2., 9., 3., 7.])
        signals = trajectories(theta, offers)
        np.testing.assert_allclose(signals['expectation_before_offer'][:, 0], theta[:, 2])
        np.testing.assert_allclose(signals['signed_prediction_error'] + signals['expectation_before_offer'],
                                   np.broadcast_to(offers, (2, 4)))
        np.testing.assert_allclose(signals['updated_norm'][:, :-1], signals['expectation_before_offer'][:, 1:])
        expected_logits = theta[:, 1, None]*(offers-theta[:, 0, None]*np.maximum(signals['updated_norm']-offers, 0))
        np.testing.assert_allclose(logits(theta, offers, 'rw_free'), expected_logits)
        self.assertFalse(np.allclose(signals['updated_norm'].mean(0), trajectories(theta.mean(0), offers)['updated_norm']))

    def test_missed_offer_updates_but_is_excluded_from_correlations(self):
        units = units_fixture()
        theta = np.broadcast_to([.95, .8, 8., .02], (8, 9, 4)).copy()
        rows, metrics = audit_signals(theta, units, 'test', False)
        self.assertEqual(len(rows), 9*48)
        self.assertEqual(rows[0]['responded'], 0)
        self.assertAlmostEqual(rows[1]['expectation_before_offer_mean'], 8+.02*(0-8))
        self.assertEqual(metrics[0]['n_responded'], 23)
        self.assertFalse(rows[0]['diagnostics_passed'])
        # Same history carries across the run boundary, rather than resetting f0.
        self.assertAlmostEqual(rows[24]['expectation_before_offer_mean'], rows[23]['updated_norm_mean'])
        other = [dict(r, fit='other') for r in rows]
        comparisons = compare_signals(rows+other)
        self.assertEqual(len(comparisons), 9*2*3)
        self.assertTrue(all(abs(r['rmse']) < 1e-12 for r in comparisons))
        with self.assertRaises(ValueError):
            compare_signals(rows+other+[other[0]])

    def test_true_participant_average_and_contrast_recovery(self):
        units = units_fixture()
        data = build_data(units, 'rw_free')
        rng = np.random.default_rng(3)
        truth = rng.uniform(.1, .9, (9, 4)); truth[:, 2] *= 20
        theta = np.broadcast_to(truth, (20, 9, 4))
        metrics = recovery_metrics(theta, truth, data)
        self.assertEqual(len(metrics), 16)
        self.assertTrue(all(abs(r['spearman_r']-1) < 1e-12 for r in metrics))
        self.assertTrue(all(r['coverage_95'] == 1 for r in metrics))
        avg = next(r for r in metrics if r['target'] == 'participant_average')
        self.assertEqual(avg['n'], 3)
        reorder = [5, 0, 8, 1, 7, 2, 4, 6, 3]
        permuted = dict(data, subject=[data['subject'][i] for i in reorder], partner=[data['partner'][i] for i in reorder])
        second = recovery_metrics(theta[:, reorder], truth[reorder], permuted)
        for a, b in zip(metrics, second):
            self.assertAlmostEqual(a['rmse'], b['rmse'])

    def test_truth_selection_is_joint_balanced_reproducible(self):
        chains = [(i, np.arange(60).reshape(10, 3, 2)+i*100) for i in (1, 2, 3, 4)]
        selected = list(truth_draws(chains, 7))
        self.assertEqual(len(selected), 8)
        for chain, half, draw, truth in selected:
            self.assertEqual(half, 1 if draw < 5 else 2)
            np.testing.assert_array_equal(truth, chains[chain-1][1][draw])
        repeated = list(truth_draws(chains, 7))
        self.assertEqual([r[:3] for r in selected], [r[:3] for r in repeated])

    def test_constants_not_reported_as_correlated(self):
        self.assertIsNone(correlation([1, 1, 1], [2, 3, 4]))
        self.assertIsNone(correlation([], []))
        self.assertIsNone(correlation([1, 2], [2, 3]))

    def test_default_source_names(self):
        paths = dict(source_jobs(Path('/tmp/test')))
        self.assertEqual(paths['repeat_prior0.5'].name, 'fit-rw_free-full-prior0.5')


class ProvenanceTests(unittest.TestCase):
    def fixture(self, root):
        job = root/'fit-rw_free-full-prior1'; job.mkdir()
        attempt = job/'attempt-test'; attempt.mkdir()
        units = units_fixture(); data = build_data(units, 'rw_free')
        config = dict(model='rw_free', stage='full', age_mode='none', phase='fit',
            parameters=['alpha', 'gamma', 'f0', 'epsilon'], subjects=['sub-0', 'sub-1', 'sub-2'],
            prior_scale=1., chains=4, samples=2, data_hash=digest(data))
        (job/'configuration.json').write_text(json.dumps(config))
        (attempt/'data.json').write_text(json.dumps(data))
        (attempt/'diagnostics.json').write_text(json.dumps({'passed': False}))
        columns = [f'theta.{u+1}.{k+1}' for u in range(9) for k in range(4)]
        row = ','.join(str(x) for _ in range(9) for x in [.95, .8, 8., .02])
        for c in range(1, 5):
            (attempt/f'chain-{c}.csv').write_text(f'# id = {c}\n# save_warmup = 0\n'+','.join(columns)+'\n'+row+'\n'+row+'\n')
        record = dict(job=job.name, config_hash=digest(config), attempt=attempt.name,
            diagnostics={'passed': False}, files={str(p.relative_to(job)): sha(p) for p in attempt.iterdir()})
        (job/'completed.json').write_text(json.dumps(record))
        return job, units

    def test_reads_flagged_fit_and_checks_all_chain_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            job, units = self.fixture(Path(d))
            item = read_job(job, units)
            self.assertEqual(len(item['chains']), 4)
            self.assertFalse(item['record']['diagnostics']['passed'])
            path = job/'attempt-test/chain-2.csv'
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'modified completed input'):
                read_job(job, units)

    def test_trial_order_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as d:
            job, units = self.fixture(Path(d))
            units[0]['offers'] = list(reversed(units[0]['offers']))
            with self.assertRaisesRegex(ValueError, 'trial identity/order'):
                read_job(job, units, draws=False)

    def test_signal_export_end_to_end_and_changed_input_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); jobs = []
            for i in range(4):
                folder = root/str(i); folder.mkdir()
                job, units = self.fixture(folder)
                jobs.append((f'fit{i}', job))
            output = root/'audit'
            with patch('audit_gu_trial_signals.load_trials', return_value=(units, [], {})), \
                    patch('audit_gu_trial_signals.source_jobs', return_value=jobs):
                execute_audit('signals', root, output)
                provenance = json.loads((output/'provenance.json').read_text())
                self.assertFalse(provenance['imaging_covariates_released'])
                self.assertEqual(len(provenance['outputs']), 4)
                self.assertEqual(len((output/'trial_signals.tsv').read_text().splitlines()), 4*9*48+1)
                execute_audit('signals', root, output)
                cp = jobs[0][1]/'completed.json'
                cp.write_text(cp.read_text()+'\n')
                with self.assertRaisesRegex(ValueError, 'audit inputs changed'):
                    execute_audit('signals', root, output)

    def test_targeted_runner_builds_16_reproducible_jobs(self):
        import types
        import run_gu_targeted_recovery as runner
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cmdstan = root/'cmdstan'; cmdstan.mkdir()
            (cmdstan/'makefile').write_text('CMDSTAN_VERSION := 2.40.0\n')
            units = units_fixture()
            theta = np.broadcast_to([.95, .8, 8., .02], (10, 9, 4)).copy()
            item = dict(chains=[(i, theta) for i in range(1, 5)],
                        record={'diagnostics': {'passed': False}}, source_hashes={})
            fake = types.SimpleNamespace(__version__='1.3.0', set_cmdstan_path=lambda p: None,
                CmdStanModel=lambda **kw: types.SimpleNamespace(exe_file='not-executed'))
            captured = []
            def fit(job, *_):
                captured.append(job)
                return dict(job=job[0].name, diagnostics={'passed': False})
            def collect(records, work, output):
                output.mkdir(parents=True, exist_ok=True)
                (output/'provenance.json').write_text('{}')
            args = ['run_gu_targeted_recovery.py', '--scratch-base', str(root/'scratch'),
                    '--cmdstan', str(cmdstan), '--collect-to', str(root/'export'), '--execute']
            with patch.object(sys, 'argv', args), patch.dict(sys.modules, {'cmdstanpy': fake}), \
                    patch.object(runner, 'load_trials', return_value=(units, [], {})), \
                    patch.object(runner, 'read_job', return_value=item), \
                    patch.object(runner, 'fit_job', side_effect=fit), patch.object(runner, 'collect', side_effect=collect):
                self.assertEqual(runner.main(), 2)
                self.assertEqual(len(captured), 16)
                configs = {j[0].name: j[2] for j in captured}
                self.assertEqual(len({c['seed'] for c in configs.values()}), 16)
                for directory, data, config, truth in captured:
                    self.assertEqual(config['data_hash'], digest(data))
                    self.assertEqual(config['truth_hash'], digest(truth.tolist()))
                    self.assertFalse(config['truth_source']['diagnostics_passed'])
                    self.assertEqual(data['observed'], build_data(units, 'rw_free')['observed'])
                captured.clear()
                self.assertEqual(runner.main(), 2)
                self.assertEqual(configs, {j[0].name: j[2] for j in captured})


class WrapperTests(unittest.TestCase):
    def invoke(self, mode, execute=True, status=0):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'code').mkdir(); (root/'bin').mkdir()
            shutil.copyfile(ROOT/'code/run_gu_signal_validation.sh', root/'code/run_gu_signal_validation.sh')
            fake = root/'bin/python'
            fake.write_text('#!/bin/sh\necho "$*" >> "$TEST_CALLS"\n'
                'case "$*" in *run_gu_targeted_recovery*) exit "$TEST_STATUS" ;; *) exit 0 ;; esac\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(root/'bin')+os.pathsep+os.environ['PATH'],
                TEST_CALLS=str(root/'calls'), TEST_STATUS=str(status))
            args = ['bash', str(root/'code/run_gu_signal_validation.sh'), mode]+(['--execute'] if execute else [])
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            calls = (root/'calls').read_text().splitlines() if (root/'calls').exists() else []
            return result.returncode, calls

    def test_targeted_dry_run(self):
        status, calls = self.invoke('targeted', execute=False)
        self.assertEqual(status, 0); self.assertEqual(len(calls), 1)
        self.assertNotIn('--execute', calls[0]); self.assertIn('--jobs 40', calls[0])

    def test_targeted_flags_exported(self):
        status, calls = self.invoke('targeted', status=2)
        self.assertEqual(status, 2); self.assertEqual(len(calls), 3)
        self.assertIn('--expected-jobs 16', calls[1])
        self.assertIn('audit_gu_trial_signals.py targeted', calls[2])

    def test_targeted_failure_stops(self):
        status, calls = self.invoke('targeted', status=1)
        self.assertEqual(status, 1); self.assertEqual(len(calls), 1)

    def test_signals_never_samples(self):
        status, calls = self.invoke('signals')
        self.assertEqual(status, 0); self.assertEqual(len(calls), 1)
        self.assertIn('audit_gu_trial_signals.py signals', calls[0])


if __name__ == '__main__':
    unittest.main()
