import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from gu_norm_model import predictions
from run_gu_hierarchical import (MODELS, CONTRAST, build_data, logits, draw_prior,
    difference_draws, config_guard, completed, digest, sha, atomic_json, cpu_plan)
from run_gu_norm_learning import load_trials


class HierarchicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, _, _ = load_trials(ROOT)

    def test_data_and_heldout_mask(self):
        full = build_data(self.units, 'rw_free')
        train = build_data(self.units, 'rw_free', 'run1')
        self.assertEqual((full['S'], full['U'], full['T']), (47, 141, 48))
        np.testing.assert_equal(full['observed'], full['use_choice'])
        np.testing.assert_equal(train['use_choice'], np.array(full['observed'])*(np.array(full['run']) == 1))
        self.assertEqual(full['offers'], train['offers'])
        self.assertEqual(full['choice'], train['choice'])
        self.assertTrue(all(full['subject'].count(s) == 3 for s in range(1, 48)))

    def test_likelihood_parity(self):
        offers = np.array([2., 8., 4., 3., 9.])
        for model in MODELS:
            if model == 'logistic':
                theta = [.3, .8]
                old = theta
            elif model == 'fs_free':
                theta = [.7, .4, 12.]
                old = theta
            else:
                theta = [.7, .4, 12., .2]  # Stan order differs from old MLE API.
                old = [.7, .4, .2, 12.]
            expected = predictions(old, offers, model)[0]
            np.testing.assert_allclose(logits(np.array(theta), offers, model), expected)
            np.testing.assert_allclose(logits(np.array([theta]*3), offers, model), [expected]*3)

    def test_future_offers_and_missed_offers(self):
        theta = np.array([.7, .4, 12., .2])
        z = logits(theta, [2, 8, 4, 3], 'rw_free')
        z2 = logits(theta, [2, 8, 10, 10], 'rw_free')
        np.testing.assert_equal(z[:2], z2[:2])
        self.assertNotEqual(z[2], logits(theta, [2, 4, 3], 'rw_free')[1])

    def test_contrasts_and_row_order(self):
        np.testing.assert_allclose(CONTRAST[1]-CONTRAST[2], [0, 1])
        np.testing.assert_allclose(.5*(CONTRAST[1]+CONTRAST[2])-CONTRAST[0], [1, 0])
        units = list(reversed(self.units[:6]))
        data = build_data(units, 'rw_free')
        theta = np.array([[p]*4 for p in data['partner']], float)
        diff = difference_draws(theta, data)
        np.testing.assert_equal(diff['similar_minus_dissimilar'], -np.ones((2, 4)))
        np.testing.assert_equal(diff['human_minus_computer'], np.full((2, 4), 1.5))

    def test_prior_bounds_and_determinism(self):
        for model in MODELS:
            data = build_data(self.units, model)
            a = draw_prior(data, model, np.random.default_rng(123))
            b = draw_prior(data, model, np.random.default_rng(123))
            np.testing.assert_equal(a, b)
            if model == 'logistic':
                self.assertTrue((np.abs(a) < [30, 10]).all())
            else:
                self.assertTrue((a > 0).all())
                self.assertTrue((a[:, 2] < 20).all())
                self.assertTrue((a[:, [0, 1]] < 1).all())

    def test_cpu_cap(self):
        self.assertEqual(cpu_plan(40, 4, 2), 5)
        self.assertEqual(cpu_plan(40, 4, 1), 10)
        with self.assertRaises(ValueError):
            cpu_plan(4, 4, 2)

    def test_configuration_and_completed_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'job'
            cfg = {'data': 'abc'}
            config_guard(out, cfg)
            config_guard(out, cfg)
            with self.assertRaises(ValueError):
                config_guard(out, {'data': 'changed'})
            self.assertIsNone(completed(out, digest(cfg)))
            csv = out/'chain.csv'
            csv.write_text('complete output')
            record = {'config_hash': digest(cfg), 'files': {'chain.csv': sha(csv)}}
            atomic_json(out/'completed.json', record)
            self.assertEqual(completed(out, digest(cfg)), record)
            csv.write_text('truncated')
            with self.assertRaises(ValueError):
                completed(out, digest(cfg))


if __name__ == '__main__':
    unittest.main()
