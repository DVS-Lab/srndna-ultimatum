import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from gu_norm_model import SPECS, trajectory, predictions, nll_gradient, fit_family
from run_gu_norm_learning import load_trials, unit_seed


class GuNormTests(unittest.TestCase):
    def test_hand_calculated_history(self):
        before, after, _, _ = trajectory([2, 8, 4], .5, 10)
        np.testing.assert_equal(before, [10, 6, 7])
        np.testing.assert_equal(after, [6, 7, 5.5])
        z, _, _ = predictions([.5, .4, .5, 10], [2, 8, 4], 'rw_free')
        np.testing.assert_allclose(z, [0, 3.2, 1.3])
        z_prior, _, _ = predictions([.5, .4, .5, 10], [2, 8, 4], 'rw_free_prior')
        np.testing.assert_allclose(z_prior, [-.8, 3.2, 1.])

    def test_analytic_gradients_all_models(self):
        offers = np.array([1., 7., 3., 9., 2., 8., 4., 10.])
        y = np.array([0., 1., np.nan, 1., 0., 1., 0., 1.])
        values = {'alpha': .62, 'gamma': .43, 'epsilon': .23, 'f0': 12.4,
                  'intercept_at_5': .3, 'offer_slope': .7}
        for model, spec in SPECS.items():
            theta = np.array([values[name] for name, _, _ in spec])
            with self.subTest(model=model):
                expected = approx_derivative(lambda x: nll_gradient(x, offers, y, model)[0], theta)
                np.testing.assert_allclose(nll_gradient(theta, offers, y, model)[1], expected.ravel(),
                                           atol=2e-6, rtol=2e-6)

    def test_zero_learning_is_static_and_full_learning_removes_shortfall(self):
        offers = np.array([1., 9., 2., 8.])
        static = predictions([.8, .6, 12.], offers, 'fs_free')[0]
        learned = predictions([.8, .6, 0., 12.], offers, 'rw_free')[0]
        np.testing.assert_equal(static, learned)
        np.testing.assert_allclose(predictions([.8, .6, 1., 12.], offers, 'rw_free')[0], offers * .6)

    def test_missed_choices_update_but_do_not_enter_likelihood(self):
        theta = np.array([.8, .6, .4, 12.])
        offers, y = np.array([1., 10., 2.]), np.array([0., np.nan, 0.])
        z = predictions(theta, offers, 'rw_free')[0]
        expected = np.logaddexp(0., z[[0, 2]]).sum()
        self.assertAlmostEqual(nll_gradient(theta, offers, y, 'rw_free')[0], expected)
        dropped = predictions(theta, offers[[0, 2]], 'rw_free')[0]
        self.assertNotAlmostEqual(z[2], dropped[-1])

    def test_future_offers_cannot_affect_training_predictions(self):
        theta = [.8, .6, .4, 12.]
        a = predictions(theta, [1, 8, 2, 9], 'rw_free')[0]
        b = predictions(theta, [1, 8, 10, 1], 'rw_free')[0]
        np.testing.assert_equal(a[:2], b[:2])

    def test_nested_fits_and_deterministic_seeds(self):
        offers = np.tile([1., 2., 3., 8., 9., 10.], 8)
        probabilities = expit(predictions([.8, .6, .2, 14.], offers, 'rw_free')[0])
        y = np.random.default_rng(12).binomial(1, probabilities)
        results = fit_family(offers, y, 12, starts=4)
        self.assertLessEqual(results['rw_free'][0]['nll'], results['fs_free'][0]['nll'] + 1e-6)
        self.assertLessEqual(results['rw_fixed'][0]['nll'], results['fs_fixed'][0]['nll'] + 1e-6)
        self.assertEqual(unit_seed(1, 'sub-104', 'similar'), unit_seed(1, 'sub-104', 'similar'))
        self.assertNotEqual(unit_seed(1, 'sub-104', 'similar'), unit_seed(1, 'sub-104', 'dissimilar'))

    def test_corrected_event_sequences(self):
        units, trials, provenance = load_trials(ROOT)
        self.assertEqual(len(units), 141)
        self.assertEqual(len(trials), 47 * 144)
        self.assertEqual(len(provenance), 95)
        self.assertTrue(all(len(u['offers']) == 48 for u in units))
        for subject in ('sub-143', 'sub-144'):
            self.assertEqual(sum(t['subject'] == subject for t in trials), 144)
        self.assertTrue(any(t['missed'] for t in trials))


if __name__ == '__main__':
    unittest.main()
