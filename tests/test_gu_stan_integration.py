"""Optional compiled-Stan checks; set SRNDNA_GU_STAN_EXE and CMDSTAN.

Fixed-parameter draws validate equations without performing scientific fitting.
"""
import os
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from run_gu_hierarchical import MODELS, build_data, logits
from run_gu_norm_learning import load_trials


@unittest.skipUnless(os.environ.get('SRNDNA_GU_STAN_EXE'), 'compiled Stan executable not supplied')
class StanIntegrationTests(unittest.TestCase):
    def test_all_models_against_python_and_holdout(self):
        import cmdstanpy
        model = cmdstanpy.CmdStanModel(exe_file=os.environ['SRNDNA_GU_STAN_EXE'])
        units = load_trials(ROOT)[0][:6]
        for name in MODELS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as tmp:
                data = build_data(units, name, 'run1')
                k, s = data['K'], data['S']
                init = dict(mu=np.linspace(-.4, .3, k).tolist(),
                    effect=np.full((2, k), .2).tolist(), sigma_subject=[.4]*k,
                    sigma_contrast=np.full((2,k), .3).tolist(),
                    z_subject=np.full((s,k), .1).tolist(),
                    z_contrast=np.full((2,s,k), -.1).tolist())
                fit = model.sample(data=data, inits=init, chains=1, iter_warmup=0,
                    iter_sampling=2, fixed_param=True, adapt_engaged=False, seed=2, output_dir=tmp,
                    show_progress=False, sig_figs=16)
                theta = fit.stan_variable('theta')[0]
                z = logits(theta, data['offers'], name)
                ll = np.array(data['choice'])*z - np.logaddexp(0, z)
                train = np.array(data['use_choice'], bool)
                test = np.array(data['observed'], bool) & ~train
                np.testing.assert_allclose(fit.stan_variable('train_log_lik')[0],
                    np.where(train, ll, 0).sum(1), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(fit.stan_variable('heldout_log_lik')[0],
                    np.where(test, ll, 0).sum(1), rtol=1e-12, atol=1e-12)
                self.assertTrue(np.isfinite(fit.stan_variable('expected_accept')).all())
                # Check the actual reduce_sum target, not just generated quantities.
                prior_only = copy.deepcopy(data)
                prior_only['use_choice'] = np.zeros_like(data['use_choice']).tolist()
                posterior_lp = model.log_prob(params=init, data=data, sig_figs=16)['lp__'].iloc[0]
                prior_lp = model.log_prob(params=init, data=prior_only, sig_figs=16)['lp__'].iloc[0]
                self.assertAlmostEqual(posterior_lp-prior_lp, np.where(train, ll, 0).sum(), places=8)


if __name__ == '__main__':
    unittest.main()
