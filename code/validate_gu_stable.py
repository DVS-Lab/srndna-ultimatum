#!/usr/bin/env python3
"""Compiled likelihood/gradient equivalence and overflow regression checks."""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import run_gu_choice_extensions as ext

ROOT = ext.ROOT
STAN_DIR = ROOT/'code/stan'


def compile_model(name, work):
    import cmdstanpy
    paths = [STAN_DIR/f'{name}.stan']
    if 'stable' in name:
        paths.append(STAN_DIR/'gu_stable_functions.stan')
    fingerprint = ext.digest({p.name: ext.sha(p) for p in paths})
    build = work/'build'/fingerprint[:16]
    build.mkdir(parents=True, exist_ok=True)
    for source in paths:
        target = build/source.name
        if target.exists() and ext.sha(target) != ext.sha(source):
            raise ValueError(f'changed cached source: {target}')
        if not target.exists():
            shutil.copyfile(source, target)
    return cmdstanpy.CmdStanModel(stan_file=str(build/paths[0].name),
                                  cpp_options={'STAN_THREADS': 'true'})


def reference_kernel(params, offer, y):
    """Independent extended-precision scalar formula and analytic gradient."""
    a, g, n, b = [np.longdouble(params[k]) for k in ('log_alpha','log_gamma','norm','bias')]
    offer = np.longdouble(offer)
    positive = np.exp(g)*offer if offer > 0 else np.longdouble(0)
    penalty = np.exp(a+g)*(n-offer) if n > offer else np.longdouble(0)
    eta = b+(positive-penalty)
    ll = -np.logaddexp(0, -eta if y else eta)
    # Compute the score without subtracting a probability rounded to one.
    score = np.exp(-np.logaddexp(0, eta)) if y else -np.exp(-np.logaddexp(0,-eta))
    grad = score*np.array([-penalty, positive-penalty,
        -np.exp(a+g) if n>offer else 0, 1], dtype=np.longdouble)
    return np.array([ll, *grad], dtype=float)


def validate(stable, original, harness, units, work):
    ext.validate_compiled(stable, units, work)
    rng = np.random.default_rng(20261006)
    comparisons, maximum = 0, 0.
    for model in ext.MODELS:
        for stage in ('full', 'run1'):
            data = ext.build_data(units[:6], model, stage)
            for case in range(6):
                init = ext.validation_initialization(data)
                init['mu'] = rng.uniform(-2, 2, 4).tolist()
                init['bias_mu'] = [(-6., 0., 6.)[case % 3]]*data['B']
                if case == 5:
                    init['mu'] = [8., -5., 1., -8.]
                old = original.log_prob(data=data, params=init, sig_figs=18)
                new = stable.log_prob(data=data, params=init, sig_figs=18)
                if list(old.columns) != list(new.columns):
                    raise ValueError('parameter/gradient ordering differs')
                np.testing.assert_allclose(new.to_numpy(), old.to_numpy(), rtol=2e-8, atol=2e-8)
                maximum = max(maximum, float(np.max(np.abs(new.to_numpy()-old.to_numpy()))))
                comparisons += 1
    # Include both choice outcomes, both bias signs, cancellation, extreme
    # but finite combined effects, and saturation with representable gradients.
    cases = [
        ('ordinary', .7, -.2, 8., 4., -3.),
        ('zero_bias_cancellation', 0., 0., 2., 1., 0.),
        ('zero_offer', 1., 2., 5., 0., 2.),
        ('no_deficit_overflow_alpha', 1000., 0., 0., 1., -2.),
        ('compensating_log_scales', 1000., -1000., 2., 1., 0.),
        ('zero_offer_large_gamma', -1000., 1000., 2., 0., 3.),
        ('large_finite', 1., 300., 8., 4., 3.),
        ('huge_cancellation', 0., 705., 2., 1., 0.),
        ('huge_cancellation_positive_bias', 0., 705., 2., 1., 3.),
        ('huge_cancellation_negative_bias', 0., 705., 2., 1., -3.),
    ]
    rows = []
    for label, a, g, n, offer, bias in cases:
        for y in (0, 1):
            params = dict(log_alpha=a, log_gamma=g, norm=n, bias=bias)
            result = harness.log_prob(data=dict(y=y,offer=offer), params=params, sig_figs=18)
            actual = result.to_numpy()[0]
            expected = reference_kernel(params, offer, y)
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
            rows.append(dict(case=label, choice=y, passed=True))
    return dict(passed=True, hierarchy_likelihood_gradient_comparisons=comparisons,
        max_absolute_target_or_gradient_difference=maximum, kernel_checks=rows,
        equations_priors_and_hinge_unchanged=True,
        caveat='Arithmetic equivalence is not evidence of convergence or parameter recovery.')
