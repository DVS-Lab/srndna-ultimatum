"""Gu et al. 2015 norm-adaptation likelihoods, fitted without brain data.

DOI: 10.1523/JNEUROSCI.2906-14.2015, p.469. Monetary inputs remain in dollars.
Gu's printed RW equations use the updated f_i in V_i(s_i). The explicitly
labelled rw_free_prior sensitivity instead uses f_(i-1) for the current choice.
No response is needed to observe an offer and update the norm.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit

SPECS = {
    'logistic': [('intercept_at_5', -30., 30.), ('offer_slope', -10., 10.)],
    'fs_fixed': [('alpha', 0., 1.), ('gamma', 0., 1.)],
    'fs_free': [('alpha', 0., 1.), ('gamma', 0., 1.), ('f0', 0., 20.)],
    'rw_fixed': [('alpha', 0., 1.), ('gamma', 0., 1.), ('epsilon', 0., 1.)],
    'rw_free': [('alpha', 0., 1.), ('gamma', 0., 1.), ('epsilon', 0., 1.), ('f0', 0., 20.)],
    'rw_free_prior': [('alpha', 0., 1.), ('gamma', 0., 1.), ('epsilon', 0., 1.), ('f0', 0., 20.)],
}


def trajectory(offers, epsilon, f0):
    """After-offer norms and analytic derivatives, including both endpoints."""
    offers = np.asarray(offers, dtype=float)
    after, _ = lfilter([epsilon], [1., -(1. - epsilon)], offers, zi=[(1. - epsilon) * f0])
    before = np.r_[f0, after[:-1]]
    d_epsilon = lfilter([1.], [1., -(1. - epsilon)], offers - before)
    d_f0 = (1. - epsilon) ** np.arange(1, len(offers) + 1)
    return before, after, d_epsilon, d_f0


def predictions(theta, offers, model):
    """Return logits, their Jacobian, and before/choice/after norms."""
    offers = np.asarray(offers, dtype=float)
    if model == 'logistic':
        jac = np.column_stack([np.ones(len(offers)), offers - 5.])
        return jac @ theta, jac, np.full((len(offers), 3), np.nan)
    params = {name: value for (name, _, _), value in zip(SPECS[model], theta)}
    alpha, gamma = params['alpha'], params['gamma']
    f0, epsilon = params.get('f0', 10.), params.get('epsilon', 0.)
    before, after, d_eps, d_f0 = trajectory(offers, epsilon, f0)
    if model == 'rw_free_prior':
        norm = before
        d_eps, d_f0 = np.r_[0., d_eps[:-1]], np.r_[1., d_f0[:-1]]
    else:
        norm = after
    shortfall = np.maximum(norm - offers, 0.)
    active = (norm > offers).astype(float)
    utility = offers - alpha * shortfall
    columns = {'alpha': -gamma * shortfall, 'gamma': utility,
               'epsilon': -gamma * alpha * active * d_eps,
               'f0': -gamma * alpha * active * d_f0}
    jac = np.column_stack([columns[name] for name, _, _ in SPECS[model]])
    return gamma * utility, jac, np.column_stack([before, norm, after])


def nll_gradient(theta, offers, choices, model, likelihood_mask=None):
    logits, jac, _ = predictions(theta, offers, model)
    observed = np.isfinite(choices)
    if likelihood_mask is not None:
        observed &= likelihood_mask
    z, y, j = logits[observed], choices[observed], jac[observed]
    return float(np.sum(np.logaddexp(0., z) - y * z)), j.T @ (expit(z) - y)


def fit(offers, choices, model, seed, starts=24, likelihood_mask=None, extra_starts=()):
    offers, choices = np.asarray(offers, float), np.asarray(choices, float)
    if offers.shape != choices.shape or offers.ndim != 1 or not len(offers):
        raise ValueError('offers and choices must be matching nonempty vectors')
    if not np.isfinite(offers).all() or ((offers < 0) | (offers > 10)).any():
        raise ValueError('expected observed disadvantageous/equal offers in [0,10] from a $20 stake')
    if not np.isin(choices[np.isfinite(choices)], [0, 1]).all():
        raise ValueError('choices must be 0=reject, 1=accept, or missing')
    observed = np.isfinite(choices)
    if likelihood_mask is not None:
        observed &= likelihood_mask
    n = int(observed.sum())
    if n <= len(SPECS[model]) + 1:
        raise ValueError('too few observed responses for model comparison')
    bounds = [(lo, hi) for _, lo, hi in SPECS[model]]
    lows, widths = np.array(bounds)[:, 0], np.diff(np.array(bounds), axis=1)[:, 0]
    rng = np.random.default_rng(seed)
    initials = [lows + widths * .5, *[np.asarray(x, float) for x in extra_starts]]
    while len(initials) < starts:
        x = lows + widths * rng.uniform(.02, .98, len(bounds))
        if model == 'logistic':
            x = np.array([rng.uniform(-2, 2), rng.uniform(0, 2)])
        initials.append(x)
    attempts = []
    for x0 in initials:
        result = minimize(nll_gradient, x0, args=(offers, choices, model, likelihood_mask),
                          method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 60})
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            attempts.append(result)
    if not attempts:
        raise RuntimeError('all optimizer starts failed')
    best = min(attempts, key=lambda r: r.fun)
    theta = np.asarray(best.x)
    z, jac, norms = predictions(theta, offers, model)
    probabilities = expit(z)
    fisher_design = jac[observed] * widths * np.sqrt(probabilities[observed] * (1 - probabilities[observed]))[:, None]
    singular = np.linalg.svd(fisher_design, compute_uv=False)
    rank = int((singular > max(singular[0] * 1e-6, 1e-10)).sum())
    near = np.array([r.x for r in attempts if r.fun <= best.fun + .1])
    boundary = ((theta - lows) / widths < .001) | ((lows + widths - theta) / widths < .001)
    k = len(theta)
    record = dict(model=model, n_choices=n, n_parameters=k, nll=float(best.fun),
                  aic=2 * best.fun + 2 * k,
                  aicc=2 * best.fun + 2 * k + 2 * k * (k + 1) / (n - k - 1),
                  bic=2 * best.fun + k * np.log(n),
                  best_converged=int(best.success), converged_starts=sum(r.success for r in attempts),
                  starts=len(initials), best_message=str(best.message),
                  fisher_rank=rank, fisher_condition=float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
                  boundary_parameters='|'.join(s[0] for s, flag in zip(SPECS[model], boundary) if flag),
                  near_optimum_starts=len(near))
    for (name, _, _), value, spread in zip(SPECS[model], theta, np.ptp(near, axis=0)):
        record[name] = float(value)
        record[name + '_near_optimum_span'] = float(spread)
    return record, theta, probabilities, norms


def fit_family(offers, choices, seed, starts=24, likelihood_mask=None):
    """Include static-optimum starts in nested learning models."""
    result = {}
    for index, model in enumerate(SPECS):
        extra = []
        if model == 'rw_fixed':
            extra = [np.r_[result['fs_fixed'][1], 0.]]
        elif model.startswith('rw_free'):
            a, g, f0 = result['fs_free'][1]
            extra = [np.array([a, g, 0., f0])]
        result[model] = fit(offers, choices, model, seed + index, starts, likelihood_mask, extra)
    for learned, static in [('rw_fixed', 'fs_fixed'), ('rw_free', 'fs_free'), ('rw_free_prior', 'fs_free')]:
        if result[learned][0]['nll'] > result[static][0]['nll'] + 1e-6:
            raise RuntimeError('nested learning optimum worse than static starting point')
    return result
