#!/usr/bin/env python3
"""Summarize completed Gu fits and recovery without promoting them to L3."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from run_gu_norm_learning import write_table, PARTNERS


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def correlation(x, y):
    return float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else None


def summarize(root, figures=True):
    completion = json.loads((root / 'completion.json').read_text())
    if completion['status'] != 'completed':
        raise ValueError('fit run is not complete')
    fits = read(root / 'fits.tsv')
    recovery = read(root / 'parameter_recovery.tsv')
    comparison = read(root / 'model_comparison.tsv')
    if len(fits) != completion['fitted_models'] or len(recovery) != completion['simulations']:
        raise ValueError('result counts differ from completion record')
    gu = [r for r in fits if r['model'] == 'rw_free']
    paired = {}
    for r in recovery:
        paired.setdefault((r['subject'], r['replicate']), {})[r['partner']] = r
    difference_recovery = []
    for parameter in ('alpha', 'gamma', 'epsilon', 'f0'):
        true, estimated = [], []
        for pair in paired.values():
            if not {'similar', 'dissimilar'} <= pair.keys():
                raise ValueError('missing partner in recovery pair')
            true.append(float(pair['similar'][parameter + '_true']) - float(pair['dissimilar'][parameter + '_true']))
            estimated.append(float(pair['similar'][parameter + '_estimated']) - float(pair['dissimilar'][parameter + '_estimated']))
        difference_recovery.append(dict(parameter=parameter, contrast='similar-minus-dissimilar',
            simulations=len(true), spearman_r=correlation(true, estimated),
            rmse=float(np.sqrt(np.mean((np.array(estimated) - true)**2)))))
    write_table(root / 'partner_difference_recovery.tsv', difference_recovery)
    report = dict(participants=completion['participants'], partner_fits=len(gu),
        gu_best_nonconverged=sum(r['best_converged'] != '1' for r in gu),
        gu_information_rank_deficient=sum(int(r['fisher_rank']) < 4 for r in gu),
        gu_boundary_fits=sum(bool(r['boundary_parameters']) for r in gu),
        gu_parameters_on_boundary={p: sum(p in r['boundary_parameters'].split('|') for r in gu)
                                   for p in ('alpha', 'gamma', 'epsilon', 'f0')},
        gu_epsilon_near_zero=sum(float(r['epsilon']) < .001 for r in gu),
        gu_alpha_near_upper_bound=sum(float(r['alpha']) > .999 for r in gu),
        recovery_simulations=len(recovery), partner_difference_recovery=difference_recovery,
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in ('fits.tsv', 'parameter_recovery.tsv', 'model_comparison.tsv')},
        summary_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        imaging_covariates_released=False)
    (root / 'diagnostic_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        models = list(dict.fromkeys(r['model'] for r in comparison))
        labels = ['Logistic', 'Static\nfixed\nnorm', 'Static\nfree\nnorm', 'RW\nfixed\nnorm', 'RW\nfree\nnorm', 'RW\nprior\nnorm']
        colors = ['#666666', '#97a6b4', '#7591a8', '#73a69a', '#237c73', '#b89a6a']
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True, sharey='row')
        for col, partner in enumerate(PARTNERS):
            rows = {r['model']: r for r in comparison if r['partner'] == partner}
            for row, field, label in [(0, 'median_delta_bic', 'Median BIC above best individual model'),
                                      (1, 'total_heldout_nll', 'Run-2 log loss per choice')]:
                values = [float(rows[m][field]) for m in models]
                if row:
                    values = [v / int(rows[m]['heldout_choices']) for m, v in zip(models, values)]
                ax = axes[row, col]
                ax.bar(range(len(models)), values, color=colors)
                ax.set_xticks(range(len(models)), labels, fontsize=8)
                ax.set_ylabel(label if col == 0 else '')
                if row == 0:
                    ax.set_title(partner.capitalize())
                ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle('Norm-learning model comparison (47 participants; lower is better)\n'
                     'Unregularized logistic fits can separate and give unstable held-out predictions.', fontsize=12)
        fig.savefig(root / 'model_comparison.png', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
        titles = {'alpha': 'Norm-violation aversion', 'gamma': 'Inverse temperature',
                  'epsilon': 'Learning rate', 'f0': 'Initial norm ($)'}
        for ax, parameter in zip(axes.flat, titles):
            x = np.array([float(r[parameter + '_true']) for r in recovery])
            y = np.array([float(r[parameter + '_estimated']) for r in recovery])
            rho = correlation(x, y)
            ax.scatter(x, y, s=10, alpha=.35, color='#237c73', edgecolors='none')
            limit = 20 if parameter == 'f0' else 1
            ax.plot([0, limit], [0, limit], '--', color='gray', lw=1)
            ax.set(xlim=(-.03 * limit, 1.03 * limit), ylim=(-.03 * limit, 1.03 * limit), xlabel='Generating value', ylabel='Recovered estimate',
                   title=f'{titles[parameter]}: Spearman r = {rho:.2f}' if rho is not None else titles[parameter] + ': undefined r')
            ax.spines[['top', 'right']].set_visible(False)
            ax.title.set_fontsize(11)
        fig.suptitle(f'RW free-norm parameter recovery ({len(recovery)} simulations)')
        fig.savefig(root / 'parameter_recovery.png', dpi=180)
        plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--no-figures', action='store_true')
    args = p.parse_args()
    summarize(args.results, not args.no_figures)
