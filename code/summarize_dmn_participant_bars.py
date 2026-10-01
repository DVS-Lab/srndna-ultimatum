#!/usr/bin/env python3
"""Descriptive weighted participant means/SEMs in the selected DMN cluster.

This is not voxelwise FLAME. Inverse cluster-mean VARCOPE is a precision proxy,
not the inverse variance of a spatial ROI mean (voxel covariance is unavailable).
Nuisance slopes are fitted per condition. SEM describes adjusted participant
dispersion conditional on those slopes, not uncertainty of a new independent
ROI test. The primary FLAME display is preserved for author comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_corrected_dmn_influence import align_rows, read_rows
from audit_l1_designs import read_vest_matrix, write_tsv
from prepare_dmn_condition_bar_models import parse_inputs, SOURCE_FSF_RELATIVE
from prepare_dmn_revision_checks import digest


def weighted_summary(y, w):
    y, w = np.asarray(y, float), np.asarray(w, float)
    if len(y)<2 or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('invalid participant values or precision weights')
    w = w / w.sum()
    mean = float(w @ y)
    neff = float(1/(w@w))
    variance = float(w @ (y-mean)**2 / (1-w@w))
    return mean, float(np.sqrt(variance/neff)), neff


def summarize(repo):
    source = repo/'results/manuscript/source_data/figure4_dmn_condition_input_roi.tsv'
    design = repo/SOURCE_FSF_RELATIVE
    inputs = parse_inputs(design.read_text().splitlines())
    x = read_vest_matrix(design.with_suffix('.mat'))
    if x.shape != (47,6) or np.linalg.matrix_rank(x)!=6:
        raise ValueError('unexpected group design')
    rows = read_rows(source)
    results, participants = [], []
    for condition in ('similar','dissimilar'):
        ordered = align_rows([r for r in rows if r['condition']==condition], inputs)
        if any(r['age_group'] != ('younger' if x[i,0]==1 else 'older') for i,r in enumerate(ordered)):
            raise ValueError('age group mismatch')
        y = np.array([float(r['cluster_mean_cope']) for r in ordered])
        v = np.array([float(r['cluster_mean_varcope']) for r in ordered])
        if np.any(v<=0) or not np.isfinite(v).all() or not np.isfinite(y).all():
            raise ValueError('invalid COPE or VARCOPE')
        w = 1/v
        weighted_x = x*np.sqrt(w[:,None])
        if np.linalg.matrix_rank(weighted_x)!=6:
            raise ValueError('weighted design lost rank')
        beta = np.linalg.lstsq(weighted_x, y*np.sqrt(w), rcond=None)[0]
        adjusted = y-x[:,2:]@beta[2:]
        for group in ('younger','older'):
            index = 0 if group=='younger' else 1
            select = x[:,index]==1
            mean, sem, neff = weighted_summary(adjusted[select], w[select])
            if not np.isclose(mean,beta[index], atol=1e-7):
                raise ValueError('weighted adjusted mean does not equal group coefficient')
            results.append(dict(age_group=group, condition=condition, n=int(select.sum()),
                                weighted_adjusted_mean=mean, conditional_weighted_participant_sem=sem,
                                effective_n=neff, reference_covariates='all_zero_in_corrected_design'))
        for i,row in enumerate(ordered):
            participants.append(dict(participant=row['participant'], age_group=row['age_group'], condition=condition,
                                     original_roi_cope=y[i], adjusted_roi_cope=adjusted[i], precision_proxy=w[i]))
    output=repo/'results/reviewer/dmn_participant_bars'
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output/'summary.tsv',results)
    write_tsv(output/'participants.tsv',participants)
    (output/'provenance.json').write_text(json.dumps({
        'sources':{str(p.relative_to(repo)):digest(p) for p in (source,design,design.with_suffix('.mat'),Path(__file__))},
        'adjustment':'Separate condition-specific WLS, corrected six-column group design, inverse mean VARCOPE weights.',
        'sem_formula':'w normalized to sum 1; n_eff=1/sum(w^2); s2=sum(w*(adjusted_y-weighted_mean)^2)/(1-sum(w^2)); SEM=sqrt(s2/n_eff)',
        'limits':__doc__, 'primary_flame_figure_replaced':False
    },indent=2)+'\n')
    print(f'PASS: descriptive weighted participant means and SEMs: {output}')


def plot_candidate(repo):
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/srndna-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = read_rows(repo/'results/reviewer/dmn_participant_bars/summary.tsv')
    lookup = {(r['age_group'],r['condition']):r for r in rows}
    fig, ax = plt.subplots(figsize=(8.5,5))
    cells=[('younger','similar'),('younger','dissimilar'),('older','similar'),('older','dissimilar')]
    x=[0,1,2.5,3.5]
    for position, key in zip(x,cells):
        row=lookup[key]
        ax.bar(position,float(row['weighted_adjusted_mean']),width=.7,
               yerr=float(row['conditional_weighted_participant_sem']),capsize=5,
               color='#28666E' if key[1]=='similar' else '#B55239')
    ax.set_xticks(x,['Similar\nYoung partner','Dissimilar\nOlder partner',
                     'Similar\nOlder partner','Dissimilar\nYoung partner'])
    ax.axhline(0,color='#777777',linewidth=.8)
    ax.set_ylabel('Covariate-adjusted, precision-weighted ROI estimate')
    ax.set_title('Corrected DMN cluster: descriptive participant means ± SEM')
    ax.spines[['top','right']].set_visible(False)
    ax.text(.22,-.18,'Younger participants (n=25)',transform=ax.transAxes,ha='center')
    ax.text(.78,-.18,'Older participants (n=22)',transform=ax.transAxes,ha='center')
    fig.text(.5,.015,'Weights: inverse mean VARCOPE. Conditional weighted participant SEM; not voxelwise FLAME inference.',ha='center',fontsize=8)
    fig.subplots_adjust(bottom=.25,top=.9)
    output=repo/'results/reviewer/figures/dmn_weighted_participant_sem.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=250,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f'PASS: candidate participant-SEM plot (primary figure preserved): {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--plot',action='store_true')
    a=p.parse_args()
    summarize(a.repository.resolve())
    if a.plot:
        plot_candidate(a.repository.resolve())
