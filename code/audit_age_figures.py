#!/usr/bin/env python3
"""Descriptive age figures, anatomical localization, and influence checks.

Reads retained estimates only. No imaging/permutation fits, exclusions, new
thresholds, or changes to the historical figures. Selected-ROI summaries are
descriptive and provide no independent significance tests.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import tempfile
import traceback
import xml.etree.ElementTree as ET

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import stats
from scipy.ndimage import label

import audit_palm_partner_motion as pm
import analyze_corrected_vmpfc as activation_validation
import collect_palm_progress as cp
import plot_palm_age_bars as bars
import plot_permutation_followup as pf

ROOT = Path(__file__).resolve().parents[1]
COLORS = ('#397b9d', '#b86c3e')
SUPPLEMENTAL = (
    ('offer-mean-positive', 'taskwide-offer', 1, 'Task-wide positive offer-size response'),
    ('human-computer-mean', 'human-computer-response', 1, 'Human > computer task response'),
    ('computer-human-mean', 'human-computer-response', 2, 'Computer > human task response'),
    ('offer-age', 'taskwide-offer', 4, 'Task-wide offer-size response: younger > older'),
    ('human-computer-age', 'human-computer-response', 4, 'Human − computer response: younger > older'),
)


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def write(path, rows):
    if not rows:
        raise ValueError(f'empty table: {path}')
    cp.write_rows(path, rows, list(rows[0]))


def record(path, sources):
    path = Path(path)
    sources[str(path)] = cp.digest(path)
    return path


def verified(root, relative, inventory, sources):
    path = root/relative
    expected = inventory.get(relative)
    if expected is None or cp.digest(path) != expected:
        raise ValueError(f'absent or changed inventoried artifact: {path}')
    return record(path, sources)


def inventory(root, filename, sources):
    path = record(root/filename, sources)
    return {r.get('path', r.get('artifact')): r['sha256'] for r in read(path)}


def same(a, b):
    if a.shape[:3] != b.shape[:3] or not np.allclose(a.affine, b.affine, atol=1e-5, rtol=0):
        raise ValueError('image grids differ; no implicit statistical resampling')


def dmn_roi(repo, sources):
    root = repo/'results/dmn_followup/v1/coverage_check/v1'
    inv = inventory(root, 'inventory.tsv', sources)
    prefix = 'maps/palm/cluster-z2.6/'
    stat = nib.load(verified(root, prefix+'palm_vox_zvstat_c3.nii', inv, sources))
    prob = nib.load(verified(root, prefix+'palm_clustere_zvstat_fwep_c3.nii', inv, sources))
    coverage = nib.load(verified(root, 'preparation/mask.nii.gz', inv, sources))
    same(stat, prob); same(stat, coverage)
    selected = (prob.get_fdata() > .95) & (stat.get_fdata() > 0) & (coverage.get_fdata() > 0)
    old = nib.load(verified(root, 'preparation/previous_roi.nii.gz', inv, sources))
    same(stat, old)
    other = nib.load(verified(root,
        'maps/randomise-contrast3/cluster-t3.1/randomise_clustere_corrp_tstat1.nii.gz', inv, sources))
    same(stat, other)
    if selected.sum() != 49 or not np.array_equal(selected, (old.get_fdata() > 0) & (coverage.get_fdata() > 0)):
        raise ValueError('not the locked coverage-corrected 49-voxel DMN region')
    if int(np.sum((other.get_fdata() > .95) & (coverage.get_fdata() > 0))) != 49:
        raise ValueError('randomise coverage-check summary changed')
    return nib.Nifti1Image(selected.astype('uint8'), stat.affine)


def catalog(repo, standardized, sources):
    root = repo/('results/dmn_covariate_standardized/v1/randomise' if standardized else 'results/permutation_audit/v1')
    config = json.loads(record(root/'config.json', sources).read_text())
    hashes = json.loads(record(root/'input_provenance.json', sources).read_text())
    return root, {m['name']: m for m in config['models']}, hashes


def validate_model(model, sources):
    x = np.asarray(model['x'], float)
    if len(model['subjects']) != 47 or len(set(model['subjects'])) != 47 or len(model['images']) != 47:
        raise ValueError('expected exactly 47 uniquely aligned participants')
    if not np.isfinite(x).all() or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError('invalid retained group design')
    for name, expected in model['source_hashes'].items():
        if cp.digest(Path(name)) != expected:
            raise ValueError(f'retained group source changed: {name}')
        record(name, sources)


def supplemental_rois(repo, sources):
    report, models, hashes = catalog(repo, False, sources)
    inv = inventory(report, 'map_inventory.tsv', sources)
    summary = read(record(report/'model_contrast_summary.tsv', sources))
    result = []
    for name, model_name, contrast, title in SUPPLEMENTAL:
        row, = [r for r in summary if r['model'] == model_name and int(r['contrast']) == contrast and r['method'] == 'cluster-t3.1']
        folder = f'maps/{row["test"]}/cluster-t3.1/'
        stat = nib.load(verified(report, folder+'randomise_tstat1.nii.gz', inv, sources))
        prob = nib.load(verified(report, folder+'randomise_clustere_corrp_tstat1.nii.gz', inv, sources))
        same(stat, prob)
        sig = (prob.get_fdata() > .95) & (stat.get_fdata() > 0)
        if sig.sum() != int(row['significant_voxels']) or not sig.any():
            raise ValueError(f'summary and significant map differ: {name}')
        result.append(dict(name=name, title=title, model=models[model_name], contrast=contrast,
            roi=nib.Nifti1Image(sig.astype('uint8'), stat.affine), statistic=stat,
            p=float(row['minimum_spatial_fwe_p']), hashes=hashes))
    return result


def parietal_roi(repo, sources):
    report, _, _ = catalog(repo, False, sources)
    inv = inventory(report, 'map_inventory.tsv', sources)
    summary = read(record(report/'model_contrast_summary.tsv', sources))
    selected = []
    for timing in pf.TIMINGS:
        name = f'crossed__{timing}_act_acceptance-intercept'
        row, = [r for r in summary if r['model'] == name and r['contrast'] == '1' and r['method'] == 'cluster-t3.1']
        folder = f'maps/{row["test"]}/cluster-t3.1/'
        stat = nib.load(verified(report, folder+'randomise_tstat1.nii.gz', inv, sources))
        prob = nib.load(verified(report, folder+'randomise_clustere_corrp_tstat1.nii.gz', inv, sources))
        same(stat, prob)
        if selected:
            same(reference, stat)
        reference = stat
        mask = (prob.get_fdata() > .95) & (stat.get_fdata() > 0)
        if mask.sum() != int(row['significant_voxels']):
            raise ValueError('parietal summary and map differ')
        selected.append(mask)
    roi = np.logical_and.reduce(selected)
    previous = nib.load(record(repo/'results/permutation_followup/v1/figures_and_extractions/activation-age-interaction/shared_selected_roi.nii.gz', sources))
    same(previous, reference)
    if roi.sum() != 125 or not np.array_equal(roi, previous.get_fdata() > 0):
        raise ValueError('expected the existing 125-voxel randomise intersection; the 61-voxel FLAME intersection is a different display summary')
    return nib.Nifti1Image(roi.astype('uint8'), reference.affine)


def run_dmn(repo, output, sources):
    roi_image = dmn_roi(repo, sources); roi = roi_image.get_fdata() > 0
    nib.save(roi_image, output/'selected_age_roi_49.nii.gz')
    comparison = nib.load(repo/'results/dmn_followup/v1/coverage_check/v1/maps/randomise-contrast3/cluster-t3.1/randomise_clustere_corrp_tstat1.nii.gz').get_fdata() > .95
    cp.write_json(output/'selection.json', dict(selection='PALM cluster-Z2.6 age region: original 51 intersected with valid coverage',
        palm_voxels=int(roi.sum()), randomise_voxels=int(comparison.sum()), overlap_voxels=int(np.sum(roi & comparison)),
        union_voxels=int(np.sum(roi | comparison)), note='The two 49-voxel regions are not identical; do not substitute their intersection.'))
    _, models, hashes = catalog(repo, True, sources)
    model = models['full-rt__dmn-age']; validate_model(model, sources)
    native, matched, audit = pm.extract_partners(model, roi_image, roi, hashes, sources)
    write(output/'run_weight_reconstruction.tsv', audit)
    policies, columns = pm.designs(model)
    summaries, participants, diagnostics = [], [], []
    matched_ok = all(r['verified'] == 1 for r in audit)
    for representation, values in [('native', native)]+([('matched', matched)] if matched_ok else []):
        current = []
        for j, metric in enumerate(list(pm.PARTNERS)+['similar-minus-dissimilar']):
            for policy, (x, c) in policies.items():
                rows, check, adjusted = pm.fit_diagnostic(x, c, values[:, j])
                current.extend(dict(representation=representation, metric=metric, policy=policy, **r) for r in rows)
                diagnostics.append(dict(representation=representation, metric=metric, policy=policy, **check))
                participants.extend(dict(subject=s, representation=representation, metric=metric, policy=policy,
                    raw_roi_estimate=float(a), adjusted_roi_estimate=float(b)) for s, a, b in zip(model['subjects'], values[:, j], adjusted))
        summaries.extend(current)
        partner_bars(current, representation, output)
        if representation == 'matched':
            lookup = {(r['policy'], r['metric'], r['age_group']): r['adjusted_mean'] for r in current}
            for policy in policies:
                for group in ('younger', 'older'):
                    if not np.isclose(lookup[policy, 'similar', group]-lookup[policy, 'dissimilar', group],
                                      lookup[policy, 'similar-minus-dissimilar', group], atol=1e-4, rtol=1e-5):
                        raise ValueError('matched adjusted bars fail COPE7 decomposition')
    write(output/'bar_summary.tsv', summaries); write(output/'participant_estimates.tsv', participants)
    write(output/'age_contrast_diagnostics.tsv', diagnostics)
    cp.write_json(output/'motion_adjustment_identity.json', pm.motion_identity(model, native[:, 3], columns['fd_mean']))
    if not matched_ok:
        raise RuntimeError('native 49-voxel bars saved; matched decomposition failed; inspect reconstruction table; no exclusions')


def partner_bars(rows, representation, output):
    lookup = {(r['policy'], r['metric'], r['age_group']): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    for ax, policy in zip(axes, ('age-only', 'no-group-FD', 'full')):
        for group, offset, color in zip(('younger', 'older'), (-.18, .18), COLORS):
            values = [lookup[policy, metric, group] for metric in pm.PARTNERS]
            ax.bar(np.arange(3)+offset, [r['adjusted_mean'] for r in values], width=.34,
                yerr=[r['conditional_participant_sem'] for r in values], capsize=3, color=color, label=group.title())
        ax.set_xticks(range(3), ['Computer', 'Similar\nOwn-age', 'Dissimilar\nOther-age'])
        ax.set_title(pm.POLICY_NAMES[policy], fontsize=11); ax.axhline(0, color='.5', lw=.8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Offer-related DMN coupling (model units)'); axes[-1].legend(frameon=False)
    fig.suptitle('Coverage-corrected DMN region: 49 voxels; 47 participants\n'+
        ('Common COPE7 run weights; exact contrast decomposition' if representation == 'matched' else 'Native condition-specific L2 run weights'), fontsize=13)
    fig.text(.05, .025, 'Means ± conditional participant SEM; descriptive selected-ROI summaries, not independent tests.\n'
        'Computer is contextual and not part of COPE7. Removing group FD does not remove L1 motion adjustment.\n'
        'Native partner bars cannot be subtracted to recover COPE7; only the verified common-weight representation has that property.', fontsize=8)
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.78, wspace=.15)
    fig.savefig(output/f'{representation}_three_partner_bars_49.png', dpi=170); plt.close(fig)
    selected = [lookup['full', 'similar-minus-dissimilar', g] for g in ('younger', 'older')]
    simple_bars(selected, 'DMN similar − dissimilar offer-related coupling\n49-voxel selected region; full covariate adjustment',
        output/f'{representation}_age_contrast_bars_49.png')


def simple_bars(rows, title, output):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(range(len(rows)), [r['adjusted_mean'] for r in rows],
        yerr=[r['conditional_participant_sem'] for r in rows], color=['#777777', *COLORS] if len(rows) == 3 else COLORS, capsize=4, width=.6)
    ax.set_xticks(range(len(rows)), [f'{r["age_group"].title()}\nN = {r["n"]}' for r in rows])
    ax.axhline(0, color='.5', lw=.8); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(title, fontsize=12); ax.set_ylabel('Adjusted contrast estimate (model units)')
    fig.text(.06, .035, 'Mean ± conditional participant SEM. Selected using the same data: descriptive only.\nUnweighted OLS adjustment; no independent ROI p-values or new imaging inference.', fontsize=8)
    fig.subplots_adjust(left=.17, right=.96, top=.82, bottom=.23)
    fig.savefig(output, dpi=170); plt.close(fig)


def adjusted_sample_bars(model, y):
    x, c = np.asarray(model['x']), np.asarray(model['c'])
    mean_col, = np.flatnonzero(c[0]); age_col, = np.flatnonzero(c[2])
    if c[0, mean_col] != 1 or c[2, age_col] != 1 or not np.allclose(x[:, mean_col], 1):
        raise ValueError('unexpected supplemental mean/age design')
    age = x[:, age_col]; groups = age > 0
    if len(age) != 47 or groups.sum() != 22 or not np.isclose(np.ptp(age), 1):
        raise ValueError('supplemental age coding differs')
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    nuisance = [j for j in range(x.shape[1]) if j not in (mean_col, age_col)]
    adjusted = y-x[:, nuisance]@beta[nuisance]
    rows = []
    for name, keep in [('all', np.ones(len(y), bool)), ('younger', ~groups), ('older', groups)]:
        v = adjusted[keep]
        rows.append(dict(age_group=name, n=int(keep.sum()), adjusted_mean=float(v.mean()),
            conditional_participant_sem=float(v.std(ddof=1)/np.sqrt(len(v)))))
    return rows, adjusted


def extract_checked(model, roi_image, hashes, sources, coverage):
    roi = roi_image.get_fdata() > 0; y = []
    for subject, name in zip(model['subjects'], model['images']):
        p = Path(name)
        if cp.digest(p) != hashes.get(name):
            raise ValueError(f'participant COPE changed or unrecorded: {p}')
        im = nib.load(record(p, sources)); same(im, roi_image); values = im.get_fdata()[roi]
        if not np.isfinite(values).all():
            raise ValueError(f'nonfinite selected-ROI COPE: {subject}')
        var = nib.load(record(p.with_name('varcope1.nii.gz'), sources)); same(var, im)
        mask = nib.load(record(p.parent.parent/'mask.nii.gz', sources)); same(mask, im)
        v = var.get_fdata()[roi]; m = mask.get_fdata()[roi]
        valid = np.isfinite(v) & (v > 0) & np.isfinite(m) & (m > 0)
        coverage.append(dict(model=model['name'], subject=subject, image=str(p), contrast_folder=p.parent.parent.name, roi_voxels=int(roi.sum()),
            valid_variance_and_mask_voxels=int(valid.sum()), coverage_complete=int(valid.all()),
            policy='fixed ROI and all participants retained; flagged means are descriptive and require review'))
        y.append(float(values.mean()))
    return np.asarray(y)


def run_effects(repo, output, sources):
    coverage = []; summaries = []; participants = []
    for item in supplemental_rois(repo, sources):
        model = item['model']; validate_model(model, sources)
        nib.save(item['roi'], output/f'{item["name"]}_selected_roi.nii.gz')
        coverage_start = len(coverage)
        y = extract_checked(model, item['roi'], item['hashes'], sources, coverage)
        # Negative contrasts are displayed in their named direction; the saved
        # input and design still encode human minus computer.
        sign = -1 if item['contrast'] == 2 else 1
        rows, adjusted = adjusted_sample_bars(model, sign*y)
        flagged = sum(r['coverage_complete'] == 0 for r in coverage[coverage_start:])
        simple_bars(rows, item['title']+f'\nSelected cluster p = {item["p"]:.4f}; coverage flags: {flagged}', output/f'{item["name"]}_bars.png')
        summaries.extend(dict(target=item['name'], direction_multiplier=sign, **r) for r in rows)
        participants.extend(dict(target=item['name'], subject=s, raw_selected_contrast=float(a), adjusted_selected_contrast=float(b))
            for s, a, b in zip(model['subjects'], sign*y, adjusted))
    write(output/'bar_summary.tsv', summaries); write(output/'participant_estimates.tsv', participants)
    write(output/'coverage.tsv', coverage)
    cp.write_json(output/'interpretation.json', dict(
        selection='all significant voxels from the named signed randomise cluster-t3.1 test; no further peak/cluster selection',
        error_bars='SD of adjusted participant values / sqrt(n); conditional on fitted nuisance slopes; not coefficient SE',
        full_sample='Pooled sample mean at the retained covariate adjustment; age-split bars are descriptive, not additional tests',
        coverage_flags=sum(r['coverage_complete'] == 0 for r in coverage),
        limitations='Same-data region selection; no independent ROI inference. Task responses may reflect visual/attentional differences.'))


def huber_fit(x, y, tuning=1.345):
    """Fixed-MAD-scale Huber IRLS; response-outlier diagnostic, not a test."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y-x@beta
    scale = 1.4826*np.median(np.abs(residual-np.median(residual)))
    if scale < np.finfo(float).eps:
        return beta, np.ones(len(y)), 0
    for iteration in range(1000):
        residual = y-x@beta
        weights = np.minimum(1., tuning*scale/np.maximum(np.abs(residual), np.finfo(float).tiny))
        new = np.linalg.lstsq(x*np.sqrt(weights[:, None]), y*np.sqrt(weights), rcond=None)[0]
        if np.linalg.norm(new-beta) <= 1e-9*(1+np.linalg.norm(beta)):
            return new, weights, iteration+1
        beta = new
    raise RuntimeError('Huber IRLS did not converge; no approximate result silently accepted')


def deletion_checks(x, y, contrast, subjects):
    beta = np.linalg.lstsq(x, y, rcond=None)[0]; value = float(contrast@beta)
    checks = []
    for size in (1, 2):
        for omit in itertools.combinations(range(len(y)), size):
            keep = np.ones(len(y), bool); keep[list(omit)] = False
            full_rank = np.linalg.matrix_rank(x[keep]) == x.shape[1]
            estimate = float(contrast@np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]) if full_rank else None
            checks.append(dict(omitted_count=size, omitted_subjects=';'.join(subjects[i] for i in omit), full_rank=int(full_rank),
                estimate=estimate if full_rank else '', change_from_full=estimate-value if full_rank else '',
                sign_reversal=int(np.sign(estimate) != np.sign(value)) if full_rank else ''))
    return checks


def influence_audit(x, y, c, subjects):
    x, y, c = np.asarray(x, float), np.asarray(y, float), np.asarray(c, float)
    if x.shape != (47, 8) or y.shape != (47,) or c.shape != (8,) or np.linalg.matrix_rank(x) != 8:
        raise ValueError('expected full-rank 47 x 8 crossed design')
    if not np.array_equal(c, [0, 0, 0, 0, 0, 0, -1, 1]):
        raise ValueError('unexpected older-minus-younger behavioral slope contrast')
    if not np.array_equal(x[:, 4]+x[:, 5], np.ones(47)) or not np.array_equal(x[:, 4:6].sum(0), [25, 22]):
        raise ValueError('age membership changed')
    for group in (0, 1):
        if np.any(x[x[:, 4+group] == 0, 6+group] != 0):
            raise ValueError('behavior not nested within group')
    d = pf.diagnostic(x, y, c); beta, weights, iterations = huber_fit(x, y)
    pinv = np.linalg.pinv(x); e = y-x@d['beta']; h = d['leverage']
    hc3 = pinv@np.diag((e/(1-h))**2)@pinv.T
    checks = deletion_checks(x, y, c, subjects)
    detail = []
    for i, s in enumerate(subjects):
        detail.append(dict(subject=s, age_group='older' if x[i, 5] else 'younger', behavior=x[i, 6]+x[i, 7],
            raw_cope=float(y[i]), adjusted_cope=float(d['adjusted'][i]), leverage=float(h[i]),
            cooks_distance=float(d['cooks'][i]), studentized_residual=float(d['studentized'][i]),
            huber_weight=float(weights[i]), full_minus_leave_one_out=float(d['full']-d['loo'][i])))
    summaries = []
    for method, coefficients in [('OLS', d['beta']), ('Huber', beta)]:
        summaries.append(dict(method=method, younger_slope=float(coefficients[6]), older_slope=float(coefficients[7]),
            older_minus_younger=float(c@coefficients), younger_hc3_se=float(np.sqrt(hc3[6, 6])) if method == 'OLS' else '',
            older_hc3_se=float(np.sqrt(hc3[7, 7])) if method == 'OLS' else '',
            interaction_hc3_se=float(np.sqrt(c@hc3@c)) if method == 'OLS' else '', iterations=iterations if method == 'Huber' else 0))
    behavior = x[:, 6]+x[:, 7]
    slopes = [float(stats.theilslopes(d['adjusted'][x[:, 4+g] > .5], behavior[x[:, 4+g] > .5]).slope) for g in (0, 1)]
    summaries.append(dict(method='Theil-Sen on OLS-nuisance-adjusted outcome', younger_slope=slopes[0], older_slope=slopes[1],
        older_minus_younger=slopes[1]-slopes[0], younger_hc3_se='', older_hc3_se='', interaction_hc3_se='', iterations=0))
    return summaries, detail, checks, d, beta


def validate_activation_partners(repo, timing, model, sources):
    reports = {'full-rt': 'results/reviewer/rt_coverage_corrected',
        'pooled-post': 'results/decision_postresponse/imaging',
        'partner-post': 'results/decision_postresponse_partner/imaging'}
    manifest = record(repo/reports[timing]/'revision_jobs.tsv', sources)
    rows = [r for r in read(manifest) if r['stage'] == 'l1' and r['model'] == 'act']
    expected = {(s, run) for s in model['subjects'] for run in ('01', '02')}
    if len(rows) != 94 or {(r['subject'], r['run']) for r in rows} != expected:
        raise ValueError('activation source manifest must contain both runs for all 47 participants')
    for row in rows:
        activation_validation.validate_l1(row, sources)
    # This reuses contrast validation only; no vmPFC ROI analysis is run.
    record(Path(activation_validation.__file__), sources)


def run_influence(repo, output, sources):
    roi = parietal_roi(repo, sources); nib.save(roi, output/'parietal_shared_roi.nii.gz')
    _, models, hashes = catalog(repo, False, sources)
    summary = []; detail = []; deletions = []; coverage = []; diagnostics = {}; condition_rows = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    for row, timing in enumerate(pf.TIMINGS):
        model = models[f'crossed__{timing}_act_acceptance-intercept']; validate_model(model, sources)
        y = extract_checked(model, roi, hashes, sources, coverage)
        # The permutation catalog canonicalizes column order. Use the retained
        # original design and its ID-aligned contrast, not positional assumptions
        # about the canonicalized array.
        directory = repo/'results/crossed_behavior_l3/v1/imaging'/f'{timing}_act_acceptance-intercept/design'
        x, original_contrasts = pf.aligned_design(directory, model['subjects'])
        c = original_contrasts[0]
        canonical_x, canonical_c = np.asarray(model['x']), np.asarray(model['c'][0])
        if not np.allclose(x@np.linalg.pinv(x), canonical_x@np.linalg.pinv(canonical_x), atol=1e-6, rtol=0):
            raise ValueError('canonical and retained original design spaces differ')
        if not np.allclose(c@np.linalg.pinv(x), canonical_c@np.linalg.pinv(canonical_x), atol=1e-6, rtol=0):
            raise ValueError('canonical and original contrast estimators differ')
        estimates, people, deleted, d, robust = influence_audit(x, y, c, model['subjects'])
        summary.extend(dict(timing=timing, **r) for r in estimates)
        detail.extend(dict(timing=timing, **r) for r in people); deletions.extend(dict(timing=timing, **r) for r in deleted)
        diagnostics[timing] = (x, y, d, robust)
        ax, deletion_ax = axes[row]
        behavior = x[:, 6]+x[:, 7]
        # Each line uses the adjustment from its OWN full-design fit.
        for g, color in enumerate(COLORS):
            keep = x[:, 4+g] > .5
            ax.scatter(behavior[keep], d['adjusted'][keep], color=color, s=23, alpha=.75, label=('Younger','Older')[g])
            xx = np.linspace(behavior[keep].min(), behavior[keep].max(), 100)
            ax.plot(xx, d['beta'][4+g]+xx*d['beta'][6+g], color=color, label='OLS' if g == 0 else None)
            # Dashed lines displayed on the OLS-adjusted scale would otherwise
            # confound different nuisance fits. Show robust fits in a separate
            # exported plot below; do not overlay incompatible adjustments.
        for i in np.argsort(d['cooks'])[-4:]:
            ax.annotate(model['subjects'][i].removeprefix('sub-'), (behavior[i], d['adjusted'][i]), xytext=(3, 3), textcoords='offset points', fontsize=8)
        ax.axhline(0, color='.7', lw=.6); ax.axvline(0, color='.7', lw=.6)
        ax.set_title(pf.LABELS[row], fontsize=11); ax.set_xlabel('Similar − dissimilar acceptance intercept')
        ax.set_ylabel('OLS-adjusted similar − dissimilar offer slope')
        if row == 0: ax.legend(frameon=False)
        for size, color in [(1, '#397b9d'), (2, '#b86c3e')]:
            values = [r['estimate'] for r in deleted if r['omitted_count'] == size and r['full_rank']]
            deletion_ax.hist(values, bins=30, alpha=.6, color=color, label=f'Leave {size} out')
        deletion_ax.axvline(d['full'], color='black', label='Full sample'); deletion_ax.axvline(0, color='.5', ls=':')
        deletion_ax.set_xlabel('Older − younger behavioral slope'); deletion_ax.set_ylabel('Number of deletion combinations')
        deletion_ax.set_title('Every single and pair deletion; nuisance terms refitted', fontsize=10)
        if row == 0: deletion_ax.legend(frameon=False, fontsize=9)
        for a in (ax, deletion_ax): a.spines[['top', 'right']].set_visible(False)
        # Native partner decomposition, not a substitute for continuous slopes.
        validate_activation_partners(repo, timing, model, sources)
        group_c = np.vstack([np.eye(8)[4], np.eye(8)[5], np.eye(8)[4]-np.eye(8)[5], np.eye(8)[5]-np.eye(8)[4]])
        for partner, cope in pm.PARTNERS.items():
            condition = dict(model, images=[v.replace('/cope7.feat/', f'/cope{cope}.feat/') for v in model['images']])
            new_hashes = dict(hashes)
            for path in condition['images']: new_hashes[path] = cp.digest(Path(path))
            values = extract_checked(condition, roi, new_hashes, sources, coverage)
            rows, _, _, _ = bars.adjusted_bars(x, group_c, values)
            condition_rows.extend(dict(timing=timing, partner=partner, **r) for r in rows)
    fig.suptitle('Parietal age × acceptance-tendency interaction: influence audit', fontsize=15)
    fig.text(.05, .025, 'Selected using these data: descriptive diagnostics only, no independent ROI tests or automatic exclusions.\n'
        'Deletion sign stability does not establish whole-brain significance. No participant was removed from the primary analyses.', fontsize=9)
    fig.tight_layout(rect=[0, .08, 1, .95]); fig.savefig(output/'parietal_influence.png', dpi=160); plt.close(fig)
    influence_plots(diagnostics, summary, condition_rows, output)
    write(output/'slope_comparison.tsv', summary); write(output/'participant_influence.tsv', detail)
    write(output/'all_single_pair_deletions.tsv', deletions); write(output/'native_partner_bar_summary.tsv', condition_rows)
    write(output/'coverage.tsv', coverage)
    checks = []
    for timing in pf.TIMINGS:
        for size in (1, 2):
            r = [v for v in deletions if v['timing'] == timing and v['omitted_count'] == size]
            values = [v['estimate'] for v in r if v['full_rank']]
            checks.append(dict(timing=timing, omitted_count=size, combinations=len(r), rank_losses=sum(not v['full_rank'] for v in r),
                estimate_min=min(values), estimate_max=max(values), sign_reversals=sum(v['sign_reversal'] for v in r if v['full_rank'])))
    write(output/'deletion_summary.tsv', checks)
    cp.write_json(output/'limitations.json', dict(
        independent_inference=False, exclusions=0,
        huber='Fixed OLS-residual MAD scale, tuning 1.345; full-design refit. Downweights response residuals, not high-leverage predictor values.',
        theil_sen='Within-group slopes on OLS-nuisance-adjusted outcomes; diagnostic only; nuisance fit itself remains OLS.',
        error_bars='Slope-coefficient bars: HC3 SE for OLS only; Huber/Theil-Sen bars have no uncertainty bars. Condition means: participant SEM.',
        selection='Existing 125-voxel randomise intersection across all three timing-model cluster-t3.1 results; circular/descriptive. '
                  'The previously reported 61 voxels also required FLAME overlap and are not the scatterplot extraction region.',
        coverage_flags=sum(r['coverage_complete'] == 0 for r in coverage),
        next_decision='If estimates collapse or reverse under joint deletion/robust fitting, weaken the claim; do not select exclusions to preserve significance.'))


def influence_plots(diagnostics, rows, conditions, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
    methods = ('OLS', 'Huber', 'Theil-Sen on OLS-nuisance-adjusted outcome')
    for timing, ax in zip(pf.TIMINGS, axes):
        current = [next(r for r in rows if r['timing'] == timing and r['method'] == method) for method in methods]
        for g, offset, color in zip(('younger', 'older'), (-.18, .18), COLORS):
            ax.bar(np.arange(3)+offset, [r[g+'_slope'] for r in current], width=.34, color=color, label=g.title())
            ax.errorbar(offset, current[0][g+'_slope'], yerr=current[0][g+'_hc3_se'], color='black', capsize=3)
        ax.set_xticks(range(3), ['OLS', 'Huber', 'Theil–Sen']); ax.axhline(0, color='.5', lw=.8)
        ax.set_title(timing); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Brain–behavior regression slope'); axes[-1].legend(frameon=False)
    fig.suptitle('Parietal brain–behavior slopes: descriptive robustness comparison', fontsize=14)
    fig.text(.05, .025, 'OLS error bars = HC3 coefficient SE, not participant SEM. Robust bars have no uncertainty bars.\nSame-data ROI; no independent p-values. Huber does not protect against all high-leverage points; Theil–Sen uses OLS nuisance adjustment.', fontsize=8)
    fig.tight_layout(rect=[0, .13, 1, .93]); fig.savefig(output/'parietal_slope_bars.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)
    for timing, ax in zip(pf.TIMINGS, axes):
        for group, offset, color in zip(('younger', 'older'), (-.18, .18), COLORS):
            selected = [next(r for r in conditions if r['timing'] == timing and r['partner'] == p and r['age_group'] == group) for p in pm.PARTNERS]
            ax.bar(np.arange(3)+offset, [r['adjusted_mean'] for r in selected], width=.34, color=color,
                yerr=[r['conditional_participant_sem'] for r in selected], capsize=3, label=group.title())
        ax.set_xticks(range(3), ['Computer', 'Similar', 'Dissimilar']); ax.set_title(timing); ax.axhline(0, color='.5', lw=.8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Adjusted partner offer-slope estimate'); axes[-1].legend(frameon=False)
    fig.suptitle('Parietal selected region: native three-partner estimates', fontsize=14)
    fig.text(.05, .025, 'Means ± conditional participant SEM; native L2 contrasts have different run weights.\nDescriptive means do not display or independently test the continuous age × behavioral-covariate interaction. See coverage.tsv for variance/coverage flags.', fontsize=8)
    fig.tight_layout(rect=[0, .13, 1, .93]); fig.savefig(output/'parietal_three_partner_bars.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
    for timing, ax in zip(pf.TIMINGS, axes):
        x, y, _, beta = diagnostics[timing]; b = x[:, 6]+x[:, 7]; adjusted = y-x[:, :4]@beta[:4]
        for g, color in enumerate(COLORS):
            keep = x[:, 4+g] > .5; xx = np.linspace(b[keep].min(), b[keep].max(), 100)
            ax.scatter(b[keep], adjusted[keep], color=color, s=22, alpha=.7)
            ax.plot(xx, beta[4+g]+xx*beta[6+g], color=color)
        ax.set_title(timing); ax.set_xlabel('Similar − dissimilar acceptance intercept'); ax.axhline(0, color='.6', lw=.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Huber-adjusted similar − dissimilar offer slope')
    fig.suptitle('Huber full-design fits: each plot uses its own nuisance adjustment', fontsize=14)
    fig.text(.05, .025, 'Descriptive only. Predictor leverage can remain influential even when Huber weights are large. No exclusions or independent ROI tests.', fontsize=8)
    fig.tight_layout(rect=[0, .11, 1, .93]); fig.savefig(output/'parietal_huber_scatter.png', dpi=170); plt.close(fig)


def atlas_labels(xml):
    labels = {int(e.attrib['index']): e.text.strip() for e in ET.parse(xml).findall('.//label')}
    if not labels or sorted(labels) != list(range(len(labels))):
        raise ValueError('atlas labels must have contiguous zero-based probability-volume indices')
    return labels


def tissue_categories(labels):
    categories = {}
    for index, name in labels.items():
        if 'White Matter' in name:
            categories[index] = 'white_matter'
        elif 'Lateral Ventricle' in name:
            categories[index] = 'ventricular_csf'
        elif 'Brain-Stem' in name or 'Brainstem' in name:
            categories[index] = 'brainstem_other'
        else:
            categories[index] = 'gray_matter'
    if not {'white_matter', 'ventricular_csf', 'gray_matter'}.issubset(categories.values()):
        raise ValueError('atlas does not include expected cortical/white-matter/ventricular labels')
    return categories


def atlas_samples(image, roi, expected_volumes):
    if len(image.shape) != 4 or image.shape[3] != expected_volumes:
        raise ValueError('probability atlas volume/label count differs')
    target = (roi.shape, roi.affine)
    selection = roi.get_fdata() > 0
    samples = []
    for i in range(expected_volumes):
        volume = nib.Nifti1Image(np.asarray(image.dataobj[..., i]), image.affine)
        samples.append(resample_from_to(volume, target, order=1).get_fdata()[selection])
    values = np.stack(samples, axis=1)
    if not np.isfinite(values).all() or np.any(values < -1e-5) or values.max() > 100.0001:
        raise ValueError('atlas probability range is not 0..100 percent')
    return values


def run_localization(repo, output, sources, fsldir, standard):
    atlas_root = fsldir/'data/atlases'
    sub_path = atlas_root/'HarvardOxford/HarvardOxford-sub-prob-2mm.nii.gz'
    cortical_path = atlas_root/'HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz'
    sub_xml = atlas_root/'HarvardOxford-Subcortical.xml'; cortical_xml = atlas_root/'HarvardOxford-Cortical.xml'
    for p in (sub_path, cortical_path, sub_xml, cortical_xml, standard):
        if not p.is_file():
            raise FileNotFoundError(f'FSL atlas/anatomy missing: {p}; supply --fsl-root and --standard for an installation containing the Harvard-Oxford probability atlases')
        record(p, sources)
    sub_labels = atlas_labels(sub_xml); cortical_labels = atlas_labels(cortical_xml)
    categories = tissue_categories(sub_labels)
    sub = nib.load(sub_path); cortical = nib.load(cortical_path)
    bg = nib.as_closest_canonical(nib.load(standard))
    targets = supplemental_rois(repo, sources)
    targets += [dict(name='dmn-age-49', title='Coverage-corrected DMN age region', roi=dmn_roi(repo, sources), statistic=None, p=.0451),
        dict(name='parietal-interaction-61', title='Shared parietal age × acceptance interaction', roi=parietal_roi(repo, sources), statistic=None, p=None)]
    summaries = []; voxel_rows = []; region_rows = []; component_rows = []; target_rows = []
    for item in targets:
        roi = item['roi']; selected = roi.get_fdata() > 0
        nib.save(roi, output/f'{item["name"]}_selected_roi.nii.gz')
        target_rows.append(dict(target=item['name'], title=item['title'], selected_voxels=int(selected.sum()),
            model=item['model']['name'] if 'model' in item else '', contrast=item.get('contrast', ''),
            existing_minimum_spatial_fwe_p=item['p'] if item['p'] is not None else '',
            selection='signed cluster-t3.1' if 'model' in item else ('PALM cluster-Z2.6 age' if item['name'] == 'dmn-age-49' else 'intersection across timing models; no separate p-value')))
        values = atlas_samples(sub, roi, len(sub_labels)); region = atlas_samples(cortical, roi, len(cortical_labels))
        tissue_names = ('gray_matter', 'white_matter', 'ventricular_csf', 'brainstem_other')
        tissue = np.stack([values[:, [i for i in sub_labels if categories[i] == name]].sum(1) for name in tissue_names], axis=1)
        winner = np.argmax(tissue, axis=1); maximum = tissue.max(1)
        # No ROI trimming or categorical GM mask: retain full probability
        # distributions and flag voxels with little atlas support as unassigned.
        labels = np.array(tissue_names, object)[winner]; labels[maximum == 0] = 'unassigned'
        for name in (*tissue_names, 'unassigned'):
            summaries.append(dict(target=item['name'], tissue=name, roi_voxels=int(selected.sum()),
                mean_atlas_probability_percent=float(tissue[:, tissue_names.index(name)].mean()) if name != 'unassigned' else '',
                dominant_voxels=int(np.sum(labels == name)), dominant_fraction=float(np.mean(labels == name))))
        indices = np.argwhere(selected); xyz = nib.affines.apply_affine(roi.affine, indices)
        for k, coord in enumerate(xyz):
            voxel_rows.append(dict(target=item['name'], x=float(coord[0]), y=float(coord[1]), z=float(coord[2]),
                dominant_tissue=labels[k], maximum_tissue_probability_percent=float(maximum[k]),
                **{name+'_probability_percent': float(tissue[k, j]) for j, name in enumerate(tissue_names)}))
        for atlas, samples, names in [('subcortical_tissue', values, sub_labels), ('cortical', region, cortical_labels)]:
            for i in sorted(names, key=lambda i: float(samples[:, i].mean()), reverse=True)[:8]:
                region_rows.append(dict(target=item['name'], atlas=atlas, label=names[i], mean_probability_percent=float(samples[:, i].mean()),
                    maximum_probability_percent=float(samples[:, i].max())))
        components, number = label(selected, structure=np.ones((3, 3, 3)))
        for j in range(1, number+1):
            keep = components[selected] == j
            component_rows.append(dict(target=item['name'], component=j, voxels=int(keep.sum()),
                **{name+'_mean_probability_percent': float(tissue[keep, k].mean()) for k, name in enumerate(tissue_names)}))
        localization_figure(item, tissue, bg, output)
    write(output/'tissue_summary.tsv', summaries); write(output/'voxel_tissue_probabilities.tsv', voxel_rows)
    write(output/'atlas_region_summary.tsv', region_rows); write(output/'component_tissue_summary.tsv', component_rows)
    write(output/'targets.tsv', target_rows)
    cp.write_json(output/'limitations.json', dict(
        atlas='FSL Harvard-Oxford probability atlases, 2 mm, linearly sampled at native statistical voxel centres',
        probability='Original 0..100 atlas percentages; not normalized to sum to 100 and not subject-specific tissue fractions',
        categories='Cortex and named subcortical gray structures; cerebral white matter; lateral ventricles; brainstem separate',
        dominant='Largest summed category probability; zero support is unassigned. Relative winner alone does not establish anatomical certainty.',
        limitations='Ventricular CSF labels do not include all sulcal/extra-axial CSF. Template registration, interpolation and partial volume limit tissue assignments. '
                    'Atlas summaries do not validate participant registration, repair low-signal estimates, or independently establish functional specificity.',
        statistical_changes=0, exclusions=0))


def localization_figure(item, tissue, background, output):
    selected = item['roi'].get_fdata() > 0
    # Largest component centroid, fixed without reference to tissue preference.
    components, _ = label(selected, structure=np.ones((3, 3, 3)))
    counts = np.bincount(components.ravel()); counts[0] = 0
    cuts = nib.affines.apply_affine(item['roi'].affine, np.argwhere(components == np.argmax(counts)).mean(0))
    overlay = resample_from_to(item['roi'], background, order=0).get_fdata()
    anatomy = background.get_fdata(); fig, axes = plt.subplots(1, 4, figsize=(14, 5.3))
    for axis in range(3):
        ax = axes[axis]; base, extent, cut = pf.slice_plane(anatomy, background.affine, axis, cuts[axis])
        values, _, _ = pf.slice_plane(overlay, background.affine, axis, cuts[axis])
        ax.imshow(base, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=np.percentile(anatomy[anatomy > 0], 99.5))
        ax.imshow(np.ma.masked_where(values == 0, values), origin='lower', extent=extent, cmap='autumn', vmin=0, vmax=1, alpha=.85, interpolation='nearest')
        ax.set_title(f'{"xyz"[axis]} = {cut:.0f} mm'); ax.set_xticks([]); ax.set_yticks([])
        ax.text(.03, .04, 'P' if axis == 0 else 'L', transform=ax.transAxes, color='white')
        ax.text(.93, .04, 'A' if axis == 0 else 'R', transform=ax.transAxes, color='white')
    axes[3].bar(range(4), tissue.mean(0), color=['#bb7755', '#cccccc', '#5599bb', '#888877'])
    axes[3].set_xticks(range(4), ['GM', 'WM', 'Ventricle', 'Brainstem'], rotation=35, ha='right')
    axes[3].set_ylabel('Mean atlas probability (%)'); axes[3].set_ylim(0, 100); axes[3].spines[['top', 'right']].set_visible(False)
    fig.suptitle(item['title']+f'\n{int(selected.sum())} native voxels; template atlas localization', fontsize=14)
    fig.text(.04, .025, 'Display only: no smoothing or ROI trimming. Harvard-Oxford atlas probabilities are not subject-specific tissue fractions.\n'
        'Cuts: centroid of largest connected component. Full component and voxel tables retain all significant voxels.', fontsize=8)
    fig.tight_layout(rect=[0, .13, 1, .86]); fig.savefig(output/f'{item["name"]}_localization.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'dmn', 'effects', 'influence', 'localization'), default='all')
    parser.add_argument('--fsl-root', type=Path, default=Path(os.environ.get('FSLDIR', '/usr/local/fsl')))
    parser.add_argument('--standard', type=Path)
    args = parser.parse_args(); repo = args.repository.resolve(); output = args.output_root.resolve()
    standard = (args.standard or args.fsl_root/'data/standard/MNI152_T1_2mm_brain.nii.gz').resolve()
    if output.exists():
        raise FileExistsError(f'report already exists; choose a new output root to preserve it: {output}')
    output.mkdir(parents=True)
    phases = ('dmn', 'effects', 'influence', 'localization') if args.phase == 'all' else (args.phase,)
    status = []
    for phase in phases:
        folder = output/phase; folder.mkdir()
        sources = {str(p): cp.digest(p) for p in (Path(__file__), Path(pm.__file__), Path(bars.__file__), Path(pf.__file__))}
        try:
            if phase == 'localization':
                run_localization(repo, folder, sources, args.fsl_root.resolve(), standard)
            else:
                {'dmn': run_dmn, 'effects': run_effects, 'influence': run_influence}[phase](repo, folder, sources)
            result = dict(phase=phase, complete=True, error='')
            print(f'PASS: {phase}; saved {folder}', flush=True)
        except Exception as exc:
            traceback.print_exc(); result = dict(phase=phase, complete=False, error=str(exc))
            print(f'FAILED: {phase}; preserved partial outputs in {folder}', flush=True)
        cp.write_json(folder/'provenance.json', dict(sources=sources, new_imaging_fits=0, exclusions=0))
        cp.write_json(folder/'status.json', result); status.append(result)
        files = [dict(path=str(p.relative_to(folder)), sha256=cp.digest(p), bytes=p.stat().st_size) for p in sorted(folder.rglob('*')) if p.is_file()]
        write(folder/'inventory.tsv', files)
    write(output/'phase_status.tsv', status)
    cp.write_json(output/'status.json', dict(complete=all(r['complete'] for r in status), phases=status,
        no_new_imaging_fits=True, exclusions=0, interpretation='Descriptive selected-ROI figures and diagnostics; existing inference unchanged.'))
    return int(not all(r['complete'] for r in status))


if __name__ == '__main__':
    raise SystemExit(main())
