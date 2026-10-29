#!/usr/bin/env python3
"""Selected-ROI three-partner decomposition and group-FD sensitivity.

Descriptive only. Preserve the locked ROI, participant set, L1 motion correction
and primary model. No FEAT, PALM, threshold changes or automatic exclusions.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import collect_palm_progress as cp
import plot_palm_age_bars as bars
from audit_l1_designs import read_vest_matrix
from prepare_sub144_imaging_repair import fsf_value

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BARS = ROOT/'results/dmn_followup/v1/age_bars/attempt-20261029T010101Z-MWqld9/outputs'
PARTNERS = {'computer': 2, 'similar': 4, 'dissimilar': 6}
POLICIES = ('age-only', 'no-group-FD', 'no-group-FD-or-tSNR', 'full')
POLICY_NAMES = {'age-only': 'Age groups only', 'no-group-FD': 'Sex + tSNR + mean RT\nNo group FD',
    'no-group-FD-or-tSNR': 'Sex + mean RT\nNo group FD/tSNR', 'full': 'Full model\nSex + tSNR + FD + mean RT'}
COLORS = ['#397b9d', '#b86c3e']


def verify_inventory(root):
    if not json.loads((root/'status.json').read_text())['complete']:
        raise ValueError(f'incomplete source export: {root}')
    for row in bars.anatomy.read_rows(root/'inventory.tsv'):
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or cp.digest(path) != row['sha256']:
            raise ValueError(f'source export checksum differs: {path}')


def designs(model):
    x, c = np.asarray(model['x']), np.asarray(model['c'])
    group_cols = [int(np.flatnonzero(row)[0]) for row in c[:2]]
    # Delegate full validation, including canonical-column invariance.
    bars.adjusted_bars(x, c, np.arange(len(x), dtype=float))
    columns = {}
    for name, values in model['nuisance'].items():
        matches = [j for j in range(x.shape[1]) if np.allclose(x[:, j], values, atol=1e-12, rtol=0)]
        if len(matches) != 1:
            raise ValueError(f'nuisance not uniquely ID-aligned in canonical design: {name}')
        columns[name] = matches[0]
    if set(columns) != {'ismale', 'tsnr', 'fd_mean', 'RT'} or len(set(columns.values())) != 4:
        raise ValueError('unexpected group nuisance design')
    choices = {'age-only': group_cols, 'no-group-FD': [j for j in range(x.shape[1]) if j != columns['fd_mean']],
        'no-group-FD-or-tSNR': [j for j in range(x.shape[1]) if j not in (columns['fd_mean'], columns['tsnr'])],
        'full': list(range(x.shape[1]))}
    return {name: (x[:, idx], c[:, idx]) for name, idx in choices.items()}, columns


def fit_diagnostic(x, c, y):
    summaries, adjusted, groups, difference = bars.adjusted_bars(x, c, y)
    inverse = np.linalg.pinv(x); residual = y-x@(inverse@y)
    leverage = np.einsum('ij,ji->i', x, inverse)
    if np.any(leverage >= 1-1e-10):
        raise ValueError('unit leverage prevents HC3/leave-one-out diagnostic')
    hc3 = float(np.sqrt(np.sum(((c[2]@inverse)*residual/(1-leverage))**2)))
    loo = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        if np.linalg.matrix_rank(x[keep]) != x.shape[1]:
            raise ValueError('leave-one-out design rank loss')
        loo.append(float(c[2]@np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]))
    return summaries, dict(younger_minus_older=difference, contrast_hc3_se=hc3,
        loo_min=min(loo), loo_max=max(loo), loo_sign_changes=int(np.sum(np.sign(loo) != np.sign(difference)))), adjusted


def load_roi(path, reference, roi, sources, expected=None):
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = cp.digest(path)
    if expected is not None and digest != expected:
        raise ValueError(f'participant COPE7 differs from primary PALM input: {path}')
    image = nib.load(path)
    if not bars.anatomy.palm.pa.compatible(image, reference):
        raise ValueError(f'ROI/input geometry differs: {path}')
    values = image.get_fdata()[roi]
    if not np.isfinite(values).all():
        raise ValueError(f'nonfinite ROI input: {path}')
    sources[str(path)] = digest
    return values


def validate_l1(folder, sources):
    fsf, contrast = folder/'design.fsf', folder/'design.con'
    text = fsf.read_text(); matrix = read_vest_matrix(contrast)
    if int(fsf_value(text, 'evs_real')) != 28 or matrix.shape[0] < 7 or matrix.shape[1] < 28:
        raise ValueError(f'unexpected original-style nPPI design: {folder}')
    if [fsf_value(text, f'evtitle{i}') for i in (12, 14, 16)] != ['comp_p', 'in_p', 'out_p']:
        raise ValueError(f'partner PPI EV ordering differs: {folder}')
    for cope, column in ((2, 12), (4, 14), (6, 16)):
        expected = np.zeros(matrix.shape[1]); expected[column-1] = 1
        if not np.array_equal(matrix[cope-1], expected):
            raise ValueError(f'wrong condition contrast {cope}: {folder}')
    if not np.array_equal(matrix[6], matrix[3]-matrix[5]):
        raise ValueError(f'COPE7 is not the within-run similar-minus-dissimilar contrast: {folder}')
    sources.update({str(p): cp.digest(p) for p in (fsf, contrast)})


def common_weights(conditions, cope7, variances):
    """Use the *difference contrast's* voxel/run weights for every partner."""
    conditions, cope7, variances = map(lambda a: np.asarray(a, float), (conditions, cope7, variances))
    if (conditions.ndim != 3 or conditions.shape[:2] != (2, 3)
            or cope7.shape != conditions[:, 0].shape or variances.shape != cope7.shape):
        raise ValueError('unexpected run/partner/voxel dimensions')
    if not np.isfinite(conditions).all() or not np.isfinite(cope7).all() or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError('nonfinite values or nonpositive run VARCOPE7; no zero-variance substitution/exclusion')
    if not np.allclose(conditions[:, 1]-conditions[:, 2], cope7, atol=1e-4, rtol=1e-5):
        raise ValueError('within-run COPE4-minus-COPE6 does not reproduce COPE7')
    weights = 1/variances; weights /= weights.sum(axis=0)
    combined = np.sum(weights[:, None, :]*conditions, axis=0)
    difference = np.sum(weights*cope7, axis=0)
    if not np.allclose(combined[1]-combined[2], difference, atol=1e-4, rtol=1e-5):
        raise ValueError('common-weight partner reconstruction failed')
    return combined, difference, weights


def extract_partners(model, reference, roi, hashes, sources):
    native, matched, audit = [], [], []
    for subject, value in zip(model['subjects'], model['images']):
        image = Path(value)
        if '/cope7.feat/stats/cope1.nii.gz' not in value or str(hashes.get(value, '')) == '':
            raise ValueError('unrecognized/missing frozen L2 COPE7 input')
        difference = load_roi(image, reference, roi, sources, hashes[value])
        values = [load_roi(Path(value.replace('/cope7.feat/', f'/cope{number}.feat/')),
                  reference, roi, sources) for number in PARTNERS.values()]
        native.append([float(v.mean()) for v in values]+[float(difference.mean())])
        # Do not use separately weighted L2 COPE4-minus-COPE6 as COPE7.
        try:
            run_conditions, run_difference, run_variance = [], [], []
            for run in ('01', '02'):
                folder = image.parents[3]/f'L1_run-{run}.feat'
                validate_l1(folder, sources)
                run_conditions.append([load_roi(folder/'stats'/f'cope{number}.nii.gz', reference, roi, sources)
                                       for number in PARTNERS.values()])
                run_difference.append(load_roi(folder/'stats/cope7.nii.gz', reference, roi, sources))
                run_variance.append(load_roi(folder/'stats/varcope7.nii.gz', reference, roi, sources))
            combined, reconstructed, weights = common_weights(run_conditions, run_difference, run_variance)
            error = float(np.max(np.abs(reconstructed-difference)))
            if not np.allclose(reconstructed, difference, atol=1e-4, rtol=1e-5):
                raise ValueError(f'reconstructed run-weighted contrast differs from retained L2 COPE7; max error {error:g}')
            matched.append([float(v.mean()) for v in combined]+[float(difference.mean())])
            audit.append(dict(subject=subject, verified=1, maximum_voxel_reconstruction_error=error,
                native_l2_difference_mismatch=float(np.mean(values[1]-values[2]-difference)),
                minimum_run_weight=float(weights.min()), maximum_run_weight=float(weights.max()), problem=''))
        except (ValueError, FileNotFoundError) as exc:
            # Preserve native estimates and explicit failure evidence; never mix
            # native and matched weighting or silently discard a participant.
            matched.append([np.nan]*4)
            audit.append(dict(subject=subject, verified=0, maximum_voxel_reconstruction_error='',
                native_l2_difference_mismatch=float(np.mean(values[1]-values[2]-difference)),
                minimum_run_weight='', maximum_run_weight='', problem=str(exc)))
        print(f'EXTRACTED: {subject}; common-weight reconstruction={audit[-1]["verified"]}', flush=True)
    return np.array(native), np.array(matched), audit


def motion_identity(model, y, fd_col):
    x, c = np.asarray(model['x']), np.asarray(model['c'])
    keep = [j for j in range(x.shape[1]) if j != fd_col]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    reduced = np.linalg.lstsq(x[:, keep], y, rcond=None)[0]
    gamma = np.linalg.lstsq(x[:, keep], x[:, fd_col], rcond=None)[0]
    full_gap = float(c[2]@beta); nofd_gap = float(c[2, keep]@reduced)
    product = float(beta[fd_col]*(c[2, keep]@gamma))
    if not np.isclose(nofd_gap-full_gap, product, atol=1e-8, rtol=1e-8):
        raise ValueError('omitted-FD regression identity failed')
    return dict(full_age_gap=full_gap, refitted_no_group_fd_age_gap=nofd_gap,
        change_when_adding_group_fd=full_gap-nofd_gap, full_fd_coefficient=float(beta[fd_col]),
        fd_age_gap_adjusted_for_other_covariates=float(c[2, keep]@gamma),
        omitted_fd_product=product, identity='no-FD age gap - full age gap = full FD coefficient x adjusted FD age gap')


def render(partner_rows, age_rows, model, y, fd_col, output, prefix):
    lookup = {(r['policy'], r['metric'], r['age_group']): r for r in partner_rows}
    display_policies = ('age-only', 'no-group-FD', 'full')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    for ax, policy in zip(axes, display_policies):
        for group, offset, color in zip(('younger', 'older'), (-.18, .18), COLORS):
            values = [lookup[policy, p, group] for p in PARTNERS]
            ax.bar(np.arange(3)+offset, [r['adjusted_mean'] for r in values], width=.34,
                yerr=[r['conditional_participant_sem'] for r in values], capsize=3, color=color, label=group.title())
        ax.set_xticks(range(3), ['Computer', 'Similar\nOwn-age', 'Dissimilar\nOther-age'])
        ax.axhline(0, color='.5', linewidth=.8); ax.set_title(POLICY_NAMES[policy], fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Offer-related DMN coupling (model units)'); axes[-1].legend(frameon=False)
    fig.suptitle('Selected right ventral frontal ROI: three-partner estimates\n'+
        ('Common run weights from COPE7; validated decomposition of the age contrast' if prefix == 'matched' else
         'Native condition-specific L2 estimates; different run weights across contrasts'), fontsize=12)
    fig.text(.04, .025, 'Means ± conditional participant SEM. Same 51 voxels and 47 participants; descriptive selected-ROI displays, not independent tests.\n'
        'No group FD removes only that covariate; L1 motion regressors and preprocessing remain unchanged. Computer is a contextual comparison, not part of COPE7.\n'
        'No additional smoothing, imaging fits, exclusions or new whole-brain inference. Anatomical boundary/coverage concern remains unresolved.', fontsize=8)
    fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.79, wspace=.15)
    fig.savefig(output/f'{prefix}_three_partner_bars.png', dpi=180); plt.close(fig)

    if prefix != 'native':
        return
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for group, offset, color in zip(('younger', 'older'), (-.18, .18), COLORS):
        values = [lookup[p, 'similar-minus-dissimilar', group] for p in POLICIES]
        ax.bar(np.arange(4)+offset, [r['adjusted_mean'] for r in values], width=.34,
               yerr=[r['conditional_participant_sem'] for r in values], capsize=4, color=color, label=group.title())
    ax.axhline(0, color='.5', linewidth=.8); ax.set_xticks(range(4), [POLICY_NAMES[p] for p in POLICIES], fontsize=9)
    ax.set_ylabel('Similar − dissimilar DMN offer coupling\nRetained participant COPE7 (model units)')
    ax.set_title('Same ROI and participant contrast: group-covariate sensitivity'); ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.05, .035, 'Means ± conditional participant SEM; nuisance terms are refitted for each policy.\n'
        'Selected-ROI sensitivity only: no new PALM p-values and no replacement of the primary full model.', fontsize=9)
    fig.subplots_adjust(left=.13, right=.98, bottom=.25, top=.88)
    fig.savefig(output/'age_contrast_motion_sensitivity.png', dpi=180); plt.close(fig)

    x = np.asarray(model['x']); fd = x[:, fd_col]
    keep = [j for j in range(x.shape[1]) if j != fd_col]
    yr = y-x[:, keep]@np.linalg.lstsq(x[:, keep], y, rcond=None)[0]
    fr = fd-x[:, keep]@np.linalg.lstsq(x[:, keep], fd, rcond=None)[0]
    young = x@np.asarray(model['c'][0]) == 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    for selection, color, name in ((young, COLORS[0], 'Younger'), (~young, COLORS[1], 'Older')):
        axes[0].scatter(fd[selection], y[selection], color=color, label=name, s=24)
        axes[1].scatter(fr[selection], yr[selection], color=color, s=24)
    axes[0].set(xlabel='Standardized mean FD', ylabel='Raw participant ROI COPE7', title='Raw age/FD association')
    axes[0].legend(frameon=False)
    slope = float(fr@yr/(fr@fr)); cuts = np.array([fr.min(), fr.max()])
    axes[1].plot(cuts, slope*cuts, color='black')
    axes[1].set(xlabel='FD residualized for age + other covariates', ylabel='COPE7 residualized for age + other covariates',
                title=f'Within-design partial FD association\nSlope = {slope:.3g}')
    for ax in axes: ax.spines[['top', 'right']].set_visible(False)
    fig.text(.04, .03, 'Descriptive selected-ROI associations, not causal motion effects; no exclusions. Partial plot includes all remaining design columns.', fontsize=8)
    fig.tight_layout(rect=[0, .08, 1, 1]); fig.savefig(output/'motion_association.png', dpi=180); plt.close(fig)


def build(repo, baseline, output):
    if output.exists():
        raise FileExistsError(f'will not overwrite diagnostic report: {output}')
    verify_inventory(baseline)
    config = repo/'results/dmn_covariate_standardized/v1/randomise/config.json'
    input_hashes = config.with_name('input_provenance.json')
    old_provenance = json.loads((baseline/'provenance.json').read_text())
    for path in (config, input_hashes):
        # Source paths were absolute on Linux1; do not guess remappings.
        if cp.digest(path) != old_provenance['sources'].get(str(path)):
            raise ValueError(f'primary bar source changed: {path}')
    model = next(m for m in json.loads(config.read_text())['models'] if m['name'] == 'full-rt__dmn-age')
    previous = bars.anatomy.read_rows(baseline/'participant_estimates.tsv')
    if [r['subject'] for r in previous] != model['subjects'] or len(previous) != 47:
        raise ValueError('primary extraction participant identity/order differs')
    mask = baseline/'selected_age_roi.nii.gz'; reference = nib.load(mask); roi = reference.get_fdata() > 0
    if int(roi.sum()) != 51:
        raise ValueError('exact primary 51-voxel ROI required')
    policies, columns = designs(model)
    output.mkdir(parents=True); cp.write_json(output/'status.json', dict(complete=False))
    sources = {str(p): cp.digest(p) for p in (config, input_hashes, mask, baseline/'inventory.tsv',
        baseline/'participant_estimates.tsv', Path(__file__), Path(bars.__file__))}
    native, matched, audits = extract_partners(model, reference, roi, json.loads(input_hashes.read_text()), sources)
    if not np.allclose(native[:, 3], [float(r['raw_roi_cope7']) for r in previous], atol=1e-8, rtol=1e-8):
        raise ValueError('re-extracted primary COPE7 differs from saved age bars')
    cp.write_rows(output/'run_weight_reconstruction.tsv', audits, list(audits[0]))
    matched_ok = all(r['verified'] == 1 for r in audits)
    metrics = list(PARTNERS)+['similar-minus-dissimilar']
    summary_rows, diagnostic_rows, participant_rows = [], [], []
    for representation, values in [('native', native)]+([('matched', matched)] if matched_ok else []):
        current = []
        for index, metric in enumerate(metrics):
            for policy in POLICIES:
                x, c = policies[policy]
                summaries, diagnostic, adjusted = fit_diagnostic(x, c, values[:, index])
                current.extend(dict(representation=representation, metric=metric, policy=policy, **r) for r in summaries)
                diagnostic_rows.append(dict(representation=representation, metric=metric, policy=policy, **diagnostic))
                for i, subject in enumerate(model['subjects']):
                    participant_rows.append(dict(subject=subject, representation=representation, metric=metric,
                        policy=policy, raw_roi_estimate=float(values[i, index]), adjusted_roi_estimate=float(adjusted[i])))
        summary_rows.extend(current)
        if representation == 'matched':
            lookup = {(r['policy'], r['metric'], r['age_group']): r for r in current}
            for p in POLICIES:
                for g in ('younger', 'older'):
                    predicted = lookup[p, 'similar', g]['adjusted_mean']-lookup[p, 'dissimilar', g]['adjusted_mean']
                    if not np.isclose(predicted, lookup[p, 'similar-minus-dissimilar', g]['adjusted_mean'], atol=1e-4, rtol=1e-5):
                        raise ValueError('matched adjusted condition bars do not reproduce primary difference')
        render(current, diagnostic_rows, model, native[:, 3], columns['fd_mean'], output, representation)
    baseline_full = {r['age_group']: r for r in bars.anatomy.read_rows(baseline/'bar_summary.tsv')}
    for row in summary_rows:
        if row['representation'] == 'native' and row['metric'] == 'similar-minus-dissimilar' and row['policy'] == 'full':
            if not np.isclose(row['adjusted_mean'], float(baseline_full[row['age_group']]['adjusted_mean']), atol=1e-8):
                raise ValueError('full model no longer reproduces the primary age bars')
    cp.write_rows(output/'bar_summary.tsv', summary_rows, list(summary_rows[0]))
    cp.write_rows(output/'age_contrast_diagnostics.tsv', diagnostic_rows, list(diagnostic_rows[0]))
    cp.write_rows(output/'participant_estimates.tsv', participant_rows, list(participant_rows[0]))
    identity = motion_identity(model, native[:, 3], columns['fd_mean'])
    cp.write_json(output/'motion_adjustment_identity.json', identity)
    groups = np.asarray(model['x'])@np.asarray(model['c'][0]) == 1
    covariate_rows = []
    for name, column in columns.items():
        v = np.asarray(model['x'])[:, column]
        for group, keep in [('younger', groups), ('older', ~groups)]:
            covariate_rows.append(dict(covariate=name, age_group=group, n=int(keep.sum()), mean=float(v[keep].mean()),
                sd=float(v[keep].std(ddof=1)), minimum=float(v[keep].min()), maximum=float(v[keep].max())))
    cp.write_rows(output/'covariate_distributions.tsv', covariate_rows, list(covariate_rows[0]))
    cp.write_json(output/'provenance.json', dict(sources=sources, roi='same selected 51 voxels; 47 participants retained',
        motion='Delete only group FD and refit sex/tSNR/RT and group terms. L1 motion confounds, preprocessing and images unchanged.',
        policies='Age only; full minus FD; full minus FD/tSNR; original full. No policy replaces primary inference.',
        native='Original partner-specific L2 COPE2/4/6 have contrast-specific fixed-effect run weights; do not subtract native bars to reproduce COPE7.',
        matched='All partners use inverse L1 VARCOPE7 weights at each ROI voxel; accepted only if reconstruction matches retained L2 COPE7 for every participant. Computer is NOT in COPE7.',
        matched_all_participants_verified=matched_ok,
        limitations='Same selected ROI: descriptive/circular, not independent p-values or whole-brain sensitivity. HC3 SE is descriptive; bars use conditional participant SEM. '
        'No exclusions, weight substitutions, imaging fits, new permutation tests or causal motion claim. Anatomical boundary/coverage question remains unresolved.'))
    cp.write_json(output/'status.json', dict(complete=matched_ok, native_diagnostics_complete=True,
        matched_decomposition_complete=matched_ok, n=47, excluded_participants=0, new_imaging_fits=0))
    files = [dict(path=str(p.relative_to(output)), sha256=cp.digest(p), bytes=p.stat().st_size)
             for p in sorted(output.rglob('*')) if p.is_file()]
    cp.write_rows(output/'inventory.tsv', files, ['path', 'sha256', 'bytes'])
    print(json.dumps(identity, indent=2)); print(f'Saved partner/motion diagnostics: {output}')
    if not matched_ok:
        raise RuntimeError('native plots saved, but exact common-weight decomposition needs review; inspect run_weight_reconstruction.tsv. No participants excluded or fits rerun.')
    print('PASS: all three partners; nested group-FD sensitivity; exact contrast decomposition verified for all 47 participants')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--baseline-bars', type=Path, default=DEFAULT_BARS)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    build(args.repository.resolve(), args.baseline_bars.resolve(), args.output_root.resolve())
