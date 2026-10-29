#!/usr/bin/env python3
"""Descriptive participant-SEM bars in the exact selected PALM age-effect ROI.

Unweighted OLS with the frozen PALM design, not FLAME/precision weighting. The
selected-ROI plot is circular/descriptive and supplies no independent p-value.
No imaging/permutation fits are launched and PALM scratch is read-only.
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
import plot_completed_palm_anatomy as anatomy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANATOMY = ROOT/'results/dmn_followup/v1/anatomy/attempt-20261029T003733Z-mGOh6z/outputs'


def adjusted_bars(x, contrasts, y):
    x, contrasts, y = np.asarray(x, float), np.asarray(contrasts, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid design or participant estimates')
    if np.linalg.matrix_rank(x) != x.shape[1] or contrasts.shape != (4, x.shape[1]):
        raise ValueError('unexpected rank or contrast dimensions')
    columns = []
    for c in contrasts[:2]:
        nonzero = np.flatnonzero(c)
        if len(nonzero) != 1 or c[nonzero[0]] != 1:
            raise ValueError('group contrast is not one group mean')
        columns.append(int(nonzero[0]))
    if len(set(columns)) != 2 or not np.array_equal(contrasts[2], contrasts[0]-contrasts[1]):
        raise ValueError('unexpected age comparison')
    group = x[:, columns]
    if not np.isin(group, [0, 1]).all() or not np.all(group.sum(axis=1) == 1):
        raise ValueError('age membership is not one-hot')
    nuisance = [i for i in range(x.shape[1]) if i not in columns]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    adjusted = y-x[:, nuisance]@beta[nuisance]
    rows = []
    for label, column in zip(('younger', 'older'), columns):
        keep = x[:, column] == 1
        if keep.sum() < 2:
            raise ValueError('not enough participants for SEM')
        mean = float(adjusted[keep].mean())
        if not np.isclose(mean, beta[column], atol=1e-8, rtol=1e-8):
            raise ValueError('adjusted group mean differs from OLS coefficient')
        rows.append(dict(age_group=label, n=int(keep.sum()), adjusted_mean=mean,
            conditional_participant_sem=float(adjusted[keep].std(ddof=1)/np.sqrt(keep.sum())),
            raw_mean=float(y[keep].mean()), raw_participant_sem=float(y[keep].std(ddof=1)/np.sqrt(keep.sum()))))
    return rows, adjusted, ['younger' if value else 'older' for value in group[:, 0]], float(contrasts[2]@beta)


def extract(model, mask, hashes):
    reference = nib.load(mask); roi = reference.get_fdata() > 0
    if len(reference.shape) != 3 or not roi.any():
        raise ValueError('invalid selected ROI')
    if len(model['images']) != len(model['subjects']) or len(set(model['subjects'])) != len(model['subjects']):
        raise ValueError('participant/image alignment differs')
    values, sources = [], {}
    for subject, name in zip(model['subjects'], model['images']):
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f'{subject}: missing retained participant image: {path}; run extraction on Linux1')
        current = cp.digest(path)
        if current != hashes.get(str(path)):
            raise ValueError(f'participant input differs from frozen PALM source: {path}')
        image = nib.load(path)
        if not anatomy.palm.pa.compatible(image, reference):
            raise ValueError(f'participant/ROI grids differ: {subject}')
        data = image.get_fdata()
        if not np.isfinite(data[roi]).all():
            raise ValueError(f'nonfinite participant values within selected ROI: {subject}')
        # Include genuine zero COPE values; no FSL -M/nonzero-voxel mean.
        values.append(float(data[roi].mean())); sources[str(path)] = current
    return np.array(values), sources


def plot(rows, output):
    fig, ax = plt.subplots(figsize=(7.4, 6.1))
    ax.bar([0, 1], [r['adjusted_mean'] for r in rows],
           yerr=[r['conditional_participant_sem'] for r in rows],
           color=['#397b9d', '#b86c3e'], width=.58, capsize=6,
           error_kw=dict(elinewidth=1.4, capthick=1.4))
    ax.set_xticks([0, 1], [f'Younger\n(n = {rows[0]["n"]})', f'Older\n(n = {rows[1]["n"]})'])
    ax.axhline(0, color='.4', linewidth=.8)
    ax.set_ylabel('Similar − dissimilar offer-related DMN coupling\nCovariate-adjusted participant COPE7 (model units)')
    ax.set_title('Right ventral frontal selected region\nMean ± participant SEM', fontsize=13, pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.09, .065, 'Selected using the same younger > older contrast: descriptive, not independent inference.\n'
             'Adjusted for sex, tSNR, motion and mean RT; unweighted, matching the PALM design.\n'
             'SEM describes participant dispersion conditional on fitted nuisance slopes; not a corrected test.\n'
             'Full-display + RT-height model. Anatomical coverage/alignment remains under review.', fontsize=8)
    fig.subplots_adjust(left=.19, right=.96, top=.84, bottom=.25)
    fig.savefig(output/'dmn_age_selected_roi_bars.png', dpi=200, facecolor='white')
    plt.close(fig)


def build(repo, anatomical_report, output):
    if output.exists():
        raise FileExistsError(f'will not overwrite existing bar report: {output}')
    # Validate every saved anatomical artifact before selecting this exact mask.
    if not json.loads((anatomical_report/'status.json').read_text())['complete']:
        raise ValueError('anatomical export incomplete')
    for row in anatomy.read_rows(anatomical_report/'inventory.tsv'):
        path = (anatomical_report/row['path']).resolve()
        if not path.is_relative_to(anatomical_report.resolve()) or cp.digest(path) != row['sha256']:
            raise ValueError('anatomical export checksum mismatch')
    provenance = json.loads((anatomical_report/'provenance.json').read_text())
    contract = json.loads(Path(provenance['collection']).joinpath('contract.json').read_text())
    config_path = repo/'results/dmn_covariate_standardized/v1/randomise/config.json'
    hashes_path = config_path.with_name('input_provenance.json')
    if cp.digest(config_path) != contract['source'] or cp.digest(hashes_path) != contract['input_provenance']:
        raise ValueError('PALM source configuration or input provenance differs')
    model = next(m for m in json.loads(config_path.read_text())['models'] if m['name'] == 'full-rt__dmn-age')
    if model['labels'] != ['younger', 'older', 'younger-older', 'older-younger'] or len(model['subjects']) != 47:
        raise ValueError('unexpected age model/participant sample')
    mask = anatomical_report/'maps/full-rt__dmn-age/cluster-z2.6/contrast3_significant_mask.nii.gz'
    reference = nib.load(mask)
    selected = reference.get_fdata() > 0
    summary = next(r for r in anatomy.read_rows(anatomical_report/'native_summary.tsv')
        if r['model'] == model['name'] and r['method'] == 'cluster-z2.6' and r['contrast'] == '3')
    if selected.sum() != 51 or int(summary['significant_voxels']) != 51 or not np.isclose(float(summary['minimum_spatial_fwe_p']), .0434, atol=1e-12, rtol=0):
        raise ValueError('not the locked 51-voxel p=.0434 age-effect region')
    y, sources = extract(model, mask, json.loads(hashes_path.read_text()))
    rows, adjusted, groups, difference = adjusted_bars(model['x'], model['c'], y)
    if [r['n'] for r in rows] != [25, 22]:
        raise ValueError('expected 25 younger and 22 older participants')
    output.mkdir(parents=True)
    cp.write_json(output/'status.json', dict(complete=False))
    cp.write_rows(output/'bar_summary.tsv', rows, list(rows[0]))
    participants = [dict(subject=s, age_group=g, raw_roi_cope7=float(a), adjusted_roi_cope7=float(b))
        for s, g, a, b in zip(model['subjects'], groups, y, adjusted)]
    cp.write_rows(output/'participant_estimates.tsv', participants, list(participants[0]))
    cp.write_json(output/'group_design.json', dict(subjects=model['subjects'], x=model['x'], c=model['c'], labels=model['labels']))
    nib.save(reference, output/'selected_age_roi.nii.gz')
    plot(rows, output)
    for path in (config_path, hashes_path, mask, anatomical_report/'inventory.tsv',
                 Path(__file__), Path(cp.__file__)):
        sources[str(path)] = cp.digest(path)
    cp.write_json(output/'provenance.json', dict(sources=sources, selected_native_voxels=51,
        group_difference_ols=difference, selection='PALM full-rt DMN-age contrast3, cluster Z=2.6, spatial FWE p<.05',
        contrast='younger minus older of within-participant similar minus dissimilar offer-related network PPI COPE7',
        adjustment='Unweighted OLS using frozen PALM canonical group design; subtract four nuisance contributions, leaving group coefficients',
        error_bars='Within-group SD of adjusted participant estimates / sqrt(n), ddof=1; conditional on fitted nuisance slopes; not inferential coefficient SE',
        interpretation='ROI selected using same age difference: descriptive/circular visualization, no independent ROI p-value. '
        'No FLAME/variance weights, no new permutation test, no new participant exclusions. Does not separately isolate decision, post-response, fairness versus own monetary value, or causality.',
        anatomy_status='Boundary/coverage concern unresolved; a bar plot does not validate localization.'))
    cp.write_json(output/'status.json', dict(complete=True, n=47, new_imaging_fits=0, excluded_participants=0))
    files = [dict(path=str(p.relative_to(output)), sha256=cp.digest(p), bytes=p.stat().st_size)
             for p in sorted(output.rglob('*')) if p.is_file()]
    cp.write_rows(output/'inventory.tsv', files, ['path', 'sha256', 'bytes'])
    print('PASS: exact 51-voxel PALM ROI; 47 verified participant images; descriptive adjusted means ± participant SEM')
    print(json.dumps(rows, indent=2)); print(f'Saved bar plot and source estimates: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--anatomical-report', type=Path, default=DEFAULT_ANATOMY)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    build(args.repository.resolve(), args.anatomical_report.resolve(), args.output_root.resolve())
