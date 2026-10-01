#!/usr/bin/env python3
"""Descriptive influence diagnostics in the selected corrected DMN cluster.

Uses the actual corrected cope-7 extraction and the exact corrected six-column
group design. OLS here is a diagnostic, NOT a replacement for voxelwise FLAME.
No p values or participant exclusion decisions are produced from this selected ROI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_l1_designs import read_vest_matrix, write_tsv
from prepare_dmn_condition_bar_models import parse_inputs, SOURCE_FSF_RELATIVE


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def align_rows(rows, inputs):
    indexed = {row['participant']: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {s for _, s, _ in inputs}:
        raise ValueError('participant membership or uniqueness differs from group design')
    ordered = [indexed[s] for _, s, _ in inputs]
    if any(int(row['design_index']) != index for row, (index, _, _) in zip(ordered, inputs)):
        raise ValueError('extraction design_index differs from corrected FSF')
    return ordered


def fit_diagnostics(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, p = x.shape
    if n <= p + 1 or np.linalg.matrix_rank(x) != p:
        raise ValueError('diagnostic design must be full rank with residual degrees of freedom')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite diagnostic input')
    pinv = np.linalg.pinv(x)
    beta = pinv @ y
    residual = y - x @ beta
    mse = float(residual @ residual / (n - p))
    leverage = np.einsum('ij,ji->i', x, pinv)
    if mse <= 0 or np.any(leverage >= 1 - 1e-10):
        raise ValueError('degenerate influence fit')
    student = residual / np.sqrt(mse * (1 - leverage))
    cooks = residual**2 * leverage / (p * mse * (1 - leverage)**2)
    c = np.array([1., -1., *([0.] * (p - 2))])
    full = float(c @ beta)
    loo = []
    for i in range(n):
        keep = np.arange(n) != i
        if np.linalg.matrix_rank(x[keep]) != p:
            raise ValueError(f'leave-one-out design loses rank at row {i+1}')
        loo.append(float(c @ np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]))
    return beta, residual, leverage, student, cooks, full, np.asarray(loo)


def audit(repo, output):
    design = repo / SOURCE_FSF_RELATIVE
    mat = design.with_suffix('.mat')
    roi = repo / 'results/manuscript/source_data/figure4_dmn_corrected_roi.tsv'
    inputs = parse_inputs(design.read_text().splitlines())
    rows = align_rows(read_rows(roi), inputs)
    x = read_vest_matrix(mat)
    if x.shape != (47, 6):
        raise ValueError('expected corrected 47 x 6 group design')
    if not np.array_equal(x[:, :2], [[int(r['age_group']=='younger'), int(r['age_group']=='older')] for r in rows]):
        raise ValueError('age labels do not match design')
    y = np.array([float(r['similar_minus_dissimilar']) for r in rows])
    beta, residual, h, student, cooks, full, loo = fit_diagnostics(x, y)
    output.mkdir(parents=True, exist_ok=True)
    details = []
    for i, row in enumerate(rows):
        details.append(dict(participant=row['participant'], age_group=row['age_group'],
                            design_index=i+1, corrected_cope7_roi=y[i],
                            fitted= float(x[i] @ beta), residual=residual[i], leverage=h[i],
                            internally_studentized_residual=student[i], cooks_distance=cooks[i],
                            cooks_above_4_over_n=int(cooks[i] > 4/len(y)),
                            loo_younger_minus_older=loo[i],
                            loo_change_from_full=loo[i]-full))
    write_tsv(output / 'participants.tsv', details)
    summary = dict(n=47, design_rank=6, diagnostic='OLS_corrected_selected_ROI_not_FLAME',
                   younger_minus_older=full, cooks_threshold=4/47,
                   n_cooks_flags=int(np.sum(cooks > 4/47)),
                   max_cooks=float(cooks.max()), max_abs_studentized=float(abs(student).max()),
                   loo_estimate_min=float(loo.min()), loo_estimate_max=float(loo.max()),
                   loo_sign_changes=int(np.sum(np.sign(loo) != np.sign(full))),
                   excluded_participants=0)
    write_tsv(output / 'summary.tsv', [summary])
    quality = []
    for col, name in ((3, 'tsnr'), (4, 'fd_mean')):
        target, predictors = x[:, col], np.delete(x, col, axis=1)
        error = target - predictors @ np.linalg.lstsq(predictors, target, rcond=None)[0]
        quality.append(dict(covariate=name, vif=float(np.sum((target-target.mean())**2)/np.sum(error**2)),
                            tsnr_fd_correlation=float(np.corrcoef(x[:, 3], x[:, 4])[0,1])))
    write_tsv(output / 'covariate_diagnostics.tsv', quality)
    sources = [design, mat, roi, Path(__file__)]
    metadata = {'sources': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'numpy_version': np.__version__,
                'interpretation': 'Selected-cluster descriptive diagnostics. No independent ROI inference, no exclusions, no proof of whole-brain robustness.'}
    (output / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    audit(args.repository.resolve(), args.repository / 'results/reviewer/dmn_corrected_influence')


if __name__ == '__main__':
    main()
