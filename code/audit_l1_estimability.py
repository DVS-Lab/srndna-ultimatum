#!/usr/bin/env python3
"""Read-only L1 rank attribution and saved-contrast estimability audit.

Reads design.mat/.con/.fsf only; writes compact diagnostics, never FEAT output.
Exact-zero columns are distinguished from constant nonzero columns. Contrasts
are tested against the row space of the column-normalized design, with the
contrast transformed into the same parameterization. A second, deliberately
stricter singular-value cutoff is a numerical-sensitivity flag, not a new fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from audit_l1_designs import read_sample, read_vest_matrix, write_tsv


def settings(path):
    values = {}
    for line in path.read_text().splitlines():
        match = re.match(r'^set fmri\(([^)]+)\)\s+(.*)$', line.strip())
        if match:
            values[match[1]] = match[2].strip('"')
    return values


def column_labels(config, ncolumns):
    original = int(config.get('evs_orig', 0))
    real = int(config.get('evs_real', 0))
    direct = original == real and real > 0 and all(
        config.get(f'deriv_yn{i}', '0') == '0' for i in range(1, original + 1))
    labels = []
    for i in range(1, ncolumns + 1):
        if direct and i <= real:
            label = config.get(f'evtitle{i}', f'EV{i}')
            kind = 'interaction' if config.get(f'shape{i}') == '4' else 'EV'
            labels.append(f'{i}:{label}[{kind}]')
        else:
            labels.append(f'{i}:unmapped' if i <= real else f'{i}:appended_nuisance')
    return labels


def diagnostics(matrix, contrasts, sensitivity_rcond=1e-8, tolerance=1e-7):
    if not np.isfinite(matrix).all() or not np.isfinite(contrasts).all():
        raise ValueError('nonfinite design or contrast')
    if matrix.shape[1] != contrasts.shape[1]:
        raise ValueError('design/contrast column count mismatch')
    norms = np.linalg.norm(matrix, axis=0)
    nonzero = norms != 0
    scales = np.where(nonzero, norms, 1.0)
    normalized = matrix / scales
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    cutoff = max(matrix.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    conservative_rank = int(np.count_nonzero(singular > sensitivity_rcond * singular[0]))
    basis = vt[:rank]
    conservative_basis = vt[:conservative_rank]
    transformed = contrasts / scales

    def errors(b):
        residual = transformed - (transformed @ b.T) @ b
        sizes = np.linalg.norm(transformed, axis=1)
        return np.linalg.norm(residual, axis=1) / np.where(sizes > 0, sizes, 1.0)

    ordinary_errors = errors(basis)
    conservative_errors = errors(conservative_basis)
    projection_diagonal = np.sum(basis ** 2, axis=0)
    result = {
        'npoints': matrix.shape[0], 'ncolumns': matrix.shape[1],
        'raw_rank': int(np.linalg.matrix_rank(matrix)),
        'nonzero_columns': int(nonzero.sum()), 'scaled_rank': rank,
        'remaining_deficiency_after_zero_removal': int(nonzero.sum()) - rank,
        'sensitivity_rank_rcond_1e_8': conservative_rank,
        'condition_nonzero_columns': float(np.linalg.cond(normalized[:, nonzero])) if nonzero.any() else 'inf',
        'svd_cutoff': float(cutoff),
        'zero_columns': (np.flatnonzero(~nonzero) + 1).tolist(),
        'constant_nonzero_columns': (np.flatnonzero(nonzero & (np.ptp(matrix, axis=0) == 0)) + 1).tolist(),
        'nonzero_columns_in_nullspace': (np.flatnonzero(nonzero & (1 - projection_diagonal > tolerance)) + 1).tolist(),
    }
    checks = []
    for i, (row, error, sensitive_error) in enumerate(zip(contrasts, ordinary_errors, conservative_errors), 1):
        checks.append({
            'contrast': i, 'nonzero_weights': json.dumps({str(j+1): float(v) for j, v in enumerate(row) if v != 0}),
            'zero_contrast': int(not np.any(row)),
            'relative_projection_error': float(error),
            'estimable': int(error <= tolerance and np.any(row)),
            'sensitivity_projection_error_rcond_1e_8': float(sensitive_error),
            'sensitivity_estimable': int(sensitive_error <= tolerance and np.any(row)),
            'weights_on_zero_columns': '|'.join(str(j+1) for j in np.flatnonzero((~nonzero) & (row != 0))),
        })
    return result, checks


def inspect_design(feat):
    paths = [feat / name for name in ('design.mat', 'design.con', 'design.fsf')]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError('missing: ' + '; '.join(missing))
    matrix, contrasts = (read_vest_matrix(p) for p in paths[:2])
    config = settings(paths[2])
    padding = 0
    if contrasts.shape[1] < matrix.shape[1] and contrasts.shape[1] == int(config['evs_real']):
        padding = matrix.shape[1] - contrasts.shape[1]
        contrasts = np.pad(contrasts, ((0, 0), (0, padding)))
    result, checks = diagnostics(matrix, contrasts)
    labels = column_labels(config, matrix.shape[1])
    for field in ('zero_columns', 'constant_nonzero_columns', 'nonzero_columns_in_nullspace'):
        result[field + '_labels'] = '|'.join(labels[i-1] for i in result[field])
        result[field] = '|'.join(map(str, result[field]))
    result['contrast_padding_nuisance_zeros'] = padding
    result['column_labels'] = '|'.join(labels)
    for path in paths:
        result[path.name.replace('.', '_') + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for check in checks:
        check['contrast_name'] = config.get(f"conname_real.{check['contrast']}", '')
    return result, checks


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_tsv(path, [{key: row.get(key, '') for key in fields} for row in rows])


def audit(root, repair_root, sample, output):
    for source_root in (root, repair_root):
        if output.resolve() == source_root.resolve() or source_root.resolve() in output.resolve().parents:
            raise ValueError('audit output must be outside the input derivative trees')
    runs, contrasts = [], []
    for participant in read_sample(sample):
        for run in ('01', '02'):
            for model in ('act', 'nppi-dmn', 'nppi-ecn'):
                sources = [('production', root)]
                if participant == 'sub-144':
                    sources.append(('repaired', repair_root))
                for source, base in sources:
                    feat = base / participant / f'L1_task-ultimatum_model-02_type-{model}_run-{run}_sm-6.feat'
                    identity = dict(source=source, participant=participant, run=run, model=model)
                    row = dict(identity, feat_dir=str(feat), status='ok', error='')
                    try:
                        result, checks = inspect_design(feat)
                        row.update(result)
                        contrasts.extend(dict(identity, **check) for check in checks)
                    except (ValueError, OSError, KeyError, np.linalg.LinAlgError) as error:
                        row.update(status='error', error=str(error))
                    runs.append(row)
    summaries = []
    for scope in ('production', 'corrected'):
        for model in ('act', 'nppi-dmn', 'nppi-ecn'):
            def selected(row):
                source = 'repaired' if scope == 'corrected' and row['participant'] == 'sub-144' else 'production'
                return row['model'] == model and row['source'] == source
            selected_runs = [r for r in runs if selected(r)]
            good = [r for r in selected_runs if r['status'] == 'ok']
            selected_contrasts = [r for r in contrasts if selected(r)]
            focal = [r for r in selected_contrasts if r['contrast'] in (4, 6, 7)]
            summary = dict(
                scope=scope, model=model, runs_expected=len(selected_runs), runs_audited=len(good),
                runs_missing_or_invalid=len(selected_runs)-len(good),
                raw_rank_deficient_runs=sum(r['raw_rank'] < r['ncolumns'] for r in good),
                runs_with_zero_columns=sum(bool(r['zero_columns']) for r in good),
                runs_deficient_after_zero_removal=sum(r['remaining_deficiency_after_zero_removal'] > 0 for r in good),
                contrasts_checked=len(selected_contrasts),
                nonestimable_contrasts=sum(not r['estimable'] for r in selected_contrasts),
                focal_contrasts_expected=3*len(selected_runs), focal_contrasts_checked=len(focal),
                focal_nonestimable=sum(not r['estimable'] for r in focal),
                focal_sensitivity_flags=sum(not r['sensitivity_estimable'] for r in focal),
            )
            summaries.append(summary)
            print(f"{scope} {model}: {len(good)}/{len(selected_runs)} designs; "
                  f"remaining rank deficiencies={summary['runs_deficient_after_zero_removal']}; "
                  f"focal nonestimable={summary['focal_nonestimable']}/{len(focal)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'runs.tsv', runs)
    if contrasts:
        write_rows(output / 'contrasts.tsv', contrasts)
    else:
        (output / 'contrasts.tsv').write_text('source\tparticipant\trun\tmodel\tcontrast\testimable\n')
    write_rows(output / 'summary.tsv', summaries)
    metadata = dict(numpy_version=np.__version__, sample=str(sample),
                    sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    projection_tolerance=1e-7, sensitivity_rcond=1e-8,
                    note='Saved unwhitened design algebra only; no images fitted; no columns removed from actual models.')
    (output / 'audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    incomplete = any(r['status'] != 'ok' for r in runs) or any(s['focal_contrasts_checked'] != s['focal_contrasts_expected'] for s in summaries)
    flagged = any(s['focal_nonestimable'] or s['focal_sensitivity_flags'] for s in summaries)
    print('INCOMPLETE: missing/invalid inputs' if incomplete else 'REVIEW: focal contrast flags' if flagged else 'PASS: focal contrasts estimable at both audited cutoffs; inspect full tables for other contrasts')
    return 2 if incomplete else 1 if flagged else 0


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--l1-root', type=Path, required=True)
    parser.add_argument('--repair-l1-root', type=Path, required=True)
    parser.add_argument('--sample', type=Path, default=repo / 'behavioral_analyses/data/participant_L3_47.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    return audit(args.l1_root, args.repair_l1_root, args.sample, args.output_dir)


if __name__ == '__main__':
    raise SystemExit(main())
