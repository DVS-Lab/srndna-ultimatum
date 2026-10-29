#!/usr/bin/env python3
"""Read-only audit of nonpositive run VARCOPE7 and selected-ROI coverage.

Audit all 47 participants for context; detailed signal checks and overlays for
the three flagged cases. No masking changes, reweighting, exclusions or fits.
Available-run calculations below are comparisons with existing L2 outputs, not
a new estimator or repair. Missing optional artifacts stay explicitly unknown.
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
from nibabel.processing import resample_from_to
import numpy as np

import audit_palm_partner_motion as pm
import collect_palm_progress as cp
from plot_completed_palm_anatomy import plane

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT/'results/dmn_followup/v1/partner_motion/attempt-20261029T014616Z-OgmBdc/outputs'
FLAGGED = ('sub-108', 'sub-112', 'sub-125')


def variance_comparison(run_copes, run_variances, l2_cope, l2_variance):
    """Test numerical hypotheses; never turn a zero variance into high precision."""
    c, v = np.asarray(run_copes, float), np.asarray(run_variances, float)
    if c.shape != v.shape or c.ndim != 2 or c.shape[0] != 2:
        raise ValueError('expected two runs by selected voxels')
    valid = np.isfinite(v) & (v > 0) & np.isfinite(c)
    weights = np.zeros(v.shape)
    np.divide(1, v, out=weights, where=valid)
    total = weights.sum(axis=0)
    supported = total > 0
    predicted = np.full(total.shape, np.nan); predicted_var = predicted.copy()
    np.divide(np.sum(weights*np.where(valid, c, 0), axis=0), total, out=predicted, where=supported)
    np.divide(1, total, out=predicted_var, where=supported)
    cope_ok = supported & np.isfinite(l2_cope) & np.isclose(predicted, l2_cope, atol=1e-4, rtol=1e-5)
    var_ok = supported & np.isfinite(l2_variance) & np.isclose(predicted_var, l2_variance, atol=1e-4, rtol=1e-5)
    return dict(positive_variance_runs=valid.sum(axis=0), expected_if_only_positive_runs_used=predicted,
        expected_variance_if_only_positive_runs_used=predicted_var, existing_l2_cope_agrees=cope_ok,
        existing_l2_variance_agrees=var_ok)


def classify(variance, mask_value=None, temporal_sd=None):
    if not np.isfinite(variance):
        return 'nonfinite_variance'
    if variance < 0:
        return 'negative_variance'
    if variance > 0:
        return 'positive_variance_outside_run_mask' if mask_value is not None and mask_value <= 0 else 'positive_variance'
    if mask_value is None:
        return 'zero_variance_mask_unavailable'
    if not np.isfinite(mask_value):
        return 'zero_variance_nonfinite_run_mask'
    if mask_value <= 0:
        return 'zero_variance_outside_run_mask'
    if temporal_sd is None:
        return 'zero_variance_inside_run_mask_signal_unavailable'
    if not np.isfinite(temporal_sd):
        return 'zero_variance_inside_run_mask_nonfinite_signal'
    return 'zero_variance_inside_run_mask_flat_signal' if temporal_sd == 0 else 'zero_variance_inside_run_mask_variable_signal'


class Inputs:
    def __init__(self, reference, expected):
        self.reference, self.expected, self.rows = reference, expected, []

    def load(self, path, role, optional=False, time_series=False):
        if not path.is_file():
            self.rows.append(dict(path=str(path), role=role, exists=0, sha256='', shape='', expected_hash_checked=0))
            if optional:
                return None
            raise FileNotFoundError(f'required audit input missing: {path}')
        digest = cp.digest(path)
        if str(path) in self.expected and digest != self.expected[str(path)]:
            raise ValueError(f'input differs from completed analysis provenance: {path}')
        image = nib.load(path)
        shape_ok = image.shape[:3] == self.reference.shape and (len(image.shape) == 4 if time_series else len(image.shape) == 3)
        if not shape_ok or not np.allclose(image.affine, self.reference.affine, atol=1e-5, rtol=0):
            raise ValueError(f'native audit grid mismatch; no input resampling guessed: {path}')
        self.rows.append(dict(path=str(path), role=role, exists=1, sha256=digest,
            shape='x'.join(map(str, image.shape)), expected_hash_checked=int(str(path) in self.expected)))
        return image.get_fdata(dtype=np.float32)


def temporal_summary(data, roi):
    if data is None:
        return None, None
    selected = data[roi]
    if selected.shape[1] < 2:
        raise ValueError('fewer than two retained volumes')
    return np.std(selected, axis=1, ddof=1), np.sum(~np.isfinite(selected), axis=1)


def save_map(path, data, reference):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), reference.affine), path)


def overlay(subject, runs, l2, roi, reference, standard, output):
    background = nib.as_closest_canonical(nib.load(standard))
    roi_count = int(roi.sum())
    def display(data, order):
        return resample_from_to(nib.Nifti1Image(np.asarray(data, float), reference.affine), background, order=order).get_fdata()
    invalid = np.logical_or.reduce([~np.isfinite(r['variance']) | (r['variance'] <= 0) for r in runs]) & roi
    candidates = np.argwhere(invalid if invalid.any() else roi)
    center = candidates.mean(axis=0)
    selected = candidates[np.argmin(np.sum((candidates-center)**2, axis=1))]
    cut = nib.affines.apply_affine(reference.affine, selected)
    roi_display = display(roi, 0)
    means = [r['mean'] for r in runs if r['mean'] is not None]
    average = np.mean(means, axis=0) if means else None
    panels = runs+[dict(variance=l2['variance'], mask=l2['mask'], mean=average)]
    fig = plt.figure(figsize=(15, 11.6), facecolor='white')
    grid = fig.add_gridspec(3, 4, left=.035, right=.98, top=.83, bottom=.13, hspace=.65, wspace=.1)
    fig.text(.035, .965, f'{subject}: selected-ROI VARCOPE and coverage audit', fontsize=18, weight='bold')
    fig.text(.035, .925, f'Cyan: unchanged {roi_count}-voxel ROI. Red: nonfinite/nonpositive variance. Yellow: existing FEAT mask edge.', fontsize=10)
    fig.text(.035, .90, 'Cuts pass through an invalid-run voxel nearest their centroid; shared across rows. These are coverage maps, not statistical effects.', fontsize=10)
    for i, record in enumerate(panels):
        bad = roi & (~np.isfinite(record['variance']) | (record['variance'] <= 0))
        bad_display = display(bad, 0)
        mask_display = display(record['mask'] > 0, 0) if record['mask'] is not None else None
        epi = display(record['mean'], 1) if record['mean'] is not None else None
        top = grid[i, 0].get_position(fig).y1
        title = f'Run {i+1:02d}' if i < 2 else 'Retained L2 COPE7'
        desc = f'{int(bad.sum())}/{roi_count} invalid-variance voxels; '
        desc += 'FEAT mask unavailable' if record['mask'] is None else f'{int(np.sum(roi & (record["mask"] > 0)))}/{roi_count} inside FEAT mask'
        if i == 2:
            desc += '; EPI underlay is display-only average of available run means'
        fig.text(.035, top+.046, title, fontsize=12, weight='bold'); fig.text(.035, top+.025, desc, fontsize=9)
        for column, axis in enumerate((0, 1, 2, 2)):
            ax = fig.add_subplot(grid[i, column]); ax.set_facecolor('black')
            use_t1 = column == 3 or epi is None
            base = background.get_fdata() if use_t1 else epi
            finite = base[np.isfinite(base) & (base > 0)]
            vmax = float(np.percentile(finite, 99.5)) if finite.size else 1
            image, extent, actual = plane(base, background.affine, axis, cut[axis])
            ax.imshow(image, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=vmax)
            b, _, _ = plane(bad_display, background.affine, axis, cut[axis])
            ax.imshow(np.ma.masked_where(b == 0, b), origin='lower', extent=extent, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
            for data, color in ((mask_display, '#ffe35b'), (roi_display, '#24dbe0')):
                if data is not None:
                    value, _, _ = plane(data, background.affine, axis, cut[axis])
                    if value.any() and not value.all():
                        ax.contour(value, levels=[.5], origin='lower', extent=extent, colors=[color], linewidths=.8)
            ax.set_title(f'{"MNI T1" if use_t1 else "Mean EPI"}: {"xyz"[axis]} = {actual:.1f}', fontsize=10)
            if column == 3:
                ax.set_xlim(cut[0]-24, cut[0]+24); ax.set_ylim(cut[1]-24, cut[1]+24)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(.02, .03, 'P' if axis == 0 else 'L', color='white', transform=ax.transAxes)
            ax.text(.94, .03, 'A' if axis == 0 else 'R', color='white', transform=ax.transAxes)
    fig.text(.035, .05, 'Native values/counts are in TSV tables. Resampling is for display only; no mask trimming or signal repair.\n'
        'A skull-stripped template edge is not ground truth for EPI coverage. Missing mean EPI uses explicitly labeled T1 underlay.\n'
        'No participants, runs or voxels are removed from the analysis; no fits or PALM processes are changed.', fontsize=9)
    fig.savefig(output/f'{subject}_coverage_varcope.png', dpi=160); plt.close(fig)
    return dict(subject=subject, x_mni_mm=float(cut[0]), y_mni_mm=float(cut[1]), z_mni_mm=float(cut[2]),
                cut_rule='invalid-variance voxel nearest their centroid, or ROI centroid fallback')


def build(repo, source, baseline, standard, output, flagged=FLAGGED):
    if output.exists():
        raise FileExistsError(f'will not overwrite existing diagnostic: {output}')
    if not standard.is_file():
        raise FileNotFoundError(f'FSL anatomical underlay missing: {standard}; provide --standard')
    # The source partner diagnostic is intentionally partial; validate its
    # inventory and native-completion flag, not an absent matched decomposition.
    state = json.loads((source/'status.json').read_text())
    if not state['native_diagnostics_complete']:
        raise ValueError('source native diagnostic incomplete')
    for row in pm.bars.anatomy.read_rows(source/'inventory.tsv'):
        path = (source/row['path']).resolve()
        if not path.is_relative_to(source.resolve()) or cp.digest(path) != row['sha256']:
            raise ValueError(f'source diagnostic checksum mismatch: {path}')
    failures = [r['subject'] for r in pm.bars.anatomy.read_rows(source/'run_weight_reconstruction.tsv') if r['verified'] != '1']
    if set(failures) != set(flagged):
        raise ValueError('flagged participants differ from saved diagnostic; review before changing scope')
    pm.verify_inventory(baseline)
    config = repo/'results/dmn_covariate_standardized/v1/randomise/config.json'
    hash_path = config.with_name('input_provenance.json')
    previous = json.loads((source/'provenance.json').read_text())['sources']
    for p in (config, hash_path):
        if cp.digest(p) != previous.get(str(p)):
            raise ValueError('primary PALM source configuration/provenance differs')
    frozen = json.loads(hash_path.read_text()); expected = {**previous, **frozen}
    model = next(m for m in json.loads(config.read_text())['models'] if m['name'] == 'full-rt__dmn-age')
    if len(model['subjects']) != 47 or len(set(model['subjects'])) != 47:
        raise ValueError('locked 47-participant sample required')
    if not set(flagged) <= set(model['subjects']):
        raise ValueError('flagged participant missing from locked sample')
    roi_path = baseline/'selected_age_roi.nii.gz'; reference = nib.load(roi_path); roi = reference.get_fdata() > 0
    if int(roi.sum()) != 51:
        raise ValueError('exact primary 51-voxel mask required')
    x, c = np.array(model['x']), np.array(model['c']); young = x@c[0] == 1
    reader = Inputs(reference, expected)
    output.mkdir(parents=True); cp.write_json(output/'status.json', dict(complete=False))
    ijk = np.argwhere(roi); xyz = nib.affines.apply_affine(reference.affine, ijk)
    run_rows, voxel_rows, l2_rows, cuts, support_rows = [], [], [], [], []
    for index, (subject, image_name) in enumerate(zip(model['subjects'], model['images'])):
        image = Path(image_name); stats = image.parent
        l2_cope = reader.load(image, 'retained L2 COPE7')
        l2_var = reader.load(stats/'varcope1.nii.gz', 'retained L2 VARCOPE7')
        l2_mask = reader.load(stats.parent/'mask.nii.gz', 'L2 FEAT mask', optional=True)
        records = []
        for run in ('01', '02'):
            folder = image.parents[3]/f'L1_run-{run}.feat'
            variance = reader.load(folder/'stats/varcope7.nii.gz', 'L1 VARCOPE7')
            cope = reader.load(folder/'stats/cope7.nii.gz', 'L1 COPE7')
            mask = reader.load(folder/'mask.nii.gz', 'L1 FEAT mask', optional=True)
            mean, sd, nonfinite, sigma = None, None, None, None
            if subject in flagged:
                mean = reader.load(folder/'mean_func.nii.gz', 'L1 mean EPI', optional=True)
                signal = reader.load(folder/'filtered_func_data.nii.gz', 'L1 filtered signal', optional=True, time_series=True)
                sd, nonfinite = temporal_summary(signal, roi)
                del signal
                sigma = reader.load(folder/'stats/sigmasquareds.nii.gz', 'L1 residual variance', optional=True)
                for role, data in [('mean_epi', mean), ('feat_mask', mask), ('varcope7', variance), ('cope7', cope), ('residual_variance', sigma)]:
                    if data is not None:
                        save_map(output/'maps'/subject/f'run-{run}_{role}.nii.gz', data, reference)
            bad = ~np.isfinite(variance[roi]) | (variance[roi] <= 0)
            covered = None if mask is None else mask[roi] > 0
            outside = None if mask is None else np.isfinite(mask[roi]) & (mask[roi] <= 0)
            row = dict(subject=subject, age_group='younger' if young[index] else 'older', run=run, roi_voxels=51,
                variance_zero=int(np.sum(variance[roi] == 0)), variance_negative=int(np.sum(variance[roi] < 0)),
                variance_nonfinite=int(np.sum(~np.isfinite(variance[roi]))), variance_positive=int(np.sum(np.isfinite(variance[roi]) & (variance[roi] > 0))),
                run_mask_found=int(mask is not None), mask_covered_voxels='' if covered is None else int(covered.sum()),
                bad_variance_inside_mask='' if covered is None else int((bad & covered).sum()),
                bad_variance_outside_mask='' if outside is None else int((bad & outside).sum()),
                bad_variance_mask_unknown=int(bad.sum()) if mask is None else int((bad & ~np.isfinite(mask[roi])).sum()),
                bad_variance_with_variable_signal='' if sd is None else int((bad & np.isfinite(sd) & (sd > 0)).sum()),
                mean_epi_found=int(mean is not None), filtered_signal_audited=int(sd is not None), residual_variance_found=int(sigma is not None))
            run_rows.append(row)
            records.append(dict(variance=variance, cope=cope, mask=mask, mean=mean, temporal_sd=sd,
                nonfinite_signal=nonfinite, residual_variance=sigma, summary=row))
        comparison = variance_comparison(np.array([r['cope'][roi] for r in records]),
            np.array([r['variance'][roi] for r in records]), l2_cope[roi], l2_var[roi])
        counts = comparison['positive_variance_runs']
        for voxel in range(len(ijk)):
            support_rows.append(dict(subject=subject, age_group='younger' if young[index] else 'older', voxel=voxel+1,
                positive_variance_runs=int(counts[voxel]), l2_mask_value='' if l2_mask is None else float(l2_mask[roi][voxel]),
                l2_variance=float(l2_var[roi][voxel])))
        l2_rows.append(dict(subject=subject, age_group='younger' if young[index] else 'older', roi_voxels=51,
            two_positive_variance_runs=int(np.sum(counts == 2)), one_positive_variance_run=int(np.sum(counts == 1)),
            no_positive_variance_runs=int(np.sum(counts == 0)), l2_mask_found=int(l2_mask is not None),
            l2_mask_covered_voxels='' if l2_mask is None else int(np.sum(l2_mask[roi] > 0)),
            l2_positive_variance_voxels=int(np.sum(np.isfinite(l2_var[roi]) & (l2_var[roi] > 0))),
            l2_cope_matches_both_positive_runs=int(np.sum((counts == 2) & comparison['existing_l2_cope_agrees'])),
            l2_cope_matches_single_positive_run=int(np.sum((counts == 1) & comparison['existing_l2_cope_agrees'])),
            l2_variance_matches_both_positive_runs=int(np.sum((counts == 2) & comparison['existing_l2_variance_agrees'])),
            l2_variance_matches_single_positive_run=int(np.sum((counts == 1) & comparison['existing_l2_variance_agrees']))))
        if subject in flagged:
            for j, coord in enumerate(ijk):
                row = dict(subject=subject, voxel=j+1, i=int(coord[0]), j=int(coord[1]), k=int(coord[2]),
                    x_mni_mm=float(xyz[j, 0]), y_mni_mm=float(xyz[j, 1]), z_mni_mm=float(xyz[j, 2]))
                for run, record in zip(('01', '02'), records):
                    var, cop = record['variance'][roi], record['cope'][roi]
                    mask = record['mask']; mask_value = None if mask is None else float(mask[roi][j])
                    sd = record['temporal_sd']; temporal = None if sd is None else float(sd[j])
                    row.update({f'run{run}_varcope7': float(var[j]), f'run{run}_cope7': float(cop[j]),
                        f'run{run}_feat_mask': '' if mask_value is None else mask_value,
                        f'run{run}_mean_epi': '' if record['mean'] is None else float(record['mean'][roi][j]),
                        f'run{run}_temporal_sd': '' if temporal is None else temporal,
                        f'run{run}_nonfinite_signal_samples': '' if record['nonfinite_signal'] is None else int(record['nonfinite_signal'][j]),
                        f'run{run}_residual_variance': '' if record['residual_variance'] is None else float(record['residual_variance'][roi][j]),
                        f'run{run}_classification': classify(float(var[j]), mask_value, temporal)})
                row.update(l2_cope7=float(l2_cope[roi][j]), l2_varcope7=float(l2_var[roi][j]),
                    l2_feat_mask='' if l2_mask is None else float(l2_mask[roi][j]))
                row.update({name: (int(value[j]) if value.dtype.kind in 'biu' else float(value[j])) for name, value in comparison.items()})
                voxel_rows.append(row)
            for role, data in [('cope7', l2_cope), ('varcope7', l2_var), ('feat_mask', l2_mask)]:
                if data is not None:
                    save_map(output/'maps'/subject/f'L2_{role}.nii.gz', data, reference)
            cuts.append(overlay(subject, records, dict(variance=l2_var, mask=l2_mask), roi, reference, standard, output))
        print(f'AUDITED: {subject}; ROI voxels with positive run variance 2/1/0 = '
            f'{int((counts==2).sum())}/{int((counts==1).sum())}/{int((counts==0).sum())}', flush=True)
    for name, rows in [('run_coverage.tsv', run_rows), ('l2_handling.tsv', l2_rows),
                       ('flagged_voxels.tsv', voxel_rows), ('cohort_voxel_support.tsv', support_rows),
                       ('display_cuts.tsv', cuts), ('input_inventory.tsv', reader.rows)]:
        cp.write_rows(output/name, rows, list(rows[0]))
    save_map(output/'selected_roi.nii.gz', roi.astype(np.uint8), reference)
    cp.write_json(output/'provenance.json', dict(
        sources={str(p): cp.digest(p) for p in (source/'inventory.tsv', baseline/'inventory.tsv', config, hash_path,
            roi_path, standard, Path(__file__), Path(pm.__file__))},
        scope='94 runs/47 retained participants for context; detailed signal and MNI overlays only for flagged participants',
        flagged=list(flagged), hypotheses='Existing L2 compared with inverse-variance combination of finite positive run variances; '
        'single-positive-run and no-positive-run cases explicitly separated. Hypotheses are never used to replace estimates.',
        limits='ROI selected using age contrast. FEAT masks and signal variability are descriptive coverage evidence, not proof of neural validity. '
        'Missing optional masks/signals remain unknown. No primary masks, weights, runs, participants, COPEs, VARCOPEs or PALM jobs changed.',
        display='Nearest-neighbor overlays/masks and linear mean-EPI resampling for display only; counts and coordinates from native grid.'))
    cp.write_json(output/'status.json', dict(complete=True, participants=47, runs=94,
        detailed_participants=list(flagged), new_fits=0, exclusions=0, analysis_changed=False,
        meaning='Audit completed, not a declaration that flagged data are valid or repaired'))
    files = [dict(path=str(p.relative_to(output)), sha256=cp.digest(p), bytes=p.stat().st_size)
             for p in sorted(output.rglob('*')) if p.is_file()]
    cp.write_rows(output/'inventory.tsv', files, ['path', 'sha256', 'bytes'])
    print(f'PASS: read-only coverage/VARCOPE audit saved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--partner-report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--baseline-bars', type=Path, default=pm.DEFAULT_BARS)
    parser.add_argument('--standard', type=Path, default=Path(os.environ.get('FSLDIR', '/usr/local/fsl'))/'data/standard/MNI152_T1_2mm_brain.nii.gz')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    build(args.repository.resolve(), args.partner_report.resolve(), args.baseline_bars.resolve(), args.standard.resolve(), args.output_root.resolve())
