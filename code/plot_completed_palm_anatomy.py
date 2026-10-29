#!/usr/bin/env python3
"""Receipt-verified anatomical inspection of completed DMN PALM age models.

Safe while PALM runs: collect an immutable snapshot, export only completed maps,
and write a new dated report. No fits, new thresholds, atlas inference or edits
to historical results. Native inference and display resampling stay separate.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import label

import collect_palm_progress as cp
import run_dmn_heteroscedasticity as palm

ROOT = Path(__file__).resolve().parents[1]
TIMINGS = ('full-rt', 'pooled-post', 'partner-post')
METHODS = ('tfce', 'cluster-z2.6', 'cluster-z3.1')
TITLES = {'full-rt': 'Full display + complete RT height',
          'pooled-post': 'Decision + pooled post-response',
          'partner-post': 'Decision + partner post-response'}
METHOD_NAMES = {'tfce': 'TFCE', 'cluster-z2.6': 'Cluster Z = 2.6', 'cluster-z3.1': 'Cluster Z = 3.1'}
TARGETS = {1: ('younger', 'Positive effect in younger adults'),
           3: ('younger-older', 'Younger > older')}
DISPLAY_VMIN, DISPLAY_VMAX = 0., 5.


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def verify_collection(collection):
    state = json.loads((collection/'collection_status.json').read_text())
    if not state['collection_complete'] or not state.get('interim_summary_requested'):
        raise ValueError('a complete --summarize collection is required')
    for row in read_rows(collection/'inventory.tsv'):
        path = (collection/row['path']).resolve()
        if not path.is_relative_to(collection.resolve()):
            raise ValueError('collection inventory escapes its directory')
        if path.stat().st_size != int(row['bytes']) or cp.digest(path) != row['sha256']:
            raise ValueError(f'collection checksum mismatch: {path}')
    return state


def verified_artifact(receipt, attempt, pattern, provenance):
    paths = [Path(p) for p in receipt['outputs'] if Path(p).match(pattern)]
    if len(paths) != 1:
        raise ValueError(f'missing/ambiguous receipted artifact: {pattern}')
    path = paths[0].resolve()
    if path.parent != attempt.resolve() or cp.digest(path) != receipt['outputs'][str(paths[0])]:
        raise ValueError(f'completed map differs from receipt: {path}')
    provenance[str(path)] = cp.digest(path)
    return path


def native_components(sig, statistic, one_minus_p, affine):
    """26-connected significant components: descriptive, NOT new cluster tests."""
    components, count = label(sig, structure=np.ones((3, 3, 3)))
    regions = sorted(range(1, count+1), key=lambda k: (-int(np.sum(components == k)), k))
    rows = []
    for rank, region in enumerate(regions, 1):
        keep = components == region
        indices = np.argwhere(keep)
        peak = np.unravel_index(np.argmax(np.where(keep, statistic, -np.inf)), statistic.shape)
        xyz = nib.affines.apply_affine(affine, peak)
        centroid = nib.affines.apply_affine(affine, indices.mean(0))
        bounds = nib.affines.apply_affine(affine, indices)
        rows.append(dict(component=rank, native_voxels=int(keep.sum()),
            volume_mm3=float(keep.sum()*abs(np.linalg.det(affine[:3, :3]))),
            minimum_spatial_fwe_p=float(1-one_minus_p[keep].max()),
            peak_z_equivalent=float(statistic[peak]),
            **{f'peak_{a}_mni_mm': float(xyz[i]) for i, a in enumerate('xyz')},
            **{f'centroid_{a}_mni_mm': float(centroid[i]) for i, a in enumerate('xyz')},
            **{f'bound_{a}_{end}_mm': float(getattr(bounds[:, i], end)())
               for i, a in enumerate('xyz') for end in ('min', 'max')}))
    return rows


def load_completed(collection, work, config_path, output):
    state = verify_collection(collection)
    contract = json.loads((collection/'contract.json').read_text())
    if json.loads((work/'contract.json').read_text()) != contract:
        raise ValueError('scratch and snapshot PALM contracts differ')
    if cp.digest(config_path) != contract['source']:
        raise ValueError('source configuration differs from PALM contract')
    if cp.digest(Path(palm.__file__)) != contract['code'] or cp.digest(Path(palm.pa.__file__)) != contract['helper']:
        raise ValueError('statistical/helper code differs from PALM contract')
    models = json.loads(config_path.read_text())['models']
    if [m['name'] for m in models] != contract['models']:
        raise ValueError('model names/order differ from PALM contract')
    by_name = {m['name']: m for m in models}
    summary = {(r['model'], r['method'], int(r['contrast'])): r
               for r in read_rows(collection/'interim_results.tsv')}
    provenance = {str(p): cp.digest(p) for p in (config_path, collection/'contract.json',
        collection/'inventory.tsv', collection/'interim_results.tsv', Path(__file__),
        Path(cp.__file__), Path(palm.__file__), Path(palm.pa.__file__))}
    payload, summaries, regions, pending = {}, [], [], []
    complete = {(r['model'], r['method']): r for r in read_rows(collection/'completed_jobs.tsv')}
    for timing in TIMINGS:
        name = timing+'__dmn-age'
        if name not in by_name:
            raise ValueError(f'locked DMN age model missing: {name}')
        model = by_name[name]
        if len(model['subjects']) != 47 or len(set(model['subjects'])) != 47:
            raise ValueError('expected the locked 47 unique participants')
        if model['labels'] != ['younger', 'older', 'younger-older', 'older-younger'] or len(model['c']) != 4:
            raise ValueError('unexpected age contrasts; positional reinterpretation forbidden')
        for method in METHODS:
            if (name, method) not in complete:
                pending.append(dict(model=name, method=method, status='pending_at_snapshot; no partial maps displayed'))
                continue
            copied_receipt = collection/'completed'/name/method/'complete.json'
            saved = json.loads(copied_receipt.read_text())
            attempt = Path(saved['attempt']).resolve()
            if not attempt.is_relative_to((work/'jobs'/name/method).resolve()):
                raise ValueError('completed attempt outside expected PALM job')
            block_receipt = work/'blocks'/model['block']/'complete.json'
            block = json.loads(block_receipt.read_text())
            mask_path = (work/'designs'/name/'mask.nii.gz').resolve()
            mask_hashes = {str(Path(p).resolve()): h for p, h in block['outputs'].items()}
            if cp.digest(mask_path) != mask_hashes.get(str(mask_path)):
                raise ValueError('original PALM analysis mask differs')
            provenance.update({str(p): cp.digest(p) for p in (copied_receipt, block_receipt, mask_path)})
            mask = nib.load(mask_path); keep = mask.get_fdata() > 0
            if not keep.any():
                raise ValueError('empty PALM analysis mask')
            folder = output/'maps'/name/method
            folder.mkdir(parents=True)
            shutil.copyfile(copied_receipt, folder/'source_complete.json')
            nib.save(mask, folder/'analysis_mask.nii.gz')
            unit = 'tfce' if method == 'tfce' else 'clustere'
            for contrast in range(1, 5):
                raw_path = verified_artifact(saved, attempt, f'palm_vox_zvstat_c{contrast}.nii*', provenance)
                p_path = verified_artifact(saved, attempt, f'palm_{unit}_zvstat_fwep_c{contrast}.nii*', provenance)
                raw, prob = nib.load(raw_path), nib.load(p_path)
                if not palm.pa.compatible(raw, mask) or not palm.pa.compatible(prob, mask):
                    raise ValueError('PALM mask/statistic/FWE grids differ')
                z, values = raw.get_fdata(), prob.get_fdata()
                if (not np.isfinite(z[keep]).all() or not np.isfinite(values[keep]).all()
                        or np.any((values[keep] < 0) | (values[keep] > 1))):
                    raise ValueError('invalid native statistic or PALM 1-p values')
                sig = keep & (values > .95)
                stats = dict(minimum_spatial_fwe_p=float(1-values[keep].max()), significant_voxels=int(sig.sum()))
                original = summary[name, method, contrast]
                if (stats['significant_voxels'] != int(original['significant_voxels'])
                        or not np.isclose(stats['minimum_spatial_fwe_p'], float(original['minimum_spatial_fwe_p']), atol=1e-12, rtol=0)):
                    raise ValueError('native maps disagree with verified interim summary')
                summaries.append(dict(model=name, method=method, contrast=contrast,
                    contrast_name=model['labels'][contrast-1], **stats))
                for suffix, data in (('z_equivalent', np.where(keep, z, 0)),
                                     ('one_minus_p_fwe', np.where(keep, values, 0)),
                                     ('significant_z_equivalent', np.where(sig, z, 0)),
                                     ('significant_mask', sig.astype(np.uint8))):
                    nib.save(nib.Nifti1Image(data, raw.affine), folder/f'contrast{contrast}_{suffix}.nii.gz')
                pieces = native_components(sig, z, values, raw.affine)
                regions.extend(dict(model=name, method=method, contrast=contrast,
                    contrast_name=model['labels'][contrast-1], **r) for r in pieces)
                payload[timing, method, contrast] = dict(z=z, sig=sig, reference=raw,
                    components=pieces, **stats)
    if not payload:
        raise ValueError('no verified completed DMN age jobs at this snapshot')
    return payload, summaries, regions, pending, provenance, state


def plane(data, affine, axis, cut):
    steps = np.diag(affine)[:3]; origin = affine[:3, 3]
    if np.any(steps <= 0) or not np.allclose(affine[:3, :3], np.diag(steps)):
        raise ValueError('anatomical reference must be canonical axis-aligned RAS')
    index = int(np.rint((cut-origin[axis])/steps[axis]))
    if not 0 <= index < data.shape[axis]:
        raise ValueError('slice lies outside anatomical reference')
    extent = [v for a in range(3) if a != axis
              for v in (origin[a]-steps[a]/2, origin[a]+steps[a]*(data.shape[a]-.5))]
    return np.take(data, index, axis=axis).T, extent, origin[axis]+steps[axis]*index


def select_cut(payload, timing, contrast, fallback):
    candidates = [payload[timing, method, contrast] for method in METHODS if (timing, method, contrast) in payload]
    candidates = [p for p in candidates if p['sig'].any()]
    if not candidates:
        return np.asarray(fallback), 'No significant voxels: full-RT matched target cut, or MNI [0,0,0] fallback'
    item = max(candidates, key=lambda p: float(p['z'][p['sig']].max()))
    ijk = np.unravel_index(np.argmax(np.where(item['sig'], item['z'], -np.inf)), item['z'].shape)
    return nib.affines.apply_affine(item['reference'].affine, ijk), 'Largest Z-equivalent value in union of completed significant maps; identical cuts across methods'


def display_overlay(item, background):
    # Native mask, never interpolated probabilities, determines inference.
    image = nib.Nifti1Image(np.where(item['sig'], item['z'], 0), item['reference'].affine)
    return resample_from_to(image, background, order=0).get_fdata()


def draw_slice(ax, anatomy, overlay, background, axis, cut, vmax, zoom=False):
    ax.set_facecolor('black')
    for spine in ax.spines.values():
        spine.set_visible(False)
    base, extent, actual = plane(anatomy, background.affine, axis, cut[axis])
    ax.imshow(base, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=vmax, interpolation='bilinear')
    if overlay is not None:
        values, _, _ = plane(overlay, background.affine, axis, cut[axis])
        ax.imshow(np.ma.masked_where(values == 0, values), origin='lower', extent=extent,
                  cmap='autumn', vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX, interpolation='nearest')
    if zoom:
        h, v = [i for i in range(3) if i != axis]
        ax.set_xlim(cut[h]-24, cut[h]+24); ax.set_ylim(cut[v]-24, cut[v]+24)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{("Sagittal", "Coronal", "Axial")[axis]}{" detail" if zoom else ""}: {"xyz"[axis]} = {actual:.1f} mm', fontsize=10)
    ax.text(.02, .04, 'P' if axis == 0 else 'L', transform=ax.transAxes, color='white', fontsize=9)
    ax.text(.94, .04, 'A' if axis == 0 else 'R', transform=ax.transAxes, color='white', fontsize=9)
    return actual


def method_figure(payload, timing, contrast, cut, background, output):
    anatomy = background.get_fdata()
    vmax = np.percentile(anatomy[anatomy > 0], 99.5)
    fig = plt.figure(figsize=(15, 12), facecolor='white')
    grid = fig.add_gridspec(3, 4, left=.035, right=.975, bottom=.12, top=.82, hspace=.64, wspace=.10)
    fig.text(.035, .965, f'DMN: {TARGETS[contrast][1]}', fontsize=19, weight='bold')
    fig.text(.035, .928, TITLES[timing]+' | Similar-minus-dissimilar offer-modulated coupling | N = 47', fontsize=12)
    fig.text(.035, .901, 'Warm colors: completed PALM spatial-FWE p < .05 only. Identical cuts across methods; no display smoothing.', fontsize=10)
    records = []
    for row, method in enumerate(METHODS):
        item = payload.get((timing, method, contrast))
        top = grid[row, 0].get_position(fig).y1
        description = ('PENDING at snapshot — no partial results displayed' if item is None else
            f'{item["significant_voxels"]} native significant voxels; minimum spatial-FWE p = {item["minimum_spatial_fwe_p"]:.4f}')
        fig.text(.035, top+.046, METHOD_NAMES[method], fontsize=12, weight='bold')
        fig.text(.035, top+.024, description, fontsize=10)
        overlay = display_overlay(item, background) if item is not None else None
        for column, axis in enumerate((0, 1, 2, 2)):
            ax = fig.add_subplot(grid[row, column])
            actual = draw_slice(ax, anatomy, overlay, background, axis, cut, vmax, column == 3)
            if column == 3 and (item is None or not item['sig'].any()):
                ax.text(.5, .5, 'Pending' if item is None else 'No FWE-significant voxels',
                    ha='center', transform=ax.transAxes, color='white', fontsize=9,
                    bbox=dict(facecolor='black', alpha=.7, edgecolor='none'))
            records.append(dict(timing=timing, method=method, contrast=contrast, panel=column,
                                axis='xyz'[axis], actual_cut_mm=float(actual)))
    cax = fig.add_axes([.77, .055, .20, .013])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(DISPLAY_VMIN, DISPLAY_VMAX), cmap='autumn'),
                      cax=cax, orientation='horizontal', extend='max')
    cb.set_label('Voxelwise Z-equivalent statistic (not TFCE score or cluster extent)', fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.text(.035, .065, 'Anatomical inspection, not independent confirmation. Neurological orientation (L on left).\n'
             'Spatial FWE per signed contrast; no correction across models/methods. Null/pending panels are retained.\n'
             'Display range is fixed at 0–5; values above 5 saturate. Native coordinates/counts are in the tables.', fontsize=9)
    fig.savefig(output, dpi=160, facecolor='white'); plt.close(fig)
    return records


def component_figures(payload, timing, method, contrast, background, output):
    item = payload[timing, method, contrast]
    parts = item['components']
    if not parts:
        return
    # Every significant age-difference component, no minimum-size filtering.
    overlay = display_overlay(item, background)
    anatomy = background.get_fdata(); vmax = np.percentile(anatomy[anatomy > 0], 99.5)
    for start in range(0, len(parts), 3):
        page = parts[start:start+3]
        fig, axes = plt.subplots(len(page), 3, figsize=(12, 3.8*len(page)+1.8), squeeze=False)
        fig.subplots_adjust(left=.03, right=.98, bottom=.18, top=1-1.1/(3.8*len(page)+1.8), hspace=.75, wspace=.1)
        fig.suptitle(f'DMN younger > older | {TITLES[timing]} | {METHOD_NAMES[method]}\n'
                     'Every 26-connected significant component; descriptive subdivisions, not new cluster tests', fontsize=12)
        for row, part in enumerate(page):
            cut = np.array([part[f'peak_{a}_mni_mm'] for a in 'xyz'])
            for axis in range(3):
                draw_slice(axes[row, axis], anatomy, overlay, background, axis, cut, vmax)
        fig.canvas.draw()
        for row, part in enumerate(page):
            bottom = min(ax.get_position().y0 for ax in axes[row])
            fig.text(.035, bottom-.035, f'Component {part["component"]}: {part["native_voxels"]} voxels; '
                f'min FWE p = {part["minimum_spatial_fwe_p"]:.4f}; peak Z-equivalent = {part["peak_z_equivalent"]:.2f}',
                ha='left', va='top', fontsize=9)
        fig.text(.03, .025, 'Only original spatial-FWE p < .05 voxels; no display smoothing. Coordinates in native-grid MNI mm.\n'
                 'No correction across models/methods; anatomical appearance does not establish robustness.', fontsize=9)
        cb = fig.colorbar(ScalarMappable(norm=Normalize(DISPLAY_VMIN, DISPLAY_VMAX), cmap='autumn'),
            cax=fig.add_axes([.78, .055, .18, .015]), orientation='horizontal', extend='max')
        cb.set_label('Voxelwise Z-equivalent', fontsize=8); cb.ax.tick_params(labelsize=8)
        fig.savefig(output/f'{timing}_{method}_age_components_page{start//3+1:02d}.png', dpi=160)
        plt.close(fig)


def build(collection, work, standard, output, config_path):
    if output.resolve().is_relative_to(work.resolve()):
        raise ValueError('anatomical reports must not be written inside PALM scratch')
    if output.exists():
        raise FileExistsError(f'will not overwrite an anatomical report: {output}')
    output.mkdir(parents=True)
    cp.write_json(output/'status.json', dict(complete=False))
    payload, summaries, regions, pending, provenance, state = load_completed(collection, work, config_path, output)
    background = nib.as_closest_canonical(nib.load(standard))
    anatomy = background.get_fdata()
    if len(background.shape) != 3 or not np.isfinite(anatomy).all() or not np.any(anatomy > 0):
        raise ValueError('invalid anatomical reference')
    # Validate orientation before generating any panels.
    for axis in range(3):
        plane(anatomy, background.affine, axis, background.affine[axis, 3])
    provenance[str(standard)] = cp.digest(standard)
    cp.write_rows(output/'native_summary.tsv', summaries, list(summaries[0]))
    cp.write_rows(output/'native_components.tsv', regions, list(regions[0]) if regions else
                  ['model', 'method', 'contrast', 'contrast_name', 'component', 'native_voxels'])
    cp.write_rows(output/'pending_jobs.tsv', pending, ['model', 'method', 'status'])
    cuts = []
    for contrast in TARGETS:
        fallback, _ = select_cut(payload, 'full-rt', contrast, np.zeros(3))
        for timing in TIMINGS:
            cut, rule = select_cut(payload, timing, contrast, fallback)
            filename = f'{timing}_{TARGETS[contrast][0]}_methods.png'
            cuts.extend(dict(cut_rule=rule, **r) for r in method_figure(payload, timing, contrast, cut, background, output/filename))
    for timing, method, contrast in payload:
        if contrast == 3:
            component_figures(payload, timing, method, contrast, background, output)
    cp.write_rows(output/'display_cuts.tsv', cuts, list(cuts[0]))
    cp.write_json(output/'provenance.json', dict(sources=provenance, collection=str(collection),
        verified_completed_jobs_in_collection=state['verified_completed_jobs'],
        rendered_models='three locked DMN-age models; all four signed contrasts exported, panels for younger and younger-minus-older',
        native_inference='analysis mask AND PALM saved 1-p > .95, strictly; no new tests, refits or exclusions',
        components='26-connected significant sets for descriptive localization only; not PALM cluster labels or new cluster p-values',
        display='canonical MNI T1, neurological orientation, fixed Z-equivalent colors 0-5; nearest-neighbor overlay, bilinear T1 display; no smoothing',
        interpretation='Interim anatomical QC. Spatial FWE per signed contrast, not across methods/models. '
        'The p=.0434 finding is threshold-dependent; anatomy is not independent validation. No partial jobs displayed.'))
    cp.write_json(output/'status.json', dict(complete=True, completed_age_jobs=len(summaries)//4,
        signed_contrasts=len(summaries), new_fits=0, excluded_participants=0,
        snapshot_utc=state['finished_utc'], batch_lock_held_at_collection=state['batch_lock_held_at_start']))
    files = [dict(path=str(p.relative_to(output)), sha256=cp.digest(p), bytes=p.stat().st_size)
             for p in sorted(output.rglob('*')) if p.is_file()]
    cp.write_rows(output/'inventory.tsv', files, ['path', 'sha256', 'bytes'])
    print(f'PASS: exported {len(summaries)//4} completed DMN age jobs; {len(summaries)} signed contrasts; no new fits')
    print(f'Anatomical panels, native components and compressed source maps: {output}')
    for row in regions:
        if row['model'] == 'full-rt__dmn-age' and row['method'] == 'cluster-z2.6' and row['contrast'] == 3:
            print('FOCAL AGE COMPONENT:', json.dumps(row, sort_keys=True))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=cp.WORK)
    parser.add_argument('--report-root', type=Path, default=cp.REPORT)
    parser.add_argument('--source-config', type=Path, default=ROOT/'results/dmn_covariate_standardized/v1/randomise/config.json')
    parser.add_argument('--collection', type=Path, help='Existing verified --summarize collection; otherwise take a fresh snapshot')
    parser.add_argument('--standard', type=Path, default=Path(os.environ.get('FSLDIR', '/usr/local/fsl'))/'data/standard/MNI152_T1_2mm_brain.nii.gz')
    parser.add_argument('--output-root', type=Path, required=True, help='New output directory; never overwritten')
    args = parser.parse_args()
    if not args.standard.is_file():
        parser.error(f'MNI anatomical reference missing: {args.standard}')
    collection = args.collection.resolve() if args.collection else cp.collect(
        args.work_root.resolve(), args.report_root.resolve(), True, True, args.source_config.resolve())
    build(collection, args.work_root.resolve(), args.standard.resolve(), args.output_root.resolve(), args.source_config.resolve())


if __name__ == '__main__':
    main()
