#!/usr/bin/env python3
"""Isolated PALM age differences and adjusted full-sample means.

Reuse retained participant estimates; verify all-run coverage before inference.
Never acquire the original PALM lock, signal workers, or rewrite old results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
import nibabel as nib
import numpy as np
from scipy import ndimage

import run_activation_coverage as cov
import run_dmn_coverage_check as prior

pa = prior.pa
headless = prior.headless
scale = prior.scale
ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/ZPOOL/data/scratch/srndna-ultimatum-palm-age-main-v1')
REPORT = ROOT/'results/palm_age_main/v1'
TIMINGS = ('full-rt', 'pooled-post', 'partner-post')
FAMILIES = ('act', 'nppi-dmn', 'nppi-ecn')
METHODS = ('cluster-z3.1', 'cluster-z2.6')
LABELS = ('full-sample-positive', 'full-sample-negative', 'younger-minus-older', 'older-minus-younger')
COLUMNS = ('mean', 'older_centered', 'ismale', 'tsnr', 'fd_mean', 'RT')
N, SEED = 10000, 20261101


def read_receipt(path, contract=None):
    saved = json.loads(path.read_text())
    if contract is not None and saved.get('contract') != contract:
        raise ValueError(f'completed work contract differs: {path}')
    pa.verify_files(saved['outputs'])
    return saved


def code_hashes():
    # Include transitive local imports: a changed numerical/runtime helper must
    # not silently reuse an old preparation or completion receipt.
    return {str(Path(module.__file__).resolve()): pa.digest(module.__file__)
            for module in list(sys.modules.values())
            if getattr(module, '__file__', None)
            and Path(module.__file__).resolve().parent == ROOT/'code'}


def source_specs():
    suffix = {'act': 'activation-norm', 'nppi-dmn': 'dmn-age', 'nppi-ecn': 'ecn-sensitivity'}
    specs = [(f'{timing}__{family}', f'{timing}__{suffix[family]}', family, 7, timing)
             for family in FAMILIES for timing in TIMINGS]
    specs += [(name, name, name, cope, 'supplemental-fixed-display') for name, cope in
              [('taskwide-offer', 11), ('human-computer-offer', 8), ('human-computer-response', 10)]]
    return specs


def build_design(model, sample, reference):
    """Same four reported controls; verified tSNR/FD, retained corrected RT.

    Center over the actual 47-person sample, not within age groups. The mean
    is sample-weighted (25/47 younger, 22/47 older), not an equal-group mean.
    """
    subjects = model['subjects']
    if len(subjects) != 47 or len(set(subjects)) != 47 or set(subjects) != set(sample):
        raise ValueError('expected exactly the locked 47 participants')
    old = np.array([float(sample[s]['older']) for s in subjects])
    if not np.isin(old, [0, 1]).all() or old.sum() != 22:
        raise ValueError('age-group coding differs from the paper sample')
    aligned = {s: i for i, s in enumerate(reference['subjects'])}
    cols = [np.ones(47), old-old.mean()]; changes = []
    for key in COLUMNS[2:]:
        before = np.array(model['nuisance'][key], float)
        ref = np.array([reference['nuisance'][key][aligned[s]] for s in subjects], float)
        if key in ('tsnr', 'fd_mean'):
            after = np.array([float(sample[s][key]) for s in subjects])
            if not np.allclose(after, ref, atol=1e-5, rtol=0):
                raise ValueError(f'verified DMN and participant table disagree: {key}')
        else:
            # RT must be the event-corrected retained model column, NOT the
            # historical RT column still present in participant_L3_47.csv.
            after = ref
            if not np.allclose(before-before.mean(), ref-ref.mean(), atol=1e-5, rtol=0):
                raise ValueError(f'retained corrected {key} differs between models; stop for review')
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError(f'nonfinite group covariate: {key}')
        centered = after-after.mean(); cols.append(centered)
        for i, subject in enumerate(subjects):
            changes.append(dict(subject=subject, covariate=key, retained_value=float(before[i]),
                final_value=float(centered[i]), centered_change=float(centered[i]-(before[i]-before.mean()))))
    x = np.column_stack(cols)
    c = np.zeros((4, 6)); c[0, 0] = 1; c[1] = -c[0]; c[2, 1] = -1; c[3] = -c[2]
    if np.linalg.matrix_rank(x) != 6 or not np.allclose(x[:, 1:].mean(axis=0), 0, atol=1e-12):
        raise ValueError('noncentered or rank-deficient final design')
    return x, c, (1+old).astype(int), changes


def load_catalog(repo):
    original, sample, table = pa.catalog(repo)
    standardized, _, _ = pa.standardized_dmn_catalog(repo, repo/'results/dmn_covariate_standardized/v1/imaging')
    current = {m['name']: m for m in original}
    current.update({m['name']: m for m in standardized if m['name'].endswith('__dmn-age')})
    paths = [repo/'results/permutation_audit/v1/config.json',
             repo/'results/dmn_covariate_standardized/v1/randomise/config.json']
    frozen = {}; expected = {}; sources = {str(table): pa.digest(table)}
    for path in paths:
        config = json.loads(path.read_text())
        frozen.update({m['name']: m for m in config['models']})
        provenance = path.with_name('input_provenance.json')
        expected.update(json.loads(provenance.read_text()))
        sources.update({str(p): pa.digest(p) for p in (path, provenance)})
    reference = current['full-rt__dmn-age']
    selected = []
    for name, source_name, family, cope, timing in source_specs():
        model = current[source_name]; saved = frozen[source_name]
        for key in ('x', 'c', 'subjects', 'images', 'nuisance', 'block', 'group'):
            if model[key] != saved[key]:
                raise ValueError(f'retained source differs from frozen catalog: {source_name}/{key}')
        x, c, groups, changes = build_design(model, sample, reference)
        selected.append(dict(model, name=name, source_model=source_name, family=family, timing=timing,
            primary_cope=cope, x=x.tolist(), c=c.tolist(), labels=list(LABELS),
            variance_groups=groups.tolist(), covariate_changes=changes))
        sources.update(model['source_hashes'])
    return selected, sources, expected


def prepare_inputs(model, folder, expected, sources):
    """All participants, both runs, positive variance and FEAT support."""
    folder.mkdir(parents=True)
    reference = nib.load(model['mask']); retained = reference.get_fdata() > 0
    reader = cov.coverage.Inputs(reference, expected)
    n = model['primary_cope']; data = []; rows = []
    runs = np.zeros(reference.shape, np.uint16); l2 = runs.copy()
    sources[model['mask']] = pa.digest(model['mask'])
    for subject, value in zip(model['subjects'], model['images']):
        path = Path(value)
        if path.parent.parent.name != f'cope{n}.feat':
            raise ValueError(f'wrong retained participant contrast: {path}')
        cope = reader.load(path, f'L2 COPE{n}')
        variance = reader.load(path.with_name('varcope1.nii.gz'), f'L2 VARCOPE{n}')
        mask = reader.load(path.parent.parent/'mask.nii.gz', 'L2 FEAT mask')
        l2 += prior.valid_estimate(cope, variance, mask)
        if not np.isfinite(cope).all():
            raise ValueError('nonfinite merged COPE data; no value replacement permitted')
        data.append(cope)
        for number, source in cov.l1_inputs(path, subject, n, sources):
            cov.validate_contrast(source, n, sources)
            c = reader.load(source/'stats'/f'cope{n}.nii.gz', f'L1 COPE{n}')
            v = reader.load(source/'stats'/f'varcope{n}.nii.gz', f'L1 VARCOPE{n}')
            m = reader.load(source/'mask.nii.gz', 'L1 FEAT mask')
            valid = prior.valid_estimate(c, v, m); runs += valid
            rows.append(dict(subject=subject, run=number, valid_original_voxels=int(np.sum(retained & valid))))
        print(f'CHECKED: {model["name"]} {subject}', flush=True)
    data = np.stack(data, axis=3)
    valid = retained & (runs == 94) & (l2 == 47) & (np.ptp(data, axis=3) > 0)
    for name, values in [('copes', data), ('original_mask', retained.astype('uint8')),
                         ('individual_valid_mask', valid.astype('uint8')),
                         ('valid_run_count', runs), ('valid_l2_count', l2)]:
        prior.save(folder/f'{name}.nii.gz', values, reference)
    pa.write_rows(folder/'run_coverage.tsv', rows)
    pa.write_rows(folder/'input_inventory.tsv', reader.rows)
    sources.update({r['path']: r['sha256'] for r in reader.rows if r['exists']})
    return reference, retained, valid


def prepare(repo, work):
    marker = work/'prepared.json'
    if marker.is_file():
        saved = read_receipt(marker); pa.verify_files(saved['sources'])
        if saved['helpers'] != code_hashes():
            raise ValueError('preparation code changed; preserve completed work and use a new versioned root')
        print('SKIP_COMPLETE: verified age/main-effect preparation', flush=True)
        return saved
    models, sources, expected = load_catalog(repo)
    root = Path(tempfile.mkdtemp(prefix='preparation-', dir=work))
    arrays = {}; references = {}
    for model in models:
        ref, retained, valid = prepare_inputs(model, root/model['name'], expected, sources)
        references[model['name']] = ref; arrays[model['name']] = (retained, valid)
    # A common whole-brain search across timing variants within each family;
    # no selected cluster, p-value or age effect enters mask construction.
    common = {}
    for family in FAMILIES:
        members = [m for m in models if m['family'] == family]
        for model in members[1:]:
            if not pa.compatible(references[members[0]['name']], references[model['name']]):
                raise ValueError(f'timing-model grids differ: {family}')
        common[family] = np.logical_and.reduce([arrays[m['name']][1] for m in members])
    summary = []; changes = []
    for model in models:
        name = model['name']; folder = root/name; ref = references[name]
        retained, valid = arrays[name]
        tested = common.get(model['family'], valid)
        if not tested.any() or np.any(tested & ~retained):
            raise ValueError(f'empty or expanded analysis mask: {name}')
        prior.save(folder/'mask.nii.gz', tested.astype('uint8'), ref)
        pa.vest(folder/'design.mat', np.array(model['x']))
        pa.vest(folder/'design.con', np.array(model['c']), contrast=True)
        np.savetxt(folder/'variance_groups.csv', model['variance_groups'], fmt='%d')
        pa.write_rows(folder/'participants.tsv', [dict(subject=s, variance_group=g)
            for s, g in zip(model['subjects'], model['variance_groups'])])
        pa.write_rows(folder/'design_columns.tsv', [dict(column=i, name=n) for i, n in enumerate(COLUMNS, 1)])
        pa.write_rows(folder/'contrast_names.tsv', [dict(contrast=i, name=n) for i, n in enumerate(LABELS, 1)])
        summary.append(dict(model=name, source_model=model['source_model'], participants=47,
            l1_runs=94, primary_cope=model['primary_cope'], original_voxels=int(retained.sum()),
            individual_valid_voxels=int(valid.sum()), tested_voxels=int(tested.sum()),
            design_rank=int(np.linalg.matrix_rank(model['x'])), residual_df=41))
        changes.extend(dict(model=name, **r) for r in model['covariate_changes'])
    pa.write_rows(root/'coverage_summary.tsv', summary)
    pa.write_rows(root/'covariate_changes.tsv', changes)
    pa.write_json(root/'models.json', models)
    saved = dict(root=str(root), models=models, sources=sources, helpers=code_hashes(),
        outputs={str(p): pa.digest(p) for p in root.rglob('*') if p.is_file()})
    pa.write_json(marker, saved)
    return saved


def export_preparation(prepared, report):
    root = Path(prepared['root'])
    for path in root.rglob('*'):
        if path.is_file() and path.name != 'copes.nii.gz':
            prior.compact_copy(path, report/'preparation'/path.relative_to(root))
    pa.write_json(report/'source_inventory.json', prepared['sources'])


def clusters(path, model, method, contrast):
    image = nib.load(path); values = image.get_fdata(); keep = values > .95
    labels, count = ndimage.label(keep, structure=ndimage.generate_binary_structure(3, 1))
    rows = []
    for number in range(1, count+1):
        indices = np.argwhere(labels == number)
        point = indices[np.argmax(values[tuple(indices.T)])]
        xyz = nib.affines.apply_affine(image.affine, point)
        rows.append(dict(model=model, method=method, contrast=contrast,
            contrast_name=LABELS[contrast-1], component=number, voxels=len(indices),
            minimum_spatial_fwe_p=float(1-values[tuple(point)]),
            representative_x=float(xyz[0]), representative_y=float(xyz[1]), representative_z=float(xyz[2])))
    return rows


def export_job(model, method, saved, report):
    attempt = Path(saved['attempt']); destination = report/'maps'/model['name']/method
    destination.mkdir(parents=True, exist_ok=True)
    # The attempt name prevents overwriting failed/retried logs elsewhere.
    # Scientific products become immutable after completion validation.
    for path in attempt.iterdir():
        if path.is_file() and not path.is_symlink():
            prior.compact_copy(path, destination/path.name)
    prior.compact_copy(Path(saved['receipt']), destination/'complete.json')


def run_job(model, method, prepared, work, report, contract, rt):
    folder = work/'jobs'/model['name']/method; folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'complete.json'; inputs = Path(prepared['root'])/model['name']
    mask = nib.load(inputs/'mask.nii.gz')
    if marker.is_file():
        saved = read_receipt(marker, contract)
        print(f'SKIP_COMPLETE: {model["name"]} {method}', flush=True)
    else:
        attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=folder))
        args = prior.palm.arguments(inputs, method, attempt/'palm', N, SEED)
        print(f'RUN: {model["name"]} {method}', flush=True)
        try:
            headless.launch(rt['engine'], Path(rt['palm_root']), args, attempt, mode=rt['mode'])
            prior.palm.probability_maps(attempt, method, 4, mask)
            saved = dict(attempt=str(attempt), contract=contract,
                outputs={str(p): pa.digest(p) for p in attempt.iterdir() if p.is_file()})
            pa.write_json(marker, saved)
        except BaseException:
            # Preserve failures and partial evidence, never pretend a job is null.
            headless.snapshot_report(attempt, report/'attempts'/model['name']/method/attempt.name)
            raise
        print(f'COMPLETED: {model["name"]} {method}', flush=True)
    export_job(model, method, dict(saved, receipt=str(marker)), report)
    return [dict(model=model['name'], method=method, contrast_name=LABELS[r['contrast']-1], **r)
            for _, r in prior.palm.probability_maps(Path(saved['attempt']), method, 4, mask)]


def collect(prepared, work, report, contract):
    rows = []; components = []; progress = []; complete = 0
    for model in prepared['models']:
        mask = nib.load(Path(prepared['root'])/model['name']/'mask.nii.gz')
        for method in METHODS:
            marker = work/'jobs'/model['name']/method/'complete.json'
            if not marker.is_file():
                progress.append(dict(model=model['name'], method=method, status='unfinished'))
                continue
            saved = read_receipt(marker, contract)
            export_job(model, method, dict(saved, receipt=str(marker)), report)
            for path, row in prior.palm.probability_maps(Path(saved['attempt']), method, 4, mask):
                rows.append(dict(model=model['name'], method=method, contrast_name=LABELS[row['contrast']-1], **row))
                components.extend(clusters(path, model['name'], method, row['contrast']))
            complete += 1
            progress.append(dict(model=model['name'], method=method, status='verified_complete'))
    if rows:
        pa.write_rows(report/'summary.tsv', rows)
    if components:
        pa.write_rows(report/'significant_components.tsv', components)
    else:
        (report/'significant_components.tsv').write_text('model\tmethod\tcontrast\tcontrast_name\tcomponent\tvoxels\tminimum_spatial_fwe_p\trepresentative_x\trepresentative_y\trepresentative_z\n')
    pa.write_rows(report/'progress.tsv', progress)
    total = len(prepared['models'])*len(METHODS)
    pa.write_json(report/'status.json', dict(complete=complete == total, completed_jobs=complete,
        expected_jobs=total, summary_rows=len(rows), expected_summary_rows=total*4,
        participants=47, exclusions=0, l1_l2_refits=0, original_palm_changed=False))
    print(f'PALM AGE/MEAN: {complete}/{total} verified jobs', flush=True)
    return complete, total


def inventory(report):
    """Snapshot compact scientific outputs; a live launch log is not immutable."""
    rows = [dict(path=str(p.relative_to(report)), bytes=p.stat().st_size, sha256=pa.digest(p))
            for p in sorted(report.rglob('*')) if p.is_file()
            and p.name != 'inventory.tsv' and not p.relative_to(report).parts[0].startswith('launch-')]
    if rows:
        pa.write_rows(report/'inventory.tsv', rows)


def active_children(work, proc=Path('/proc')):
    """Detect orphaned MATLAB attempts after a driver/SSH failure; never signal."""
    found = []
    for folder in proc.glob('[0-9]*'):
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            text = (folder/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            state = (folder/'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state != 'Z' and str(work)+'/' in text and re.search(r'matlab', text, re.I):
                found.append(int(folder.name))
        except (OSError, IndexError):
            continue
    return found


def smoke(prepared, work, report, contract, rt):
    marker = work/'smoke_complete.json'
    if marker.is_file():
        read_receipt(marker, contract)
        print('SKIP_COMPLETE: four-contrast PALM smoke', flush=True)
        return
    folder = work/'smoke'; folder.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=folder))
    inputs = Path(prepared['root'])/prepared['models'][0]['name']
    reference = nib.load(inputs/'mask.nii.gz'); keep = reference.get_fdata() > 0
    selected = np.zeros(reference.shape, 'uint8'); points = np.argwhere(keep)[:64]
    selected[tuple(points.T)] = 1; prior.save(attempt/'mask.nii.gz', selected, reference)
    for name in ('copes.nii.gz', 'design.mat', 'design.con', 'variance_groups.csv'):
        (attempt/name).symlink_to((inputs/name).resolve())
    # Same new design and variance groups, both inference thresholds. No TFCE.
    try:
        for method in METHODS:
            target = attempt/method; target.mkdir()
            headless.launch(rt['engine'], Path(rt['palm_root']),
                prior.palm.arguments(attempt, method, target/'palm', 20, 42), target, mode=rt['mode'])
            prior.palm.probability_maps(target, method, 4, nib.load(attempt/'mask.nii.gz'))
        pa.write_json(marker, dict(contract=contract, not_scientific_results=True,
            outputs={str(p): pa.digest(p) for p in attempt.rglob('*') if p.is_file() and not p.is_symlink()}))
        print('PASS: new design, both cluster thresholds, all four contrasts', flush=True)
    finally:
        for path in attempt.rglob('*'):
            if path.is_file() and not path.is_symlink():
                prior.compact_copy(path, report/'smoke'/attempt.name/path.relative_to(attempt))


def run_pending(tasks, worker, limit, after_job):
    """Bound scheduling, drain active jobs after failure, preserve receipts."""
    remaining = list(tasks); failures = []
    with ThreadPoolExecutor(max_workers=limit) as pool:
        active = {}
        def fill():
            while remaining and len(active) < limit and not failures:
                item = remaining.pop(0); active[pool.submit(worker, item)] = item
        fill()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                model, method = active.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    failures.append(dict(model=model['name'], method=method, error=str(exc)))
                    print(f'STOP_SCHEDULING: {model["name"]} {method}: {exc}', flush=True)
            after_job()
            fill()
    return failures, [dict(model=m['name'], method=method) for m, method in remaining]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=WORK)
    parser.add_argument('--report-root', type=Path, default=REPORT)
    parser.add_argument('--original-palm-work', type=Path, default=prior.PALM_SOURCE)
    parser.add_argument('--palm-root', type=Path, required=True)
    parser.add_argument('--jobs', type=int, choices=range(1, 41), default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--collect-only', action='store_true')
    mode.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    work, report, old = args.work_root.resolve(), args.report_root.resolve(), args.original_palm_work.resolve()
    if work == ROOT or ROOT in work.parents or work in ROOT.parents:
        parser.error('work must be outside the code repository')
    if work == old or old in work.parents or work in old.parents:
        parser.error('new scratch must be separate from all original PALM work')
    if ROOT/'results/palm_age_main' not in report.parents:
        parser.error('use a versioned report below results/palm_age_main')
    work.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    with (work/'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        orphans = active_children(work)
        if orphans and (args.execute or args.smoke_only):
            raise RuntimeError(f'MATLAB attempts still active in this new work root: {orphans}; '
                               'no duplicate jobs launched or workers signalled; let them finish first')
        prepared = prepare(ROOT, work)
        export_preparation(prepared, report)
        rt = prior.runtime(old, args.palm_root.resolve())
        contract = dict(preparation=pa.digest(work/'prepared.json'), runtime=rt,
            shuffles=N, seed=SEED, methods=list(METHODS), labels=list(LABELS),
            primary_method='cluster-z3.1', helpers=code_hashes(),
            covariates='verified participant-table tSNR/FD; retained corrected sex/RT; all centered',
            mean='sample-weighted adjusted mean of 25 younger and 22 older participants',
            inference='age variance groups; independent symmetric-error sign flips; Freedman-Lane; no FLAME VARCOPE weighting',
            multiplicity='spatial FWE per signed contrast; not corrected across models, directions or thresholds')
        path = work/'contract.json'
        if path.is_file() and json.loads(path.read_text()) != contract:
            raise ValueError('batch contract changed; completed work preserved; use a new versioned work root')
        pa.write_json(path, contract); prior.compact_copy(path, report/'contract.json')
        identity = pa.identity(contract)
        done, total = collect(prepared, work, report, identity)
        pending = [(m, method) for method in METHODS for m in prepared['models']
                   if not (work/'jobs'/m['name']/method/'complete.json').is_file()]
        if args.collect_only or not (args.execute or args.smoke_only):
            inventory(report)
            print(f'PASS: {len(prepared["models"])} models; {total} jobs; {total*4} signed result cells; no jobs launched', flush=True)
            return
        try:
            if pending or args.smoke_only:
                smoke(prepared, work, report, identity, rt)
            if args.smoke_only or not pending:
                return
            capacity = scale.choose_jobs(args.jobs, scale.memory_budget(), 0, os.cpu_count() or 1)
            # Existing jobs consume MemAvailable already. No old driver is
            # drained/signalled. This is an independent bounded scheduler.
            pa.write_json(report/'capacity.json', capacity)
            print('CAPACITY: '+json.dumps(capacity), flush=True)
            worker = lambda item: run_job(item[0], item[1], prepared, work, report, identity, rt)
            failures, not_attempted = run_pending(pending, worker, capacity['selected_jobs'],
                lambda: collect(prepared, work, report, identity))
            pa.write_json(report/'execution.json', dict(failures=failures, not_attempted=not_attempted))
            if failures:
                raise RuntimeError(f'{len(failures)} PALM jobs failed; completed jobs preserved')
        finally:
            collect(prepared, work, report, identity)
            inventory(report)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
