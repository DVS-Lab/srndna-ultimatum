#!/usr/bin/env python3
"""Separate whole-brain coverage-restricted activation inference.

Four frozen models, all retained contrasts, all 47 participants. L1/L2 values
and FLAME Z maps remain unchanged. Recompute spatial inference and randomise
in new scratch; never modify historical outputs or running PALM workers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import nibabel as nib
import numpy as np

import audit_age_figures as figures
import audit_palm_varcope_coverage as coverage
from audit_l1_designs import read_vest_matrix
from prepare_sub144_imaging_repair import fsf_value
from prepare_ultimatum_l3_repair import parse_inputs
import run_dmn_coverage_check as maskcheck
import run_permutation_audit as pa

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/ZPOOL/data/scratch/srndna-ultimatum-activation-coverage-v1')
REPORT = ROOT/'results/activation_coverage/v1'
AGE_AUDIT = ROOT/'results/age_figure_audit/v1/attempt-20261030T161854Z-2NzpRx/outputs'
N = 10000
METHODS = ('cluster-t2.6', 'cluster-t3.1')
NAMES = ['human-computer-response']+[f'crossed__{t}_act_acceptance-intercept' for t in figures.pf.TIMINGS]


def save(path, values, reference):
    maskcheck.save(path, np.asarray(values), reference)


def copy_small(path, destination):
    maskcheck.compact_copy(Path(path), Path(destination))


def helpers():
    return {str(p): pa.digest(p) for p in (Path(__file__), Path(figures.__file__), Path(coverage.__file__),
        Path(maskcheck.__file__), Path(pa.__file__), Path(figures.pf.__file__), Path(figures.bars.__file__))}


def receipt(path, contract=None):
    value = json.loads(path.read_text())
    if contract is not None and value.get('contract') != contract:
        raise ValueError(f'completed job contract differs: {path}')
    pa.verify_files(value['outputs'])
    return value


def l1_inputs(image, subject, cope, sources):
    """Follow actual retained L2 declarations, not assumed scratch names."""
    candidates = [image.parents[2]/'design.fsf', image.parents[1]/'design.fsf']
    declarations = []
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(); inputs = parse_inputs(text.splitlines())
        if len(inputs) != 2 or [i for i, _, _ in inputs] != [1, 2] or {s for _, s, _ in inputs} != {subject}:
            raise ValueError(f'not a two-run participant L2 design: {path}')
        if any(fsf_value(text, f'copeinput.{j}') != '1' for j in range(1, cope+1)):
            raise ValueError(f'L2 does not preserve source COPE numbering 1..{cope}: {path}')
        sources[str(path)] = pa.digest(path)
        declarations.append(tuple(Path(p).resolve() for _, _, p in inputs))
    if not declarations or any(v != declarations[0] for v in declarations):
        raise ValueError(f'no unique retained L2-to-L1 source declarations: {image}')
    runs = []
    for folder in declarations[0]:
        match = re.search(r'(?:_run-|run-)(\d+)', folder.name)
        if not folder.is_dir() or not match:
            raise ValueError(f'missing/non-run L1 source in retained L2 FSF: {folder}')
        runs.append((int(match.group(1)), folder))
    if sorted(i for i, _ in runs) != [1, 2]:
        raise ValueError('expected exactly run 1 and run 2')
    return sorted(runs)


def validate_contrast(folder, number, sources):
    fsf, path = folder/'design.fsf', folder/'design.con'
    text = fsf.read_text(); matrix = read_vest_matrix(path)
    nreal = int(fsf_value(text, 'evs_real'))
    if matrix.ndim != 2 or matrix.shape[0] < number or matrix.shape[1] < nreal:
        raise ValueError(f'L1 contrast dimensions differ: {folder}, COPE{number}')
    expected = np.zeros(matrix.shape[1])
    for j in range(1, nreal+1):
        value = fsf_value(text, f'con_real{number}.{j}')
        if value is None:
            raise ValueError(f'L1 FSF lacks explicit contrast weight: {folder}, COPE{number}, EV{j}')
        expected[j-1] = float(value)
    if not np.any(expected) or not np.allclose(matrix[number-1], expected, atol=1e-6, rtol=0):
        raise ValueError(f'compiled L1 contrast differs from FSF: {folder}, COPE{number}')
    sources.update({str(p): pa.digest(p) for p in (fsf, path)})


def source_catalog(repo, audit):
    sources = {}; report, catalog, expected = figures.catalog(repo, False, sources)
    models = [catalog[name] for name in NAMES]
    if any(m['subjects'] != models[0]['subjects'] for m in models):
        raise ValueError('model participant identities/order differ; no positional remapping guessed')
    for model in models:
        figures.validate_model(model, sources)
    # Check completed figure-audit files; their image hashes provide additional
    # frozen L2/VARCOPE/mask and native-partner provenance.
    for phase in ('effects', 'influence'):
        root = audit/phase
        inv = figures.inventory(root, 'inventory.tsv', sources)
        if not json.loads((root/'status.json').read_text())['complete']:
            raise ValueError(f'figure audit incomplete: {phase}')
        for relative in inv:
            figures.verified(root, relative, inv, sources)
        previous = json.loads((root/'provenance.json').read_text())['sources']
        expected.update(previous)
    return models, sources, expected, report


def prepare_model(model, folder, expected, sources):
    folder.mkdir(parents=True)
    reference = nib.load(model['mask']); original = reference.get_fdata() > 0
    reader = coverage.Inputs(reference, expected)
    primary = 10 if model['name'] == 'human-computer-response' else 7
    contrasts = (10,) if primary == 10 else (2, 4, 6, 7)
    l2_count = np.zeros(reference.shape, np.uint16)
    run_count = np.zeros(reference.shape, np.uint16)
    data = []; coverage_rows = []; contrast_rows = []
    for subject, name in zip(model['subjects'], model['images']):
        image = Path(name)
        if image.parent.parent.name != f'cope{primary}.feat':
            raise ValueError(f'unexpected participant-level contrast: {image}')
        participant_valid = original.copy(); l2_cope = None
        for number in contrasts:
            p = Path(name.replace(f'/cope{primary}.feat/', f'/cope{number}.feat/'))
            c = reader.load(p, f'L2 COPE{number}'); v = reader.load(p.with_name('varcope1.nii.gz'), f'L2 VARCOPE{number}')
            m = reader.load(p.parent.parent/'mask.nii.gz', f'L2 mask COPE{number}')
            valid = maskcheck.valid_estimate(c, v, m); participant_valid &= valid
            contrast_rows.append(dict(subject=subject, stage='l2', run='', cope=number, valid_original_voxels=int(np.sum(original & valid))))
            if number == primary:
                if not np.isfinite(c).all():
                    raise ValueError('nonfinite COPE data outside mask; no zero replacement/imputation guessed')
                l2_cope = c
        l2_count += participant_valid
        data.append(l2_cope)
        for run, source in l1_inputs(image, subject, primary, sources):
            run_valid = original.copy()
            m = reader.load(source/'mask.nii.gz', 'L1 FEAT mask')
            for number in contrasts:
                validate_contrast(source, number, sources)
                c = reader.load(source/'stats'/f'cope{number}.nii.gz', f'L1 COPE{number}')
                v = reader.load(source/'stats'/f'varcope{number}.nii.gz', f'L1 VARCOPE{number}')
                valid = maskcheck.valid_estimate(c, v, m); run_valid &= valid
                contrast_rows.append(dict(subject=subject, stage='l1', run=run, cope=number,
                    valid_original_voxels=int(np.sum(original & valid))))
            run_count += run_valid
            coverage_rows.append(dict(subject=subject, run=run, valid_original_voxels=int(run_valid.sum())))
        print(f'CHECKED: {model["name"]} {subject}; both runs and L2', flush=True)
    data = np.stack(data, axis=3)
    complete = original & (run_count == 94) & (l2_count == 47)
    nonconstant = np.ptp(data, axis=3) > 0
    save(folder/'copes.nii.gz', data, reference)
    save(folder/'original_mask.nii.gz', original.astype('uint8'), reference)
    save(folder/'valid_mask.nii.gz', complete.astype('uint8'), reference)
    save(folder/'nonconstant.nii.gz', nonconstant.astype('uint8'), reference)
    save(folder/'valid_run_count.nii.gz', run_count, reference)
    save(folder/'valid_l2_count.nii.gz', l2_count, reference)
    pa.write_rows(folder/'run_coverage.tsv', coverage_rows)
    pa.write_rows(folder/'contrast_coverage.tsv', contrast_rows)
    pa.write_rows(folder/'input_inventory.tsv', reader.rows)
    for row in reader.rows:
        if row['exists']: sources[row['path']] = row['sha256']
    return reference, complete, original, nonconstant


def prepare(repo, work, audit=AGE_AUDIT):
    marker = work/'prepared.json'
    if marker.exists():
        saved = receipt(marker); pa.verify_files(saved['sources'])
        if saved['helpers'] != helpers():
            raise ValueError('preparation code changed; completed work preserved; choose a new versioned root')
        print('SKIP_COMPLETE: verified coverage preparation', flush=True)
        return saved
    models, sources, expected, original_report = source_catalog(repo, audit)
    root = Path(tempfile.mkdtemp(prefix='preparation-', dir=work))
    arrays = {}; references = {}
    for model in models:
        reference, valid, original, nonconstant = prepare_model(model, root/model['name'], expected, sources)
        references[model['name']] = reference
        arrays[model['name']] = dict(valid=valid, original=original, nonconstant=nonconstant)
    parietal = models[1:]
    for model in parietal[1:]:
        figures.same(references[parietal[0]['name']], references[model['name']])
    common = np.logical_and.reduce([arrays[m['name']]['valid'] & arrays[m['name']]['nonconstant'] for m in parietal])
    if not common.any():
        raise ValueError('no common valid parietal model search domain')
    frozen = []; coverage_summary = []
    seeds = pa.read_rows(original_report/'model_contrast_summary.tsv')
    sources[str(original_report/'model_contrast_summary.tsv')] = pa.digest(original_report/'model_contrast_summary.tsv')
    for model in models:
        name = model['name']; folder = root/name; reference = references[name]
        retained = arrays[name]['original']; tested = (arrays[name]['valid'] & arrays[name]['nonconstant']) if name == NAMES[0] else common
        if not tested.any() or np.any(tested & ~retained):
            raise ValueError('coverage mask empty or expands historical search')
        save(folder/'mask.nii.gz', tested.astype('uint8'), reference)
        old_roi = (figures.supplemental_rois(repo, sources)[1]['roi'] if name == NAMES[0] else figures.parietal_roi(repo, sources))
        figures.same(old_roi, reference); selected = old_roi.get_fdata() > 0
        valid_selected = selected & tested
        if not valid_selected.any(): raise ValueError('selected region has no common valid voxels; no replacement region selected')
        save(folder/'historical_roi.nii.gz', selected.astype('uint8'), reference)
        save(folder/'descriptive_roi.nii.gz', valid_selected.astype('uint8'), reference)
        coverage_summary.append(dict(model=name, participants=47, runs=94, original_voxels=int(retained.sum()),
            individual_valid_voxels=int(arrays[name]['valid'].sum()), tested_voxels=int(tested.sum()),
            historical_selected_voxels=int(selected.sum()), valid_selected_voxels=int(valid_selected.sum()),
            additional_contrasts_checked='10' if name == NAMES[0] else '2;4;6;7',
            policy='human own common support; parietal shared support across all three timing models; all participants and both runs'))
        new = dict(model, block=pa.identity([model['block'], pa.digest(folder/'mask.nii.gz')]))
        new['group'] = pa.identity([new['block'], new['x']])
        block = work/'blocks'/new['block']; block.mkdir(parents=True, exist_ok=True)
        for filename in ('copes.nii.gz', 'mask.nii.gz'):
            target = block/filename
            if target.exists() or target.is_symlink():
                if not target.is_file() or pa.digest(target) != pa.digest(folder/filename):
                    raise ValueError('incomplete preparation has different input links; preserve it and use a new work root')
            else: target.symlink_to(folder/filename)
        group = work/'groups'/new['group']; group.mkdir(parents=True, exist_ok=True)
        design = group/'design.mat'
        if design.exists():
            if not np.allclose(read_vest_matrix(design), new['x'], atol=1e-10, rtol=0):
                raise ValueError('incomplete preparation group design changed')
        else: pa.vest(design, np.array(new['x']))
        pa.vest(folder/'design.mat', np.array(new['x'])); pa.vest(folder/'design.con', np.array(new['c']), contrast=True)
        # Retain every signed original contrast, original seed, and original N.
        historical = [r for r in seeds if r['model'] == name]
        for i in range(1, len(new['c'])+1):
            rs = [r for r in historical if int(r['contrast']) == i]
            if len(rs) != 3 or len({r['seed'] for r in rs}) != 1 or any(int(r['permutations_recorded']) != N for r in rs):
                raise ValueError(f'historical seeds/counts differ: {name}, contrast {i}')
        frozen.append(new)
    tests, mapping = pa.plan_tests(frozen, 1)
    pa.retain_reference_seeds(tests, mapping, original_report)
    for test in tests:
        path = work/'tests'/test['test']; path.mkdir(parents=True, exist_ok=True)
        con = path/'design.con'; matrix = np.array(test['contrast'])[None, :]
        if con.exists():
            if not np.allclose(read_vest_matrix(con), matrix, atol=1e-10, rtol=0):
                raise ValueError('incomplete preparation contrast changed')
        else: pa.vest(con, matrix, contrast=True)
    pa.write_rows(root/'coverage_summary.tsv', coverage_summary)
    pa.write_rows(root/'test_mapping.tsv', mapping)
    pa.write_json(root/'models.json', frozen)
    pa.write_json(root/'policy.json', dict(
        mask='Historical whole-brain mask intersected with finite COPE, finite positive VARCOPE and FEAT mask in all L1 runs and all L2 estimates. Remove constants.',
        parietal='One intersection across all three timing models and native partner contrasts 2/4/6 plus focal 7; no significance-based mask choice.',
        changes='Coverage/search domain only. No participant exclusions, variance substitutions, coefficient/L1/L2/FLAME refits, or new covariates.',
        flame='Recompute cluster inference on retained FLAME Z fields at Z=3.1. Re-estimate Z-field smoothness in each mask. '
              'Original-mask calibration is recomputed with the same procedure to isolate the mask effect; compare it with historical maps.',
        randomise='Original designs, signed contrasts, seeds, -R -N, 10000 permutations; cluster t=2.6 and 3.1. No new TFCE jobs.',
        bars='Historical selected regions intersected with common validity; NOT newly significant region selection. Participant SEM for means; HC3 SE for OLS coefficients.'))
    outputs = {str(p): pa.digest(p) for p in root.rglob('*') if p.is_file()}
    for directory in ('blocks', 'groups', 'tests'):
        outputs.update({str(p):pa.digest(p) for p in (work/directory).rglob('*') if p.is_file()})
    saved = dict(root=str(root), models=frozen, tests=tests, mapping=mapping, sources=sources, outputs=outputs, helpers=helpers())
    pa.write_json(marker, saved)
    return saved


def group_directory(model):
    paths = [Path(p).parent.parent for p in model['source_hashes'] if p.endswith('/design/design.fsf')]
    if len(paths) != 1: raise ValueError('ambiguous retained group report')
    return paths[0]


def software():
    result = {}
    for name in ('randomise', 'smoothest', 'fsl-cluster'):
        candidate = shutil.which(name)
        if name == 'fsl-cluster' and candidate is None:
            fsldir = Path(os.environ.get('FSLDIR', '/usr/local/fsl'))
            candidate = next((str(p) for p in [fsldir/'bin/fsl-cluster', fsldir/'share/fsl/bin/fsl-cluster', fsldir/'bin/cluster', fsldir/'share/fsl/bin/cluster'] if p.is_file()), None)
        if candidate is None: raise FileNotFoundError(f'FSL executable missing: {name}')
        result[name] = dict(path=str(Path(candidate).resolve()), sha256=pa.digest(candidate))
    return result


def historical_maps(models):
    sources = {}
    for model in models:
        historical = group_directory(model)
        rows = pa.read_rows(historical.parent/'inventory.tsv')
        index = {r.get('artifact', r.get('destination')): r['sha256'] for r in rows}
        for i in range(1, len(model['c'])+1):
            for filename in (f'zstat{i}.nii.gz', f'cluster_mask_zstat{i}.nii.gz'):
                path = historical/f'contrast{i}'/filename
                digest = pa.digest(path)
                if index.get(str(path.relative_to(historical.parent))) != digest:
                    raise ValueError(f'retained FLAME map differs from collected inventory: {path}')
                sources[str(path)] = digest
    return sources


def smoothness(text, expected_volume):
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in ('DLH', 'VOLUME', 'RESELS'):
            values[fields[0]] = float(fields[1])
    if set(values) != {'DLH', 'VOLUME', 'RESELS'} or not all(np.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError('missing/nonpositive FSL smoothness estimates')
    if values['VOLUME'] != expected_volume:
        raise ValueError('smoothest search volume differs from explicit coverage mask')
    return values


def run_flame(model, contrast, mode, prepared, work, report, contract, binaries):
    parent = work/'flame'/model['name']/f'contrast{contrast}'/mode; parent.mkdir(parents=True, exist_ok=True)
    marker = parent/'complete.json'
    if marker.exists():
        saved = receipt(marker, contract)
        print(f'SKIP_COMPLETE: FLAME cluster {model["name"]} contrast{contrast} {mode}', flush=True)
    else:
        attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
        source = group_directory(model)/f'contrast{contrast}'/f'zstat{contrast}.nii.gz'
        reference = nib.load(Path(prepared['root'])/model['name']/('original_mask.nii.gz' if mode == 'original-mask' else 'mask.nii.gz'))
        keep = reference.get_fdata() > 0; z = nib.load(source); figures.same(reference, z)
        values = z.get_fdata()
        if not np.isfinite(values[keep]).all(): raise ValueError('nonfinite retained FLAME Z field')
        save(attempt/'masked_zstat.nii.gz', np.where(keep, values, 0).astype('float32'), reference)
        save(attempt/'mask.nii.gz', keep.astype('uint8'), reference)
        cmd = [binaries['smoothest']['path'], '-z', str(attempt/'masked_zstat.nii.gz'), '-m', str(attempt/'mask.nii.gz')]
        commands = [cmd]
        try:
            with (attempt/'smoothness.txt').open('w') as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'smoothest failed: {attempt}/smoothness.txt')
            estimates = smoothness((attempt/'smoothness.txt').read_text(), int(keep.sum()))
            cmd = [binaries['fsl-cluster']['path'], f'--in={attempt}/masked_zstat.nii.gz', '--thresh=3.1', '--pthresh=0.05',
                f'--dlh={estimates["DLH"]}', f'--volume={int(estimates["VOLUME"])}',
                f'--oindex={attempt}/cluster_index.nii.gz', f'--othresh={attempt}/thresh_zstat.nii.gz',
                f'--olmax={attempt}/local_maxima.txt', '--mm']
            commands.append(cmd)
            with (attempt/'clusters.tsv').open('w') as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'fsl-cluster failed: {attempt}/clusters.tsv')
            rows = pa.read_rows(attempt/'clusters.tsv')
            if any('P' not in r or 'Voxels' not in r for r in rows):
                raise ValueError('cluster table lacks probability/extent columns')
            image = nib.load(attempt/'thresh_zstat.nii.gz'); figures.same(image, reference)
            significant = image.get_fdata() != 0
            if np.any(significant & ~keep): raise ValueError('cluster output exceeds testing mask')
            save(attempt/'significant_mask.nii.gz', significant.astype('uint8'), reference)
            pa.write_json(attempt/'summary.json', dict(model=model['name'], engine='retained-FLAME-Z-cluster', mode=mode,
                contrast=contrast, contrast_name=model['labels'][contrast-1], clusters=len(rows), significant_voxels=int(significant.sum()),
                minimum_spatial_fwe_p=min(float(r['P']) for r in rows) if rows else None, smoothness=estimates,
                source_zstat=str(source), source_zstat_sha256=pa.digest(source), coefficient_refits=0))
        finally:
            pa.write_json(attempt/'commands.json', commands)
            for p in attempt.iterdir():
                if p.is_file(): copy_small(p, report/'attempt_logs/flame'/model['name']/f'contrast{contrast}'/mode/attempt.name/p.name)
        saved = dict(contract=contract, attempt=str(attempt), outputs={str(p):pa.digest(p) for p in attempt.iterdir() if p.is_file()})
        pa.write_json(marker, saved)
    for p in Path(saved['attempt']).iterdir():
        if p.is_file(): copy_small(p, report/'flame'/model['name']/f'contrast{contrast}'/mode/p.name)
    copy_small(marker, report/'flame'/model['name']/f'contrast{contrast}'/mode/'complete.json')


def run_randomise(test, method, prepared, work, report, contract, binaries):
    path = work/'jobs'/test['test']/method/'complete.json'
    if path.exists():
        saved = receipt(path)
        if saved.get('coverage_contract') not in (None, contract): raise ValueError('randomise receipt contract differs')
        expected_command = pa.command(test, method, work, Path(saved['attempt'])/'randomise', N, binaries['randomise']['path'])
        if saved['command'] != expected_command:
            raise ValueError('completed randomise command differs from the locked coverage plan')
    try:
        saved = pa.run_test(test, method, work, N, binaries['randomise']['path'])
        if saved['permutations_recorded'] != N:
            raise ValueError('randomise did not record exactly 10000 permutations')
        saved['coverage_contract'] = contract; pa.write_json(path, saved)
    finally:
        parent = work/'jobs'/test['test']/method
        for attempt in parent.glob('attempt-*'):
            for p in attempt.iterdir():
                if p.is_file(): copy_small(p, report/'attempt_logs/randomise'/test['test']/method/attempt.name/p.name)
    for p in Path(saved['attempt']).iterdir():
        if p.is_file(): copy_small(p, report/'randomise'/test['test']/method/p.name)
    copy_small(path, report/'randomise'/test['test']/method/'complete.json')


def export_preparation(prepared, report):
    root = Path(prepared['root'])
    for p in root.rglob('*'):
        if p.is_file() and p.name != 'copes.nii.gz':
            copy_small(p, report/'preparation'/p.relative_to(root))


def summarize(prepared, work, report, contract):
    rows = []; complete = 0; calibration = []
    for model in prepared['models']:
        historical_root = group_directory(model)
        for contrast in range(1, len(model['c'])+1):
            for mode in ('original-mask', 'coverage-mask'):
                marker = work/'flame'/model['name']/f'contrast{contrast}'/mode/'complete.json'
                if not marker.exists(): continue
                saved = receipt(marker, contract); attempt = Path(saved['attempt']); complete += 1
                value = json.loads((attempt/'summary.json').read_text())
                rows.append({k:v for k,v in value.items() if k not in ('smoothness', 'source_zstat_sha256', 'source_zstat')})
                if mode == 'original-mask':
                    oldpath = historical_root/f'contrast{contrast}'/f'cluster_mask_zstat{contrast}.nii.gz'
                    old = nib.load(oldpath); new = nib.load(attempt/'significant_mask.nii.gz'); figures.same(old, new)
                    a, b = old.get_fdata() > 0, new.get_fdata() > 0
                    calibration.append(dict(model=model['name'], contrast=contrast, historical_voxels=int(a.sum()),
                        reestimated_original_mask_voxels=int(b.sum()), overlap_voxels=int(np.sum(a & b)),
                        identical_significant_mask=int(np.array_equal(a, b)),
                        note='Original-mask rerun calibrates any smoothness/rounding difference; not a coefficient refit.'))
    for mapping in prepared['mapping']:
        model = next(m for m in prepared['models'] if m['name'] == mapping['model'])
        keep = nib.load(Path(prepared['root'])/model['name']/'mask.nii.gz').get_fdata() > 0
        for method in METHODS:
            marker = work/'jobs'/mapping['test']/method/'complete.json'
            if not marker.exists(): continue
            saved = receipt(marker)
            if saved.get('coverage_contract') is None: continue  # brief receipt/annotation window, or interrupted export
            if saved.get('coverage_contract') != contract: raise ValueError('randomise contract differs in collection')
            attempt = Path(saved['attempt']); im = nib.load(attempt/'randomise_clustere_corrp_tstat1.nii.gz')
            p = im.get_fdata(); complete += 1
            rows.append(dict(model=mapping['model'], engine='randomise', mode=method, contrast=mapping['contrast'],
                contrast_name=mapping['contrast_name'], clusters='', significant_voxels=int(np.sum((p > .95) & keep)),
                minimum_spatial_fwe_p=float(1-p[keep].max()), coefficient_refits=0))
    if rows: pa.write_rows(report/'summary.tsv', rows)
    if calibration: pa.write_rows(report/'flame_original_mask_calibration.tsv', calibration)
    expected = sum(len(m['c']) for m in prepared['models'])*4
    pa.write_json(report/'status.json', dict(complete=complete == expected, completed_jobs=complete, expected_jobs=expected,
        participants=47, exclusions=0, l1_l2_refits=0, flame_coefficient_refits=0,
        original_palm_changed=False, calibration_mismatches=sum(not r['identical_significant_mask'] for r in calibration)))
    return complete, expected


def export_figures(prepared, report):
    output = report/'figures'
    # Repeatable export: completed static files are verified, never replaced.
    marker = output/'complete.json'
    if marker.exists():
        receipt(marker); print('SKIP_COMPLETE: coverage-restricted bars and influence tables', flush=True); return
    folder = Path(tempfile.mkdtemp(prefix='figures-', dir=Path(prepared['root']).parent))
    summaries = []; people = []; comparisons = []; deletions = []; conditions = []; diagnostics = {}
    for model in prepared['models']:
        root = Path(prepared['root'])/model['name']; roi = nib.load(root/'descriptive_roi.nii.gz'); keep = roi.get_fdata() > 0
        values = nib.load(root/'copes.nii.gz').get_fdata()[keep, :].mean(0)
        if model['name'] == NAMES[0]:
            rows, adjusted = figures.adjusted_sample_bars(model, values)
            summaries.extend(dict(model=model['name'], voxels=int(keep.sum()), **r) for r in rows)
            figures.simple_bars(rows, f'Human > computer task response\nHistorical region intersected with valid coverage: {int(keep.sum())} voxels', folder/'human_computer_bars.png')
            people.extend(dict(model=model['name'], subject=s, raw_roi_cope=float(a), adjusted_roi_cope=float(b)) for s,a,b in zip(model['subjects'],values,adjusted))
            continue
        timing = model['name'].removeprefix('crossed__').removesuffix('_act_acceptance-intercept')
        x, contrasts = figures.pf.aligned_design(group_directory(model)/'design', model['subjects'])
        rows, details, deleted, diag, robust = figures.influence_audit(x, values, contrasts[0], model['subjects'])
        comparisons.extend(dict(timing=timing, voxels=int(keep.sum()), **r) for r in rows)
        deletions.extend(dict(timing=timing, **r) for r in deleted)
        people.extend(dict(model=model['name'], subject=r['subject'], raw_roi_cope=r['raw_cope'], adjusted_roi_cope=r['adjusted_cope']) for r in details)
        diagnostics[timing] = (x, values, diag, robust)
        group_c = np.vstack([np.eye(8)[4], np.eye(8)[5], np.eye(8)[4]-np.eye(8)[5], np.eye(8)[5]-np.eye(8)[4]])
        for partner, cope in figures.pm.PARTNERS.items():
            condition = []
            for name in model['images']:
                p = Path(name.replace('/cope7.feat/', f'/cope{cope}.feat/'))
                condition.append(nib.load(p).get_fdata()[keep].mean())
            bars, _, _, _ = figures.bars.adjusted_bars(x, group_c, np.asarray(condition))
            conditions.extend(dict(timing=timing, partner=partner, **r) for r in bars)
    figures.influence_plots(diagnostics, comparisons, conditions, folder)
    pa.write_rows(folder/'human_computer_bar_summary.tsv', summaries)
    pa.write_rows(folder/'participant_estimates.tsv', people)
    pa.write_rows(folder/'parietal_slope_comparison.tsv', comparisons)
    pa.write_rows(folder/'parietal_deletions.tsv', deletions)
    pa.write_rows(folder/'parietal_native_partner_bars.tsv', conditions)
    output.mkdir(parents=True)
    for path in folder.iterdir(): copy_small(path, output/path.name)
    pa.write_json(marker, dict(outputs={str(p):pa.digest(p) for p in output.iterdir() if p.is_file()},
        interpretation='Descriptive fixed historical regions intersected with whole-brain validity, not newly significant regions.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=WORK)
    parser.add_argument('--report-root', type=Path, default=REPORT)
    parser.add_argument('--audit-root', type=Path, default=AGE_AUDIT)
    parser.add_argument('--jobs', type=int, default=8, choices=range(1, 25))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); work = args.work_root.resolve(); report = args.report_root.resolve()
    if work == ROOT or ROOT in work.parents or work in ROOT.parents:
        parser.error('scratch must be outside repository')
    if report != REPORT:
        parser.error('use the dedicated results/activation_coverage/v1 report root')
    work.parent.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    with work.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); work.mkdir(exist_ok=True)
        prepared = prepare(ROOT, work, args.audit_root.resolve())
        binaries = software()
        # Freeze original Z/mask sources for all retained contrasts before any
        # jobs, including original-mask smoothness calibration.
        extra = historical_maps(prepared['models'])
        contract = dict(prepared_sha256=pa.digest(work/'prepared.json'), helpers=helpers(), software=binaries,
            z_fields_and_historical_masks=extra, permutations=N, methods=METHODS)
        path = work/'contract.json'
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(contract)):
            raise ValueError('coverage contract changed; existing results preserved; use a new version')
        pa.write_json(path, contract); key = pa.identity(contract)
        copy_small(path, report/'contract.json'); export_preparation(prepared, report)
        completed, expected = summarize(prepared, work, report, key)
        print(f'PASS: common coverage prepared; {completed}/{expected} inference jobs completed; all 47 participants retained', flush=True)
        if not args.execute:
            print('Dry run: no statistical executables launched; historical outputs and PALM unchanged', flush=True); return
        tasks = []
        for model in prepared['models']:
            for i in range(1, len(model['c'])+1):
                for mode in ('original-mask', 'coverage-mask'):
                    tasks.append(('flame', model, i, mode))
        for test in prepared['tests']:
            for method in METHODS: tasks.append(('randomise', test, 0, method))
        def run(task):
            engine, item, i, mode = task
            if engine == 'flame': run_flame(item, i, mode, prepared, work, report, key, binaries)
            else: run_randomise(item, mode, prepared, work, report, key, binaries)
        failures = []; queued = []
        # Small single-threaded FSL workers, bounded independently of PALM.
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                if future.cancelled():
                    queued.append(dict(engine=task[0], target=task[1].get('name', task[1].get('test')), contrast=task[2], method=task[3]))
                    continue
                try: future.result()
                except Exception as exc:
                    traceback.print_exc(); failures.append(dict(engine=task[0], target=task[1].get('name', task[1].get('test')), contrast=task[2], method=task[3], error=str(exc)))
                    for pending in futures: pending.cancel()
                finally: summarize(prepared, work, report, key)
        # Figure export is independent of inferential success and uses only
        # fully audited coverage/inputs. Recheck frozen sources first.
        pa.verify_files(prepared['sources']); pa.verify_files(extra); export_figures(prepared, report)
        completed, expected = summarize(prepared, work, report, key)
        pa.write_json(report/'batch_outcome.json', dict(failures=failures, queued=queued, completed=completed, expected=expected))
        # The outer shell is still writing launch/run.txt; freeze data artifacts
        # only. Launch logs remain tracked separately without a premature hash.
        files = [dict(path=str(p.relative_to(report)), sha256=pa.digest(p), bytes=p.stat().st_size)
            for p in sorted(report.rglob('*')) if p.is_file() and p.name != 'inventory.tsv'
            and not p.relative_to(report).parts[0].startswith('launch-')]
        pa.write_rows(report/'inventory.tsv', files)
        if failures or completed != expected:
            raise RuntimeError('coverage inference incomplete; push logs and repeat the same command to resume verified jobs')
        print('PASS: all coverage-restricted inference, bars and influence checks complete; original analyses/PALM preserved', flush=True)


if __name__ == '__main__':
    main()
