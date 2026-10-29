#!/usr/bin/env python3
"""Separate full-RT DMN-age coverage check; never modify the running PALM batch.

Same 47 people, original estimates/designs/seeds, 10,000 resamples. Restrict the
whole-brain search to finite, positive-variance estimates in every run and L2.
No L1/L2 refits, no variance substitution, no selected-cluster mask construction.
"""
import argparse
import fcntl
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import nibabel as nib
import numpy as np

import audit_palm_varcope_coverage as coverage
import scale_palm_recovery as scale

headless = scale.headless
palm = scale.palm
pa = palm.pa
ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'results/dmn_followup/v1/coverage_audit/attempt-20261029T025636Z-F9W34I/outputs'
PALM_SOURCE = headless.logs.WORK
N = 10000


def film_thresholds(text):
    """Extract explicit thresholds only, never guess from a current default."""
    # A character class such as [-+\d.eE]+ also accepts '--'. In a modern
    # command, design.mat is commonly followed by another option, not a legacy
    # positional threshold. Require an entire real-number token instead.
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    end = r'(?=$|[\s<>\"\';&])'
    result = []
    for line in html.unescape(text).splitlines():
        if not re.search(r'\bfilm_gls\b', line):
            continue
        modern = re.findall(r'--thr(?:eshold)?(?:=|\s+)(' + number + ')' + end, line)
        old = re.findall(r'\bdesign\.mat[\"\']?\s+(' + number + ')' + end, line)
        for value in modern+old:
            result.append(float(value))
    return result


def film_evidence(folder, snapshot=None):
    paths = [folder/'report_log.html', folder/'stats/logfile']
    if (folder/'logs').is_dir():
        paths += sorted(p for p in (folder/'logs').rglob('*') if p.is_file())
    evidence, candidates = [], []
    for path in sorted(set(paths)):
        if not path.is_file() or path.stat().st_size > 8_000_000:
            continue
        text = path.read_text(errors='replace')
        lines = [html.unescape(s).strip() for s in text.splitlines() if re.search(r'\bfilm_gls\b', s)]
        if lines:
            candidates.append(dict(path=str(path), sha256=pa.digest(path), commands=lines))
            # Retain real command evidence even if parsing/uniqueness later fails.
            if snapshot is not None:
                pa.write_json(snapshot, candidates)
        values = film_thresholds(text)
        if values:
            evidence.append(dict(path=str(path), sha256=pa.digest(path), thresholds=values, commands=lines))
    thresholds = sorted({v for row in evidence for v in row['thresholds']})
    if len(thresholds) != 1 or not np.isfinite(thresholds[0]):
        raise ValueError(f'no unique explicit FILM threshold found: {folder}; preserve logs for review, no fits launched')
    return thresholds[0], evidence


def valid_estimate(cope, variance, mask):
    return np.isfinite(cope) & np.isfinite(variance) & (variance > 0) & np.isfinite(mask) & (mask > 0)


def save(path, data, reference):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), reference.affine), path)


def verify_receipt(path, contract=None):
    saved = json.loads(path.read_text())
    if contract is not None and saved.get('contract') != contract:
        raise ValueError(f'changed coverage-check contract: {path}')
    pa.verify_files(saved['outputs'])
    return saved


def prepare(repo, work, audit=AUDIT):
    marker = work/'prepared.json'
    if marker.exists():
        saved = verify_receipt(marker)
        pa.verify_files(saved['sources'])
        if saved['code_sha256'] != pa.digest(Path(__file__)):
            raise ValueError('coverage-check code changed; use a new versioned work/report root')
        return saved
    coverage.pm.verify_inventory(audit)
    config = repo/'results/dmn_covariate_standardized/v1/randomise/config.json'
    hashes_path = config.with_name('input_provenance.json')
    recorded = json.loads((audit/'provenance.json').read_text())['sources']
    for p in (config, hashes_path):
        if pa.digest(p) != recorded.get(str(p)):
            raise ValueError('audited source configuration/provenance changed')
    model = next(m for m in json.loads(config.read_text())['models'] if m['name'] == 'full-rt__dmn-age')
    x, c = np.array(model['x']), np.array(model['c'])
    if (len(model['subjects']) != 47 or len(set(model['subjects'])) != 47 or len(model['images']) != 47
        or x.shape != (47, 6) or c.shape != (4, 6) or np.linalg.matrix_rank(x) != 6
        or model['labels'] != ['younger', 'older', 'younger-older', 'older-younger']):
        raise ValueError('unexpected frozen primary DMN-age design')
    younger = x@c[0]; older = x@c[1]
    if not (np.isin(younger, [0, 1]).all() and np.array_equal(younger+older, np.ones(47))
            and younger.sum() == 25 and older.sum() == 22 and np.array_equal(c[2], c[0]-c[1])
            and np.array_equal(c[3], -c[2])):
        raise ValueError('age coding/contrasts changed')
    root = Path(tempfile.mkdtemp(prefix='preparation-', dir=work))
    inputs = root/'inputs'; inputs.mkdir()
    reference = nib.load(model['mask']); retained = reference.get_fdata() > 0
    expected = json.loads(hashes_path.read_text())
    expected.update({r['path']: r['sha256'] for r in pa.read_rows(audit/'input_inventory.tsv') if r['exists'] == '1'})
    reader = coverage.Inputs(reference, expected)
    sources = {str(p): pa.digest(p) for p in (config, hashes_path, audit/'inventory.tsv', Path(model['mask']))}
    run_support = np.zeros(reference.shape, np.uint16); l2_support = run_support.copy()
    images, records, film_rows, film_sources = [], [], [], []
    flagged = json.loads((audit/'status.json').read_text())['detailed_participants']
    for subject, path in zip(model['subjects'], model['images']):
        image = Path(path)
        cope = reader.load(image, 'L2 COPE7'); var = reader.load(image.with_name('varcope1.nii.gz'), 'L2 VARCOPE7')
        mask = reader.load(image.parent.parent/'mask.nii.gz', 'L2 mask')
        l2_support += valid_estimate(cope, var, mask)
        # Preserve the original data; the new mask, not zero replacement, controls testing.
        if not np.isfinite(cope).all():
            raise ValueError(f'nonfinite merged COPE data: {subject}; no substitution guessed')
        images.append(cope)
        for run in ('01', '02'):
            folder = image.parents[3]/f'L1_run-{run}.feat'
            v = reader.load(folder/'stats/varcope7.nii.gz', 'L1 VARCOPE7')
            b = reader.load(folder/'stats/cope7.nii.gz', 'L1 COPE7')
            m = reader.load(folder/'mask.nii.gz', 'L1 mask')
            valid = valid_estimate(b, v, m); run_support += valid
            records.append(dict(subject=subject, run=run, original_search_voxels=int(retained.sum()),
                valid_estimates_in_original_search=int(np.sum(retained & valid))))
            if subject in flagged:
                threshold, evidence = film_evidence(folder, root/'film_command_evidence.json')
                mean = reader.load(folder/'mean_func.nii.gz', 'flagged mean EPI')
                # Confirm the observed whole-volume boundary, not only selected ROI voxels.
                bad = (m > 0) & (v <= 0); positive = (m > 0) & (v > 0)
                violations = int(np.sum(bad & (mean >= threshold)) + np.sum(positive & (mean < threshold)))
                film_rows.append(dict(subject=subject, run=run, recorded_film_threshold=threshold,
                    zero_or_negative_varcope_voxels_inside_mask=int(bad.sum()),
                    threshold_pattern_violations=violations, command_sources=len(evidence)))
                film_sources.extend(evidence)
                pa.write_rows(root/'film_thresholds.tsv', film_rows)
                pa.write_json(root/'film_command_evidence.json', film_sources)
                if threshold != 1000 or violations:
                    raise ValueError(f'FILM threshold/pattern differs from audit: {subject} run-{run}; inspect saved evidence')
        print(f'CHECKED: {subject}: both L1 runs and retained L2', flush=True)
    data = np.stack(images, axis=3)
    statistical_coverage = retained & (run_support == 94) & (l2_support == 47)
    tested = statistical_coverage & (np.ptp(data, axis=3) > 0)
    if not tested.any():
        raise ValueError('no voxels with valid estimates in all runs/participants')
    # Selected ROI is used only AFTER constructing the whole-brain mask, for descriptive accounting.
    roi = reader.load(audit/'selected_roi.nii.gz', 'previous selected ROI') > 0
    save(inputs/'copes.nii.gz', data, reference)
    for name, values in [('mask', tested.astype(np.uint8)), ('original_mask', retained.astype(np.uint8)),
                         ('valid_run_count', run_support), ('valid_l2_count', l2_support),
                         ('previous_roi', roi.astype(np.uint8))]:
        save(root/f'{name}.nii.gz', values, reference)
    shutil.copy2(root/'mask.nii.gz', inputs/'mask.nii.gz')
    pa.vest(inputs/'design.mat', x); pa.vest(inputs/'design.con', c, contrast=True)
    np.savetxt(inputs/'variance_groups.csv', (1+older).astype(int), fmt='%d')
    pa.write_rows(root/'participants.tsv', [dict(subject=s, variance_group=int(1+o)) for s, o in zip(model['subjects'], older)])
    pa.write_rows(root/'run_coverage.tsv', records)
    pa.write_rows(root/'input_inventory.tsv', reader.rows)
    for row in reader.rows:
        if row['exists']: sources[row['path']] = row['sha256']
    for row in film_sources: sources[row['path']] = row['sha256']
    seeds_path = config.with_name('model_contrast_summary.tsv')
    historical = [r for r in pa.read_rows(seeds_path) if r['model'] == model['name']]
    seeds = {}
    for i in range(1, 5):
        rows = [r for r in historical if int(r['contrast']) == i]
        if len(rows) != 3 or {r['method'] for r in rows} != set(pa.METHODS) or len({r['seed'] for r in rows}) != 1:
            raise ValueError('historical randomise contrast seed/method mismatch')
        if any(int(r['permutations_recorded']) != N for r in rows):
            raise ValueError('historical randomise count differs from 10000')
        seeds[str(i)] = int(rows[0]['seed'])
        folder = inputs/f'contrast{i}'; folder.mkdir()
        pa.vest(folder/'design.con', c[i-1][None, :], contrast=True)
    sources[str(seeds_path)] = pa.digest(seeds_path)
    pa.write_rows(root/'historical_randomise.tsv', historical)
    summary = dict(participants=47, runs=94, original_search_voxels=int(retained.sum()),
        original_nonconstant_search_voxels=int(np.sum(retained & (np.ptp(data, axis=3) > 0))),
        common_valid_estimation_voxels=int(statistical_coverage.sum()), tested_voxels=int(tested.sum()),
        constant_voxels_removed=int(np.sum(statistical_coverage & ~tested)),
        previous_selected_roi_voxels=int(roi.sum()), previous_roi_valid_voxels=int(np.sum(roi & tested)),
        policy='Original whole-brain mask INTERSECT finite COPE, finite positive VARCOPE and FEAT mask in all 94 L1 runs and all 47 L2 estimates; remove between-participant constants.',
        changes='Search mask only. No imputation, participant exclusion, L1/L2 refit, covariate change or significance-based mask choice.',
        limitations='Strict complete-coverage sensitivity analysis changes spatial search domain and FWE calibration; not a repair of low-signal voxels or independent replication.')
    pa.write_json(root/'coverage_summary.json', summary)
    outputs = {str(p): pa.digest(p) for p in root.rglob('*') if p.is_file()}
    receipt = dict(root=str(root), model=model, randomise_seeds=seeds, sources=sources,
        outputs=outputs, code_sha256=pa.digest(Path(__file__)), permutations=N, coverage=summary)
    pa.write_json(marker, receipt)
    print(json.dumps(summary, indent=2), flush=True)
    return receipt


def runtime(original_work, palm_root):
    # Read the successful original smoke/canary receipts; never acquire its batch lock.
    profile = scale.validated_runtime(original_work)
    frozen = json.loads((original_work/'contract.json').read_text())
    if frozen['shuffles'] != N or pa.digest(Path(frozen['engine'])) != frozen['engine_sha256']:
        raise ValueError('validated MATLAB engine/shuffle count changed')
    commit = subprocess.check_output(['git', '-C', str(palm_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != palm.PALM_COMMIT or subprocess.check_output(['git', '-C', str(palm_root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('PALM must remain clean at the original pinned commit; no checkout performed')
    mode = json.loads((profile/'startup_selection.json').read_text())['mode']
    return dict(profile=str(profile), mode=mode, engine=frozen['engine'],
        engine_sha256=frozen['engine_sha256'], palm_root=str(palm_root), palm_commit=commit,
        evidence={str(p): pa.digest(p) for p in [original_work/'contract.json', profile/'contract.json',
            profile/'startup_selection.json', profile/'smoke_complete.json', profile/'canary_complete.json']})


def compact_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size > 20_000_000:
        raise ValueError(f'not a compact report file: {source}')
    if destination.exists():
        if pa.digest(destination) != pa.digest(source):
            raise ValueError(f'will not replace changed report artifact: {destination}')
    else:
        shutil.copy2(source, destination)


def run_job(engine, method, contrast, prepared, work, report, contract, rt, binary):
    label = engine if engine == 'palm' else f'randomise-contrast{contrast}'
    folder = work/'jobs'/label/method; folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'complete.json'
    if marker.is_file():
        saved = verify_receipt(marker, contract)
        print(f'SKIP_COMPLETE: coverage {label} {method}', flush=True)
    else:
        attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=folder))
        inputs = Path(prepared['root'])/'inputs'; prefix = attempt/engine
        print(f'RUN: coverage {label} {method} -> {attempt}', flush=True)
        try:
            if engine == 'palm':
                args = palm.arguments(inputs, method, prefix, N, 20261018)
                headless.launch(rt['engine'], Path(rt['palm_root']), args, attempt, mode=rt['mode'])
                palm.probability_maps(attempt, method, 4, nib.load(inputs/'mask.nii.gz'))
            else:
                cmd = [binary, '-i', str(inputs/'copes.nii.gz'), '-o', str(prefix),
                    '-d', str(inputs/'design.mat'), '-t', str(inputs/f'contrast{contrast}/design.con'),
                    '-m', str(inputs/'mask.nii.gz'), '-n', str(N),
                    f'--seed={prepared["randomise_seeds"][str(contrast)]}', '-R', '-N', *pa.METHODS[method]]
                pa.write_json(attempt/'command.json', cmd)
                with (attempt/'run.txt').open('w') as stream:
                    result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError(f'randomise exited {result.returncode}; {attempt}/run.txt')
                expected = pa.expected_outputs(attempt, method)
                null = np.loadtxt(expected[2], ndmin=1)
                if null.size != N or not np.isfinite(null).all():
                    raise ValueError('randomise did not save all 10000 null values')
                reference = nib.load(inputs/'mask.nii.gz'); keep = reference.get_fdata() > 0
                for path in expected[:2]:
                    image = nib.load(path); values = image.get_fdata()
                    if not pa.compatible(image, reference) or not np.isfinite(values[keep]).all():
                        raise ValueError('invalid randomise output')
                    if 'corrp' in path.name and not np.all((values[keep] >= 0) & (values[keep] <= 1)):
                        raise ValueError('invalid randomise 1-p range')
            saved = dict(contract=contract, engine=engine, method=method, contrast=contrast,
                attempt=str(attempt), outputs={str(p): pa.digest(p) for p in attempt.iterdir() if p.is_file()})
            pa.write_json(marker, saved)
        finally:
            for path in attempt.iterdir():
                if path.is_file(): compact_copy(path, report/'attempts'/label/method/attempt.name/path.name)
        print(f'COMPLETED: coverage {label} {method}', flush=True)
    for path in Path(saved['attempt']).iterdir():
        if path.is_file(): compact_copy(path, report/'maps'/label/method/path.name)
    compact_copy(marker, report/'maps'/label/method/'complete.json')
    return []


def summarize(prepared, work, report, contract):
    reference = nib.load(Path(prepared['root'])/'mask.nii.gz'); keep = reference.get_fdata() > 0
    rows = []; completed = 0
    for engine, methods in [('palm', palm.METHODS), ('randomise', pa.METHODS)]:
        for contrast in ([0] if engine == 'palm' else range(1, 5)):
            label = engine if engine == 'palm' else f'randomise-contrast{contrast}'
            for method in methods:
                marker = work/'jobs'/label/method/'complete.json'
                if not marker.is_file(): continue
                saved = verify_receipt(marker, contract); attempt = Path(saved['attempt']); completed += 1
                if engine == 'palm':
                    results = [row for _, row in palm.probability_maps(attempt, method, 4, reference)]
                else:
                    unit = 'tfce' if method == 'tfce' else 'clustere'
                    data = nib.load(attempt/f'randomise_{unit}_corrp_tstat1.nii.gz').get_fdata()
                    results = [dict(contrast=contrast, minimum_spatial_fwe_p=float(1-data[keep].max()), significant_voxels=int(np.sum(data[keep] > .95)))]
                rows.extend(dict(engine=engine, method=method, contrast=r['contrast'],
                    contrast_name=prepared['model']['labels'][r['contrast']-1], **{k:v for k,v in r.items() if k != 'contrast'}) for r in results)
    if rows: pa.write_rows(report/'summary.tsv', rows)
    pa.write_json(report/'status.json', dict(complete=completed == 15, completed_jobs=completed,
        expected_jobs=15, signed_result_cells=len(rows), participants=47, exclusions=0,
        l1_l2_refits=0, running_original_palm_changed=False))
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, default=Path('/ZPOOL/data/scratch/srndna-ultimatum-dmn-coverage-v1'))
    parser.add_argument('--report-root', type=Path, default=ROOT/'results/dmn_followup/v1/coverage_check/v1')
    parser.add_argument('--audit-root', type=Path, default=AUDIT)
    parser.add_argument('--original-palm-work', type=Path, default=PALM_SOURCE)
    parser.add_argument('--palm-root', type=Path, default=PALM_SOURCE.parent/'tools/PALM')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    work, report = args.work_root.resolve(), args.report_root.resolve()
    original = args.original_palm_work.resolve()
    if work == ROOT or ROOT in work.parents or work in ROOT.parents or work == original or original in work.parents or work in original.parents:
        parser.error('new scratch must be outside the repository and separate from original PALM work')
    if report != ROOT/'results/dmn_followup/v1/coverage_check/v1':
        parser.error('use the dedicated coverage_check/v1 report root; do not reuse historical reports')
    binary = shutil.which('randomise')
    if not binary: parser.error('randomise must be on PATH')
    work.parent.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    with work.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work.mkdir(exist_ok=True)
        prepared = prepare(ROOT, work, args.audit_root.resolve())
        rt = runtime(original, args.palm_root.resolve())
        contract = dict(prepared_sha256=pa.digest(work/'prepared.json'), runtime=rt,
            code_sha256=pa.digest(Path(__file__)), randomise=str(Path(binary).resolve()),
            randomise_sha256=pa.digest(Path(binary).resolve()), permutations=N,
            helper_hashes={str(p):pa.digest(p) for p in [Path(coverage.__file__), Path(headless.__file__), Path(palm.__file__), Path(pa.__file__), Path(scale.__file__)]})
        key = pa.identity(contract); path = work/'contract.json'
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError('coverage-check software/runtime contract changed; completed outputs preserved')
        pa.write_json(path, contract); compact_copy(path, report/'contract.json')
        root = Path(prepared['root'])
        for p in root.iterdir():
            if p.is_file(): compact_copy(p, report/'preparation'/p.name)
        for name in ('design.mat', 'design.con', 'variance_groups.csv'):
            compact_copy(root/'inputs'/name, report/'design'/name)
        compact_copy(work/'prepared.json', report/'preparation/prepared.json')
        completed = summarize(prepared, work, report, key)
        print(f'PASS: FILM thresholds verified; separate whole-brain coverage check; {completed}/15 completed jobs', flush=True)
        if not args.execute:
            print('Dry run: no MATLAB or randomise launched; original PALM remains untouched', flush=True); return
        tasks = [(dict(name='palm', engine='palm', contrast=0), m) for m in palm.METHODS]
        tasks += [(dict(name=f'randomise-contrast{i}', engine='randomise', contrast=i), m) for i in range(1, 5) for m in pa.METHODS]
        pending = [(job, method) for job, method in tasks if not (work/'jobs'/job['name']/method/'complete.json').is_file()]
        def worker(item):
            job, method = item
            return run_job(job['engine'], method, job['contrast'], prepared, work, report, key, rt, binary)
        capacity = scale.choose_jobs(args.jobs, scale.memory_budget(), 0, os.cpu_count() or 1) if pending else dict(selected_jobs=1)
        pa.write_json(report/'capacity.json', capacity)
        print(f'Additional coverage-check workers: {capacity["selected_jobs"]}; original PALM batch untouched', flush=True)
        _, failures, queued = scale.parallel_jobs(pending, worker, capacity['selected_jobs'])
        completed = summarize(prepared, work, report, key)
        pa.write_json(report/'batch_outcome.json', dict(failures=failures, queued=queued, completed_jobs=completed))
        files = [dict(path=str(p.relative_to(report)), sha256=pa.digest(p), bytes=p.stat().st_size)
                 for p in sorted(report.rglob('*')) if p.is_file() and p.name != 'inventory.tsv']
        pa.write_rows(report/'inventory.tsv', files)
        if failures: raise RuntimeError('coverage check stopped scheduling after a failure; completed jobs preserved; push logs')
        if completed != 15: raise RuntimeError('incomplete coverage check')
        print('PASS: coverage-restricted PALM/randomise complete; 24 signed result cells; original batch unchanged', flush=True)


if __name__ == '__main__':
    main()
