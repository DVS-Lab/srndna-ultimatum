#!/usr/bin/env python3
"""Cross both retained behavioral measures with activation, DMN and ECN L2 COPE7.

Default: prepare and validate only. No L1/L2 reruns, exclusions, or overwrites.
The acceptance-intercept measure is a legacy norm proxy, NOT a Gu learning
parameter or an indifference threshold. The unified choice-slope estimator is
not substituted for the corrected separate-model estimator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from audit_l1_designs import write_tsv
from collect_dmn_revision_checks import collect
from prepare_dmn_revision_checks import digest, safe_empty
from prepare_sub144_imaging_repair import fsf_value, replace_fsf_value
from prepare_ultimatum_l3_repair import parse_evs, parse_inputs
from run_dmn_revision_stage import compile_designs, verify_inputs, verify_stage_outputs
from run_ultimatum_repair_jobs import run_jobs

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    'partner-post': 'results/decision_postresponse_partner/imaging',
    'pooled-post': 'results/decision_postresponse/imaging',
    'full-rt': 'results/reviewer/rt_coverage_corrected',
}
FAMILIES = {'act': 'activation-norm', 'nppi-dmn': 'dmn-age', 'nppi-ecn': 'ecn-sensitivity'}
METRICS = {'choice-slope': 'corrected_sensitivity', 'acceptance-intercept': 'corrected_norm'}
TEMPLATE = 'templates/revision/L3_task-ultimatum_behavioral-association.fsf'
COVARIATES = 'results/reviewer/tables/l3_event_corrected_covariates.tsv'
CONTRASTS = [
    ('older-minus-younger_behavior', [0, 0, 0, 0, 0, 0, -1, 1]),
    ('younger-minus-older_behavior', [0, 0, 0, 0, 0, 0, 1, -1]),
    ('younger-minus-older_at-mean-behavior', [0, 0, 0, 0, 1, -1, 0, 0]),
    ('older-minus-younger_at-mean-behavior', [0, 0, 0, 0, -1, 1, 0, 0]),
    ('younger_behavior-positive', [0, 0, 0, 0, 0, 0, 1, 0]),
    ('younger_behavior-negative', [0, 0, 0, 0, 0, 0, -1, 0]),
    ('older_behavior-positive', [0, 0, 0, 0, 0, 0, 0, 1]),
    ('older_behavior-negative', [0, 0, 0, 0, 0, 0, 0, -1]),
]


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def matrix(text):
    n = int(fsf_value(text, 'evs_real'))
    cells = parse_evs(text.splitlines())
    result = np.array([[cells[r, c] for c in range(1, n + 1)] for r in range(1, 48)])
    if len(cells) != 47 * n or not np.isfinite(result).all():
        raise ValueError('expected a complete finite 47-participant group matrix')
    return result


def members(text):
    inputs = parse_inputs(text.splitlines())
    if [r for r, _, _ in inputs] != list(range(1, 48)) or len({s for _, s, _ in inputs}) != 47:
        raise ValueError('expected 47 unique participants in contiguous rows')
    return inputs


def verified_source(report, relative):
    path = report / relative
    records = [r for r in read_rows(report / 'inventory.tsv') if r['artifact'] == relative]
    if len(records) != 1 or digest(path) != records[0]['sha256']:
        raise ValueError(f'collected source does not match its inventory: {path}')
    return path


def build_plan(repo, timings, work, standard):
    """Pure design construction; input image existence is checked separately."""
    template = repo / TEMPLATE
    table = repo / COVARIATES
    rows = read_rows(table)
    covars = {r['subjID']: r for r in rows}
    if len(rows) != 47 or len(covars) != 47:
        raise ValueError('behavior table must contain exactly 47 unique participants')
    provenance = {str(p): digest(p) for p in (template, table, Path(__file__).resolve())}
    plans = []
    for timing in timings:
        report = repo / REPORTS[timing]
        manifest = verified_source(report, 'revision_jobs.tsv')
        provenance[str(manifest)] = digest(manifest)
        old_jobs = read_rows(manifest)
        sources = {family: verified_source(report, f'{name}/design/design.fsf')
                   for family, name in FAMILIES.items()}
        provenance.update({str(p): digest(p) for p in sources.values()})
        texts = {f: p.read_text() for f, p in sources.items()}
        order = [s for _, s, _ in members(texts['nppi-ecn'])]
        if set(order) != set(covars):
            raise ValueError('behavior and imaging participant membership differ')
        base = matrix(texts['nppi-ecn'])[:, :6]
        if not (np.isin(base[:, 4:6], [0, 1]).all()
                and np.array_equal(base[:, 4] + base[:, 5], np.ones(47))
                and np.array_equal(base[:, 4:6].sum(axis=0), [25, 22])):
            raise ValueError('age coding differs from the 25 younger / 22 older contract')
        nuisance_by_family = {}
        for family, text in texts.items():
            if [s for _, s, _ in members(text)] != order:
                raise ValueError('family participant orders differ; do not align by position')
            existing = matrix(text)
            nuisance = existing[:, [2, 3, 4, 5, 0, 1]] if family == 'nppi-dmn' else existing[:, :6]
            # Retained DMN tSNR/FD differ from activation/ECN. Preserve this
            # explicitly; crossing a behavior measure is not authority to
            # replace the nuisance-data source at the same time.
            if not np.allclose(nuisance[:, [0, 3, 4, 5]], base[:, [0, 3, 4, 5]], atol=1e-8, rtol=0):
                raise ValueError(f'family sex/RT/age coding differs: {timing} {family}')
            nuisance_by_family[family] = nuisance
        expected_rt = np.array([float(covars[s]['corrected_mean_rt_z']) for s in order])
        if not np.allclose(base[:, 3], expected_rt, atol=1e-8, rtol=0):
            raise ValueError('retained group RT is not the corrected RT covariate')
        for metric, prefix in METRICS.items():
            behavior = np.array([[float(covars[s][prefix + '_' + age]) for age in ('young', 'old')]
                                 for s in order])
            design = np.column_stack([base, behavior])
            if not np.isfinite(design).all() or np.linalg.matrix_rank(design) != 8:
                raise ValueError('behavioral design is nonfinite or rank deficient')
            if not np.allclose(behavior * (1 - base[:, 4:6]), 0, atol=1e-10, rtol=0):
                raise ValueError('behavior assigned to the wrong age group')
            if abs(behavior.sum()) > 1e-8:
                raise ValueError('behavior difference must be centered across all 47 participants')
            calibration_family = 'nppi-ecn' if metric == 'choice-slope' else 'act'
            if not np.allclose(design, matrix(texts[calibration_family]), atol=1e-8, rtol=0):
                raise ValueError('canonical metric does not reproduce its retained group design')
            for family in FAMILIES:
                design = np.column_stack([nuisance_by_family[family], behavior])
                if np.linalg.matrix_rank(design) != 8:
                    raise ValueError(f'rank-deficient crossed model: {timing} {family} {metric}')
                name = f'{timing}_{family}_{metric}'
                output = work / 'outputs' / (name + '.gfeat')
                fsf = work / 'fsf' / (name + '.fsf')
                text = template.read_text()
                text = replace_fsf_value(text, 'outputdir', str(output), fmri=True)
                text = replace_fsf_value(text, 'regstandard', str(standard), fmri=True)
                paths = []
                for row, subject, path in members(texts[family]):
                    matches = [j for j in old_jobs if j['stage'] == 'l2' and j['model'] == family
                               and j['subject'] == subject]
                    if len(matches) != 1 or str(Path(matches[0]['output']) / 'cope7.feat/stats/cope1.nii.gz') != path:
                        raise ValueError(f'L3 input does not match retained L2 manifest: {family} {subject}')
                    paths.append(path)
                    text = replace_fsf_value(text, f'feat_files({row})', path, fmri=False)
                    for col in range(1, 9):
                        text = text.replace(f'EV_{row}_{col}\n', f'{design[row-1, col-1]:.15g}\n')
                for col, age in ((7, 'younger'), (8, 'older')):
                    text = replace_fsf_value(text, f'evtitle{col}', f'{age}_{metric}', fmri=True)
                for i, (label, weights) in enumerate(CONTRASTS, 1):
                    text = replace_fsf_value(text, f'conname_real.{i}', label, fmri=True)
                    actual = [float(fsf_value(text, f'con_real{i}.{j}')) for j in range(1, 9)]
                    if actual != weights:
                        raise ValueError('shared template contrast mismatch')
                if re.search(r'EV_\d+_\d+', text) or not np.allclose(matrix(text), design, atol=1e-12, rtol=0):
                    raise ValueError('incomplete or inaccurate template rendering')
                if any(fsf_value(text, k) != v for k, v in {
                    'evs_real': '8', 'ncon_real': '8', 'mixed_yn': '1',
                    'robust_yn': '0', 'z_thresh': '3.1', 'prob_thresh': '0.05',
                    'level2orth': '0', 'inputtype': '2'}.items()):
                    raise ValueError('group inference settings differ from the locked contract')
                job = dict(stage='l3', model=family, subject='', run=name, fsf=str(fsf),
                           output=str(output), inputs='|'.join(paths + [str(standard)]),
                           source_fsf=str(template), source_sha256=digest(template),
                           fsf_sha256='', n_evs=8, expected_zstats=8, group_contract='locked-source',
                           timing=timing, behavior=metric, source_report=str(report),
                           calibration=int(family == calibration_family),
                           nuisance_policy='retained-within-imaging-family',
                           tsnr_max_abs_vs_ecn=float(np.max(abs(design[:, 1] - base[:, 1]))),
                           fd_max_abs_vs_ecn=float(np.max(abs(design[:, 2] - base[:, 2]))))
                plans.append((job, text))
    return plans, provenance


def check_images(plans, standard):
    import nibabel as nib
    if not standard.is_file():
        raise FileNotFoundError(standard)
    checked, reference = {}, None
    for job, _ in plans:
        for name in job['inputs'].split('|')[:-1]:
            for path in (Path(name), Path(name).with_name('varcope1.nii.gz')):
                if str(path) in checked:
                    continue
                image = nib.load(path)
                if len(image.shape) != 3:
                    raise ValueError(f'expected a 3D participant COPE/VARCOPE: {path}')
                if reference is None:
                    reference = image
                if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
                    raise ValueError(f'group image geometry mismatch: {path}')
                data = np.asarray(image.dataobj)
                if not np.isfinite(data).all() or (path.name.startswith('varcope') and (data < 0).any()):
                    raise ValueError(f'nonfinite image or negative variance: {path}')
                checked[str(path)] = {'sha256': digest(path)}
    checked[str(standard)] = {'sha256': digest(standard)}
    return checked


def prepare(repo, timings, work, standard):
    plans, provenance = build_plan(repo, timings, work, standard)
    config = dict(timings=timings, standard=str(standard), sources=provenance,
                  work_root=str(work), jobs=len(plans))
    config_path = work / 'preparation_config.json'
    manifest = work / 'revision_jobs.tsv'
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            raise ValueError('preparation sources/config changed; use a new scratch root')
        expected_jobs = []
        for job, text in plans:
            path = Path(job['fsf'])
            if path.read_text() != text:
                raise ValueError(f'prepared design changed: {path}')
            job['fsf_sha256'] = digest(path)
            expected_jobs.append({k: str(v) for k, v in job.items()})
        if read_rows(manifest) != expected_jobs:
            raise ValueError('prepared manifest changed')
        verify_inputs(manifest)
        print(f'PASS: resume {len(plans)} unchanged L3 specifications', flush=True)
        return manifest
    # Validate every input before writing the scratch tree.
    image_provenance = check_images(plans, standard)
    protected = [repo, standard]
    protected.extend(Path(j['inputs'].split('|')[0]).parents[6] for j, _ in plans)
    safe_empty(work, protected)
    for job, text in plans:
        path = Path(job['fsf'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        job['fsf_sha256'] = digest(path)
    write_tsv(manifest, [j for j, _ in plans])
    image_provenance.update({p: {'sha256': h} for p, h in provenance.items()})
    (work / 'input_provenance.json').write_text(json.dumps(image_provenance, indent=2) + '\n')
    (work / 'analysis_plan.json').write_text(json.dumps(dict(
        estimand='similar minus dissimilar offer-slope COPE7',
        metrics=METRICS, contrasts=CONTRASTS, timing_models=timings,
        exclusions=0, participants=47, inference='FLAME 1+2, Z>3.1, cluster FWE .05',
        calibration_jobs=sum(j['calibration'] for j, _ in plans),
        interpretation='Separate exploratory models; acceptance-intercept is not a Gu parameter.',
        multiplicity='Spatial FWE only; not corrected across these model/contrast searches.',
        permutation_audit='Pending: TFCE, cluster t=2.6 and t=3.1; not run by this script.'
    ), indent=2) + '\n')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    print(f'PASS: prepared {len(plans)} L3 jobs; no L1/L2 jobs or images changed', flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timing', choices=list(REPORTS) + ['all'], default='partner-post')
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--standard', type=Path)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--collect-to', type=Path)
    a = p.parse_args()
    if a.jobs < 1 or a.jobs > 45:
        p.error('--jobs must be between 1 and 45')
    if a.collect_to and not a.execute:
        p.error('--collect-to requires --execute (completed jobs are resumed)')
    if not a.standard and not os.environ.get('FSLDIR'):
        p.error('set FSLDIR or specify --standard')
    standard = (a.standard or Path(os.environ['FSLDIR']) / 'data/standard/MNI152_T1_2mm_brain.nii.gz').resolve()
    timings = list(REPORTS) if a.timing == 'all' else [a.timing]
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    manifest = prepare(ROOT, timings, a.work_root.resolve(), standard)
    compile_designs(manifest, 'l3')
    status = run_jobs(manifest, 'l3', a.jobs, resume=True, dry_run=not a.execute)
    if status == 0 and a.execute:
        verify_stage_outputs(manifest, 'l3')
        if a.collect_to:
            collect(manifest, a.collect_to.resolve(), expected_jobs=6 * len(timings))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
