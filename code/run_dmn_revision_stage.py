#!/usr/bin/env python3
"""Guarded preparation checks and execution for the final DMN sensitivity jobs.

Default is dry-run. --execute is explicit. L1 designs must pass focal-contrast
estimability at both cutoffs before ANY L1 jobs are launched. Uses the established
FEAT runner for bounded concurrency and no-overwrite/resume behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from audit_l1_designs import read_vest_matrix, write_tsv
from audit_l1_estimability import diagnostics
from prepare_dmn_revision_checks import CONTRASTS, digest, group_design
from prepare_sub144_imaging_repair import fsf_value
from prepare_ultimatum_l3_repair import parse_evs
from run_ultimatum_repair_jobs import run_jobs


def verify_inputs(manifest):
    path = manifest.parent / 'input_provenance.json'
    if path.exists():
        for name, expected in json.loads(path.read_text()).items():
            source = Path(name)
            if not source.is_file():
                raise FileNotFoundError(source)
            current = {'sha256': digest(source)} if 'sha256' in expected else {
                'bytes': source.stat().st_size, 'mtime_ns': source.stat().st_mtime_ns}
            if current != expected:
                raise ValueError(f'prepared input changed; use a new scratch root: {source}')


def baseline_agreement(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return False, None
    difference = float(np.max(np.abs(reference-candidate)))
    return bool(np.allclose(reference, candidate, atol=1e-5, rtol=1e-6)), difference


def validate_baselines(rows, manifest):
    recovering = [row for row in rows if row.get('baseline_fsf')]
    if not recovering:
        return
    if len(recovering) != len(rows):
        raise ValueError('mixed baseline-recovery policy within the L1 batch')
    results, seen = [], {}
    for row in recovering:
        fsf, source = Path(row['baseline_fsf']), Path(row['source_design'])
        if digest(fsf) != row['baseline_sha256'] or digest(source) != row['source_design_sha256']:
            raise ValueError(f'baseline/reference design changed: {fsf}')
        identity = (str(source),row['baseline_sha256'],row['source_design_sha256'])
        if fsf in seen:
            if seen[fsf] != identity:
                raise ValueError('conflicting baseline manifest entries')
            continue
        seen[fsf] = identity
        result = subprocess.run(['feat_model',str(fsf.with_suffix('')),fsf_value(fsf.read_text(),'confoundev_files(1)')],
                                capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(f'baseline feat_model failed: {fsf}\n{result.stdout}\n{result.stderr}')
        reference = read_vest_matrix(source)
        compiled = read_vest_matrix(fsf.with_suffix('.mat'))
        passed, maximum = baseline_agreement(reference, compiled)
        results.append(dict(model=row.get('model',''),subject=row['subject'],run=row['run'],baseline_fsf=str(fsf),
                            source_design=str(source),source_design_sha256=digest(source),
                            compiled_sha256=digest(fsf.with_suffix('.mat')),reference_rows=reference.shape[0],
                            reference_columns=reference.shape[1],compiled_rows=compiled.shape[0],
                            compiled_columns=compiled.shape[1],maximum_abs_difference=maximum,
                            atol=1e-5,rtol=1e-6,passed=int(passed)))
        print(f"{'PASS' if passed else 'REVIEW'}: retained-input baseline {row.get('model','')} {row['subject']} run-{row['run']}",flush=True)
    write_tsv(manifest.parent/'baseline_preflight.tsv',results)
    if not all(row['passed'] for row in results):
        raise ValueError('recovered inputs do not reproduce every retained design within tolerance; NO sensitivity fits launched; inspect baseline_preflight.tsv')


def compile_designs(manifest, stage):
    verify_inputs(manifest)
    with manifest.open() as stream:
        rows = [r for r in csv.DictReader(stream, delimiter='\t') if r['stage'] == stage]
    if not rows:
        raise ValueError(f'no {stage} jobs')
    if stage == 'l1':
        validate_baselines(rows, manifest)
    results = []
    for row in rows:
        fsf = Path(row['fsf'])
        source = Path(row['source_fsf'])
        if digest(fsf) != row['fsf_sha256'] or digest(source) != row['source_sha256']:
            raise ValueError(f'prepared/source FSF changed: {fsf}')
        for value in row['inputs'].split('|'):
            if not Path(value).exists():
                raise FileNotFoundError(f'missing job input: {value}')
            if stage == 'l3' and value.endswith('/stats/cope1.nii.gz'):
                variance = Path(value).with_name('varcope1.nii.gz')
                if not variance.is_file():
                    raise FileNotFoundError(f'missing group input VARCOPE: {variance}')
        text = fsf.read_text()
        command = ['feat_model', str(fsf.with_suffix(''))]
        if stage == 'l1':
            command.append(fsf_value(text, 'confoundev_files(1)'))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'feat_model failed for {fsf}: {result.stdout}\n{result.stderr}')
        matrix = read_vest_matrix(fsf.with_suffix('.mat'))
        con = read_vest_matrix(fsf.with_suffix('.con'))
        task_columns = int(row.get('n_evs') or 28)
        if stage == 'l1' and con.shape[1] == task_columns and matrix.shape[1] > task_columns:
            con = np.pad(con, ((0, 0), (0, matrix.shape[1]-task_columns)))
        metrics, checks = diagnostics(matrix, con)
        relevant = checks if stage != 'l1' else [r for r in checks if r['contrast'] in (4, 6, 7)]
        if stage == 'l3':
            if row.get('group_contract') == 'locked-source':
                n = int(fsf_value(text, 'evs_real'))
                evs = parse_evs(text.splitlines())
                expected_matrix = np.array([[evs[(r,c)] for c in range(1,n+1)] for r in range(1,48)])
                expected_con = np.array([[float(fsf_value(text, f'con_real{i}.{j}'))
                                         for j in range(1,n+1)]
                                        for i in range(1,int(row['expected_zstats'])+1)])
            else:
                expected_matrix = group_design(text)
                expected_con = np.array([v for _, v in CONTRASTS])
            if matrix.shape != expected_matrix.shape or not np.allclose(matrix, expected_matrix, atol=1e-5, rtol=0):
                raise ValueError('compiled group matrix differs from corrected FSF')
            if not np.array_equal(con, expected_con):
                raise ValueError('compiled group contrasts differ from the locked plan')
        expected = 3 if stage == 'l1' else int(row['expected_zstats']) if stage == 'l3' else len(checks)
        passed = len(relevant) == expected and all(r['estimable'] and r['sensitivity_estimable'] for r in relevant)
        passed = passed and metrics['remaining_deficiency_after_zero_removal'] == 0
        results.append(dict(stage=stage, model=row['model'], subject=row.get('subject', ''), run=row['run'],
                            compiled_design_sha256=digest(fsf.with_suffix('.mat')),
                            ncolumns=metrics['ncolumns'], rank=metrics['scaled_rank'],
                            zero_columns='|'.join(map(str, metrics['zero_columns'])),
                            residual_deficiency=metrics['remaining_deficiency_after_zero_removal'],
                            condition_nonzero_columns=metrics['condition_nonzero_columns'],
                            focal_checked=len(relevant), passed=int(passed)))
        print(f"{'PASS' if passed else 'REVIEW'}: {stage} {row['model']} {row.get('subject', '')} {row['run']}", flush=True)
    write_tsv(manifest.parent / f'{stage}_preflight.tsv', results)
    if not all(r['passed'] for r in results):
        raise ValueError('design preflight failed; no jobs have been launched; inspect *_preflight.tsv')


def verify_stage_outputs(manifest, stage):
    with manifest.open() as stream:
        rows = [r for r in csv.DictReader(stream, delimiter='\t') if r['stage']==stage]
    for row in rows:
        root = Path(row['output'])
        count = int(row['expected_zstats'] if stage=='l3' else row['expected_copes'])
        for i in range(1,count+1):
            stats = root/'stats' if stage=='l1' else root/f'cope{i}.feat/stats' if stage=='l2' else root/'cope1.feat/stats'
            number = 1 if stage=='l2' else i
            for stem in ('cope','varcope','zstat'):
                path = stats/f'{stem}{number}.nii.gz'
                if not path.is_file() or path.stat().st_size==0:
                    raise FileNotFoundError(f'incomplete stage output (never overwrite automatically): {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--stage', choices=['l1', 'l2', 'l3'], required=True)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    compile_designs(a.manifest, a.stage)
    status = run_jobs(a.manifest, a.stage, a.jobs, resume=True, dry_run=not a.execute)
    if status == 0 and a.execute:
        verify_stage_outputs(a.manifest, a.stage)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
