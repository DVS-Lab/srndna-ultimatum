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


def compile_designs(manifest, stage):
    verify_inputs(manifest)
    with manifest.open() as stream:
        rows = [r for r in csv.DictReader(stream, delimiter='\t') if r['stage'] == stage]
    if not rows:
        raise ValueError(f'no {stage} jobs')
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
        if stage == 'l1' and con.shape[1] == 28 and matrix.shape[1] > 28:
            con = np.pad(con, ((0, 0), (0, matrix.shape[1]-28)))
        metrics, checks = diagnostics(matrix, con)
        relevant = checks if stage != 'l1' else [r for r in checks if r['contrast'] in (4, 6, 7)]
        if stage == 'l3':
            if not np.allclose(matrix, group_design(text), atol=1e-5, rtol=0):
                raise ValueError('compiled group matrix differs from corrected FSF')
            if not np.array_equal(con, np.array([v for _, v in CONTRASTS])):
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
