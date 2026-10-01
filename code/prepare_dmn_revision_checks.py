#!/usr/bin/env python3
"""Prepare scratch-only DMN simple-effect and full-sample robustness L3 models."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from audit_l1_designs import write_tsv
from prepare_dmn_condition_bar_models import parse_inputs, SOURCE_FSF_RELATIVE
from prepare_sub144_imaging_repair import fsf_value, replace_fsf_value, replace_fsf_number
from prepare_ultimatum_l3_repair import parse_evs

CONTRASTS = (
    ('younger-positive', (1, 0, 0, 0, 0, 0)),
    ('older-positive', (0, 1, 0, 0, 0, 0)),
    ('younger-minus-older', (1, -1, 0, 0, 0, 0)),
    ('older-minus-younger', (-1, 1, 0, 0, 0, 0)),
    ('younger-negative', (-1, 0, 0, 0, 0, 0)),
    ('older-negative', (0, -1, 0, 0, 0, 0)),
)

REFERENCE_NAMES = {
    0: 'L3_task-ultimatum_type-nppi-dmn_age_simple-effects.fsf',
    1: 'L3_task-ultimatum_type-nppi-dmn_age_outlier-deweighted.fsf',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_design(text):
    for key, value in (('evs_real', '6'), ('evs_orig', '6'), ('npts', '47'), ('mixed_yn', '1')):
        if fsf_value(text, key) != value:
            raise ValueError(f'unexpected corrected group setting: {key}')
    for i, title in enumerate(('younger', 'older', 'ismale', 'tsnr', 'fd_mean', 'RT'), 1):
        if fsf_value(text, f'evtitle{i}') != title:
            raise ValueError(f'unexpected EV {i}')
    values = parse_evs(text.splitlines())
    matrix = np.array([[values[(r, c)] for c in range(1, 7)] for r in range(1, 48)])
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 6:
        raise ValueError('corrected group design is nonfinite or rank deficient')
    return matrix


def render_group(text, output, standard, robust=0, replacements=None):
    before = group_design(text)
    inputs = parse_inputs(text.splitlines())
    if sum(s == 'sub-144' and 'sub144-repair-v2' in p for _, s, p in inputs) != 1:
        raise ValueError('source must contain the corrected sub-144 input')
    text = replace_fsf_value(text, 'outputdir', str(output), fmri=True)
    text = replace_fsf_value(text, 'regstandard', str(standard), fmri=True)
    for key, value in [('ncon_orig', 6), ('ncon_real', 6), ('robust_yn', robust), ('poststats_yn', 1)]:
        text = replace_fsf_number(text, key, value)
    # Rebuild all contrast and contrast-mask fields, not just ncon_real.
    pattern = r'^set fmri\((?:conname_(?:real|orig)\.|conpic_(?:real|orig)\.|con_(?:real|orig)\d+\.|conmask\d+_\d+)'
    lines = [line for line in text.splitlines() if not re.match(pattern, line)]
    for kind in ('real', 'orig'):
        for i, (name, vector) in enumerate(CONTRASTS, 1):
            lines += [f'set fmri(conpic_{kind}.{i}) 1', f'set fmri(conname_{kind}.{i}) "{name}"']
            lines += [f'set fmri(con_{kind}{i}.{j}) {value}' for j, value in enumerate(vector, 1)]
    lines += [f'set fmri(conmask{i}_{j}) 0' for i in range(1, 7) for j in range(1, 7)]
    text = '\n'.join(lines) + '\n'
    if replacements is not None:
        if set(replacements) != {s for _, s, _ in inputs}:
            raise ValueError('replacement L2 membership differs from corrected design')
        for i, subject, _ in inputs:
            text = replace_fsf_value(text, f'feat_files({i})', str(replacements[subject]), fmri=False)
    if not np.array_equal(before, group_design(text)):
        raise ValueError('group covariates changed')
    return text


def safe_empty(root, protected):
    root = root.resolve()
    for path in protected:
        path = path.resolve()
        if root == path or path in root.parents or root in path.parents:
            raise ValueError(f'work root overlaps protected path: {path}')
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'work root is not empty: {root}')
    root.mkdir(parents=True, exist_ok=True)


def group_job(repo, work, name, standard, robust=0, replacements=None):
    source = repo / SOURCE_FSF_RELATIVE
    reference = repo / 'templates/revision' / REFERENCE_NAMES[robust]
    expected = render_group(source.read_text(), Path('OUTPUTDIR'), Path('STANDARD_IMAGE'), robust)
    if not reference.is_file() or reference.read_text() != expected:
        raise ValueError(f'reference template missing or out of sync with renderer: {reference}')
    output = work / 'outputs' / name
    fsf = work / 'fsf' / f'{name}.fsf'
    text = render_group(source.read_text(), output, standard, robust, replacements)
    fsf.parent.mkdir(parents=True, exist_ok=True)
    fsf.write_text(text)
    inputs = [p for _, _, p in parse_inputs(text.splitlines())]
    return dict(stage='l3', model='dmn', run=name, subject='', fsf=str(fsf),
                output=str(output)+'.gfeat', inputs='|'.join(inputs+[str(standard)]),
                source_fsf=str(source), source_sha256=digest(source), fsf_sha256=digest(fsf),
                n_evs=6, design_rank=6, expected_zstats=6, expected_copes='',
                robust_yn=robust)


def write_reference_templates(repo):
    """Mechanical reference export; placeholders must be rendered before FEAT."""
    text = (repo / SOURCE_FSF_RELATIVE).read_text()
    for robust, name in REFERENCE_NAMES.items():
        path = repo / 'templates/revision' / name
        path.write_text(render_group(text, Path('OUTPUTDIR'), Path('STANDARD_IMAGE'), robust))


def prepare(repo, work, standard):
    if not standard.is_file():
        raise FileNotFoundError(standard)
    safe_empty(work, [repo])
    rows = [group_job(repo, work, 'simple-effects', standard),
            group_job(repo, work, 'outlier-deweighted', standard, robust=1)]
    write_tsv(work / 'revision_jobs.tsv', rows)
    (work / 'analysis_plan.json').write_text(json.dumps({
        'contrasts': CONTRASTS, 'N': 47, 'exclusions': [], 'mixed_yn': 1,
        'primary_analysis_unchanged': True,
        'interpretation': 'Additional contrasts and full-sample outlier deweighting sensitivity; selected before inspecting these results.'
    }, indent=2) + '\n')
    print(f'PASS: two scratch L3 jobs (6 contrasts each): {work / "revision_jobs.tsv"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--standard-image', type=Path, required=True)
    a = p.parse_args()
    prepare(a.repository.resolve(), a.work_root.resolve(), a.standard_image.resolve())


if __name__ == '__main__':
    main()
