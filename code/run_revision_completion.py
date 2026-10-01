#!/usr/bin/env python3
"""One entry point for the remaining Linux1 DMN checks; dry-run by default.

quick: saved-L1 estimability, corrected descriptive influence, two L3 checks.
rt: two DMN RT sensitivities (188 L1, 94 L2, 2 L3); no primary overwrite.
No upload, publication, Git commit, or participant exclusion is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import platform
from pathlib import Path

from prepare_dmn_revision_checks import prepare as prepare_group, digest
from prepare_dmn_rt_sensitivity import prepare as prepare_rt
from collect_dmn_revision_checks import collect


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('RUN: '+' '.join(map(str, command)), flush=True)
    # No shell pipe; the child's exit code is retained even when writing a log.
    with log.open('w') as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f'stopped: command exited {status}; see {log}')


def validate_saved_audit(repo):
    # Historical focal flags do not silently disappear; user must review those.
    path = repo/'results/reviewer/l1_estimability/summary.tsv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if len(rows) != 6:
        raise ValueError('incomplete saved-design audit')
    for row in rows:
        if int(row['runs_audited']) != 94 or int(row['focal_contrasts_checked']) != 282:
            raise ValueError('incomplete saved-design audit')
        if any(int(row[k]) for k in ('focal_nonestimable','focal_sensitivity_flags','runs_deficient_after_zero_removal')):
            raise ValueError('saved-design audit needs review before launching new fits')


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['quick','rt'], required=True)
    p.add_argument('--production-fsl-root', type=Path, default=Path('/ZPOOL/data/projects/srndna-ultimatum/derivatives/fsl'))
    p.add_argument('--repaired-fsl-root', type=Path, default=Path('/ZPOOL/data/scratch/srndna-ultimatum-sub144-repair-v2/derivatives/fsl'))
    p.add_argument('--work-root', type=Path, default=Path('/ZPOOL/data/scratch/srndna-ultimatum-final-review-v1'))
    p.add_argument('--standard-image', type=Path)
    p.add_argument('--jobs', type=int, default=40, help='L1 concurrency; L2/L3 capped at two')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.jobs < 1 or a.jobs > 45:
        p.error('--jobs must be between 1 and 45')
    if a.execute and a.phase=='quick':
        # Fail before running hours of models if the figure dependencies are absent.
        import nibabel, matplotlib, nilearn  # noqa: F401
    standard = a.standard_image or Path(os.environ.get('FSLDIR','/usr/local/fsl'))/'data/standard/MNI152_T1_2mm_brain.nii.gz'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    base, work = a.work_root.resolve(), (a.work_root/a.phase).resolve()
    logs = base/'logs'
    run([sys.executable, repo/'code/audit_l1_estimability.py', '--l1-root', a.production_fsl_root,
         '--repair-l1-root', a.repaired_fsl_root, '--output-dir', repo/'results/reviewer/l1_estimability'], logs/f'{a.phase}-estimability.log')
    validate_saved_audit(repo)
    run([sys.executable, repo/'code/audit_corrected_dmn_influence.py'], logs/f'{a.phase}-influence.log')
    manifest = work/'revision_jobs.tsv'
    config = dict(production=str(a.production_fsl_root.resolve()), repaired=str(a.repaired_fsl_root.resolve()),
                  standard=str(standard.resolve()), phase=a.phase,
                  preparer_sha256=digest(repo/('code/prepare_dmn_revision_checks.py' if a.phase=='quick' else 'code/prepare_dmn_rt_sensitivity.py')),
                  group_renderer_sha256=digest(repo/'code/prepare_dmn_revision_checks.py'))
    version_file = Path(os.environ.get('FSLDIR','/usr/local/fsl'))/'etc/fslversion'
    software = dict(python=sys.version, platform=platform.platform(),
                    fsl_version=version_file.read_text().strip() if version_file.is_file() else 'unavailable',
                    fsl_dir=os.environ.get('FSLDIR',''),
                    thread_limits={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    if manifest.exists():
        if json.loads((work/'preparation_config.json').read_text()) != config:
            raise ValueError('preparation configuration/code changed; use a new --work-root')
    else:
        if a.phase=='quick':
            prepare_group(repo, work, standard.resolve())
        else:
            prepare_rt(repo, a.production_fsl_root.resolve(), a.repaired_fsl_root.resolve(), work, standard.resolve())
        (work/'preparation_config.json').write_text(json.dumps(config, indent=2)+'\n')
    if (work/'software.json').exists() and json.loads((work/'software.json').read_text()) != software:
        raise ValueError('execution software/environment changed; use a new --work-root')
    (work/'software.json').write_text(json.dumps(software,indent=2)+'\n')
    stages = ('l3',) if a.phase=='quick' else ('l1','l2','l3') if a.execute else ('l1',)
    for stage in stages:
        command = [sys.executable, repo/'code/run_dmn_revision_stage.py', '--manifest', manifest,
                   '--stage', stage, '--jobs', str(a.jobs if stage=='l1' else 2)]
        if a.execute:
            command.append('--execute')
        run(command, logs/f'{a.phase}-{stage}.log')
    if not a.execute:
        print('DRY RUN COMPLETE: no FEAT fits launched. L2/L3 inputs in RT phase will exist only after preceding stages finish.')
        return 0
    destination = repo/'results/reviewer'/('dmn_revision_checks' if a.phase=='quick' else 'dmn_rt_sensitivity')
    collect(manifest, destination)
    if a.phase=='quick':
        run([sys.executable, repo/'code/plot_dmn_revision_simple_effects.py'], logs/'simple-effect-figure.log')
    print(f'BATCH COMPLETE: {a.phase}; inspect and commit compact outputs in {destination}. This is not publication approval.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
