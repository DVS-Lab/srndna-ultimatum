#!/usr/bin/env python3
"""Rebuild all three legacy imaging families with complete RT coverage on Linux1.

Dry-run by default. Full display durations, original contrasts, network signals,
confounds and smoothing are retained. No response-duration experiment, uploads,
participant exclusions or overwrites of historical derivatives are performed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np

from audit_l1_designs import write_tsv
from collect_dmn_revision_checks import collect
from make_ultimatum_3col import read_events
from prepare_dmn_revision_checks import digest, safe_empty, group_job
from prepare_dmn_rt_sensitivity import (baseline_copy, retained_inputs, resolve_input,
                                        parse_input_maps, render_l1, decisions)
from prepare_sub144_imaging_repair import fsf_value, replace_fsf_value, render_l2
from prepare_ultimatum_l3_repair import parse_inputs, parse_evs
from run_revision_completion import run


MODELS = {
    'act': 'L3_task-ultimatum_type-act_norm_reported-covariates-corrected.fsf',
    'nppi-dmn': 'L3_task-ultimatum_type-nppi-dmn_age_reported-covariates-corrected.fsf',
    'nppi-ecn': 'L3_task-ultimatum_type-nppi-ecn_sensitivity_reported-covariates-corrected.fsf',
}
NAMES = {'act': 'activation-norm', 'nppi-dmn': 'dmn-age', 'nppi-ecn': 'ecn-sensitivity'}


def group_replacement(text, output, standard, l2_outputs):
    """Change paths only; preserve every numerical group EV/contrast/setting."""
    inputs = parse_inputs(text.splitlines())
    if len(inputs) != 47 or len({s for _,s,_ in inputs}) != 47:
        raise ValueError('group must contain 47 unique participants')
    if set(l2_outputs) != {s for _,s,_ in inputs}:
        raise ValueError('group replacement membership mismatch')
    before = parse_evs(text.splitlines())
    text = replace_fsf_value(text, 'outputdir', str(output), fmri=True)
    text = replace_fsf_value(text, 'regstandard', str(standard), fmri=True)
    for i, subject, path in inputs:
        match = re.search(r'/(cope\d+\.feat/stats/cope1\.nii\.gz)$', path)
        if not match:
            raise ValueError(f'unexpected source COPE path: {path}')
        text = replace_fsf_value(text, f'feat_files({i})', str(l2_outputs[subject]/match[1]), fmri=False)
    if parse_evs(text.splitlines()) != before:
        raise ValueError('group covariates changed')
    n = int(fsf_value(text, 'evs_real'))
    matrix = np.array([[before[(r,c)] for c in range(1,n+1)] for r in range(1,48)])
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != n:
        raise ValueError('group design is nonfinite/rank deficient')
    return text


def prepare(repo, production, repaired, work, standard, maps):
    if not standard.is_file():
        raise FileNotFoundError(standard)
    records, groups = [], {}
    # Resolve ALL runs before creating a model tree. No guessed substitute inputs.
    for family, template in MODELS.items():
        source_group = repo/'templates/revision'/template
        members = parse_inputs(source_group.read_text().splitlines())
        if len(members) != 47 or len({s for _,s,_ in members}) != 47:
            raise ValueError('incomplete group membership')
        groups[family] = source_group
        for _, subject, _ in members:
            root = repaired if subject == 'sub-144' else production
            l2_source = root/subject/f'L2_task-ultimatum_model-02_type-{family}_sm-6.gfeat/design.fsf'
            if not l2_source.is_file():
                raise FileNotFoundError(l2_source)
            for run_id in ('01','02'):
                source = root/subject/f'L1_task-ultimatum_model-02_type-{family}_run-{run_id}_sm-6.feat/design.fsf'
                text = source.read_text()
                nevs = 9 if family == 'act' else 28
                if fsf_value(text,'evs_orig') != str(nevs) or fsf_value(text,'evs_real') != str(nevs):
                    raise ValueError(f'unexpected design size: {source}')
                if not source.with_suffix('.mat').is_file():
                    raise FileNotFoundError(source.with_suffix('.mat'))
                events = repo/f'source_data/bids/{subject}/func/{subject}_task-ultimatum_run-{run_id}_events.tsv'
                decisions(read_events(events))
                for role in retained_inputs(text, include_rt=True):
                    resolve_input(fsf_value(text,role) or '',standard,maps,source,role)
                records.append((family,subject,run_id,source,l2_source,events,nevs))
    if len(records) != 282:
        raise ValueError('expected 282 first-level jobs')
    safe_empty(work,[repo,production,repaired,*[new for _,new in maps]])
    jobs, counts, provenance, resolutions = [], [], {}, {}
    l1_outputs, l2_sources, l2_outputs = {}, {}, {f:{} for f in MODELS}
    def record(paths):
        for path in paths:
            provenance[str(path)] = ({'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
                                      if path.name.endswith('.nii.gz') else {'sha256':digest(path)})
    for family,subject,run_id,source,l2_source,events,nevs in records:
        name = f'{family}_{subject}_run-{run_id}'
        baseline = work/'baseline'/f'{name}.fsf'
        baseline_inputs = baseline_copy(source,baseline,standard,maps,resolutions)
        output = work/'derivatives'/family/subject/f'L1_run-{run_id}'
        fsf = work/'fsf'/f'{name}.fsf'
        text, required, n = render_l1(source.read_text(),output,work/'EVfiles'/family/subject/run_id,
                                     read_events(events),'all-trial-rt',standard,maps,resolutions,source)
        fsf.parent.mkdir(parents=True,exist_ok=True)
        fsf.write_text(text)
        required += baseline_inputs+[source,events,fsf]
        record(required)
        jobs.append(dict(stage='l1',model=family,subject=subject,run=run_id,fsf=str(fsf),
                         output=str(output)+'.feat',inputs='|'.join(map(str,required)),
                         source_fsf=str(source),source_sha256=digest(source),fsf_sha256=digest(fsf),
                         baseline_fsf=str(baseline),baseline_sha256=digest(baseline),
                         source_design=str(source.with_suffix('.mat')),source_design_sha256=digest(source.with_suffix('.mat')),
                         n_evs=nevs,expected_copes=int(fsf_value(text,'ncon_real'))))
        l1_outputs[(family,subject,run_id)] = Path(str(output)+'.feat')
        l2_sources[(family,subject)] = l2_source
        counts.append(dict(family=family,subject=subject,run=run_id,responded_trials=n,rt_rows=n,
                           events_sha256=digest(events),rt_policy='all_responded_height',duration_policy='retained_display_epoch'))
    for (family,subject),source in l2_sources.items():
        output = work/'derivatives'/family/subject/'L2'
        fsf = work/'fsf'/f'{family}_{subject}_L2.fsf'
        inputs = [l1_outputs[(family,subject,r)] for r in ('01','02')]
        render_l2(source,fsf,output,*inputs)
        fsf.write_text(replace_fsf_value(fsf.read_text(),'regstandard',str(standard),fmri=True))
        record([source,fsf])
        jobs.append(dict(stage='l2',model=family,subject=subject,run='combined',fsf=str(fsf),
                         output=str(output)+'.gfeat',inputs='|'.join(map(str,inputs)),
                         source_fsf=str(source),source_sha256=digest(source),fsf_sha256=digest(fsf),
                         expected_copes=10 if family=='act' else 11))
        l2_outputs[family][subject] = Path(str(output)+'.gfeat')
    for family,source in groups.items():
        name = NAMES[family]
        output, fsf = work/'outputs'/name, work/'fsf'/f'{name}.fsf'
        text = group_replacement(source.read_text(),output,standard,l2_outputs[family])
        fsf.write_text(text)
        record([source,fsf])
        inputs = [p for _,_,p in parse_inputs(text.splitlines())]
        jobs.append(dict(stage='l3',model=family,subject='',run=name,fsf=str(fsf),
                         output=str(output)+'.gfeat',inputs='|'.join(inputs+[str(standard)]),
                         source_fsf=str(source),source_sha256=digest(source),fsf_sha256=digest(fsf),
                         n_evs=int(fsf_value(text,'evs_real')),expected_zstats=int(fsf_value(text,'ncon_real')),
                         group_contract='locked-source'))
    replacements = {s:p/'cope7.feat/stats/cope1.nii.gz' for s,p in l2_outputs['nppi-dmn'].items()}
    for robust,name in [(0,'dmn-simple-effects'),(1,'dmn-outlier-deweighted')]:
        job = group_job(repo,work,name,standard,robust,replacements)
        job['group_contract']='locked-source'
        jobs.append(job)
        record([Path(job['source_fsf']),Path(job['fsf'])])
    fields = list(dict.fromkeys(k for row in jobs for k in row))
    write_tsv(work/'revision_jobs.tsv',[{k:r.get(k,'') for k in fields} for r in jobs])
    write_tsv(work/'rt_construction.tsv',counts)
    (work/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (work/'input_path_resolution.json').write_text(json.dumps(resolutions,indent=2)+'\n')
    (work/'analysis_plan.json').write_text(json.dumps(dict(
        purpose='Primary RT-coverage correction; not a response-duration sensitivity',
        expected_jobs=dict(l1=282,l2=141,l3=5),subjects=47,
        rt_policy='all responded substantive rows; zero-duration impulses, heights 1 and raw RT',
        preserved='task durations/amplitudes; contrasts; network signals; confounds; smoothing; corrected group covariates',
        historical_outputs='preserved, not overwritten',
        unaffected='Behavioral analyses and the already all-trial-RT fairness-main/social-computer activation pipeline',
        publication_status='pending result inspection and new figure/source-data integration'),indent=2)+'\n')


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-fsl-root',type=Path,default=Path('/ZPOOL/data/projects/srndna-ultimatum/derivatives/fsl'))
    parser.add_argument('--repaired-fsl-root',type=Path,default=Path('/ZPOOL/data/scratch/srndna-ultimatum-sub144-repair-v2/derivatives/fsl'))
    parser.add_argument('--work-root',type=Path,default=Path('/ZPOOL/data/scratch/srndna-ultimatum-rt-correction-v1'))
    parser.add_argument('--standard-image',type=Path)
    parser.add_argument('--input-map',action='append',default=[],metavar='OLD=NEW')
    parser.add_argument('--jobs',type=int,default=40)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output-root',type=Path)
    a=parser.parse_args()
    if not 1 <= a.jobs <= 45:
        parser.error('--jobs must be 1 through 45')
    standard=(a.standard_image or Path(os.environ.get('FSLDIR','/usr/local/fsl'))/'data/standard/MNI152_T1_2mm_brain.nii.gz').resolve()
    maps=parse_input_maps(a.input_map)
    work=a.work_root.resolve()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    # Lock all Python workflow modules, plus templates, across dry-run/resume.
    config=dict(production=str(a.production_fsl_root.resolve()),repaired=str(a.repaired_fsl_root.resolve()),
                standard=str(standard),input_maps=[[str(x),str(y)] for x,y in maps],
                code={str(p.relative_to(repo)):digest(p) for p in sorted((repo/'code').glob('*.py'))},
                templates={str(p.relative_to(repo)):digest(p) for p in sorted((repo/'templates/revision').glob('*.fsf'))})
    manifest=work/'revision_jobs.tsv'
    if manifest.exists():
        if json.loads((work/'preparation_config.json').read_text()) != config:
            raise ValueError('code/configuration changed; preserve this run and use a new --work-root')
    else:
        prepare(repo,a.production_fsl_root.resolve(),a.repaired_fsl_root.resolve(),work,standard,maps)
        (work/'preparation_config.json').write_text(json.dumps(config,indent=2)+'\n')
    version=Path(os.environ.get('FSLDIR','/usr/local/fsl'))/'etc/fslversion'
    software=dict(python=sys.version,platform=platform.platform(),fsl_dir=os.environ.get('FSLDIR',''),
                  fsl_version=version.read_text().strip() if version.is_file() else 'unavailable')
    if (work/'software.json').exists() and json.loads((work/'software.json').read_text()) != software:
        raise ValueError('execution environment changed; use a new work root')
    (work/'software.json').write_text(json.dumps(software,indent=2)+'\n')
    for stage in (('l1','l2','l3') if a.execute else ('l1',)):
        command=[sys.executable,repo/'code/run_dmn_revision_stage.py','--manifest',manifest,
                 '--stage',stage,'--jobs',str(a.jobs if stage=='l1' else 2)]
        if a.execute:
            command.append('--execute')
        run(command,work/'logs'/f'{stage}.log')
    if not a.execute:
        print('DRY RUN COMPLETE: all 282 baseline/new L1 designs checked; no FEAT fits launched. L2/L3 checks occur after their inputs exist.')
        return 0
    destination=a.output_root or repo/'results/reviewer/rt_coverage_corrected'
    collect(manifest,destination,expected_jobs=5)
    print(f'COMPLETE: inspect {destination}/summary.tsv before manuscript/figure updates. Historical results remain untouched.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
