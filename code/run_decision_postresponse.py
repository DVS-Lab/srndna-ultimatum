#!/usr/bin/env python3
"""Prepare/audit a separate decision/post-response model; fitting is opt-in."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

from audit_l1_designs import write_tsv, read_vest_matrix
from collect_dmn_revision_checks import collect
from decision_postresponse_model import render_l1, validate_source, split_epochs
from make_ultimatum_3col import read_events
from phase_design_diagnostics import diagnose_design
from prepare_dmn_revision_checks import digest, safe_empty, group_job
from prepare_dmn_rt_sensitivity import baseline_copy, retained_inputs, resolve_input, parse_input_maps, decisions
from prepare_sub144_imaging_repair import fsf_value, replace_fsf_value, render_l2
from prepare_ultimatum_l3_repair import parse_inputs
from run_full_rt_correction import MODELS, NAMES, group_replacement
from run_dmn_revision_stage import compile_designs, verify_stage_outputs
from run_ultimatum_repair_jobs import run_jobs

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
                validate_source(text)
                split_epochs(read_events(events))
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
        text, required, stats = render_l1(source.read_text(),output,work/'EVfiles'/family/subject/run_id,
                                     read_events(events),standard,maps,resolutions,source)
        fsf.parent.mkdir(parents=True,exist_ok=True)
        fsf.write_text(text)
        required += baseline_inputs+[source,events,fsf]
        record(required)
        jobs.append(dict(stage='l1',model=family,subject=subject,run=run_id,fsf=str(fsf),
                         output=str(output)+'.feat',inputs='|'.join(map(str,required)),
                         source_fsf=str(source),source_sha256=digest(source),fsf_sha256=digest(fsf),
                         baseline_fsf=str(baseline),baseline_sha256=digest(baseline),
                         source_design=str(source.with_suffix('.mat')),source_design_sha256=digest(source.with_suffix('.mat')),
                         n_evs=stats['n_evs'],expected_copes=int(fsf_value(text,'ncon_real'))))
        l1_outputs[(family,subject,run_id)] = Path(str(output)+'.feat')
        l2_sources[(family,subject)] = l2_source
        counts.append(dict(family=family,subject=subject,run=run_id,events_sha256=digest(events),**stats))
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
    write_tsv(work/'phase_construction.tsv',counts)
    (work/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (work/'input_path_resolution.json').write_text(json.dumps(resolutions,indent=2)+'\n')
    (work/'analysis_plan.json').write_text(json.dumps(dict(
        model='decision-postresponse',expected_jobs=dict(l1=282,l2=141,l3=5),subjects=47,
        decision='offer onset to response; original offer amplitudes and orthogonalization',
        post_response='response to recorded display offset; one pooled unit-height EV',
        nppi='all eight psychological EVs interacted with the retained main network signal',
        removed='dedicated RT onset/height pair and their two PPI interactions',
        preserved='sample, contrasts, miss EV, network signals, confounds, smoothing, corrected group covariates',
        interpretation='changed estimand; post partner, offer and choice effects are not separately modeled',
        historical_outputs='preserved; all new derivatives are scratch-only'),indent=2)+'\n')

def audit_compiled(manifest, destination):
    """Supplementary IID diagnostics in the actual orthogonalized FEAT basis."""
    with manifest.open() as stream:
        jobs = [r for r in csv.DictReader(stream, delimiter='\t') if r['stage']=='l1']
    contrasts, pairs = [], []
    for row in jobs:
        fsf = Path(row['fsf'])
        matrix = read_vest_matrix(fsf.with_suffix('.mat'))
        con = read_vest_matrix(fsf.with_suffix('.con'))
        diag = diagnose_design(matrix,con,orthogonalized=True)
        identity = dict(model=row['model'],subject=row['subject'],run=row['run'])
        for result in diag['contrasts']:
            contrasts.append(dict(**identity,contrast=result['contrast'],estimable=int(result['estimable']),
                                  status=result['status'],iid_contrast_variance=result['iid_contrast_variance'],
                                  compiled_basis_variance_ratio=result['actual_basis_variance_ratio']))
        for a,b,name in [(1,8,'computer'),(3,8,'ingroup'),(5,8,'outgroup')]:
            indices=[(a,b,'activity')]
            if row['model']!='act':
                indices.append((a+9,b+9,'ppi'))
            for left,right,kind in indices:
                x,y=matrix[:,left-1],matrix[:,right-1]
                correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
                pairs.append(dict(**identity,partner=name,component=kind,decision_ev=left,
                                  post_ev=right,compiled_pair_correlation=correlation))
    destination.mkdir(parents=True,exist_ok=True)
    write_tsv(destination/'contrast_diagnostics.tsv',contrasts)
    write_tsv(destination/'phase_correlations.tsv',pairs)
    for filename in ('phase_construction.tsv','baseline_preflight.tsv','l1_preflight.tsv','analysis_plan.json'):
        shutil.copyfile(manifest.parent/filename,destination/filename)
    (destination/'diagnostic_scope.json').write_text(json.dumps(dict(
        basis='compiled FEAT, including original orthogonalization and all confounds',
        interpretation='IID conditional-basis variance ratios; not original-stimulus cVIF or FILM precision',
        decision='review correlations, contrast variance and rank checks before explicitly authorizing fitting',
        limitations=['Adjacent phases may be difficult to separate after HRF convolution.',
                     'One pooled post mean does not separately model partner, offer or choice post effects.',
                     'Statistical estimability does not establish physiological specificity.',
                     'Decision and full-display contrasts have different temporal definitions.']),indent=2)+'\n')
    print(f'DESIGN AUDIT: {destination}',flush=True)


def resolve_config(args):
    inherited = json.loads(args.reuse_input_config.read_text()) if args.reuse_input_config else {}
    fields = [('production_fsl_root','production','/ZPOOL/data/projects/srndna-ultimatum/derivatives/fsl'),
              ('repaired_fsl_root','repaired','/ZPOOL/data/scratch/srndna-ultimatum-sub144-repair-v2/derivatives/fsl'),
              ('standard_image','standard',str(Path(os.environ.get('FSLDIR','/usr/local/fsl'))/'data/standard/MNI152_T1_2mm_brain.nii.gz'))]
    resolved = []
    for attr,key,default in fields:
        supplied = getattr(args,attr)
        if supplied and key in inherited and supplied.resolve()!=Path(inherited[key]).resolve():
            raise ValueError(f'conflicting {attr} with --reuse-input-config')
        resolved.append(Path(supplied or inherited.get(key,default)).resolve())
    if args.input_map and inherited:
        raise ValueError('do not combine --input-map with --reuse-input-config')
    maps = parse_input_maps(args.input_map or [f'{old}={new}' for old,new in inherited.get('input_maps',[])])
    return (*resolved,maps)


def main(argv=None):
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-fsl-root',type=Path)
    parser.add_argument('--repaired-fsl-root',type=Path)
    parser.add_argument('--standard-image',type=Path)
    parser.add_argument('--input-map',action='append',default=[])
    parser.add_argument('--reuse-input-config',type=Path,help='Reuse exact roots and maps from a completed preparation_config.json')
    parser.add_argument('--work-root',type=Path,default=Path('/ZPOOL/data/scratch/srndna-ultimatum-decision-postresponse-v1'))
    parser.add_argument('--output-root',type=Path,default=repo/'results/decision_postresponse')
    parser.add_argument('--jobs',type=int,default=40)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--accept-design-diagnostics',action='store_true',
                        help='Explicitly accept the inspected phase-design diagnostics before fitting')
    args=parser.parse_args(argv)
    if not 1<=args.jobs<=45:
        parser.error('--jobs must be 1 through 45')
    if args.execute and not args.accept_design_diagnostics:
        parser.error('run the default design audit first, inspect it, then use --execute --accept-design-diagnostics')
    production,repaired,standard,maps=resolve_config(args)
    work=args.work_root.resolve()
    destination=args.output_root.resolve()
    # A run's report tree must not contain (or be inside) its scratch derivatives.
    if work==destination or work in destination.parents or destination in work.parents:
        raise ValueError('report and work roots must be separate')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    config=dict(production=str(production),repaired=str(repaired),standard=str(standard),
                input_maps=[[str(x),str(y)] for x,y in maps],
                code={str(p.relative_to(repo)):digest(p) for p in sorted((repo/'code').glob('*.py'))},
                templates={str(p.relative_to(repo)):digest(p) for p in sorted((repo/'templates/revision').glob('*.fsf'))})
    manifest=work/'revision_jobs.tsv'
    if manifest.exists():
        if json.loads((work/'preparation_config.json').read_text())!=config:
            raise ValueError('code/configuration changed; preserve this run and use a new --work-root')
    else:
        if args.execute:
            raise ValueError('prepare and inspect a dry-run model tree before execution')
        prepare(repo,production,repaired,work,standard,maps)
        (work/'preparation_config.json').write_text(json.dumps(config,indent=2)+'\n')
    version=Path(os.environ.get('FSLDIR','/usr/local/fsl'))/'etc/fslversion'
    software=dict(python=sys.version,platform=platform.platform(),fsl_dir=os.environ.get('FSLDIR',''),
                  fsl_version=version.read_text().strip() if version.is_file() else 'unavailable')
    if (work/'software.json').exists() and json.loads((work/'software.json').read_text())!=software:
        raise ValueError('execution environment changed; use a new work root')
    (work/'software.json').write_text(json.dumps(software,indent=2)+'\n')
    compile_designs(manifest,'l1')
    audit_compiled(manifest,destination/'design_audit')
    stages=('l1','l2','l3') if args.execute else ('l1',)
    for stage in stages:
        if stage!='l1':
            compile_designs(manifest,stage)
        status=run_jobs(manifest,stage,args.jobs if stage=='l1' else 2,resume=True,dry_run=not args.execute)
        if status:
            return status
        if args.execute:
            verify_stage_outputs(manifest,stage)
    if not args.execute:
        print('DRY RUN COMPLETE: no FEAT fits launched. Inspect design_audit before fitting. L2/L3 checks await new inputs.')
        return 0
    collect(manifest,destination/'imaging',expected_jobs=5)
    print(f'COMPLETE: inspect {destination}/imaging; no primary results or figures replaced.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
