#!/usr/bin/env python3
"""Rerun retained DMN L3 models with verified common tSNR/FD covariates.

Only those two columns change. Historical fits, L2 inputs, contrasts, RT,
behavioral covariates, robust settings and age coding are preserved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

import run_permutation_audit as pa
from collect_dmn_revision_checks import collect
from prepare_dmn_revision_checks import safe_empty
from prepare_sub144_imaging_repair import fsf_value, replace_fsf_value
from prepare_ultimatum_l3_repair import parse_inputs, parse_evs
from run_crossed_behavior_l3 import check_images
from run_dmn_revision_stage import compile_designs, verify_inputs, verify_stage_outputs
from run_ultimatum_repair_jobs import run_jobs

ROOT=Path(__file__).resolve().parents[1]
POLICY='verified-participant-table-tsnr-fd-only'


def matrix(text):
    n=int(fsf_value(text,'evs_real'))
    cells=parse_evs(text.splitlines())
    return np.array([[cells[r,c] for c in range(1,n+1)] for r in range(1,48)])


def render(text, sample, output, standard):
    """Align by subject ID; prove no other numerical or FSF settings changed."""
    inputs=parse_inputs(text.splitlines()); subjects=[s for _,s,_ in inputs]
    if len(subjects)!=47 or len(set(subjects))!=47 or set(subjects)!=set(sample):
        raise ValueError('expected exactly the locked 47 participants')
    before=matrix(text); expected=before.copy(); changes=[]; allowed=[]
    titles=[fsf_value(text,f'evtitle{i}') for i in range(1,before.shape[1]+1)]
    for metric in ('tsnr','fd_mean'):
        if titles.count(metric)!=1: raise ValueError(f'expected one {metric} EV')
        col=titles.index(metric)+1
        for row,subject,_ in inputs:
            value=float(sample[subject][metric])
            if not np.isfinite(value): raise ValueError('nonfinite standardized covariate')
            expected[row-1,col-1]=value
            key=f'evg{row}.{col}'; allowed.append(key)
            pattern=rf'^set fmri\({re.escape(key)}\) .*$'
            text,count=re.subn(pattern,f'set fmri({key}) {value:.15g}',text,flags=re.MULTILINE)
            if count!=1: raise ValueError(f'expected one {key}')
            changes.append(dict(subject=subject,metric=metric,column=col,
                                historical=before[row-1,col-1],standardized=value,
                                difference=value-before[row-1,col-1]))
    if not np.isfinite(expected).all() or np.linalg.matrix_rank(expected)!=expected.shape[1]:
        raise ValueError('standardized design is rank deficient or nonfinite')
    text=replace_fsf_value(text,'outputdir',str(output),fmri=True)
    text=replace_fsf_value(text,'regstandard',str(standard),fmri=True)
    if not np.allclose(matrix(text),expected,atol=1e-12,rtol=0):
        raise ValueError('rendered covariates differ from plan')
    return text,changes,allowed


def unchanged_settings(before,after,allowed):
    ignore=re.compile(r'^set fmri\(('+'|'.join(re.escape(x) for x in allowed+['outputdir','regstandard'])+r')\) ')
    if [x for x in before.splitlines() if not ignore.match(x)] != [x for x in after.splitlines() if not ignore.match(x)]:
        raise ValueError('unauthorized FSF change beyond tSNR/FD and destination paths')


def build_plan(repo,work,standard):
    models,sample,table=pa.catalog(repo)
    selected=[m for m in models if 'dmn' in m['name']]
    if len(selected)!=15: raise ValueError('expected 15 retained DMN models')
    # The prior measurement audit must refer to the exact covariate table used.
    audit=repo/'results/permutation_followup/v1/covariate_audit'
    inventory=pa.read_rows(audit/'source_inventory.tsv')
    if not any(r['path'].endswith('/srndna-ultimatum/behavioral_analyses/data/participant_L3_47.csv')
               and r['sha256']==pa.digest(table) for r in inventory):
        raise ValueError('participant table differs from verified covariate audit')
    matches=pa.read_rows(audit/'candidate_matches.tsv')
    for family in ('act','ecn'):
        for metric in ('tsnr','fd_mean'):
            if not any('direct MRIQC JSON aggregation' in r['source'] and r['target_family']==family
                       and r['metric']==metric and r['policy']=='ultimatum_mean_all_runs'
                       and r['full_positive_affine_match']=='1' and r['n_overlap']=='47' for r in matches):
                raise ValueError('missing independently reconstructed quality covariate evidence')
    sources={str(p):pa.digest(p) for p in (table,audit/'source_inventory.tsv',audit/'candidate_matches.tsv')}
    for name in ('run_dmn_covariate_standardization.py','run_permutation_audit.py',
                 'run_dmn_revision_stage.py','run_ultimatum_repair_jobs.py',
                 'collect_dmn_revision_checks.py','run_crossed_behavior_l3.py',
                 'prepare_sub144_imaging_repair.py','prepare_ultimatum_l3_repair.py',
                 'prepare_dmn_revision_checks.py','audit_l1_designs.py','audit_l1_estimability.py'):
        p=repo/'code'/name; sources[str(p)]=pa.digest(p)
    plans=[]; differences=[]
    # Verify the same table still reproduces the activation/ECN reference designs.
    for m in models:
        if m['name'].endswith(('__activation-norm','__ecn-sensitivity')):
            for metric in ('tsnr','fd_mean'):
                # FEAT's compiled VEST matrix rounds the full-precision FSF.
                if not np.allclose(m['nuisance'][metric],[float(sample[s][metric]) for s in m['subjects']],atol=1e-5,rtol=0):
                    raise ValueError('activation/ECN reference no longer matches verified table')
            sources.update(m['source_hashes'])
    for m in selected:
        source=next(Path(p) for p in m['source_hashes'] if p.endswith('/design.fsf'))
        original=source.read_text(); name=m['name']
        fsf=work/'fsf'/f'{name}.fsf'; output=work/'outputs'/f'{name}.gfeat'
        text,changes,allowed=render(original,sample,output,standard)
        unchanged_settings(original,text,allowed)
        sources.update(m['source_hashes'])
        images=[p for _,_,p in parse_inputs(original.splitlines())]
        job=dict(stage='l3',model='nppi-dmn',subject='',run=name,fsf=str(fsf),output=str(output),
                 inputs='|'.join(images+[str(standard)]),source_fsf=str(source),source_sha256=pa.digest(source),
                 fsf_sha256='',n_evs=len(m['x'][0]),expected_zstats=len(m['c']),group_contract='locked-source',
                 historical_scope=m['scope'],historical_mask=m['mask'],nuisance_policy=POLICY)
        plans.append((job,text)); differences.extend(dict(model=name,**r) for r in changes)
    return plans,sources,differences


def prepare(repo,work,standard):
    plans,sources,differences=build_plan(repo,work,standard)
    version=Path(os.environ['FSLDIR'])/'etc/fslversion'
    software=dict(fsl_version=version.read_text().strip(),
                  binaries={tool:pa.digest(Path(shutil.which(tool))) for tool in ('feat','feat_model','randomise')})
    config=dict(sources=sources,work=str(work),standard=str(standard),software=software,covariate_policy=POLICY)
    config_path=work/'preparation_config.json'; manifest=work/'revision_jobs.tsv'
    if config_path.exists():
        if json.loads(config_path.read_text())!=config: raise ValueError('sources/software changed; use a new versioned work/report root')
        expected=[]
        for job,text in plans:
            path=Path(job['fsf'])
            if path.read_text()!=text: raise ValueError(f'prepared design changed: {path}')
            job['fsf_sha256']=pa.digest(path); expected.append({k:str(v) for k,v in job.items()})
        if pa.read_rows(manifest)!=expected: raise ValueError('prepared manifest changed')
        verify_inputs(manifest)
        return manifest
    image_provenance=check_images(plans,standard)
    protected=[repo,standard]
    protected.extend(Path(j['inputs'].split('|')[0]).parents[2] for j,_ in plans)
    safe_empty(work,protected)
    for job,text in plans:
        path=Path(job['fsf']); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
        job['fsf_sha256']=pa.digest(path)
    pa.write_rows(manifest,[j for j,_ in plans])
    pa.write_rows(work/'covariate_changes.tsv',differences)
    image_provenance.update({p:dict(sha256=h) for p,h in sources.items()})
    pa.write_json(work/'input_provenance.json',image_provenance)
    pa.write_json(work/'analysis_plan.json',dict(covariate_policy=POLICY,models=15,participants=47,
        source_contrasts=96,changes=['tsnr','fd_mean'],preserved='L2 COPE/VARCOPE, sample, RT, behavior, contrasts, age coding and FEAT settings',
        permutation='10000 permutations; TFCE, cluster t=2.6 and t=3.1; historical seeds retained',
        limitations='Spatial FWE per signed test; no multiplicity correction across model/method search. Randomise is unweighted, not FLAME robust deweighting.'))
    pa.write_json(work/'software.json',software); pa.write_json(config_path,config)
    print('PASS: prepared 15 DMN L3 models; only tSNR/FD changed; historical outputs untouched',flush=True)
    return manifest


def compare(repo,report):
    """Report every old/new contrast, including nulls, without selecting results."""
    new=pa.read_rows(report/'imaging/summary.tsv'); rows=[]
    for r in new:
        scope,run=r['model'].split('__',1)
        old=pa.read_rows(repo/pa.REPORTS[scope]/'summary.tsv')
        candidates=[s for s in old if s['model']==run and s['contrast']==r['contrast']]
        if len(candidates)!=1: raise ValueError('historical FLAME contrast missing/ambiguous')
        s=candidates[0]
        rows.append(dict(model=r['model'],contrast=r['contrast'],contrast_name=r['contrast_name'],
                         historical_clusters=s['clusters'],standardized_clusters=r['clusters'],
                         historical_voxels=s['voxels'],standardized_voxels=r['voxels'],
                         historical_minimum_cluster_p=s['minimum_cluster_p'],standardized_minimum_cluster_p=r['minimum_cluster_p']))
    pa.write_rows(report/'flame_historical_comparison.tsv',rows)
    old={(r['model'],r['contrast'],r['method']):r for r in pa.read_rows(repo/'results/permutation_audit/v1/model_contrast_summary.tsv')}
    rows=[]
    for r in pa.read_rows(report/'randomise/model_contrast_summary.tsv'):
        s=old[r['model'],r['contrast'],r['method']]
        rows.append(dict(model=r['model'],contrast=r['contrast'],contrast_name=r['contrast_name'],method=r['method'],
                         historical_p=s['minimum_spatial_fwe_p'],standardized_p=r['minimum_spatial_fwe_p'],
                         historical_voxels=s['significant_voxels'],standardized_voxels=r['significant_voxels'],
                         seed=r['seed'],historical_seed=s['seed']))
    pa.write_rows(report/'randomise_historical_comparison.tsv',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root',type=Path,required=True); p.add_argument('--report-root',type=Path,required=True)
    p.add_argument('--group-jobs',type=int,default=4); p.add_argument('--jobs',type=int,default=40)
    p.add_argument('--execute',action='store_true'); a=p.parse_args()
    if not 1<=a.group_jobs<=15 or not 1<=a.jobs<=45: p.error('group jobs 1..15, randomise jobs 1..45')
    work,report=a.work_root.resolve(),a.report_root.resolve()
    if report.parent!=ROOT/'results/dmn_covariate_standardized': p.error('use a new version under results/dmn_covariate_standardized')
    if work==ROOT or ROOT in work.parents or work in ROOT.parents: p.error('scratch must be outside repository')
    for tool in ('feat','feat_model','randomise'):
        if not shutil.which(tool): p.error(f'{tool} not on PATH')
    if not os.environ.get('FSLDIR'): p.error('FSLDIR not set')
    standard=(Path(os.environ['FSLDIR'])/'data/standard/MNI152_T1_2mm_brain.nii.gz').resolve()
    # Sibling lock avoids modifying an unprepared root and serializes prepare/execute.
    import fcntl
    work.parent.mkdir(parents=True,exist_ok=True)
    with work.with_suffix('.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('standardization batch already running')
        manifest=prepare(ROOT,work,standard)
        report.mkdir(parents=True,exist_ok=True)
        for name in ('preparation_config.json','covariate_changes.tsv','analysis_plan.json'):
            destination=report/name; source=work/name
            if destination.exists() and pa.digest(destination)!=pa.digest(source): raise ValueError('report belongs to a different preparation')
            shutil.copy2(source,destination)
        compile_designs(manifest,'l3')
        shutil.copy2(work/'l3_preflight.tsv',report/'l3_preflight.tsv')
        status=run_jobs(manifest,'l3',a.group_jobs,resume=True,dry_run=not a.execute)
        if status or not a.execute: return status
        verify_stage_outputs(manifest,'l3')
        collect(manifest,report/'imaging',expected_jobs=15)
        args=[sys.executable,'-u',str(ROOT/'code/run_permutation_audit.py'),
              '--dmn-standardized-report',str(report/'imaging'),'--work-root',str(work/'randomise'),
              '--report-root',str(report/'randomise'),'--jobs',str(a.jobs),'--permutations','10000','--execute']
        subprocess.run(args,check=True)
        compare(ROOT,report)
        pa.write_json(report/'status.json',dict(complete=True,l3_models=15,source_contrasts=96,
                     historical_outputs_preserved=True,interpretation='Inspect all old/new results; no model selection performed'))
        print('PASS: DMN standardization, group fits, randomise and historical comparisons complete',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
