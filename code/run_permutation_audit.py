#!/usr/bin/env python3
"""Resumable, source-locked randomise audit of retained group analyses.

No L1/L2 fits, covariate substitutions, age exchangeability blocks, or
significance-based model selection. See docs/PERMUTATION_AUDIT_RUNBOOK.txt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import nibabel as nib
import numpy as np

from audit_l1_designs import read_vest_matrix
from prepare_sub144_imaging_repair import fsf_value
from prepare_ultimatum_l3_repair import parse_evs, parse_inputs

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'tfce': ['-T'], 'cluster-t2.6': ['-c', '2.6'], 'cluster-t3.1': ['-c', '3.1']}
REPORTS = {
    'full-rt': 'results/reviewer/rt_coverage_corrected',
    'pooled-post': 'results/decision_postresponse/imaging',
    'partner-post': 'results/decision_postresponse_partner/imaging',
    'crossed': 'results/crossed_behavior_l3/v1/imaging',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


def read_rows(path, delimiter='\t'):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'cannot silently write an empty table: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def vest(path, matrix, contrast=False):
    matrix = np.atleast_2d(matrix)
    header = f'/NumWaves\t{matrix.shape[1]}\n'
    header += f'/{"NumContrasts" if contrast else "NumPoints"}\t{matrix.shape[0]}\n'
    with path.open('w') as f:
        f.write(header + '/Matrix\n')
        np.savetxt(f, matrix, fmt='%.12g', delimiter='\t')


def canonical_design(x, c):
    """Only permute EV columns; never change scaling, signs or design span."""
    order = sorted(range(x.shape[1]), key=lambda i: tuple(x[:, i]))
    return x[:, order], c[:, order]


def inspect_model(repo, directory, name, scope, sample):
    fsf = directory / 'design/design.fsf'
    text = fsf.read_text()
    paths = [directory / 'design' / ('design.' + ext) for ext in ('fsf', 'mat', 'con', 'grp')]
    mask_path = directory / 'poststats/mask.nii.gz'
    x = read_vest_matrix(paths[1]); c = read_vest_matrix(paths[2])
    members = parse_inputs(text.splitlines())
    subjects = [s for _, s, _ in members]
    if [r for r, _, _ in members] != list(range(1, 48)) or len(set(subjects)) != 47 or set(subjects) != set(sample):
        raise ValueError(f'expected the locked 47 unique participants: {name}')
    if x.shape[0] != 47 or c.shape[1] != x.shape[1] or not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError(f'invalid group design: {name}')
    if np.linalg.matrix_rank(x) != x.shape[1] or x.shape[1] >= 47:
        raise ValueError(f'rank deficient group design: {name}')
    if not np.allclose(x @ np.linalg.lstsq(x, np.ones(47), rcond=None)[0], 1, atol=1e-5):
        raise ValueError(f'design must model the mean explicitly, no automatic -D: {name}')
    evs = parse_evs(text.splitlines())
    expected = np.array([[evs[r, j] for j in range(1, x.shape[1]+1)] for r in range(1, 48)])
    expected_c = np.array([[float(fsf_value(text, f'con_real{i}.{j}')) for j in range(1,x.shape[1]+1)] for i in range(1,c.shape[0]+1)])
    if not np.allclose(x, expected, atol=1e-5, rtol=0) or not np.array_equal(c, expected_c):
        raise ValueError(f'compiled FSF/matrix/contrast mismatch: {name}')
    images = [p for _, _, p in members]
    if any(not re.search(r'/cope\d+\.feat/stats/cope1\.nii\.gz$', p) for p in images):
        raise ValueError(f'expected one participant-level L2 COPE, not a Z map: {name}')
    reorder = np.argsort(subjects)
    subjects = [subjects[i] for i in reorder]; images = [images[i] for i in reorder]
    x = x[reorder]
    titles = [fsf_value(text, f'evtitle{i}') for i in range(1,x.shape[1]+1)]
    nuisance = {title: x[:, i].tolist() for i, title in enumerate(titles) if title in ('tsnr', 'fd_mean', 'RT', 'ismale')}
    x, c = canonical_design(x, c)
    mask = nib.load(mask_path)
    values = np.asarray(mask.dataobj)
    if len(mask.shape) != 3 or not np.isfinite(values).all() or not np.isin(values, [0,1]).all() or not values.any():
        raise ValueError(f'invalid whole-brain mask: {mask_path}')
    mask_key = identity([mask.shape, mask.affine.tolist(), hashlib.sha256((values>0).tobytes()).hexdigest()])
    block = identity([subjects, images, mask_key])
    group = identity([block, x.tolist()])
    return dict(name=name, scope=scope, subjects=subjects, images=images, x=x.tolist(), c=c.tolist(),
                labels=[fsf_value(text, f'conname_real.{i}') for i in range(1,len(c)+1)],
                mask=str(mask_path), block=block, group=group, nuisance=nuisance,
                robust=fsf_value(text,'robust_yn')=='1',
                source_hashes={str(p):digest(p) for p in paths+[mask_path]})


def catalog(repo):
    table = repo/'behavioral_analyses/data/participant_L3_47.csv'
    people = read_rows(table, ','); sample = {r['subjID']:r for r in people}
    if len(people)!=47 or len(sample)!=47 or sum(int(float(r['older'])) for r in people)!=22:
        raise ValueError('unexpected participant sample')
    models=[]
    for name, relative in REPORTS.items():
        report=repo/relative
        jobs=[r for r in read_rows(report/'revision_jobs.tsv') if r['stage']=='l3']
        if len(jobs)!=(18 if name=='crossed' else 5):
            raise ValueError(f'unexpected retained L3 count in {report}')
        indexed={r['artifact']:r['sha256'] for r in read_rows(report/'inventory.tsv')}
        for job in jobs:
            directory=report/job['run']
            m=inspect_model(repo,directory,f'{name}__{job["run"]}',name,sample)
            for path,h in m['source_hashes'].items():
                if indexed.get(str(Path(path).relative_to(report)))!=h:
                    raise ValueError(f'source inventory mismatch: {path}')
            models.append(m)
    extras=[('taskwide-offer','results/reviewer/activation_fairness_main'),
            ('human-computer-offer','results/reviewer/activation_social_computer/social_offer_slope'),
            ('human-computer-response','results/reviewer/activation_social_computer/social_task_response')]
    for name, relative in extras:
        models.append(inspect_model(repo,repo/relative,name,'supplemental-fixed-display',sample))
    return models,sample,table


def plan_tests(models, seed):
    tests={}; mapping=[]
    for model in models:
        for k,(contrast,label) in enumerate(zip(model['c'],model['labels']),1):
            c=np.array(contrast)
            if not np.any(c): raise ValueError('zero contrast')
            # Positive scalar multiples are the same one-sided test. Opposite
            # signs remain separate. Canonical EV ordering makes duplicates explicit.
            c=c/np.max(np.abs(c))
            key=identity([model['group'],c.tolist()])
            tests.setdefault(key,dict(test=key,group=model['group'],block=model['block'],contrast=c.tolist(),
                                      seed=(seed+int(key[:7],16))%2147483646+1))
            mapping.append(dict(model=model['name'],scope=model['scope'],contrast=k,contrast_name=label,
                                test=key,group=model['group'],block=model['block'],
                                flame_robust=int(model['robust']),
                                interpretation='unweighted audit; NOT FLAME outlier deweighting' if model['robust'] else 'unweighted audit of retained design'))
    return list(tests.values()),mapping


def nuisance_audit(models,sample,report):
    subject_rows=[]; summary=[]
    baseline={m['scope']:m for m in models if m['name'].endswith('__ecn-sensitivity')}
    for m in models:
        for metric in ('tsnr','fd_mean'):
            a=np.array(m['nuisance'][metric])
            reference=np.array([float(sample[s][metric]) for s in m['subjects']])
            b=baseline.get(m['scope'])
            other=np.array(b['nuisance'][metric]) if b else reference
            fit=np.column_stack([np.ones(len(a)),reference])
            affine_error=np.max(np.abs(a-fit@np.linalg.lstsq(fit,a,rcond=None)[0]))
            summary.append(dict(model=m['name'],metric=metric,
                max_abs_vs_participant_table=float(np.max(abs(a-reference))),
                max_abs_after_affine_rescaling=float(affine_error),
                max_abs_vs_ecn_or_participant_table=float(np.max(abs(a-other))),
                source_table_match=int(np.allclose(a,reference,atol=1e-5,rtol=0)),
                policy='audit only; no nuisance covariates replaced'))
            for i,s in enumerate(m['subjects']):
                subject_rows.append(dict(model=m['name'],subject=s,metric=metric,retained=a[i],participant_table=reference[i],difference=a[i]-reference[i]))
    write_rows(report/'nuisance_source_comparison.tsv',summary)
    write_rows(report/'nuisance_participant_comparison.tsv',subject_rows)


def verify_files(records):
    for path,expected in records.items():
        if not Path(path).is_file() or digest(path)!=expected:
            raise ValueError(f'input or completed output changed: {path}')


def compatible(a,b):
    return a.shape==b.shape and np.allclose(a.affine,b.affine,atol=1e-5,rtol=0)


def prepare_block(model,work):
    root=work/'blocks'/model['block']; receipt=root/'complete.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text()); verify_files(saved['sources']); verify_files(saved['outputs'])
        return root
    root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent,prefix='preparing-') as tmp:
        staging=Path(tmp); reference=nib.load(model['mask']); mask=reference.get_fdata()>0
        images=[]; source_hashes={model['mask']:digest(model['mask'])}; coverage=[]
        for subject,name in zip(model['subjects'],model['images']):
            path=Path(name); var=path.with_name('varcope1.nii.gz')
            if not path.is_file() or not var.is_file(): raise FileNotFoundError(f'missing participant COPE/VARCOPE: {path}')
            im=nib.load(path); vi=nib.load(var)
            if not compatible(im,reference) or not compatible(vi,reference): raise ValueError(f'geometry mismatch: {path}')
            data=im.get_fdata(dtype=np.float32); variance=vi.get_fdata(dtype=np.float32)
            if not np.isfinite(data).all() or not np.isfinite(variance).all() or (variance<0).any(): raise ValueError(f'nonfinite data or negative VARCOPE: {path}')
            images.append(data)
            source_hashes[str(path)]=digest(path); source_hashes[str(var)]=digest(var)
            coverage.append(dict(subject=subject,cope=str(path),varcope=str(var),mask_voxels=int(mask.sum()),
                                 zero_varcope_inside_mask=int(np.sum(variance[mask]==0))))
        data=np.stack(images,axis=3)
        # Match randomise's default removal of constant voxels explicitly and
        # retain both masks/counts. Never expand the retained analysis mask.
        active=mask & (np.ptp(data,axis=3)>0)
        if not active.any(): raise ValueError('no nonconstant voxels in retained mask')
        nib.save(nib.Nifti1Image(data,reference.affine),staging/'copes.nii.gz')
        nib.save(nib.Nifti1Image(active.astype(np.uint8),reference.affine),staging/'mask.nii.gz')
        shutil.copy2(model['mask'],staging/'retained_mask.nii.gz')
        write_rows(staging/'participants.tsv',coverage)
        write_json(staging/'coverage.json',dict(retained_voxels=int(mask.sum()),tested_voxels=int(active.sum()),
                                               constant_voxels_removed=int(np.sum(mask & ~active)),
                                               zero_varcope_policy='reported; not used for weighting or automatic exclusion'))
        if root.exists(): raise FileExistsError(f'unexpected incomplete prepared block: {root}')
        staging.rename(root)
    outputs={str(p):digest(p) for p in root.iterdir() if p.is_file()}
    write_json(receipt,dict(sources=source_hashes,outputs=outputs))
    return root


def residual_audit(x,y,older):
    """Descriptive OLS residual diagnostics, not an exchangeability proof/test."""
    inverse=np.linalg.pinv(x); residual=y-x@(inverse@y)
    leverage=np.diag(x@inverse)
    if np.any(leverage>=1-1e-8): raise ValueError('unit leverage prevents residual audit')
    squared=residual**2/(1-leverage[:,None])
    young=squared[~older].mean(0); old=squared[older].mean(0)
    valid=(young>0)&(old>0)
    if not valid.any(): raise ValueError('no residual variance for group comparison')
    ratio=old[valid]/young[valid]
    return dict(older_over_younger_variance_p05=float(np.quantile(ratio,.05)),
                older_over_younger_variance_median=float(np.median(ratio)),
                older_over_younger_variance_p95=float(np.quantile(ratio,.95)),
                fraction_ratio_outside_half_to_two=float(np.mean((ratio<.5)|(ratio>2))),
                max_leverage=float(leverage.max()),n_young=int((~older).sum()),n_old=int(older.sum()),
                interpretation='descriptive; does not establish exchangeability; no data-driven method switching'),np.mean(squared,axis=1)


def prepare(models,tests,sample,work,report):
    blocks={}; groups={}; diagnostics=[]; subject_diagnostics=[]
    for m in models:
        if m['block'] not in blocks:
            print(f'CHECK INPUTS: {m["name"]}',flush=True)
            blocks[m['block']]=prepare_block(m,work)
        if m['group'] in groups: continue
        groups[m['group']]=m
        group=work/'groups'/m['group']; group.mkdir(parents=True,exist_ok=True)
        x=np.array(m['x']); vest(group/'design.mat',x)
        block=blocks[m['block']]; im=nib.load(block/'copes.nii.gz'); mask=nib.load(block/'mask.nii.gz').get_fdata()>0
        y=im.get_fdata(dtype=np.float32)[mask,:].T
        older=np.array([bool(int(float(sample[s]['older']))) for s in m['subjects']])
        diag,subject_variance=residual_audit(x,y,older)
        diagnostics.append(dict(group=m['group'],example_model=m['name'],**diag))
        for s,value in zip(m['subjects'],subject_variance):
            subject_diagnostics.append(dict(group=m['group'],subject=s,older=int(float(sample[s]['older'])),mean_leverage_adjusted_squared_residual=float(value)))
        out=report/'designs'/m['group']; out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(group/'design.mat',out/'design.mat')
        for name in ('mask.nii.gz','retained_mask.nii.gz','participants.tsv','coverage.json'):
            shutil.copy2(block/name,out/name)
    for t in tests:
        folder=work/'tests'/t['test']; folder.mkdir(parents=True,exist_ok=True)
        vest(folder/'design.con',np.array(t['contrast'])[None,:],contrast=True)
        out=report/'contrasts'/t['test']; out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(folder/'design.con',out/'design.con')
    input_hashes={}
    for block in blocks.values():
        receipt=json.loads((block/'complete.json').read_text())
        input_hashes.update(receipt['sources'])
        input_hashes.update(receipt['outputs'])
    write_json(report/'input_provenance.json',input_hashes)
    write_rows(report/'residual_variance_by_age.tsv',diagnostics)
    write_rows(report/'participant_residual_diagnostics.tsv',subject_diagnostics)


def command(test,method,work,prefix,permutations,binary):
    return [str(binary),'-i',str(work/'blocks'/test['block']/'copes.nii.gz'),'-o',str(prefix),
            '-d',str(work/'groups'/test['group']/'design.mat'),'-t',str(work/'tests'/test['test']/'design.con'),
            '-m',str(work/'blocks'/test['block']/'mask.nii.gz'),'-n',str(permutations),
            f'--seed={test["seed"]}','-R','-N',*METHODS[method]]


def expected_outputs(root,method):
    stat='tfce' if method=='tfce' else 'clustere'
    return [root/'randomise_tstat1.nii.gz',root/f'randomise_{stat}_corrp_tstat1.nii.gz',
            root/f'randomise_{stat}_corrp_tstat1.txt']


def run_test(test,method,work,permutations,binary):
    parent=work/'jobs'/test['test']/method; parent.mkdir(parents=True,exist_ok=True)
    marker=parent/'complete.json'
    if marker.exists():
        receipt=json.loads(marker.read_text()); verify_files(receipt['outputs'])
        print(f'SKIP_COMPLETE: {test["test"]} {method}',flush=True)
        return receipt
    # An interrupted attempt stays intact. Retry into a new attempt directory.
    attempt=Path(tempfile.mkdtemp(dir=parent,prefix='attempt-'))
    cmd=command(test,method,work,attempt/'randomise',permutations,binary)
    write_json(attempt/'command.json',cmd)
    print(f'RUN: {test["test"]} {method}',flush=True)
    with (attempt/'run.txt').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'randomise exit {result.returncode}: {attempt}/run.txt\n' + (attempt/'run.txt').read_text(errors='replace')[-3000:])
    expected=expected_outputs(attempt,method)
    for p in expected:
        if not p.is_file() or p.stat().st_size==0:
            raise FileNotFoundError(f'missing permutation output: {p}; found {[x.name for x in attempt.iterdir()]}')
    reference=nib.load(work/'blocks'/test['block']/'mask.nii.gz')
    for p in expected[:2]:
        im=nib.load(p); a=im.get_fdata()
        if not compatible(im,reference) or not np.isfinite(a).all(): raise ValueError(f'invalid permutation image: {p}')
        if 'corrp' in p.name and ((a<0).any() or (a>1).any()): raise ValueError(f'invalid 1-p range: {p}')
    null=np.loadtxt(expected[2],ndmin=1)
    if not np.isfinite(null).all() or not 2<=null.size<=permutations:
        raise ValueError(f'invalid recorded permutation null distribution: {expected[2]}')
    receipt=dict(test=test['test'],method=method,attempt=str(attempt),command=cmd,
                 permutations_recorded=int(null.size),
                 outputs={str(p):digest(p) for p in attempt.iterdir() if p.is_file()})
    write_json(marker,receipt)
    print(f'COMPLETED: {test["test"]} {method}',flush=True)
    return receipt


def collect(tests,mapping,work,report):
    records=[]; inventory=[]
    for t in tests:
        for method in METHODS:
            parent=work/'jobs'/t['test']/method
            record=json.loads((parent/'complete.json').read_text()); verify_files(record['outputs'])
            source=Path(record['attempt']); target=report/'maps'/t['test']/method
            target.mkdir(parents=True,exist_ok=True)
            for p in source.iterdir():
                if not p.is_file(): continue
                if p.stat().st_size>20_000_000: raise ValueError(f'not a compact output: {p}')
                dest=target/p.name; shutil.copy2(p,dest)
                inventory.append(dict(artifact=str(dest.relative_to(report)),sha256=digest(dest),bytes=dest.stat().st_size))
            timage,pimage,_=expected_outputs(source,method)
            stat=nib.load(timage); p=nib.load(pimage).get_fdata(); values=stat.get_fdata()
            mask=nib.load(work/'blocks'/t['block']/'mask.nii.gz').get_fdata()>0
            significant=mask & (p>0.95) & (values>0)
            coords=np.unravel_index(np.argmax(np.where(mask,p,-1)),p.shape)
            xyz=nib.affines.apply_affine(stat.affine,coords)
            records.append(dict(test=t['test'],method=method,significant_voxels=int(significant.sum()),
                                minimum_spatial_fwe_p=float(1-p[mask].max()),
                                maximum_t=float(values[mask].max()),
                                max_corrp_x=float(xyz[0]),max_corrp_y=float(xyz[1]),max_corrp_z=float(xyz[2]),
                                status='complete',seed=t['seed'],permutations_recorded=record['permutations_recorded']))
    write_rows(report/'unique_test_summary.tsv',records)
    lookup={(r['test'],r['method']):r for r in records}
    expanded=[dict(**m,**{k:v for k,v in lookup[m['test'],method].items() if k!='test'}) for m in mapping for method in METHODS]
    write_rows(report/'model_contrast_summary.tsv',expanded)
    write_rows(report/'map_inventory.tsv',inventory)
    return len(records)


def software(binary):
    fsl=Path(os.environ.get('FSLDIR','/nonexistent'))/'etc/fslversion'
    return dict(python=sys.version,numpy=np.__version__,nibabel=nib.__version__,
                randomise=str(binary),randomise_sha256=digest(binary),
                fsl_version=fsl.read_text().strip() if fsl.is_file() else 'unavailable')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--report-root',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=40)
    p.add_argument('--permutations',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not 1<=a.jobs<=45 or a.permutations<100 or a.seed<1: p.error('jobs 1..45, permutations >=100, positive seed required')
    work,report=a.work_root.resolve(),a.report_root.resolve()
    if work==ROOT or ROOT in work.parents or work in ROOT.parents or work==report or work in report.parents or report in work.parents:
        p.error('scratch root must be outside the repository and separate from report root')
    binary=shutil.which('randomise')
    if not binary: p.error('randomise not found; activate FSL')
    binary=Path(binary).resolve()
    models,sample,table=catalog(ROOT); tests,mapping=plan_tests(models,a.seed)
    sources={str(table):digest(table),str(Path(__file__).resolve()):digest(__file__)}
    for module in ('audit_l1_designs.py','prepare_sub144_imaging_repair.py','prepare_ultimatum_l3_repair.py'):
        sources[str(ROOT/'code'/module)]=digest(ROOT/'code'/module)
    for m in models: sources.update(m['source_hashes'])
    config=dict(models=models,permutations=a.permutations,seed=a.seed,methods=METHODS,
                sources=sources,software=software(binary),work=str(work),report=str(report),
                exchangeability='independent participants; unrestricted nuisance-adjusted permutations; randomise automatic sign-flips when applicable',
                limitations='Assumes appropriate residual exchangeability/symmetry; age variance audit is descriptive. No FLAME weighting, no robust deweighting, no correction across tests/models/methods.')
    existing=work/'config.json'
    for m in models:
        for source in m['images']:
            protected=Path(source).parents[2]
            if work==protected or protected in work.parents or work in protected.parents:
                p.error(f'scratch overlaps existing imaging inputs: {protected}')
    report.mkdir(parents=True,exist_ok=True)
    if (report/'config.json').exists() and json.loads((report/'config.json').read_text())!=config:
        raise ValueError('report belongs to a different preparation; use a new report root')
    if existing.exists():
        if json.loads(existing.read_text())!=config: raise ValueError('configuration/source/software changed; use a new versioned work/report root')
    else:
        if work.exists() and any(work.iterdir()): raise FileExistsError(f'unrecognized nonempty scratch root: {work}')
        work.mkdir(parents=True,exist_ok=True); write_json(existing,config)
    # Lock is released by the OS even if SSH or the process dies.
    import fcntl
    with (work/'run.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('this scratch batch is already running')
        write_json(report/'config.json',config)
        write_rows(report/'test_mapping.tsv',mapping)
        nuisance_audit(models,sample,report)
        prepare(models,tests,sample,work,report)
        commands=[dict(test=t['test'],method=m,command=json.dumps(command(t,m,work,work/'jobs'/t['test']/m/'ATTEMPT/randomise',a.permutations,binary))) for t in tests for m in METHODS]
        write_rows(report/'planned_commands.tsv',commands)
        write_json(report/'status.json',dict(stage='prepared',models=len(models),source_contrasts=len(mapping),unique_tests=len(tests),jobs=len(commands),complete=False))
        print(f'PASS: {len(models)} models, {len(mapping)} source contrasts, {len(tests)} unique tests, {len(commands)} permutation jobs',flush=True)
        if not a.execute:
            print('Dry run complete; no permutations executed',flush=True); return 0
        failures=[]
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures={pool.submit(run_test,t,m,work,a.permutations,binary):(t,m) for t in tests for m in METHODS}
            for future in as_completed(futures):
                try: future.result()
                except Exception as e:
                    t,m=futures[future]; failures.append(dict(test=t['test'],method=m,error=str(e)))
                    print(f'FAILED: {t["test"]} {m}: {e}',flush=True)
        if failures:
            write_rows(report/f'failures-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.tsv',failures)
            # Keep failed attempt logs small and available to Git without copying COPEs.
            for f in failures:
                for log in (work/'jobs'/f['test']/f['method']).glob('attempt-*/run.txt'):
                    dest=report/'failure_logs'/f['test']/f['method']/log.parent.name
                    dest.mkdir(parents=True,exist_ok=True); shutil.copy2(log,dest/'run.txt')
            write_json(report/'status.json',dict(stage='execution',complete=False,failed=len(failures)))
            return 1
        count=collect(tests,mapping,work,report)
        write_json(report/'status.json',dict(stage='collected',complete=True,unique_method_tests=count,
                                            source_model_method_contrasts=len(mapping)*len(METHODS),
                                            scientific_review='pending variance, anatomy, influence and multiplicity review'))
        print(f'PASS: collected {count} unique permutation results, including nulls',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
