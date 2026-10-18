#!/usr/bin/env python3
"""PALM age-variance-group sensitivity audit; preserve all retained DMN designs.

Independent symmetric-error sign flips, not unrestricted age-label permutations.
PALM cluster thresholds are equivalent Z thresholds, NOT randomise t thresholds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import nibabel as nib
import numpy as np
import run_permutation_audit as pa

ROOT=Path(__file__).resolve().parents[1]
PALM_COMMIT='9086eff76dcacf4afde12c9af5400a1fe0033b9e'
METHODS={'tfce':['-T'], 'cluster-z2.6':['-C','2.6','-Cstat','extent'],
         'cluster-z3.1':['-C','3.1','-Cstat','extent']}


def variance_groups(model,sample):
    groups=np.array([1+int(float(sample[s]['older'])) for s in model['subjects']])
    if sorted(np.bincount(groups)[1:].tolist())!=[22,25]: raise ValueError('unexpected age groups')
    return groups


def quote(text): return "'"+str(text).replace("'","''")+"'"


def engine_command(engine,palm,args):
    expression='addpath('+quote(palm)+');'
    if 'octave' in Path(engine).name: expression+='pkg load image; pkg load statistics;'
    expression+='disp(version);palm('+','.join(quote(x) for x in args)+');'
    if 'octave' in Path(engine).name:
        return [engine,'--no-gui','--quiet','--eval',expression]
    return [engine,'-singleCompThread','-batch',expression]


def arguments(directory,method,prefix,n,seed):
    return ['-i',str(directory/'copes.nii.gz'),'-m',str(directory/'mask.nii.gz'),
        '-d',str(directory/'design.mat'),'-t',str(directory/'design.con'),
        '-vg',str(directory/'variance_groups.csv'),'-ise','-rmethod','Freedman-Lane',
        '-n',str(n),'-seed',str(seed),'-zstat','-save1-p','-savemax',
        '-noniiclass','-o',str(prefix)]+METHODS[method]


def probability_maps(directory,method,ncontrasts,mask):
    unit='tfce' if method=='tfce' else 'clustere'
    found=[]
    for c in range(1,ncontrasts+1):
        paths=list(directory.glob(f'palm_{unit}_*_fwep_c{c}.nii*'))
        if len(paths)!=1: raise ValueError(f'missing/ambiguous PALM FWE map: {directory}, contrast {c}')
        im=nib.load(paths[0])
        if not pa.compatible(im,mask): raise ValueError('PALM output grid differs')
        data=im.get_fdata(); keep=mask.get_fdata()>0
        if not np.isfinite(data[keep]).all() or np.any((data[keep]<0)|(data[keep]>1)):
            raise ValueError('invalid PALM 1-p values')
        found.append((paths[0],dict(contrast=c,minimum_spatial_fwe_p=float(1-data[keep].max()),
            significant_voxels=int(np.sum(data[keep]>.95)))))
    return found


def job(model,method,inputs,work,report,engine,palm,n):
    name=model['name']; folder=work/'jobs'/name/method;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'complete.json'; mask=nib.load(inputs/'mask.nii.gz')
    if receipt.exists():
        saved=json.loads(receipt.read_text());pa.verify_files(saved['outputs'])
        attempt=Path(saved['attempt']); print(f'SKIP_COMPLETE: {name} {method}',flush=True)
    else:
        attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=folder))
        seed=20261018
        cmd=engine_command(engine,palm,arguments(inputs,method,attempt/'palm',n,seed))
        pa.write_json(attempt/'command.json',cmd)
        print(f'RUN: {name} {method}',flush=True)
        with (attempt/'run.txt').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'PALM failed ({result.returncode}): {attempt}/run.txt')
        probability_maps(attempt,method,len(model['c']),mask)
        pa.write_json(receipt,dict(attempt=str(attempt),outputs={str(p):pa.digest(p) for p in attempt.iterdir() if p.is_file()}))
        print(f'COMPLETED: {name} {method}',flush=True)
    destination=report/'maps'/name/method;destination.mkdir(parents=True,exist_ok=True)
    for file in attempt.iterdir():
        if file.is_file():
            if file.stat().st_size>20_000_000: raise ValueError(f'oversize compact artifact: {file}')
            shutil.copy2(file,destination/file.name)
    return [dict(model=name,method=method,contrast_name=model['labels'][r['contrast']-1],**r)
            for _,r in probability_maps(attempt,method,len(model['c']),mask)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/dmn_covariate_standardized/v1')
    p.add_argument('--work-root',type=Path,required=True)
    p.add_argument('--report-root',type=Path,default=ROOT/'results/dmn_followup/v1/palm')
    p.add_argument('--palm-root',type=Path,required=True)
    p.add_argument('--engine',default=shutil.which('matlab') or shutil.which('octave') or '')
    p.add_argument('--jobs',type=int,default=15);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.engine: p.error('MATLAB or Octave required. For Octave install its image and statistics packages too.')
    if not 1<=a.jobs<=40:p.error('jobs must be 1..40')
    work=a.work_root.resolve();report=a.report_root.resolve();source=a.source.resolve();palm=a.palm_root.resolve()
    if work==ROOT or ROOT in work.parents or work in ROOT.parents:p.error('work must be outside repository')
    if report.parent!=ROOT/'results/dmn_followup/v1':p.error('use results/dmn_followup/v1/palm (or a separate version of this runner)')
    commit=subprocess.check_output(['git','-C',str(palm),'rev-parse','HEAD'],text=True).strip()
    if commit!=PALM_COMMIT or subprocess.check_output(['git','-C',str(palm),'status','--porcelain'],text=True).strip():
        p.error('PALM must be the clean pinned source checkout')
    if not json.loads((source/'status.json').read_text())['complete']:raise ValueError('source incomplete')
    config=json.loads((source/'randomise/config.json').read_text())
    pa.verify_files(json.loads((source/'randomise/input_provenance.json').read_text()))
    # Reinspect compact designs, subject identities, and inventories independently.
    models,sample,table=pa.standardized_dmn_catalog(ROOT,source/'imaging')
    frozen={m['name']:m for m in config['models']}
    for m in models:
        for key in ('x','c','subjects','images','block','group'):
            if m[key]!=frozen[m['name']][key]:raise ValueError(f'changed retained model: {m["name"]}/{key}')
    report.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    with (work/'batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract=dict(source=pa.digest(source/'randomise/config.json'),table=pa.digest(table),
            code=pa.digest(__file__),helper=pa.digest(pa.__file__),palm_commit=commit,
            engine=str(Path(a.engine).resolve()),engine_sha256=pa.digest(Path(a.engine).resolve()),
            input_provenance=pa.digest(source/'randomise/input_provenance.json'),
            source_designs={m['name']:m['source_hashes'] for m in models},
            methods=METHODS,shuffles=10000,seed=20261018,models=[m['name'] for m in models],
            assumptions='independent symmetric errors; Freedman-Lane sign flips; age variance groups',
            limitations='Unweighted, not FLAME. Spatial FWE per signed contrast; not corrected across methods/models. Cluster thresholds equivalent Z, not randomise t. No automatic exclusions.')
        path=work/'contract.json'
        if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('contract changed; do not reuse work')
        pa.write_json(path,contract);shutil.copy2(path,report/'contract.json')
        pa.write_json(report/'status.json',dict(complete=False,stage='preparation'))
        inputs={}
        for m in models:
            block=pa.prepare_block(m,work)
            directory=work/'designs'/m['name'];directory.mkdir(parents=True,exist_ok=True)
            for name in ('copes.nii.gz','mask.nii.gz'):
                target=directory/name
                if not target.exists(): target.symlink_to(block/name)
                if target.resolve()!=(block/name).resolve():raise ValueError('unexpected input link')
            pa.vest(directory/'design.mat',np.array(m['x']));pa.vest(directory/'design.con',np.array(m['c']),contrast=True)
            np.savetxt(directory/'variance_groups.csv',variance_groups(m,sample),fmt='%d')
            dest=report/'designs'/m['name'];dest.mkdir(parents=True,exist_ok=True)
            for name in ('design.mat','design.con','variance_groups.csv','mask.nii.gz'):shutil.copy2(directory/name,dest/name)
            pa.write_rows(dest/'participants.tsv',[dict(subject=s,variance_group=int(g)) for s,g in zip(m['subjects'],variance_groups(m,sample))])
            inputs[m['name']]=directory
        for block in (work/'blocks').iterdir():
            shutil.copy2(block/'complete.json',report/f'input-block-{block.name}.json')
        if not a.execute:
            print('PASS: 15 locked DMN models, 45 PALM jobs, 288 signed contrast/method summaries planned; no fits run')
            return
        # Actual engine/NIfTI/TFCE/cluster preflight before submitting the expensive batch.
        smoke=work/'smoke';smoke.mkdir(exist_ok=True)
        first=models[0]; src=inputs[first['name']]
        if (smoke/'complete.json').exists():
            pa.verify_files(json.loads((smoke/'complete.json').read_text())['outputs'])
        else:
            mask=nib.load(src/'mask.nii.gz');values=mask.get_fdata()>0
            selected=np.zeros_like(values);indices=np.argwhere(values)[:64]
            selected[tuple(indices.T)]=True
            nib.save(nib.Nifti1Image(selected.astype('uint8'),mask.affine),smoke/'mask.nii.gz')
            for name in ('copes.nii.gz','design.mat','design.con','variance_groups.csv'):
                target=smoke/name
                if not target.exists():target.symlink_to(src/name)
            args=arguments(smoke,'tfce',smoke/'palm',20,42)+['-C','3.1','-Cstat','extent']
            with (smoke/'run.txt').open('w') as log:
                result=subprocess.run(engine_command(a.engine,palm,args),stdout=log,stderr=subprocess.STDOUT)
            shutil.copy2(smoke/'run.txt',report/'smoke.txt')
            if result.returncode:raise RuntimeError(f'engine smoke failed; inspect {report}/smoke.txt')
            for method in ('tfce','cluster-z3.1'):probability_maps(smoke,method,len(first['c']),nib.load(smoke/'mask.nii.gz'))
            pa.write_json(smoke/'complete.json',dict(passed=True,shuffles=20,not_scientific_results=True,
                outputs={str(f):pa.digest(f) for f in smoke.iterdir() if f.is_file() and not f.is_symlink()}))
        tasks=[(m,method) for m in models for method in METHODS]
        summaries=[];failures=[]
        def run(item):
            m,method=item
            try:return job(m,method,inputs[m['name']],work,report,a.engine,palm,10000),None
            except Exception as exc:return [],dict(model=m['name'],method=method,error=str(exc))
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for rows,error in pool.map(run,tasks):
                summaries.extend(rows)
                if error:failures.append(error)
        if summaries:pa.write_rows(report/'summary.tsv',summaries)
        pa.write_json(report/'status.json',dict(complete=not failures,jobs=45,summary_rows=len(summaries),failures=failures))
        pa.write_rows(report/'inventory.tsv',[dict(path=str(f.relative_to(report)),sha256=pa.digest(f),bytes=f.stat().st_size)
            for f in sorted(report.rglob('*')) if f.is_file() and f.name!='inventory.tsv'])
        if failures:raise RuntimeError(f'{len(failures)} PALM jobs failed; completed jobs preserved; see status.json')
        if len(summaries)!=288:raise ValueError('incomplete summary')
        print('PASS: all 45 PALM jobs complete; 288 signed contrast/method summaries')


if __name__=='__main__':main()
