#!/usr/bin/env python3
"""Fixed-mask vmPFC offer slopes from all three corrected activation pipelines.

No L1/L2 estimation, PALM access, outcome-based ROI selection, or exclusions.
Participant contrasts retain pairing. HC3 inference and descriptive SEM bars
are deliberately separate; mean voxelwise VARCOPE is not ROI-mean variance.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_three_partner_ratings import ROOT, digest, read, write, save_json, save_figure
from audit_l1_designs import read_vest_matrix
from prepare_sub144_imaging_repair import fsf_value

REPORTS = {'full-rt':'results/reviewer/rt_coverage_corrected',
           'pooled-post':'results/decision_postresponse/imaging',
           'partner-post':'results/decision_postresponse_partner/imaging'}
LABELS = {'full-rt':'Full display + corrected RT height',
          'pooled-post':'Decision + pooled post-response',
          'partner-post':'Decision + partner post-response'}
CONDITIONS = (('computer',2,'comp_p'),('similar',4,'in_p'),('dissimilar',6,'out_p'))
ROI_SHA256 = '70ecc1378c8edd2526a3178e8bdb4018e4bb8cda9fa911ef4e5e11da8b2d3bce'
OUTCOMES = {'partner_average':[1/3,1/3,1/3], 'computer':[1,0,0],
            'similar':[0,1,0], 'dissimilar':[0,0,1],
            'human_minus_computer':[-1,.5,.5], 'similar_minus_dissimilar':[0,1,-1]}


def fit_hc3(x, y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if y.ndim==1: y=y[:,None]
    n,p=x.shape
    if len(y)!=n or n<=p+2 or np.linalg.matrix_rank(x)!=p or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid ROI model/sample')
    inv=np.linalg.inv(x.T@x); pinv=inv@x.T
    beta=pinv@y; residual=y-x@beta
    leverage=np.sum(x*pinv.T,axis=1)
    if np.any(leverage>=1-1e-10): raise ValueError('unit leverage')
    return beta, residual, leverage, pinv


def estimate(x,y,c):
    beta,e,h,pinv=fit_hc3(x,y)
    if beta.shape[1]!=1: raise ValueError('scalar outcome required')
    value=float(c@beta[:,0]); influence=(c@pinv)*e[:,0]/(1-h)
    se=float(np.sqrt(influence@influence)); df=len(y)-x.shape[1]
    if se<=0: raise ValueError('degenerate HC3 uncertainty')
    t=value/se; p=float(2*stats.t.sf(abs(t),df))
    loo=[]
    for i in range(len(y)):
        keep=np.arange(len(y))!=i
        if np.linalg.matrix_rank(x[keep])!=x.shape[1]: raise ValueError('LOO rank loss')
        loo.append(float(c@np.linalg.lstsq(x[keep],y[keep],rcond=None)[0]))
    mse=float(e[:,0]@e[:,0]/df)
    cooks=e[:,0]**2*h/(x.shape[1]*mse*(1-h)**2)
    row=dict(estimate=value,hc3_se=se,t=t,df=df,p_uncorrected=p,
             ci95_low=value-stats.t.ppf(.975,df)*se,ci95_high=value+stats.t.ppf(.975,df)*se,
             loo_min=min(loo),loo_max=max(loo),loo_sign_changes=int(np.sum(np.sign(loo)!=np.sign(value))),
             cooks_flags=int(np.sum(cooks>4/len(y))),max_cooks=float(max(cooks)))
    return row,cooks,loo


def joint_age_test(x,y):
    # Two linearly independent within-participant partner contrasts. Preserve
    # their residual covariance; do not treat 141 observations as independent.
    d=y@np.array([OUTCOMES['human_minus_computer'],OUTCOMES['similar_minus_dissimilar']]).T
    beta,e,h,pinv=fit_hc3(x,d)
    influences=pinv[1,:,None]*e/(1-h[:,None])
    cov=influences.T@influences
    if np.linalg.matrix_rank(cov)!=2: raise ValueError('singular interaction covariance')
    w=float(beta[1]@np.linalg.solve(cov,beta[1]))
    return dict(wald_chi2=w,df=2,p_uncorrected=float(stats.chi2.sf(w,2)),
                reference_distribution='asymptotic chi-square; HC3 subject-vector sandwich')


def holm(values):
    values=np.asarray(values,float); order=np.argsort(values)
    adjusted=np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values))))
    result=np.empty_like(values); result[order]=np.minimum(adjusted,1)
    return result


def make_design(subjects,sample,rt,adjusted):
    older=np.array([float(sample[s]['older']) for s in subjects])
    if sorted(np.bincount(older.astype(int)).tolist()) != [22,25]:
        raise ValueError('expected 25 younger and 22 older participants')
    values=[np.ones(len(subjects)),older-older.mean()]
    names=['intercept','older_centered']
    if adjusted:
        for key in ('ismale','tsnr','fd_mean','corrected_mean_rt_z'):
            value=np.array([float((rt if key=='corrected_mean_rt_z' else sample)[s][key]) for s in subjects])
            values.append(value-value.mean()); names.append(key)
    x=np.column_stack(values)
    if not np.isfinite(x).all() or np.linalg.matrix_rank(x)!=x.shape[1]:
        raise ValueError('rank-deficient/nonfinite nuisance design')
    return x,names,older


def fixed_roi(mask, image):
    data=mask.get_fdata()
    if len(mask.shape)!=3 or not np.isin(data,[0,1]).all() or not data.any():
        raise ValueError('ROI must be nonempty and binary')
    same=mask.shape==image.shape and np.allclose(mask.affine,image.affine,atol=1e-5,rtol=0)
    roi=(data if same else resample_from_to(mask,image,order=0).get_fdata())>0
    if not roi.any(): raise ValueError('ROI outside image grid')
    return roi,not same


def extract_mean(cope, support, mask):
    image=nib.load(cope); coverage=nib.load(support)
    if len(image.shape)!=3 or image.shape!=coverage.shape or not np.allclose(image.affine,coverage.affine,atol=1e-5,rtol=0):
        raise ValueError(f'COPE/support grid mismatch: {cope}')
    roi,resampled=fixed_roi(mask,image)
    covered=coverage.get_fdata()[roi]
    if not np.isfinite(covered).all() or not np.all(covered>0):
        raise ValueError(f'incomplete fixed vmPFC coverage: {cope}; do not silently trim mask')
    values=image.get_fdata()[roi]
    if not np.isfinite(values).all(): raise ValueError(f'nonfinite ROI: {cope}')
    return float(values.mean()),int(roi.sum()),resampled


def validate_l1(row,provenance):
    fsf=Path(row['fsf']); expected=row['fsf_sha256']
    if digest(fsf)!=expected: raise ValueError(f'prepared FSF changed: {fsf}')
    provenance[str(fsf)]=expected
    text=fsf.read_text(); conpath=Path(row['output'])/'design.con'
    matrix=read_vest_matrix(conpath); provenance[str(conpath)]=digest(conpath)
    for _,cope,title in CONDITIONS:
        if fsf_value(text,f'conname_real.{cope}')!=title or fsf_value(text,f'conname_orig.{cope}')!=title:
            raise ValueError(f'offer-slope contrast label mismatch: {fsf}, cope {cope}')
        expected_row=np.array([float(fsf_value(text,f'con_real{cope}.{j}')) for j in range(1,matrix.shape[1]+1)])
        if not np.allclose(matrix[cope-1],expected_row,atol=1e-6,rtol=0):
            raise ValueError(f'compiled L1 contrast mismatch: {conpath}, cope {cope}')
        n=int(fsf_value(text,'evs_orig'))
        orig=np.array([float(fsf_value(text,f'con_orig{cope}.{j}')) for j in range(1,n+1)])
        target=np.zeros(n);target[cope-1]=1
        if not np.array_equal(orig,target): raise ValueError('unexpected offer-slope contrast weights')


def load_sample(repo,provenance):
    sp=repo/'behavioral_analyses/data/participant_L3_47.csv'
    rp=repo/'results/reviewer/tables/l3_event_corrected_covariates.tsv'
    rows=read(sp,',');rts=read(rp)
    sample={r['subjID']:r for r in rows};rt={r['subjID']:r for r in rts}
    if len(rows)!=47 or len(sample)!=47 or len(rts)!=47 or set(sample)!=set(rt):
        raise ValueError('participant/covariate mismatch')
    for p in (sp,rp): provenance[str(p)]=digest(p)
    return sample,rt


def collect(repo,provenance):
    sample,rt=load_sample(repo,provenance); subjects=sorted(sample)
    maskpath=repo/'masks/roi-VMPFC.nii.gz';provenance[str(maskpath)]=digest(maskpath)
    if provenance[str(maskpath)]!=ROI_SHA256: raise ValueError('historical vmPFC mask changed')
    mask=nib.load(maskpath); cfgpath=repo/'results/permutation_audit/v1/config.json'
    cfg=json.loads(cfgpath.read_text());provenance[str(cfgpath)]=digest(cfgpath)
    locked={m['name']:m for m in cfg['models']}
    records=[]
    for timing,relative in REPORTS.items():
        manifest=repo/relative/'revision_jobs.tsv';provenance[str(manifest)]=digest(manifest)
        jobs=read(manifest); l2=[r for r in jobs if r['stage']=='l2' and r['model']=='act']
        l1=[r for r in jobs if r['stage']=='l1' and r['model']=='act']
        if len(l2)!=47 or {r['subject'] for r in l2}!=set(subjects): raise ValueError('L2 sample mismatch')
        if len(l1)!=94 or {(r['subject'],r['run']) for r in l1}!={(s,r) for s in subjects for r in ('01','02')}:
            raise ValueError('L1 run sample mismatch')
        for row in l1: validate_l1(row,provenance)
        model=locked[timing+'__activation-norm']
        if len(model['subjects'])!=47 or set(model['subjects'])!=set(subjects):
            raise ValueError('retained activation sample mismatch')
        roots={s:Path(p).parents[2] for s,p in zip(model['subjects'],model['images'])}
        # Check the intended nuisance policy against retained activation models.
        if set(model['nuisance'])!={'ismale','tsnr','fd_mean','RT'}:
            raise ValueError('unexpected activation nuisance columns')
        for key,values in model['nuisance'].items():
            expected=[float(rt[s]['corrected_mean_rt_z'] if key=='RT' else sample[s][key]) for s in model['subjects']]
            if not np.allclose(values,expected,atol=1e-5,rtol=0): raise ValueError('activation covariate mismatch')
        for row in sorted(l2,key=lambda r:r['subject']):
            subject=row['subject']; directory=Path(row['output'])
            if roots[subject]!=directory: raise ValueError('L2 output differs from retained activation inputs')
            if digest(row['fsf'])!=row['fsf_sha256']: raise ValueError('L2 prepared FSF changed')
            provenance[row['fsf']]=row['fsf_sha256']
            for partner,cope,_ in CONDITIONS:
                image=directory/f'cope{cope}.feat/stats/cope1.nii.gz'
                support=directory/f'cope{cope}.feat/mask.nii.gz'
                value,nvox,resampled=extract_mean(image,support,mask)
                for path in (image,support): provenance[str(path)]=digest(path)
                records.append(dict(timing=timing,subject=subject,partner=partner,cope=cope,
                    mean_cope=value,roi_voxels=nvox,mask_resampled=int(resampled),
                    age_group='older' if float(sample[subject]['older']) else 'younger',
                    input=str(image),input_sha256=provenance[str(image)]))
        print(f'PASS: {timing}, 47 participants × 3 vmPFC offer slopes',flush=True)
    # A common grid/voxel set is required across every participant and model.
    grids={(nib.load(r['input']).shape,tuple(nib.load(r['input']).affine.ravel())) for r in records}
    if len(grids)!=1: raise ValueError('participant/model grids differ; no varying ROI supports allowed')
    return subjects,sample,rt,records


def analyze(subjects,sample,rt,records,output):
    tests=[]; joint=[]; bars=[]; details=[]; influence=[]; designs=[]
    for timing in REPORTS:
        lookup={(r['subject'],r['partner']):r['mean_cope'] for r in records if r['timing']==timing}
        y=np.array([[lookup[s,p] for p,_,_ in CONDITIONS] for s in subjects])
        for adjusted in (False,True):
            policy='covariate-adjusted' if adjusted else 'unadjusted'
            x,names,older=make_design(subjects,sample,rt,adjusted)
            for s,row in zip(subjects,x): designs.append(dict(timing=timing,adjustment=policy,subject=s,**dict(zip(names,row))))
            beta,_,_,_=fit_hc3(x,y)
            plotted=y-x[:,2:]@beta[2:] if adjusted else y.copy()
            for i,(partner,_,_) in enumerate(CONDITIONS):
                for group,label in ((0,'younger'),(1,'older')):
                    values=plotted[older==group,i]
                    bars.append(dict(timing=timing,adjustment=policy,partner=partner,age_group=label,
                        n=len(values),mean=float(values.mean()),participant_sem=float(values.std(ddof=1)/np.sqrt(len(values))),
                        error_definition='SD of plotted participant values / sqrt(n); descriptive, not model SE'))
                for k,s in enumerate(subjects): details.append(dict(timing=timing,adjustment=policy,subject=s,
                    partner=partner,raw_cope=y[k,i],plotted_cope=plotted[k,i]))
            for outcome,weights in OUTCOMES.items():
                values=y@np.array(weights)
                for effect,index in (('sample_average',0),('older_minus_younger',1)):
                    c=np.zeros(x.shape[1]);c[index]=1
                    test,cooks,loo=estimate(x,values,c)
                    if effect=='sample_average': family='partner-difference' if 'minus' in outcome else 'offer-slope-vs-zero'
                    else: family='age-interaction-components' if 'minus' in outcome else ('age-average' if outcome=='partner_average' else 'age-by-condition')
                    tests.append(dict(timing=timing,adjustment=policy,outcome=outcome,effect=effect,
                        family=family,n=47,**test))
                    for s,cook,value in zip(subjects,cooks,loo): influence.append(dict(timing=timing,
                        adjustment=policy,outcome=outcome,effect=effect,subject=s,cooks=cook,leave_one_out_estimate=value))
            joint.append(dict(timing=timing,adjustment=policy,n=47,**joint_age_test(x,y)))
    for family in {r['family'] for r in tests}:
        selected=[r for r in tests if r['family']==family]
        for row,p in zip(selected,holm([r['p_uncorrected'] for r in selected])):
            row['p_holm_family_across_timings_and_adjustments']=float(p)
    for row,p in zip(joint,holm([r['p_uncorrected'] for r in joint])): row['p_holm_six_omnibus_tests']=float(p)
    write(output/'roi_tests.tsv',tests);write(output/'age_partner_omnibus.tsv',joint)
    write(output/'bar_source.tsv',bars);write(output/'plotted_participants.tsv',details)
    write(output/'influence.tsv',influence)
    # The two designs differ in width; retain one complete adjusted design.
    write(output/'adjusted_design.tsv',[r for r in designs if r['adjustment']=='covariate-adjusted'])
    for policy in ('unadjusted','covariate-adjusted'):
        for timing in REPORTS:
            fig,ax=plt.subplots(figsize=(7,5))
            for i,(group,color) in enumerate((('younger','#d9b526'),('older','#318672'))):
                subset=[next(r for r in bars if r['timing']==timing and r['adjustment']==policy and r['partner']==p and r['age_group']==group) for p,_,_ in CONDITIONS]
                ax.bar(np.arange(3)+(i-.5)*.34,[r['mean'] for r in subset],width=.32,
                       yerr=[r['participant_sem'] for r in subset],capsize=4,color=color,
                       label=f'{group.capitalize()} (n={subset[0]["n"]})')
            ax.axhline(0,color='.3',lw=.8);ax.spines[['top','right']].set_visible(False)
            ax.set_xticks(np.arange(3),['Computer','Similar','Dissimilar'])
            ax.set_ylabel('vmPFC offer-size slope\n(mean COPE; model units)')
            ax.set_title('vmPFC responses to offer size\n'+LABELS[timing]);ax.legend(frameon=False)
            fig.tight_layout(rect=(0,.13,1,1))
            fig.text(.10,.035,f'{policy.capitalize()} means ± participant SEM. Fixed historical ROI; N = 47.\n'
                     'SEM describes participant dispersion; inferential tests use HC3 uncertainty.',fontsize=8)
            for ext in ('png','svg'): save_figure(fig,output/f'vmpfc_{timing}_{policy}.{ext}')
            plt.close(fig)


def run(repo,output,execute):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        provenance={str(Path(__file__).resolve()):digest(__file__)}
        for module in ('plot_three_partner_ratings.py','audit_l1_designs.py','prepare_sub144_imaging_repair.py'):
            path=repo/'code'/module;provenance[str(path)]=digest(path)
        subjects,sample,rt,records=collect(repo,provenance)
        contract={'inputs_and_code':provenance,'timings':list(REPORTS),'n':47,
            'mask_policy':'fixed original ROI; nearest-neighbour only if needed; full coverage required',
            'inference':'participant contrasts; unweighted OLS with HC3; no automatic exclusions',
            'plot':'raw and nuisance-adjusted participant mean +/- descriptive SEM; no VARCOPE weighting'}
        cp=output/'contract.json'
        if (output/'complete.json').exists() and not cp.exists():
            raise ValueError('completed outputs lack their input contract')
        if cp.exists() and json.loads(cp.read_text())!=contract:
            raise ValueError('inputs/code changed; preserve this output and use a new version directory')
        if not execute:
            print('PASS: vmPFC preflight; 423 condition estimates available; no analysis outputs replaced')
            return
        receipt=output/'complete.json'
        if receipt.exists():
            for name,checksum in json.loads(receipt.read_text())['outputs'].items():
                if digest(output/name)!=checksum: raise ValueError(f'completed output changed: {name}')
            print('SKIP_COMPLETE: verified vmPFC outputs');return
        save_json(cp,contract)
        save_json(output/'status.json',{'complete':False})
        write(output/'participant_extractions.tsv',records)
        analyze(subjects,sample,rt,records,output)
        save_json(output/'status.json',{'complete':True,'n':47,'exclusions':0,'palm_dependency':False,
            'analysis':'fixed historical vmPFC ROI; three corrected timing models; not a voxelwise PALM analysis'})
        save_json(receipt,{'outputs':{str(p.relative_to(output)):digest(p) for p in output.iterdir()
            if p.is_file() and p.name not in ('.lock','complete.json')}})
        print(f'PASS: corrected vmPFC analyses and six bar figures saved: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/vmpfc_ratings/v1/vmpfc')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();run(ROOT,args.output.resolve(),args.execute)
