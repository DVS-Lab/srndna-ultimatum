#!/usr/bin/env python3
"""Descriptive overlays and selected-ROI diagnostics; no new imaging inference."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import label
from audit_l1_designs import read_vest_matrix
from prepare_ultimatum_l3_repair import parse_inputs

ROOT=Path(__file__).resolve().parents[1]
TIMINGS=('full-rt','pooled-post','partner-post')
LABELS=('Full display + RT height','Decision + pooled post-response','Decision + partner post-response')
METHODS=('tfce','cluster-t2.6','cluster-t3.1')
TARGETS={
    'activation-age-interaction': dict(metric='acceptance-intercept',contrast=1,
        title='Activation: age difference in the brain–behavior association',
        behavior='Similar − dissimilar acceptance intercept',
        contrast_label='Older − younger behavioral slope'),
    'activation-fairness-older': dict(metric='choice-slope',contrast=8,
        title='Activation: negative brain–behavior association in older adults',
        behavior='Similar − dissimilar choice sensitivity to offer size',
        contrast_label='Negative older-adult behavioral slope'),
}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f: return list(csv.DictReader(f,delimiter='\t'))
def table(path, values):
    if not values: raise ValueError(f'empty result: {path}')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]),delimiter='\t'); w.writeheader(); w.writerows(values)
def same_grid(a,b):
    if a.shape!=b.shape or not np.allclose(a.affine,b.affine,atol=1e-5,rtol=0): raise ValueError('image grids differ')

def diagnostic(x,y,c):
    """Unweighted full-design OLS and fixed-ROI leave-one-out diagnostics."""
    x,y,c=np.asarray(x,float),np.asarray(y,float),np.asarray(c,float)
    n,p=x.shape
    if n<=p+1 or np.linalg.matrix_rank(x)!=p or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid diagnostic design or values')
    inverse=np.linalg.pinv(x); beta=inverse@y; error=y-x@beta
    h=np.einsum('ij,ji->i',x,inverse); mse=float(error@error/(n-p))
    if mse<=0 or np.any(h>=1-1e-10): raise ValueError('degenerate residual diagnostics')
    cooks=error**2*h/(p*mse*(1-h)**2)
    student=error/np.sqrt(mse*(1-h))
    loo=[]
    for i in range(n):
        keep=np.arange(n)!=i
        if np.linalg.matrix_rank(x[keep])!=p: raise ValueError('leave-one-out rank loss')
        loo.append(float(c@np.linalg.lstsq(x[keep],y[keep],rcond=None)[0]))
    return dict(beta=beta,residual=error,leverage=h,cooks=cooks,studentized=student,
                full=float(c@beta),loo=np.array(loo),adjusted=y-x[:,:4]@beta[:4])

def aligned_design(directory, expected_subjects):
    text=(directory/'design.fsf').read_text(); inputs=parse_inputs(text.splitlines())
    order=[s for _,s,_ in inputs]
    if len(order)!=47 or len(set(order))!=47 or set(order)!=set(expected_subjects):
        raise ValueError('participant mismatch; positional alignment forbidden')
    x=read_vest_matrix(directory/'design.mat')
    c=read_vest_matrix(directory/'design.con')
    if x.shape!=(47,8): raise ValueError('expected original crossed 47x8 design')
    return x[[order.index(s) for s in expected_subjects]],c

def load_payload(report, mapping, model, contrast, method, inventory, provenance):
    matches=[r for r in mapping if r['model']==model and int(r['contrast'])==contrast and r['method']==method]
    if len(matches)!=1: raise ValueError('ambiguous or absent model/contrast/method')
    row=matches[0]; folder=report/'maps'/row['test']/method
    stem='tfce' if method=='tfce' else 'clustere'
    paths=[folder/'randomise_tstat1.nii.gz',folder/f'randomise_{stem}_corrp_tstat1.nii.gz']
    for path in paths:
        checksum=digest(path)
        if inventory[str(path.relative_to(report))]!=checksum: raise ValueError(f'map hash mismatch: {path}')
        provenance[str(path)]=checksum
    stat,prob=[nib.load(p) for p in paths]; same_grid(stat,prob)
    mp=report/'designs'/row['group']/'mask.nii.gz'
    mask=nib.load(mp); same_grid(stat,mask); provenance[str(mp)]=digest(mp)
    values=stat.get_fdata(); probabilities=prob.get_fdata()
    if not np.isfinite(values).all() or not np.isfinite(probabilities).all(): raise ValueError('nonfinite map')
    sig=(probabilities>.95)&(values>0)&(mask.get_fdata()>0)
    if int(sig.sum())!=int(row['significant_voxels']): raise ValueError('summary/map mismatch')
    return dict(row=row,stat=stat,sig=sig,values=values)

def slice_plane(data,affine,axis,coordinate):
    steps=np.diag(affine)[:3]; origin=affine[:3,3]
    if np.any(steps<=0) or not np.allclose(affine[:3,:3],np.diag(steps)):
        raise ValueError('reference must be canonical axis-aligned RAS')
    index=int(np.rint((coordinate-origin[axis])/steps[axis]))
    if not 0<=index<data.shape[axis]: raise ValueError('cut outside reference')
    other=[a for a in range(3) if a!=axis]
    extent=[v for a in other for v in (origin[a]-steps[a]/2,origin[a]+steps[a]*(data.shape[a]-.5))]
    return np.take(data,index,axis=axis).T,extent,origin[axis]+index*steps[axis]

def overlays(target, payloads, flames, background, cut, output):
    anatomy=background.get_fdata(); vmax=np.percentile(anatomy[anatomy>0],99.5)
    cuts=[]
    for method in METHODS:
        shared=np.logical_and.reduce([payloads[t,method]['sig'] for t in TIMINGS])
        chosen=cut
        rule='t=3.1 shared extraction ROI (no shared mask for this method)'
        if shared.any():
            labs,_=label(shared,structure=np.ones((3,3,3)));counts=np.bincount(labs.ravel());counts[0]=0
            chosen=nib.affines.apply_affine(payloads[TIMINGS[0],method]['stat'].affine,np.argwhere(labs==np.argmax(counts)).mean(0))
            rule=f'shared {method} mask, largest-component centroid'
        cuts.append(dict(method=method,cut_x=float(chosen[0]),cut_y=float(chosen[1]),cut_z=float(chosen[2]),selection=rule))
        fig,axes=plt.subplots(3,3,figsize=(12,10),facecolor='white')
        fig.subplots_adjust(left=.05,right=.96,bottom=.12,top=.86,wspace=.03,hspace=.35)
        fig.suptitle(target['title']+'\n'+method+'; spatial FWE p < .05',fontsize=15,y=.98)
        for i,timing in enumerate(TIMINGS):
            item=payloads[timing,method]
            current=nib.Nifti1Image(np.where(item['sig'],item['values'],0).astype('float32'),item['stat'].affine)
            current=resample_from_to(current,background,order=0).get_fdata()
            old=resample_from_to(flames[timing],background,order=0).get_fdata()>0
            for axis in range(3):
                ax=axes[i,axis]
                base,extent,actual=slice_plane(anatomy,background.affine,axis,chosen[axis])
                values,_,_=slice_plane(current,background.affine,axis,chosen[axis])
                prior,_,_=slice_plane(old,background.affine,axis,chosen[axis])
                ax.imshow(base,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=vmax)
                ax.imshow(np.ma.masked_where(values==0,values),origin='lower',extent=extent,cmap='autumn',vmin=0,vmax=6,interpolation='nearest')
                if prior.any() and not prior.all():
                    ax.contour(prior.astype(float),levels=[.5],origin='lower',extent=extent,colors=['#00c8e0'],linewidths=.8)
                ax.set_title(f'{LABELS[i]}\n{"xyz"[axis]} = {actual:.0f} mm',fontsize=10)
                ax.set_xticks([]);ax.set_yticks([])
                ax.text(.02,.05,'P' if axis==0 else 'L',transform=ax.transAxes,color='white',fontsize=9)
                ax.text(.95,.05,'A' if axis==0 else 'R',transform=ax.transAxes,color='white',fontsize=9)
            axes[i,0].set_ylabel(f'{int(item["sig"].sum())} voxels; min p={float(item["row"]["minimum_spatial_fwe_p"]):.4f}',fontsize=9)
        cb=fig.colorbar(ScalarMappable(norm=Normalize(0,6),cmap='autumn'),cax=fig.add_axes([.72,.07,.23,.014]),orientation='horizontal')
        cb.set_label('Permutation t statistic (not Z)',fontsize=9)
        fig.text(.05,.035,'Cyan: corresponding significant FLAME regions. Display resampling only.\nCuts: '+rule+'.\nExploratory tests; no correction across the model search.',fontsize=8)
        fig.savefig(output/f'overlay_{method}.png',dpi=160);plt.close(fig)
    table(output/'display_cuts.tsv',cuts)

def extract(model, roi, reference, provenance, image_hashes):
    y=[]
    for subject,path in zip(model['subjects'],model['images']):
        p=Path(path)
        if not p.is_file(): raise FileNotFoundError(f'{subject}: missing retained L2 COPE: {p}')
        value=digest(p)
        if image_hashes.get(str(p))!=value: raise ValueError(f'COPE differs from permutation input: {p}')
        provenance[str(p)]=value
        im=nib.load(p);same_grid(im,reference); data=im.get_fdata()
        if not np.isfinite(data[roi]).all(): raise ValueError(f'nonfinite ROI values: {p}')
        y.append(float(data[roi].mean()))
    return np.array(y)

def plots_and_tables(target, diagnostics, output):
    colors=('#c26a20','#2468a2'); detail=[]; summaries=[]
    fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True)
    fig.subplots_adjust(top=.88,bottom=.16,hspace=.38,wspace=.30)
    fig.suptitle(target['title'],fontsize=15,y=.98)
    fig.text(.06,.925,'Left: observed ROI means. Right: adjusted to zero on the four retained nuisance columns.',fontsize=10)
    loofig,looaxes=plt.subplots(1,3,figsize=(13,4.8))
    for i,timing in enumerate(TIMINGS):
        subjects,x,y,d=diagnostics[timing]; behavior=x[:,6]+x[:,7]
        for j,values in enumerate((y,d['adjusted'])):
            ax=axes[i,j]
            for group in (0,1):
                keep=x[:,4+group]>.5
                ax.scatter(behavior[keep],values[keep],s=27,color=colors[group],alpha=.8,label=('Younger','Older')[group])
                if j==1:
                    xx=np.linspace(behavior[keep].min(),behavior[keep].max(),100)
                    ax.plot(xx,d['beta'][4+group]+xx*d['beta'][6+group],color=colors[group])
            ax.axhline(0,color='.7',lw=.6);ax.axvline(0,color='.7',lw=.6)
            ax.set_title(LABELS[i]+(' — raw' if j==0 else ' — adjusted'),fontsize=11)
            ax.set_ylabel('Similar − dissimilar offer-slope COPE7\nROI mean (model units)',fontsize=9)
            if i==2: ax.set_xlabel(target['behavior']+'\n(sample-centered)',fontsize=9)
            ax.spines[['top','right']].set_visible(False)
            if i==0 and j==1: ax.legend(frameon=False)
        flagged=np.where(d['cooks']>4/len(y))[0]
        for k in sorted(flagged,key=lambda k:d['cooks'][k],reverse=True)[:3]:
            axes[i,1].annotate(subjects[k].removeprefix('sub-'),(behavior[k],d['adjusted'][k]),xytext=(4,3),textcoords='offset points',fontsize=8)
        for k,s in enumerate(subjects):
            detail.append(dict(timing=timing,participant=s,age_group='older' if x[k,5] else 'younger',behavior_centered=behavior[k],
                raw_cope7=y[k],adjusted_cope7=d['adjusted'][k],fitted_cope7=float(x[k]@d['beta']),
                leverage=d['leverage'][k],cooks_distance=d['cooks'][k],internally_studentized_residual=d['studentized'][k],
                cooks_above_4_over_n=int(d['cooks'][k]>4/len(y)),loo_contrast=d['loo'][k]))
        summaries.append(dict(timing=timing,n=len(y),rank=int(np.linalg.matrix_rank(x)),contrast=target['contrast_label'],estimate=d['full'],
            younger_behavior_slope=d['beta'][6],older_behavior_slope=d['beta'][7],cooks_flags=len(flagged),
            max_cooks=float(d['cooks'].max()),loo_min=float(d['loo'].min()),loo_max=float(d['loo'].max()),
            loo_sign_changes=int(np.sum(np.sign(d['loo'])!=np.sign(d['full']))),excluded_participants=0,
            interpretation='Selected-ROI unweighted OLS diagnostic, not independent inference or FLAME weighting'))
        ax=looaxes[i]; ax.scatter(np.arange(len(y))+1,d['loo'],s=18,c=[colors[int(v)] for v in x[:,5]])
        ax.axhline(d['full'],color='black',label='Full-sample estimate');ax.axhline(0,color='.7',ls=':')
        ax.set_title(LABELS[i],fontsize=10);ax.set_xlabel('Omitted participant (ID-sorted index)')
        ax.set_ylabel(target['contrast_label']);ax.spines[['top','right']].set_visible(False)
    footer='Descriptive, data-selected ROI: intersection of significant t=3.1 cluster maps across the three timings.\nOLS lines use the full eight-column group design; no inferential p values or confidence bands. No participants excluded.'
    fig.text(.06,.045,footer,fontsize=9);fig.savefig(output/'brain_behavior.png',dpi=170);plt.close(fig)
    loofig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.25,wspace=.4)
    loofig.suptitle('Fixed-ROI leave-one-out diagnostics',fontsize=14)
    loofig.text(.05,.04,'The ROI remains selected from all participants. This is not cross-validation or a leave-one-out whole-brain analysis.\nParticipant identities and flags are in participant_extractions.tsv; no automatic exclusions.',fontsize=9)
    loofig.savefig(output/'influence.png',dpi=170);plt.close(loofig)
    table(output/'participant_extractions.tsv',detail);table(output/'diagnostic_summary.tsv',summaries)

def build(report, output, standard, overlays_only=False):
    status=json.loads((report/'status.json').read_text())
    if not status.get('complete'): raise ValueError('permutation batch incomplete')
    config=json.loads((report/'config.json').read_text()); mapping=rows(report/'model_contrast_summary.tsv')
    inventory={r['artifact']:r['sha256'] for r in rows(report/'map_inventory.tsv')}
    input_hashes=json.loads((report/'input_provenance.json').read_text())
    provenance={str(p):digest(p) for p in (Path(__file__),report/'config.json',report/'model_contrast_summary.tsv',report/'map_inventory.tsv',standard)}
    background=nib.as_closest_canonical(nib.load(standard))
    contract=dict(code_sha256=digest(__file__),config_sha256=digest(report/'config.json'),
                  summary_sha256=digest(report/'model_contrast_summary.tsv'),standard_sha256=digest(standard),
                  roi_rule='intersection all three timings at cluster-t3.1, p<.05, positive signed t',overlays_only=overlays_only)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=contract:
        raise ValueError('changed source/settings: choose a new output directory')
    (output/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(dict(complete=False,stage='building'))+'\n')
    rois=[]
    for name,target in TARGETS.items():
        folder=output/name;folder.mkdir(exist_ok=True); payloads={};flames={};models={}
        for timing in TIMINGS:
            model=f'crossed__{timing}_act_{target["metric"]}'
            models[timing]=next(m for m in config['models'] if m['name']==model)
            for method in METHODS:
                payloads[timing,method]=load_payload(report,mapping,model,target['contrast'],method,inventory,provenance)
            old=ROOT/'results/crossed_behavior_l3/v1/imaging'/model.removeprefix('crossed__')/f'contrast{target["contrast"]}'/f'cluster_mask_zstat{target["contrast"]}.nii.gz'
            old_inventory={r['artifact']:r['sha256'] for r in rows(ROOT/'results/crossed_behavior_l3/v1/imaging/inventory.tsv')}
            if old_inventory[str(old.relative_to(ROOT/'results/crossed_behavior_l3/v1/imaging'))]!=digest(old): raise ValueError('FLAME mask hash mismatch')
            flames[timing]=nib.load(old);provenance[str(old)]=digest(old)
        reference=payloads[TIMINGS[0],'cluster-t3.1']['stat']
        for item in payloads.values(): same_grid(reference,item['stat'])
        roi=np.logical_and.reduce([payloads[t,'cluster-t3.1']['sig'] for t in TIMINGS])
        if not roi.any(): raise ValueError(f'no shared ROI for {name}; no replacement region selected')
        # Fixed cuts at the centroid of the largest connected part of the shared mask.
        components,n=label(roi,structure=np.ones((3,3,3)));sizes=np.bincount(components.ravel());sizes[0]=0
        cut=nib.affines.apply_affine(reference.affine,np.argwhere(components==np.argmax(sizes)).mean(0))
        nib.save(nib.Nifti1Image(roi.astype('uint8'),reference.affine),folder/'shared_selected_roi.nii.gz')
        rois.append(dict(target=name,voxels=int(roi.sum()),components=int(n),largest_component=int(sizes.max()),
                         cut_x=float(cut[0]),cut_y=float(cut[1]),cut_z=float(cut[2]),selection=contract['roi_rule']))
        overlays(target,payloads,flames,background,cut,folder)
        if not overlays_only:
            diagnostics={}
            for timing in TIMINGS:
                model=models[timing];subjects=model['subjects']
                directory=ROOT/'results/crossed_behavior_l3/v1/imaging'/model['name'].removeprefix('crossed__')/'design'
                for p in (directory/'design.fsf',directory/'design.mat',directory/'design.con'):
                    expected=[v for k,v in model['source_hashes'].items() if k.endswith(str(p.relative_to(ROOT)))]
                    if expected!=[digest(p)]: raise ValueError(f'original design hash mismatch: {p}')
                    provenance[str(p)]=digest(p)
                x,c=aligned_design(directory,subjects)
                y=extract(model,roi,reference,provenance,input_hashes)
                diagnostics[timing]=(subjects,x,y,diagnostic(x,y,c[target['contrast']-1]))
            plots_and_tables(target,diagnostics,folder)
        print(f'PASS: {name}: {roi.sum()} selected ROI voxels; overlays'+(' only' if overlays_only else ', 141 participant extractions, and diagnostics'),flush=True)
    table(output/'roi_definitions.tsv',rois)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(dict(complete=True,overlays_only=overlays_only,
        inference='none added; selected-ROI descriptive OLS, no exclusions'),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,default=ROOT/'results/permutation_audit/v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--standard',type=Path,default=Path(os.environ.get('FSLDIR','/nonexistent'))/'data/standard/MNI152_T1_2mm_brain.nii.gz')
    p.add_argument('--overlays-only',action='store_true')
    a=p.parse_args();build(a.report.resolve(),a.output.resolve(),a.standard.resolve(),a.overlays_only)

if __name__=='__main__': main()
