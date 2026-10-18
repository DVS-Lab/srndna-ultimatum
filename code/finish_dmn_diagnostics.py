#!/usr/bin/env python3
"""Standardized DMN overlays and descriptive fixed-selected-ROI influence checks.

No participant exclusion, FLAME refit, or independent ROI hypothesis test.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import label
import plot_permutation_followup as pf
from audit_l1_designs import read_vest_matrix
from prepare_ultimatum_l3_repair import parse_inputs

ROOT = Path(__file__).resolve().parents[1]


def hc3_se(x, y, c):
    inv = np.linalg.pinv(x)
    residual = y - x @ (inv @ y)
    leverage = np.einsum('ij,ji->i', x, inv)
    return float(np.sqrt(np.sum(((c @ inv) * residual / (1-leverage))**2)))


def center(mask, affine):
    components, _ = label(mask, structure=np.ones((3,3,3)))
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    return nib.affines.apply_affine(affine, np.argwhere(components == sizes.argmax()).mean(0))


def build(source, output, standard, overlays_only=False):
    report = source/'randomise'
    if not json.loads((source/'status.json').read_text())['complete']:
        raise ValueError('standardized batch incomplete')
    config = json.loads((report/'config.json').read_text())
    mapping = pf.rows(report/'model_contrast_summary.tsv')
    inventory = {r['artifact']:r['sha256'] for r in pf.rows(report/'map_inventory.tsv')}
    flame_inventory = {r['artifact']:r['sha256'] for r in pf.rows(source/'imaging/inventory.tsv')}
    hashes = json.loads((report/'input_provenance.json').read_text())
    provenance = {str(p):pf.digest(p) for p in (Path(__file__), Path(pf.__file__), standard,
        report/'config.json', report/'model_contrast_summary.tsv', report/'map_inventory.tsv',
        source/'imaging/inventory.tsv', report/'input_provenance.json')}
    contract = dict(sources=provenance.copy(), roi_rule='union of significant FLAME and three randomise masks across all three timings, separately per contrast; fixed before leave-one-out')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('sources changed: use a new report directory')
    (output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    (output/'status.json').write_text('{"complete": false}\n')
    background = nib.as_closest_canonical(nib.load(standard)); anatomy = background.get_fdata()
    summaries=[]; details=[]; regions=[]; cuts=[]; component_summaries=[]; component_details=[]
    for contrast, name in ((1,'younger-positive'), (3,'younger-minus-older')):
        folder=output/name; folder.mkdir(exist_ok=True)
        payload={}; flames={}; models={}; union=None
        for timing in pf.TIMINGS:
            model=next(m for m in config['models'] if m['name']==f'{timing}__dmn-age')
            models[timing]=model
            for method in pf.METHODS:
                item=pf.load_payload(report,mapping,model['name'],contrast,method,inventory,provenance)
                payload[timing,method]=item
                union=item['sig'].copy() if union is None else union | item['sig']
            path=source/'imaging'/model['name']/f'contrast{contrast}'/f'cluster_mask_zstat{contrast}.nii.gz'
            if pf.digest(path)!=flame_inventory[str(path.relative_to(source/'imaging'))]:
                raise ValueError('FLAME mask hash mismatch')
            provenance[str(path)]=pf.digest(path)
            image=nib.load(path); pf.same_grid(image,item['stat'])
            flames[timing]=image; union |= image.get_fdata()>0
        reference=payload[pf.TIMINGS[0],'tfce']['stat']
        for item in payload.values(): pf.same_grid(reference,item['stat'])
        if not union.any():
            regions.append(dict(target=name,voxels=0,status='no significant regions; no replacement ROI'))
            continue
        nib.save(nib.Nifti1Image(union.astype('uint8'),reference.affine),folder/'selected_union_roi.nii.gz')
        components,ncomponents=label(union,structure=np.ones((3,3,3)))
        nib.save(nib.Nifti1Image(components.astype('int16'),reference.affine),folder/'selected_union_components.nii.gz')
        regions.append(dict(target=name,voxels=int(union.sum()),status='selected descriptive ROI, not independent'))
        for method in pf.METHODS:
            fig,axes=pf.plt.subplots(3,3,figsize=(11,10))
            for row,timing in enumerate(pf.TIMINGS):
                item=payload[timing,method]; fl=flames[timing].get_fdata()>0
                region=item['sig'] | fl
                chosen=center(region if region.any() else union,reference.affine)
                cuts.append(dict(target=name,timing=timing,method=method,x=chosen[0],y=chosen[1],z=chosen[2],
                    rule='largest component of this row FLAME/permutation union; global union fallback if empty',
                    flame_voxels=int(fl.sum()),permutation_voxels=int(item['sig'].sum()),overlap_voxels=int((fl&item['sig']).sum())))
                current=resample_from_to(nib.Nifti1Image(np.where(item['sig'],item['values'],0),reference.affine),background,order=0).get_fdata()
                contour=resample_from_to(flames[timing],background,order=0).get_fdata()>0
                for axis in range(3):
                    ax=axes[row,axis]
                    base,extent,actual=pf.slice_plane(anatomy,background.affine,axis,chosen[axis])
                    val,_,_=pf.slice_plane(current,background.affine,axis,chosen[axis])
                    edge,_,_=pf.slice_plane(contour,background.affine,axis,chosen[axis])
                    ax.imshow(base,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=np.percentile(anatomy[anatomy>0],99.5))
                    ax.imshow(np.ma.masked_where(val==0,val),origin='lower',extent=extent,cmap='autumn',vmin=0,vmax=6)
                    if edge.any() and not edge.all(): ax.contour(edge,levels=[.5],origin='lower',extent=extent,colors=['cyan'],linewidths=.8)
                    ax.set_title(f'{timing}: {"xyz"[axis]}={actual:.0f}',fontsize=10); ax.set_xticks([]);ax.set_yticks([])
                    ax.text(.02,.02,'P' if axis==0 else 'L',transform=ax.transAxes,color='white')
                    ax.text(.94,.02,'A' if axis==0 else 'R',transform=ax.transAxes,color='white')
            fig.suptitle(f'DMN {name}: {method}\nWarm: randomise t; cyan: FLAME clusters; spatial FWE p<.05')
            fig.text(.05,.02,'Row-specific cuts show detected regions, not evidence of spatial replication.\nSelected descriptive displays; no correction across the model/method search.',fontsize=9)
            fig.tight_layout(rect=[0,.07,1,.93]);fig.savefig(folder/f'overlay_{method}.png',dpi=150);pf.plt.close(fig)
        if not overlays_only:
            fig,axes=pf.plt.subplots(1,3,figsize=(14,4.7))
            for ax,timing in zip(axes,pf.TIMINGS):
                model=models[timing]; x=np.array(model['x']); c=np.array(model['c'])[contrast-1]
                y=pf.extract(model,union,reference,provenance,hashes)
                d=pf.diagnostic(x,y,c)
                # Never use the legacy helper's positional nuisance adjustment for these six-column models.
                summaries.append(dict(target=name,timing=timing,n=len(y),estimate=d['full'],hc3_se=hc3_se(x,y,c),
                    loo_min=float(d['loo'].min()),loo_max=float(d['loo'].max()),
                    loo_sign_changes=int(np.sum(np.sign(d['loo'])!=np.sign(d['full']))),
                    cooks_flags=int(np.sum(d['cooks']>4/len(y))),max_cooks=float(d['cooks'].max()),
                    inference='selected-ROI OLS/HC3 diagnostic, not FLAME or independent inference'))
                for i,subject in enumerate(model['subjects']):
                    details.append(dict(target=name,timing=timing,subject=subject,raw_roi_mean=y[i],
                        leverage=d['leverage'][i],cooks=d['cooks'][i],studentized=d['studentized'][i],
                        leave_one_out_estimate=d['loo'][i],excluded=0))
                # Component-wise checks avoid concealing a local influential case
                # by averaging anatomically distinct detections into one ROI.
                for component in range(1,ncomponents+1):
                    region=components==component
                    cy=pf.extract(model,region,reference,provenance,hashes)
                    cd=pf.diagnostic(x,cy,c)
                    component_summaries.append(dict(target=name,timing=timing,component=component,voxels=int(region.sum()),
                        estimate=cd['full'],hc3_se=hc3_se(x,cy,c),loo_min=float(cd['loo'].min()),loo_max=float(cd['loo'].max()),
                        loo_sign_changes=int(np.sum(np.sign(cd['loo'])!=np.sign(cd['full']))),
                        cooks_flags=int(np.sum(cd['cooks']>4/len(cy))),max_cooks=float(cd['cooks'].max())))
                    for i,subject in enumerate(model['subjects']):
                        component_details.append(dict(target=name,timing=timing,component=component,subject=subject,
                            raw_roi_mean=cy[i],cooks=cd['cooks'][i],leverage=cd['leverage'][i],
                            studentized=cd['studentized'][i],leave_one_out_estimate=cd['loo'][i],excluded=0))
                ax.scatter(np.arange(len(y)),d['loo'],s=18);ax.axhline(d['full'],color='black');ax.axhline(0,color='.6',linestyle=':')
                for i in np.where(d['cooks']>4/len(y))[0]: ax.annotate(model['subjects'][i],(i,d['loo'][i]),fontsize=7,rotation=45)
                ax.set_title(timing);ax.set_xlabel('Omitted participant (sorted ID)');ax.set_ylabel('Fixed-ROI contrast estimate')
            fig.suptitle(f'DMN {name}: leave-one-out influence (no exclusions)')
            fig.text(.02,.01,'ROI selected with all participants; this is not cross-validation or leave-one-out whole-brain inference.',fontsize=9)
            fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(folder/'influence.png',dpi=160);pf.plt.close(fig)
    pf.table(output/'roi_definitions.tsv',regions);pf.table(output/'display_cuts.tsv',cuts)
    if summaries:
        pf.table(output/'diagnostic_summary.tsv',summaries);pf.table(output/'participant_extractions.tsv',details)
        pf.table(output/'component_diagnostics.tsv',component_summaries);pf.table(output/'component_participants.tsv',component_details)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'status.json').write_text(json.dumps(dict(complete=not overlays_only,overlays_complete=True,exclusions=0))+'\n')
    print('PASS: DMN overlays'+(' only' if overlays_only else ' and fixed-ROI influence checks'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'results/dmn_covariate_standardized/v1')
    p.add_argument('--output',type=Path,default=ROOT/'results/dmn_followup/v1/diagnostics')
    p.add_argument('--standard',type=Path,default=Path(os.environ.get('FSLDIR','/nonexistent'))/'data/standard/MNI152_T1_2mm_brain.nii.gz')
    p.add_argument('--overlays-only',action='store_true')
    a=p.parse_args();build(a.source.resolve(),a.output.resolve(),a.standard.resolve(),a.overlays_only)
