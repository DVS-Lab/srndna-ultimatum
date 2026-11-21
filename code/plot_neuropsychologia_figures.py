"""Rebuild Figures 2–4 using saved results. Usage: python SCRIPT REPO MNI_TEMPLATE OUTPUT [CONDITION_SUMMARY_TSV]. The fourth argument overrides the saved Figure 4 aggregate TSV."""
from pathlib import Path
from collections import defaultdict
import csv,json,sys,os
os.environ.setdefault('SOURCE_DATE_EPOCH', '1795219200')
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
repo,template,out=map(Path,sys.argv[1:4]);out.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(repo/'code'))
from plot_current_result_review import Review
from plot_permutation_followup import slice_plane
audit=Review(repo,out,template)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
colors=['#af4f35','#227b8c']
cmap=LinearSegmentedColormap.from_list('signed_orange',['#ffd5a3','#ff9250','#cf312a','#8d1230'])
def read(f):
 with f.open() as h:return list(csv.DictReader(h,delimiter='\t'))
def save(f,n):
 for ext in ['png','pdf']:f.savefig(out/(n+'.'+ext),dpi=350,bbox_inches='tight',pad_inches=.10)
 plt.close(f)
def plane(ax,payload,axis):
 rec,overlay,xyz=payload
 bg,extent,val=slice_plane(audit.anatomy,audit.bg.affine,axis,xyz[axis]); ov,_,_=slice_plane(overlay,audit.bg.affine,axis,xyz[axis])
 ax.imshow(bg,extent=extent,origin='lower',cmap='gray',vmin=0,vmax=audit.vmax)
 ax.imshow(np.ma.masked_equal(ov,0),extent=extent,origin='lower',cmap=cmap,vmin=0,vmax=6,interpolation='nearest')
 ax.set_axis_off();ax.set_title(f"{['Sagittal','Coronal','Axial'][axis]}   {'xyz'[axis]} = {val:.0f} mm",fontsize=9,pad=3)
 for x,t in [(0.035,'P' if axis==0 else 'L'),(.94,'A' if axis==0 else 'R')]:ax.text(x,.04,t,transform=ax.transAxes,color='white',fontsize=8)
def zbar(fig,ax,label='Z in the labeled direction'):
 c=fig.colorbar(ScalarMappable(norm=Normalize(0,6),cmap=cmap),cax=ax,orientation='horizontal',ticks=[0,3,6]);c.outline.set_linewidth(.5);c.ax.tick_params(labelsize=8,length=2);c.set_label(label,size=9)
rows=read(repo/'results/scientific_report/2026-11-02/choice_participant_source.tsv')
by=defaultdict(lambda:np.zeros(2))
for r in rows:by[r['age_group'],r['subject'],int(r['offer'])]+=[int(r['accepted']),int(r['trials'])]
assert sum(a[1] for a in by.values())==4438
fig,ax=plt.subplots(figsize=(5.6,3.5),layout='constrained')
for age,col,mark in [('younger',colors[0],'o'),('older',colors[1],'s')]:
 amounts=np.arange(1,11);samples=[[x[0]/x[1] for (g,s,a),x in by.items() if g==age and a==i] for i in amounts]
 ax.errorbar(amounts,[np.mean(x) for x in samples],yerr=[np.std(x,ddof=1)/np.sqrt(len(x)) for x in samples],color=col,marker=mark,lw=1.7,ms=4.4,capsize=2.6,label=age.capitalize())
ax.set(xticks=amounts,xlim=(.65,10.35),ylim=(-.015,1.06),yticks=[0,.25,.50,.75,1],xlabel='Offer amount ($)',ylabel='Mean proportion accepted')
ax.grid(axis='y',alpha=.19);ax.legend(frameon=False,loc='lower right');save(fig,'Figure2')
names=['Positive offer slope','Human > computer displays','Computer > human displays','Right insula · FLAME sensitivity analysis']
specs=[('taskwide-offer',1,45,.0052),('human-computer-response',1,270,.0002),('human-computer-response',2,155,.0012),('human-computer-response',3,25,.0163)]
flame=repo/'results/reviewer';summary=[]
fig=plt.figure(figsize=(7.55,9.0))
gs=fig.add_gridspec(4,3,width_ratios=[1,1,1.18],left=.025,right=.97,top=.946,bottom=.095,hspace=.63,wspace=.18)
for i in range(4):
 if i < 3:
  model,c,n,p=specs[i]
  pl=audit.payload(model,'cluster-z3.1',c,names[i]);rec=pl[0];assert rec['native_voxels']==n and abs(rec['minimum_spatial_fwe_p']-p)<1e-9
  q=repo/'results/palm_age_main/v1/maps'/model/'cluster-z3.1'/f'palm_clustere_zvstat_fwep_c{c}.nii'
  im=nib.load(q);mask=(im.get_fdata()>.95) & (nib.load(repo/'results/palm_age_main/v1/preparation'/model/'mask.nii.gz').get_fdata()>0)
  fp=flame/('activation_fairness_main/focal_contrast1/cope1.nii.gz' if i==0 else f'activation_social_computer/social_task_response/contrast{c}/cope{c}.nii.gz')
 else:
  dp=flame/'activation_fairness_main/focal_contrast1';im=nib.load(dp/'cluster_mask_zstat1.nii.gz');mask=im.get_fdata()==4
  stat=nib.load(dp/'zstat1.nii.gz');z=stat.get_fdata();n=82
  assert np.allclose(im.affine,stat.affine) and mask.sum()==n and round(float(z[mask].max()),2)==4.49
  peak=np.argwhere(mask)[z[mask].argmax()];cuts=nib.affines.apply_affine(im.affine,peak);assert np.allclose(cuts,[40.3,1.29,5.22],atol=.1)
  overlap=read(repo/'results/figure_review/2026-11-02-parametric/insula_cluster_permutation_overlap.tsv')
  assert len(overlap)==4 and all(int(r['corrected_overlap_voxels'])==0 for r in overlap)
  rec=dict(model='positive offer slope, earlier covariates and mask',method='FLAME',contrast=1,title=names[i],native_voxels=n,cluster_fwe_p=.000129,cluster_id=4,statistic_map=str(dp/'zstat1.nii.gz'),native_cut_x=float(cuts[0]),native_cut_y=float(cuts[1]),native_cut_z=float(cuts[2]))
  audit.map_records.append(rec)
  overlay=resample_from_to(nib.Nifti1Image(np.where(mask,z,0).astype('float32'),im.affine),audit.bg,order=0).get_fdata()
  pl=(rec,overlay,cuts);fp=dp/'cope1.nii.gz'
 for j,axis in enumerate([0,2]):plane(fig.add_subplot(gs[i,j]),pl,axis)
 amp=nib.load(fp);assert amp.shape==im.shape and np.allclose(amp.affine,im.affine)
 components,nc=label(mask,structure=np.ones((3,3,3)));values=[]
 for k in range(1,nc+1):
  vox=np.argwhere(components==k);xyz=nib.affines.apply_affine(im.affine,vox).mean(axis=0);vals=amp.get_fdata()[components==k];assert np.all(np.isfinite(vals))
  hemi='L' if xyz[0]<0 else 'R'
  values.append((hemi,int(len(vox)),float(vals.mean())))
 values.sort(key=lambda x:(x[0],-x[1]));assert sum(v[1] for v in values)==n
 for h,size,mean in values:summary.append(dict(panel='ABCD'[i],hemisphere=h,voxels=size,mean_cope=mean,source=str(fp.relative_to(repo))))
 ax=fig.add_subplot(gs[i,2]);xx=np.arange(len(values));axismax=max(v[2] for v in values)*1.35
 ax.bar(xx,[v[2] for v in values],width=.50,color='#57778d');ax.set_xlim(-.7,len(values)-.3);ax.set_ylim(0,axismax);ax.set_xticks(xx,[f'{h}\n{size} vx' for h,size,_ in values],fontsize=8)
 ax.set_title('Mean FLAME estimate',fontsize=9,pad=3);ax.set_ylabel('Offer slope (a.u.)' if i in [0,3] else 'Display contrast (a.u.)',fontsize=8);ax.tick_params(axis='y',labelsize=8)
 ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 bbox=gs[i,0].get_position(fig);fig.text(.027,bbox.y1+.031,f"{'ABCD'[i]}   {names[i]}",fontsize=12,weight='bold',va='bottom')
fig.text(.52,.050,'D: no corrected overlap in PALM or randomise',fontsize=8.2,ha='left');zbar(fig,fig.add_axes([.065,.052,.34,.013]));save(fig,'Figure3')
with (out/'figure3_component_estimates.tsv').open('w') as h:
 writer=csv.DictWriter(h,fieldnames=list(summary[0]),delimiter='\t');writer.writeheader();writer.writerows(summary)
pl=audit.payload('human-computer-response','cluster-z3.1',3,'Younger > older: mean human − computer display response');assert pl[0]['native_voxels']==25
# A publication rebuild always includes the six Linux1 condition estimates.
if len(sys.argv) > 4:
    condition_path = Path(sys.argv[4])
else:
    figure_dir = repo / 'results/manuscript/figure4_partner_displays'
    candidates = [
        folder / filename
        for filename in ('condition_estimates_adjusted.tsv', 'condition_estimates.tsv')
        for folder in (Path(__file__).parent, figure_dir)
    ]
    condition_path = next((path for path in candidates if path.is_file()), candidates[0])
estimates = read(condition_path)
age_groups = [('younger', 25), ('older', 22)]
conditions = ['similar', 'dissimilar', 'computer']
records = {(row['age_group'], row['condition']): row for row in estimates}
expected = {(age, partner) for age, _ in age_groups for partner in conditions}
if len(estimates) != 6 or set(records) != expected:
    raise ValueError('Expected one estimate per age group and partner condition')
adjusted_keys = {'covariate_adjusted_mean', 'covariate_adjusted_sem'}
available = adjusted_keys.intersection(estimates[0])
if available and available != adjusted_keys:
    raise ValueError('Covariate-adjusted means and SEMs must be supplied together')
use_adjusted = available == adjusted_keys
mean_key, sem_key = ('covariate_adjusted_mean', 'covariate_adjusted_sem') if use_adjusted else ('mean', 'sem')
fig = plt.figure(figsize=(7.35, 3.35))
grid = fig.add_gridspec(1, 2, left=.035, right=.98, bottom=.27, top=.83,
                        wspace=.38, width_ratios=[1, 1.25])
mapax = fig.add_subplot(grid[0, 0])
ax = fig.add_subplot(grid[0, 1])
plane(mapax, pl, 0)
mapax.set_title('Sagittal   x = −2 mm', fontsize=9, pad=4)
for i, (age, n) in enumerate(age_groups):
    rows = [records[age, partner] for partner in conditions]
    means = np.array([float(r[mean_key]) for r in rows])
    sems = np.array([float(r[sem_key]) for r in rows])
    if any(int(r['n']) != n for r in rows) or not np.isfinite([means, sems]).all() or min(sems) < 0:
        raise ValueError(f'Invalid Figure 4 estimates for {age}')
    ax.bar(np.arange(3) + (i-.5)*.36, means, width=.34, yerr=sems, color=colors[i],
           capsize=3.2, error_kw={'elinewidth': 1, 'capthick': 1}, label=f'{age.capitalize()} (n = {n})')
ax.axhline(0, color='#404040', lw=.85)
ax.grid(axis='y', alpha=.16)
ax.set_axisbelow(True)
ax.set_xticks(range(3), ['Age-similar', 'Age-dissimilar', 'Computer'], fontsize=9)
ax.set_ylabel('Adjusted display estimate (model units)' if use_adjusted else 'Display estimate (model units)', size=9)
ax.tick_params(axis='y', labelsize=8)
ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.18),
          handlelength=1.1, columnspacing=1.3, fontsize=8)
fig.text(.038, .965, 'A   Direct age contrast', weight='bold', fontsize=11, va='top')
fig.text(grid[0, 1].get_position(fig).x0, .965, 'B   Response by partner', weight='bold', fontsize=11, va='top')
zbar(fig, fig.add_axes([.083, .16, .24, .018]), 'Z (younger > older)')
save(fig, 'Figure4')
for rec in audit.map_records:rec['statistic_map']=str(Path(rec['statistic_map']).relative_to(repo))
(out/'map_provenance.json').write_text(json.dumps(audit.map_records,indent=2))
