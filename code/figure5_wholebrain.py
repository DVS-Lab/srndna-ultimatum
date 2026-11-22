"""Display the two corrected whole-brain age clusters for offer amount."""
import argparse
import os
from pathlib import Path
os.environ.setdefault("SOURCE_DATE_EPOCH","1795381200")
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from wholebrain_choice import clusters, PREP, SOURCE_SCRATCH
from extract_figure4_conditions import read_rows, same_grid, verified
from extract_figure3_responses import write_table
from plot_permutation_followup import slice_plane

FIG = Path("results/manuscript/figure5_wholebrain_offer_age")
NAMES = ("Ventromedial frontal cortex","Left precuneus")

def age_masks(repo):
    ref, components=clusters(repo,"taskwide-offer","2.6",3)
    _, strict=clusters(repo,"taskwide-offer","3.1",3)
    if strict or [v["voxels"] for _,v in components]!=[81,68]:
        raise ValueError("Whole-brain age contrast has changed")
    return ref,components

def estimates(repo,scratch,out):
    ref,components=age_masks(repo)
    prep=repo/PREP/"taskwide-offer"
    roster=read_rows(prep/"participants.tsv")
    design=np.loadtxt((prep/"design.mat").read_text().split("/Matrix")[1].splitlines())
    columns=[r["name"] for r in read_rows(prep/"design_columns.tsv")]
    if columns[:2]!=["mean","older_centered"]:
        raise ValueError(f"Unexpected group design: {columns}")
    hashes={Path(r["path"]):r["sha256"] for r in read_rows(prep/"input_inventory.tsv") if r["role"]=="L2 COPE11"}
    if len(hashes)!=47:raise ValueError("Expected one fixed-effects contrast per participant")
    group=np.array([r["variance_group"]=="2" for r in roster])
    subjects=[r["subject"] for r in roster]
    if len(subjects)!=47 or sum(group)!=22 or not np.allclose(design[:,1],group-group.mean()):
        raise ValueError("Age coding mismatch")
    vals=[]
    for s in subjects:
        path,= [p for p in hashes if s in p.parts]
        img=nib.load(verified(scratch/path.relative_to(SOURCE_SCRATCH),hashes[path]))
        same_grid(img,ref);data=img.get_fdata()
        vals.append([data[m].mean() for m,_ in components])
    vals=np.asarray(vals)
    if not np.isfinite(vals).all():raise ValueError("Invalid participant averages")
    beta,_,rank,_=np.linalg.lstsq(design,vals,rcond=None)
    if rank!=design.shape[1]:raise ValueError("Rank-deficient imaging design")
    adjusted=vals-design[:,2:]@beta[2:];rows=[]
    for j,name in enumerate(NAMES):
        for lab,mask in (("Younger",~group),("Older",group)):
            v=adjusted[mask,j]
            rows.append(dict(region=name,age_group=lab,n=len(v),mean=float(v.mean()),sem=float(v.std(ddof=1)/np.sqrt(len(v)))))
    out.mkdir(parents=True,exist_ok=True);write_table(out/"figure5_cluster_estimates.tsv",rows)

def figure5(repo,out,background,anatomy,vmax):
    repo,out=Path(repo),Path(out)
    ref,components=age_masks(repo)
    out.mkdir(parents=True,exist_ok=True)
    table=out/"figure5_cluster_estimates.tsv"
    if not table.exists():table=repo/FIG/table.name
    rows=read_rows(table) if table.exists() else []
    fig,axes=plt.subplots(1,3 if rows else 2,figsize=(8.1 if rows else 6.2,3.1))
    fig.subplots_adjust(left=.025,right=.98,top=.80,bottom=.20,wspace=.23)
    summary=[]
    for ax,name,(mask,info) in zip(axes,NAMES,components):
        x,y,z=info["peak"];summary.append(dict(region=name,**{k:info[k] for k in ("voxels","p")},peak_x=x,peak_y=y,peak_z=z))
        overlay=resample_from_to(nib.Nifti1Image(mask.astype("uint8"),ref.affine),background,order=0).get_fdata()
        brain,extent,cut=slice_plane(anatomy,background.affine,0,x)
        color,_,_=slice_plane(overlay,background.affine,0,x)
        ax.imshow(brain,extent=extent,origin="lower",cmap="gray",vmin=0,vmax=vmax)
        ax.imshow(np.ma.masked_equal(color,0),extent=extent,origin="lower",interpolation="nearest",cmap=ListedColormap(["#ec9745"]))
        ax.set_axis_off();ax.set_title(f"{name}\nx = {cut:.0f} mm",fontsize=10)
    if rows:
        axes[-1].set_title("Across-partner offer contrast",fontsize=10)
        for j,(lab,col) in enumerate((("Younger","#af4f35"),("Older","#227b8c"))):
            cells=[next(r for r in rows if r["region"]==n and r["age_group"]==lab) for n in NAMES]
            axes[-1].bar(np.arange(2)+(j-.5)*.35,[float(r["mean"]) for r in cells],width=.34,yerr=[float(r["sem"]) for r in cells],color=col,label=lab,capsize=3)
        axes[-1].axhline(0,color="#505050",lw=.6);axes[-1].set_xticks([0,1],["Frontal","Precuneus"]);axes[-1].legend(fontsize=7,frameon=False)
        axes[-1].set_ylabel("Adjusted model units",fontsize=8)
    fig.text(.04,.93,"Younger > older adults: response to increasing offer amount",fontweight="bold",fontsize=11)
    write_table(out/"figure5_clusters.tsv",summary)
    for ext in ("pdf","png"):fig.savefig(out/f"Figure5.{ext}",dpi=350,bbox_inches="tight",pad_inches=.08)
    plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--scratch",type=Path);p.add_argument("--background",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.scratch:estimates(a.repo,a.scratch,a.output)
    bg=nib.as_closest_canonical(nib.load(a.background));brain=bg.get_fdata()
    figure5(a.repo,a.output,bg,brain,np.percentile(brain[brain>0],98))
