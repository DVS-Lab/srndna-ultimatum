"""Extract whole-brain offer clusters, same-trial network expression and optional run gPPI."""
import argparse
from collections import Counter
from pathlib import Path
import json
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import label
from extract_single_trial_choice import PREP, VERSION, SOURCE_SCRATCH, release_image, trial_order, write_trials
from extract_figure4_conditions import read_rows, same_grid, verified

def clusters(repo, model, threshold, contrast):
    directory = repo / PREP.parent / "maps" / model / ("cluster-z"+threshold)
    stat = nib.load(directory / f"palm_vox_zvstat_c{contrast}.nii")
    prob = nib.load(directory / f"palm_clustere_zvstat_fwep_c{contrast}.nii")
    support = nib.load(repo / PREP / model / "mask.nii.gz")
    for img in (prob, support):
        same_grid(img, stat)
    z, conf = stat.get_fdata(), prob.get_fdata()
    lb, n = label((conf > .95) & (support.get_fdata()>0), structure=np.ones((3,3,3)))
    rows = []
    for i in range(1,n+1):
        m = lb == i
        pts = np.argwhere(m)
        peak = nib.affines.apply_affine(stat.affine,pts[np.argmax(z[m])])
        rows.append((m, dict(voxels=int(m.sum()), p=float(1-conf[m].max()), peak=peak.tolist())))
    return stat, sorted(rows,key=lambda pair:-pair[1]["voxels"])

def regions(repo):
    base, age = clusters(repo,"taskwide-offer","2.6",3)
    reg, visual = clusters(repo,"taskwide-offer","3.1",1)
    dmn, focus = clusters(repo,"full-rt__nppi-dmn","2.6",3)
    for img in (reg,dmn): same_grid(img,base)
    records = age+visual+focus
    if [a[1]["voxels"] for a in records] != [81,68,45,49]:
        raise ValueError(f"Unexpected corrected maps: {[a[1] for a in records]}")
    return base, dict(zip(("medial_frontal","precuneus","visual","frontal_target"),records))

def sample_grid(values, ref, grid, order):
    if grid[0]==ref.shape and np.allclose(grid[1],ref.affine,rtol=0,atol=1e-4): return values
    return resample_from_to(nib.Nifti1Image(values.astype(np.float32),ref.affine),grid,order=order).get_fdata()

def extract(repo, dataset, out, scratch):
    ref, rois = regions(repo)
    network = nib.load(repo/"masks/nan_rPNAS_2mm_net0003.nii.gz")
    support = nib.load(repo/PREP/"full-rt__nppi-dmn/mask.nii.gz")
    same_grid(ref,support)
    prep = repo/PREP/"full-rt__nppi-dmn"
    roster = read_rows(prep/"participants.tsv")
    design = np.loadtxt((prep/"design.mat").read_text().split("/Matrix")[1].splitlines())
    cols = [r["name"] for r in read_rows(prep/"design_columns.tsv")]
    demo = {r["subject"]:dict(age=int(r["variance_group"]=="2"),
          **{c:design[i,cols.index(c)] for c in ("ismale","tsnr","fd_mean")}) for i,r in enumerate(roster)}
    ppi = {}
    if scratch:
        for row in read_rows(prep/"input_inventory.tsv"):
            if row["role"]!="L1 COPE7": continue
            path=verified(scratch/Path(row["path"]).relative_to(SOURCE_SCRATCH),row["sha256"])
            feat=path.parents[1]
            key=(feat.parent.name,int(feat.name.removeprefix("L1_run-").removesuffix(".feat")))
            img=nib.load(path);same_grid(img,ref)
            values=img.get_fdata()[rois["frontal_target"][0]]
            if key in ppi or not np.isfinite(values).all(): raise ValueError(str(key))
            ppi[key]=float(values.mean())
        if len(ppi)!=94: raise ValueError("Missing run-specific DMN estimates")
    records=read_rows(repo/VERSION/"openneuro_2.2.0_inputs.tsv")
    if {(r["subject"],int(r["run"])) for r in records}!={(s,j) for s in demo for j in (1,2)} or len(records)!=94: raise ValueError("Incomplete roster")
    trial_rows=[]; coverage=[]; common_grid={}
    for record in records:
        s, j=record["subject"],int(record["run"])
        events=repo/"source_data/bids"/s/"func"/f"{s}_task-ultimatum_run-{j:02}_events.tsv"
        trials=trial_order(events,record["events_git_blob"])
        img=release_image(dataset/record["image"],record)
        if img.shape[-1]!=72:
            if (s,j)!=("sub-104",2): raise ValueError("Unexpected partial run")
            continue
        grid=(img.shape[:3],img.affine)
        signature=(grid[0],grid[1].tobytes())
        if signature not in common_grid:
            masks={k:sample_grid(v[0],ref,grid,0)>.5 for k,v in rois.items()}
            distributed=(sample_grid(support.get_fdata(),ref,grid,0)>.5) & ~masks["frontal_target"]
            w=resample_from_to(network,grid,order=1).get_fdata()
            common_grid[signature]=masks,distributed,w
        masks,distributed,w=common_grid[signature]
        data=np.asarray(img.dataobj,dtype=np.float32)
        valid=np.isfinite(data).all(axis=3)&(data!=0).any(axis=3)
        series={};fraction={}
        for k,m in masks.items():
            use=m&valid
            if not use.any(): raise ValueError(f"No supporting voxels {k} {s} {j}")
            series[k]=data[use].mean(axis=0,dtype=np.float64)
            fraction[k]=float(use.sum()/m.sum())
        use=distributed&valid
        weights=w[use]-w[use].mean()
        series["dmn"]=(weights/np.dot(weights,weights))@data[use]
        if any(not np.isfinite(v).all() or np.ptp(v)==0 for v in series.values()):raise ValueError(f"Invalid trial data for {s} run {j}")
        fraction["dmn"]=float(use.sum()/distributed.sum());coverage.append(fraction)
        for t,trial in enumerate(trials):
            if trial["human"]!="1" or trial["missed"]!="0":continue
            row=dict(subject=s,run=j,trial=t+1,offer=trial["offer"],accept=trial["accept"],partner=int(trial["ingroup"])-.5,
                     fair=int(trial["block_fair"]),rt=float(trial["response_time"]),**demo[s])
            for k,v in series.items():
                row[k+"_mean"]=v.mean();row[k+"_within"]=v[t]-v.mean()
            if scratch: row["opposite_run_dmn"]=ppi[s,3-j]
            trial_rows.append(row)
    if len(trial_rows)!=4382 or len(coverage)!=93 or len(demo)!=47:raise ValueError("Unexpected analysis sample")
    out.mkdir(parents=True,exist_ok=True);write_trials(out/"private_trials.tsv",trial_rows)
    audit=dict(participants=len(demo),runs=len(coverage),trials=len(trial_rows),regions={k:v[1] for k,v in rois.items()},
               minimum_coverage={k:min(d[k] for d in coverage) for k in coverage[0]},verified_gppi_images=len(ppi),
               corrected_rt_range=[float(q([r["rt"] for r in trial_rows])) for q in (min,max)])
    if ppi: audit["run_correlation"]=float(np.corrcoef([[ppi[s,j] for s in sorted(demo)] for j in (1,2)])[0,1])
    (out/"validation.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n");print(audit,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument("--dataset",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--scratch",type=Path,help="Also verify and extract corrected run-level DMN COPE7 images")
    args=ap.parse_args();extract(args.repo,args.dataset,args.output,args.scratch)
