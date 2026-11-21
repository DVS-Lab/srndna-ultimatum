"""Figure 5: the literature-defined vmPFC ROI and the separate whole-brain age contrast."""
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import label
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from extract_figure4_conditions import read_rows, same_grid
from extract_figure3_responses import write_table
from plot_permutation_followup import slice_plane


def figure5(repo, out, background, anatomy, vmax):
    project = Path(repo) / "results/palm_age_main/v1"
    folder = project / "maps/taskwide-offer/cluster-z2.6"
    stat = nib.load(folder / "palm_vox_zvstat_c3.nii")
    prob = nib.load(folder / "palm_clustere_zvstat_fwep_c3.nii")
    coverage = nib.load(project / "preparation/taskwide-offer/mask.nii.gz")
    fixed = nib.load(Path(repo) / "masks/roi-VMPFC.nii.gz")
    for image in (prob, coverage, fixed):
        same_grid(image, stat)
    supported = coverage.get_fdata() > 0
    roi = fixed.get_fdata() > 0
    signed = stat.get_fdata()
    probability = prob.get_fdata()
    significant = supported & (probability > .95)
    strict = nib.load(project / "maps/taskwide-offer/cluster-z3.1/palm_clustere_zvstat_fwep_c3.nii")
    same_grid(strict, stat)
    if (significant.sum() != 149 or roi.sum() != 19 or not supported[roi].all() or
            np.any(signed[significant] <= 0) or np.any(strict.get_fdata()[supported] > .95)):
        raise ValueError("Saved whole-brain maps or literature ROI differ from the reported analyses")
    components, n = label(significant, structure=np.ones((3, 3, 3)))
    clusters = []
    for i in range(1, n + 1):
        selected = components == i
        indices = np.argwhere(selected)
        peak = indices[np.argmax(signed[selected])]
        x, y, z = nib.affines.apply_affine(stat.affine, peak)
        clusters.append(dict(voxels=len(indices), spatial_fwe_p=1-probability[selected].max(),
                             peak_x=x, peak_y=y, peak_z=z, vmPFC_overlap_voxels=int((selected & roi).sum())))
    clusters.sort(key=lambda row: -row["voxels"])
    if [(c["voxels"], c["vmPFC_overlap_voxels"]) for c in clusters] != [(81,10), (68,0)]:
        raise ValueError("Cluster sizes or overlap with the independent ROI have changed")
    write_table(out / "figure5_clusters.tsv", clusters)
    source = Path(repo) / "results/vmpfc_ratings/v1/vmpfc/bar_source.tsv"
    rows = [row for row in read_rows(source) if row["timing"] == "full-rt" and row["adjustment"] == "covariate-adjusted"]
    lookup = {(r["age_group"],r["partner"]): r for r in rows}
    if len(rows) != 6 or len(lookup) != 6:
        raise ValueError("Expected six unique adjusted ROI estimates")
    write_table(out / "figure5_offer_slopes.tsv", rows)
    overlays = [resample_from_to(nib.Nifti1Image(mask.astype("uint8"), stat.affine), background,
                                 order=0).get_fdata() for mask in (significant, roi)]
    fig = plt.figure(figsize=(8.2,3.15))
    grid = fig.add_gridspec(1, 3, left=.025, right=.985, bottom=.24, top=.79, wspace=.12,
                            width_ratios=[1,1,1.52])
    for col, (title, cut) in enumerate([("Medial frontal", 6), ("Left precuneus", -12)]):
        ax = fig.add_subplot(grid[0,col])
        brain, extent, actual = slice_plane(anatomy, background.affine, 0, cut)
        sig, _, _ = slice_plane(overlays[0], background.affine, 0, cut)
        contour, _, _ = slice_plane(overlays[1], background.affine, 0, cut)
        ax.imshow(brain, extent=extent, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        ax.imshow(np.ma.masked_equal(sig,0), extent=extent, origin="lower", interpolation="nearest",
                  cmap=ListedColormap(["#ec9745"]))
        if contour.any():
            ax.contour(contour, levels=[.5], colors=["#52d6ef"], linewidths=1.35, origin="lower", extent=extent)
        ax.set_axis_off()
        ax.set_title(f"{title}\nx = {actual:.0f} mm", fontsize=9)
    left = grid[0,0].get_position(fig); right = grid[0,2].get_position(fig)
    fig.text(left.x0,.96,"A   Whole-brain age contrast",weight="bold",fontsize=11)
    fig.text(right.x0+.04,.96,"B   Offer slopes in the fixed vmPFC ROI",weight="bold",fontsize=11)
    fig.legend(handles=[Patch(color="#ec9745",label="Younger > older, Z > 2.6"),
                        Line2D([0],[0],color="#2795ae",lw=2,label="Literature-defined ROI")],
               loc="lower left",bbox_to_anchor=(.025,.055),fontsize=8,ncol=2,frameon=False,
               handlelength=1.3,columnspacing=1.2)
    ax = fig.add_subplot(grid[0,2])
    # Leave a physical gutter for the y label between the scans and the chart.
    box=ax.get_position();ax.set_position([box.x0+.055,box.y0,box.width-.055,box.height])
    conditions = ["similar","dissimilar","computer"]
    for i,(age,n,color) in enumerate([("younger",25,"#af4f35"),("older",22,"#227b8c")]):
        cells=[lookup[age,k] for k in conditions]
        means=[float(row["mean"]) for row in cells]
        errors=[float(row["participant_sem"]) for row in cells]
        if any(int(row["n"]) != n for row in cells) or not np.isfinite([means,errors]).all() or min(errors)<0:
            raise ValueError("Invalid ROI estimates")
        ax.bar(np.arange(3)+(i-.5)*.36,means,width=.34,yerr=errors,capsize=3,color=color,
               label=f"{age.capitalize()} (n = {n})",error_kw={"elinewidth":1,"capthick":1})
    ax.axhline(0,color=".25",lw=.8)
    ax.set_ylabel("Adjusted offer slope\n(model units)",fontsize=9)
    ax.set_xticks(range(3),["Age-similar","Age-dissimilar","Computer"],fontsize=8)
    ax.set_axisbelow(True);ax.grid(axis="y",alpha=.15)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8,frameon=False,
              handlelength=1.1,columnspacing=1.0)
    for ext in ("pdf","png"):
        fig.savefig(out / f"Figure5.{ext}",dpi=350,bbox_inches="tight",pad_inches=.10)
    plt.close(fig)
