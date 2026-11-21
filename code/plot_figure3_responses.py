"""Plot one independently defined ROI or one whole-sample display contrast per figure."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from extract_figure3_responses import read_masks
from extract_figure4_conditions import read_rows
from plot_permutation_followup import slice_plane


def panel(rows, region, title, mask, reference, background, cut, out):
    fig, (anatomy, bars) = plt.subplots(1, 2, figsize=(7.3, 3.1), layout="constrained")
    base = background.get_fdata()
    native = nib.Nifti1Image(mask.astype("uint8"), reference.affine)
    overlay = resample_from_to(native, background, order=0).get_fdata()
    brain, extent, z = slice_plane(base, background.affine, 2, cut)
    region_slice, _, _ = slice_plane(overlay, background.affine, 2, cut)
    if not np.any(region_slice):
        raise ValueError(f"Empty displayed section for {region}")
    anatomy.imshow(brain, extent=extent, origin="lower", cmap="gray", vmin=0, vmax=np.percentile(base[base>0], 99.5))
    anatomy.imshow(np.ma.masked_equal(region_slice, 0), extent=extent, origin="lower",
                   cmap=matplotlib.colors.ListedColormap(["#e99a48"]), interpolation="nearest")
    anatomy.set_title(title, fontsize=10)
    anatomy.set_xlabel(f"Axial   z = {z:.0f} mm")
    anatomy.set_xticks([])
    anatomy.set_yticks([])
    for x, label in ((.03,"L"),(.94,"R")):
        anatomy.text(x, .05, label, color="white", transform=anatomy.transAxes)
    selected = [r for r in rows if r["region"] == region and r["quantity"] == "display"]
    values = {r["partner"]:r for r in selected}
    order = ("similar","dissimilar","computer")
    if len(selected) != 3 or set(values) != set(order) or any(int(r["n"]) != 47 for r in selected):
        raise ValueError("Expected three sample-wide conditions from the same 47 people")
    y = [float(values[k]["adjusted_mean"]) for k in order]
    se = [float(values[k]["adjusted_sem"]) for k in order]
    if not np.isfinite([y,se]).all() or min(se) < 0:
        raise ValueError("Invalid figure source")
    bars.bar(range(3), y, yerr=se, width=.58, capsize=4, color="#397989",
             error_kw={"elinewidth":1.1,"capthick":1.1})
    bars.axhline(0, lw=.8, color="#333333")
    bars.set_xticks(range(3), ["Age-similar\nhuman","Age-dissimilar\nhuman","Computer"])
    bars.set_ylabel("Adjusted display estimate (model units)")
    bars.set_title("Partner displays across 47 participants", fontsize=10)
    bars.spines[["top","right"]].set_visible(False)
    for ext in ("pdf","png"):
        fig.savefig(out / f"{region}.{ext}", dpi=350, metadata={"Creator":"srndna-ultimatum"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    background = nib.as_closest_canonical(nib.load(args.template))
    reference, insula, display = read_masks(args.repo)
    rows = read_rows(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    # The published PALM display peaks at native MNI z = -10.88; the literature
    # spheres are centered at z = 0 and +2. The mask includes all voxels for extraction.
    panel(rows,"anterior_insula","Bilateral anterior insula",insula,reference,background,0,args.output)
    panel(rows,"human_gt_computer","Human > computer display contrast",display,reference,background,-10.88,args.output)


if __name__ == "__main__":
    main()
