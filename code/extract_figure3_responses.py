"""Extract corrected partner responses in a literature ROI and a full-sample display mask."""
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from analyze_corrected_vmpfc import estimate, fit_hc3, holm
from extract_figure4_conditions import (CODES, CONDITIONS, MODEL, SOURCE_ROOT, combine,
                                        read_rows, same_grid, validate_design, verified)

PREP = Path("results/palm_age_main/v1")
PROB_SHA = "4c2884e6f44b1679d7be2fc13a795ccf52051e907fbba0318fd938569dbcd4fe"
ROI_SHA = "5a1261604d598273495ec90f875a6305caa84c4d96866c60677d6b0b8d733374"
ROSTER_SHA = "9292fd8b02e9997a7a7a0cc98801787ce818222e7ec46ffd9551a6042395b5ec"
DESIGN_SHA = "8db47e786cb27b10fa1b5b5f484cf1715a01d1c9072fe62014b197c9dbc7e7ea"
INV_SHA = "539dff6e12dc6aba33f54128127dd77ef385ee11f12346428ef6f998cf0ac5d6"
MASK_SHA = "851c70d5627dcafcd22967ab694de886990e8f51f14f986b34ef3ce6dec715e0"
NAMES = ("comp", "comp_p", "in", "in_p", "out", "out_p")


def read_masks(repo):
    project = repo / PREP
    prep = project / "preparation" / MODEL
    reference = nib.load(verified(prep / "mask.nii.gz", MASK_SHA))
    roifile = verified(repo / "masks/roi-aINS.nii.gz", ROI_SHA)
    probfile = verified(project / "maps" / MODEL / "cluster-z3.1/palm_clustere_zvstat_fwep_c1.nii", PROB_SHA)
    insula, probability = (nib.load(path) for path in (roifile, probfile))
    for image in (insula, probability):
        same_grid(image, reference)
    insula = insula.get_fdata() > 0
    support = reference.get_fdata() > 0
    display = (probability.get_fdata() > .95) & support
    if insula.sum() != 38 or not support[insula].all() or display.sum() != 270:
        raise ValueError("The published masks or their coverage have changed")
    return reference, insula, display


def read_design(prep):
    roster = read_rows(verified(prep / "participants.tsv", ROSTER_SHA))
    mat = verified(prep / "design.mat", DESIGN_SHA)
    design = np.loadtxt(mat.read_text().split("/Matrix", 1)[1].splitlines(), ndmin=2)
    older = np.asarray([r["variance_group"] == "2" for r in roster])
    if (design.shape != (47, 6) or older.sum() != 22 or
            not np.allclose(design[:, 0], 1) or not np.allclose(design[:, 1], older - older.mean()) or
            not np.allclose(design[:, 1:].mean(axis=0), 0, atol=1e-6)):
        raise ValueError("Expected the original centered six-column model and 47 participants")
    return roster, design


def write_table(path, records):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def read_participant(files, reference, insula, display, hashes, scratch):
    def local(path):
        return scratch / path.relative_to(SOURCE_ROOT)

    def voxels(path, mask):
        target = local(path)
        if path in hashes:
            verified(target, hashes[path])
        image = nib.load(target)
        same_grid(image, reference)
        values = image.get_fdata()[mask]
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite ROI values in {target}")
        return values

    l2, = (p for role, p in files if role == "L2 COPE10")
    runs = sorted(p for role, p in files if role == "L1 COPE10")
    if len(runs) != 2 or any(f"_run-{j:02}_" not in str(p) for j, p in enumerate(runs, 1)):
        raise ValueError("Missing one of the two expected FEAT runs")
    # Fixed literature ROI: use each condition's native, two-run FEAT estimate.
    l2_root = l2.parents[2]
    estimates = []
    for cope in range(1, 7):
        folder = l2_root / f"cope{cope}.feat"
        if not np.all(voxels(folder / "mask.nii.gz", insula) > 0):
            raise ValueError(f"Incomplete fixed ROI coverage in {folder}")
        estimates.append(voxels(folder / "stats/cope1.nii.gz", insula).mean())

    # In the map-defined ROI, reconstruct the defining contrast with its common run weights.
    partners, contrasts, variances = [], [], []
    for cope_path in runs:
        folder = local(cope_path).parents[1]
        validate_design(folder)
        lines = (folder / "design.con").read_text().splitlines()
        labels = dict(line.split(None, 1) for line in lines if line.startswith("/ContrastName"))
        matrix = np.loadtxt(lines[lines.index("/Matrix") + 1:], ndmin=2)
        if (any(labels.get(f"/ContrastName{i}", "").strip() != name for i, name in enumerate(NAMES, 1))
                or not np.allclose(matrix[:6], np.eye(6, matrix.shape[1]))):
            raise ValueError(f"Unexpected display or offer contrast weights in {folder}")
        partners.append([voxels(cope_path.with_name(f"cope{i}.nii.gz"), display) for i in CODES])
        contrasts.append(voxels(cope_path, display))
        variances.append(voxels(cope_path.with_name("varcope10.nii.gz"), display))
    values, error = combine(np.asarray(partners), np.asarray(contrasts), np.asarray(variances), voxels(l2, display))
    return estimates, values, error


def summarize(design, datasets):
    rows, tests = [], []
    for (region, quantity), data in datasets.items():
        beta, _, _, _ = fit_hc3(design, data)
        adjusted = data - design[:, 2:] @ beta[2:]
        for i, partner in enumerate(CONDITIONS):
            values = adjusted[:, i]
            rows.append(dict(region=region, quantity=quantity, partner=partner, n=len(values),
                             adjusted_mean=values.mean(), adjusted_sem=values.std(ddof=1)/np.sqrt(len(values))))
        if region == "anterior_insula":
            # Repeated conditions contribute exactly one observation per person.
            result, _, _ = estimate(design, data.mean(axis=1), np.eye(design.shape[1])[0])
            tests.append(dict(quantity=quantity, n=len(data), **{key:result[key] for key in
                              ("estimate", "hc3_se", "df", "p_uncorrected", "ci95_low", "ci95_high")}))
    for row, p in zip(tests, holm([row["p_uncorrected"] for row in tests])):
        row["p_holm_two_tests"] = float(p)
    return rows, tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scratch-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prep = args.repo / PREP / "preparation" / MODEL
    reference, insula, display = read_masks(args.repo)
    roster, design = read_design(prep)
    inventory = read_rows(verified(prep / "input_inventory.tsv", INV_SHA))
    hashes = {Path(r["path"]): r["sha256"] for r in inventory}
    roi_values, display_values, errors = [], [], []
    for r in roster:
        files = [(row["role"], Path(row["path"])) for row in inventory if r["subject"] in Path(row["path"]).parts]
        fixed, selected, error = read_participant(files, reference, insula, display, hashes, args.scratch_root)
        roi_values.append(fixed)
        display_values.append(selected)
        errors.append(error)
    fixed = np.asarray(roi_values)
    rows, tests = summarize(design, {
        ("anterior_insula", "display"): fixed[:, ::2],
        ("anterior_insula", "offer_slope"): fixed[:, 1::2],
        ("human_gt_computer", "display"): np.asarray(display_values),
    })
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(args.output / "condition_estimates.tsv", rows)
    write_table(args.output / "insula_tests.tsv", tests)
    record = dict(participants=len(fixed), runs=2*len(fixed), insula_voxels=int(insula.sum()),
                  human_gt_computer_voxels=int(display.sum()), maximum_cope10_error=float(max(errors)),
                  insula_sha256=ROI_SHA, palm_corrected_map_sha256=PROB_SHA,
                  input_inventory_sha256=INV_SHA, design_sha256=DESIGN_SHA)
    (args.output / "validation.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
