"""Extract the six descriptive Figure 4 means from retained FEAT outputs."""
import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import nibabel as nib
import numpy as np

CONDITIONS = ("computer", "similar", "dissimilar")
CODES = (1, 3, 5)
SOURCE_ROOT = Path("/ZPOOL/data/scratch")
MODEL = "human-computer-response"

def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()

def verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f"File differs from the saved analysis: {path}")
    return path

def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))

def same_grid(image, reference):
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError(f"Image and primary ROI grids differ: {image.get_filename()}")

def combine(partners, contrasts, variances, original):
    """Apply the tested contrast's run weights to each condition and check both levels."""
    if partners.shape[:2] != (2, 3) or contrasts.shape != variances.shape or partners[:, 0].shape != contrasts.shape:
        raise ValueError("Run, condition and voxel dimensions differ")
    if not np.isfinite(partners).all() or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("Invalid condition estimate or contrast variance")
    within_run = partners[:, 1] + partners[:, 2] - 2 * partners[:, 0]
    if not np.allclose(within_run, contrasts, atol=1e-4, rtol=1e-5):
        raise ValueError("Condition estimates do not reconstruct the retained run contrast")
    weights = 1.0 / variances
    weights /= weights.sum(axis=0)
    fixed = (weights[:, None, :] * partners).sum(axis=0)
    reconstructed = fixed[1] + fixed[2] - 2 * fixed[0]
    if not np.allclose(reconstructed, original, atol=1e-4, rtol=1e-5):
        raise ValueError("Weighted conditions do not reconstruct the retained participant COPE10")
    return fixed.mean(axis=1), float(np.abs(reconstructed - original).max())

def validate_design(run_folder):
    text = (run_folder / "design.con").read_text().splitlines()
    labels = {a: b.strip() for row in text if row.startswith("/ContrastName") for a, b in [row.split(None, 1)]}
    for index, label in ((1, "comp"), (3, "in"), (5, "out")):
        if labels.get(f"/ContrastName{index}") != label:
            raise ValueError(f"Unexpected display contrast labels in {run_folder}")
    matrix = np.loadtxt(text[text.index("/Matrix") + 1:], ndmin=2)
    if not np.allclose(matrix[9], matrix[2] + matrix[4] - 2 * matrix[0], atol=1e-6, rtol=0):
        raise ValueError(f"COPE10 contrast weights do not match the partner displays in {run_folder}")

def adjusted_errors(estimates, roster, prep):
    """Return within-person SEMs and the repository's conditional covariate-adjusted means/SEMs."""
    mat = verified(prep / "design.mat", "8db47e786cb27b10fa1b5b5f484cf1715a01d1c9072fe62014b197c9dbc7e7ea")
    colfile = verified(prep / "design_columns.tsv", "b48239508bee98858f14f5fa9cd98a05e6a467f213b74863c6243963438ff397")
    columns = [row["name"] for row in read_rows(colfile)]
    if columns != ["mean", "older_centered", "ismale", "tsnr", "fd_mean", "RT"]:
        raise ValueError("Unexpected group design columns")
    design = np.loadtxt(mat.read_text().split("/Matrix", 1)[1].splitlines())
    older = np.array([row["variance_group"] == "2" for row in roster])
    if design.shape != (len(roster), len(columns)) or not np.allclose(design[:, 1], older - older.mean()):
        raise ValueError("Group design and frozen roster do not align")
    values = np.empty((len(roster), len(CONDITIONS)))
    groups = {"younger": ~older, "older": older}
    for age, members in groups.items():
        values[members] = np.asarray(estimates[age])
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != len(columns):
        raise ValueError("Group design is rank deficient")
    adjusted = values - design[:, 2:] @ coefficients[2:]
    # All three repeated partner conditions contribute to each participant's mean.
    k = len(CONDITIONS)
    morey = np.sqrt(k / (k - 1))
    cells = {}
    for age, members in groups.items():
        raw = values[members]
        cov_adjusted = adjusted[members]
        normalized = raw - raw.mean(axis=1, keepdims=True) + raw.mean()
        if not np.allclose(normalized.mean(axis=0), raw.mean(axis=0)):
            raise ValueError("Within-person normalization changed the group condition means")
        if not np.allclose(cov_adjusted.mean(axis=0), design[members, :2].mean(axis=0) @ coefficients[:2]):
            raise ValueError("Adjusted means differ from group model predictions")
        n = len(raw)
        within_sem = morey * normalized.std(axis=0, ddof=1) / np.sqrt(n)
        adjusted_mean = cov_adjusted.mean(axis=0)
        conditional_sem = cov_adjusted.std(axis=0, ddof=1) / np.sqrt(n)
        for i, partner in enumerate(CONDITIONS):
            cells[age, partner] = dict(within_person_sem=within_sem[i],
                                       covariate_adjusted_mean=adjusted_mean[i],
                                       covariate_adjusted_sem=conditional_sem[i])
    return cells

def extract(repo, scratch, include_adjusted=False):
    project = repo / "results/palm_age_main/v1"
    prep = project / "preparation" / MODEL
    probability = project / "maps" / MODEL / "cluster-z3.1/palm_clustere_zvstat_fwep_c3.nii"
    reference = nib.load(verified(probability, "94ada4c560143f05ade139952ac84b2116417b7bc73c59db22ba012bd71e3554"))
    search = nib.load(verified(prep / "mask.nii.gz", "851c70d5627dcafcd22967ab694de886990e8f51f14f986b34ef3ce6dec715e0"))
    same_grid(search, reference)
    roi = (reference.get_fdata() > .95) & (search.get_fdata() > 0)
    if roi.sum() != 25:
        raise ValueError("Expected the published 25-voxel primary posterior midline cluster")
    inventory = read_rows(verified(prep / "input_inventory.tsv", "539dff6e12dc6aba33f54128127dd77ef385ee11f12346428ef6f998cf0ac5d6"))
    roster = read_rows(verified(prep / "participants.tsv", "9292fd8b02e9997a7a7a0cc98801787ce818222e7ec46ffd9551a6042395b5ec"))
    if Counter(r["variance_group"] for r in roster) != {"1": 25, "2": 22}:
        raise ValueError("Participant groups differ from the primary model")
    lookup = {Path(r["path"]): r["sha256"] for r in inventory}
    def local(path):
        return scratch / path.relative_to(SOURCE_ROOT)
    def values(path):
        target = local(path)
        if path in lookup:
            verified(target, lookup[path])
        im = nib.load(target)
        same_grid(im, reference)
        result = im.get_fdata()[roi]
        if not np.isfinite(result).all():
            raise ValueError(f"Nonfinite estimate inside ROI: {target}")
        return result
    estimates = {"younger": [], "older": []}
    largest_error = 0.0
    for member in roster:
        sid = member["subject"]
        files = [(r["role"], Path(r["path"])) for r in inventory if sid in Path(r["path"]).parts]
        l2, = [p for role, p in files if role == "L2 COPE10"]
        l1 = sorted(p for role, p in files if role == "L1 COPE10")
        if len(l1) != 2 or any(f"_run-{run:02}_" not in str(p) for run, p in enumerate(l1, 1)):
            raise ValueError(f"Expected both archived runs for {sid}")
        original = values(l2)
        partners, contrasts, variances = [], [], []
        for p in l1:
            validate_design(local(p).parents[1])
            partners.append([values(p.with_name(f"cope{i}.nii.gz")) for i in CODES])
            contrasts.append(values(p))
            variances.append(values(p.with_name("varcope10.nii.gz")))
        means, error = combine(np.array(partners), np.array(contrasts), np.array(variances), original)
        largest_error = max(largest_error, error)
        age = "younger" if member["variance_group"] == "1" else "older"
        estimates[age].append(means)
    records = []
    for age, group in estimates.items():
        data = np.array(group)
        for i, condition in enumerate(CONDITIONS):
            samples = data[:, i]
            records.append(dict(age_group=age, condition=condition, n=len(samples), mean=samples.mean(), sem=samples.std(ddof=1)/np.sqrt(len(samples))))
    if include_adjusted:
        adjustments = adjusted_errors(estimates, roster, prep)
        for row in records:
            row.update(adjustments[row["age_group"], row["condition"]])
    return records, largest_error

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=SOURCE_ROOT, help="Local equivalent of /ZPOOL/data/scratch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjusted", action="store_true",
                        help="Also export within-participant and covariate-adjusted summaries")
    args = parser.parse_args()
    rows, error = extract(args.repo, args.scratch_root, args.adjusted)
    with NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        temporary = stream.name
    os.replace(temporary, args.output)
    print(f"Validated 47 participants, 94 runs, 25 voxels. Maximum absolute COPE10 reconstruction difference: {error:.8g}.")

if __name__ == "__main__":
    main()
