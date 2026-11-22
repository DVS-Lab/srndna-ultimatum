"""Identify retained sub-104 z images against the two archived ROI series."""
import argparse
import csv
import io
import json
import os
import re
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from extract_single_trial_choice import VERSION, archived_trial_ids, read_rows, release_image

REGIONS = ("VMPFC", "aINS", "dACC", "VS")
TOL = 1e-5  # Same ROI fingerprint tolerance as the choice extractor.


def references(repo, run):
    rows = [r for r in read_rows(repo / VERSION / "sub104_roi_reference.tsv") if int(r["run"]) == run]
    assert [int(r["trial"]) for r in rows] == list(range(1, 73))
    latest = np.array([[float(r[k]) for k in REGIONS] for r in rows])
    old = []
    for key in REGIONS:
        name = f"derivatives/singletrial/sub-104/sub-104_run-{run:02}_mask-roi-{key}.txt"
        blob = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{name}"])
        old.append(np.loadtxt(io.BytesIO(blob)))
    return {"datapaper_2025": latest, "ultimatum_2022": np.column_stack(old)}


def candidates(root, run):
    folder = root / f"sub-104/LSS-images_task-ultimatum_model-01_type-act_run-{run:02}"
    if folder.is_dir():
        numbered = {}
        for path in sorted(folder.glob("zstat_trial-*.nii.gz")):
            m = re.fullmatch(r"zstat_trial-(\d+)\.nii\.gz", path.name)
            if m:
                trial = int(m[1])
                if not 1 <= trial <= 72 or trial in numbered:
                    raise ValueError(f"Invalid or duplicate numbered trial: {path}")
                numbered[trial] = path
        if numbered:
            yield "numbered", sorted(numbered.items())
        for path in sorted(folder.glob("*merged*z*.nii.gz")):
            yield "merged", path
    for base in ("single_trials", "singletrial"):
        path = root.parent / base / "sub-104" / f"sub-104_task-ultimatum_run-{run:02}_singletrial-Act.nii.gz"
        if path.is_file():
            yield "packed", path


def compare(kind, source, public_img, public_data, known, masks, refs):
    result = {"format": kind}
    grid = public_img.shape[:3]

    def check_grid(img):
        if img.shape[:3] != grid or not np.allclose(img.affine, public_img.affine, atol=1e-4, rtol=0):
            raise ValueError("spatial grid differs")

    if kind == "numbered":
        def volumes():
            for trial, path in source:
                img = nib.load(path)
                check_grid(img)
                if len(img.shape) != 3:
                    raise ValueError("numbered image is not 3D")
                yield trial, np.asarray(img.dataobj, dtype=np.float32)
        iterator = volumes()
    else:
        img = nib.load(source)
        check_grid(img)
        if len(img.shape) != 4 or img.shape[-1] not in (len(known), 72):
            return dict(result, status="unexpected volume count", shape=list(img.shape))
        # For a partial merge, validate the putative public order below.
        ids = known if img.shape[-1] == len(known) else list(range(1, 73))
        data = np.asarray(img.dataobj, dtype=np.float32)
        iterator = ((trial, data[..., i]) for i, trial in enumerate(ids))

    public_order = {j: i for i, j in enumerate(known)}
    observed, ids, exact, diffs, correlations = [], [], [], [], []
    for trial, volume in iterator:
        if not np.isfinite(volume).all():
            raise ValueError("nonfinite voxel values")
        ids.append(trial)
        observed.append([np.mean(volume[m], dtype=np.float64) for m in masks])
        if trial in public_order:
            other = public_data[..., public_order[trial]]
            exact.append(np.array_equal(volume, other))
            diffs.append(float(np.max(np.abs(volume - other))))
            active = (volume != 0) | (other != 0)
            a, b = volume[active].astype(float), other[active].astype(float)
            if a.size > 1 and a.std() > 0 and b.std() > 0:
                correlations.append(float(np.corrcoef(a, b)[0, 1]))
    result.update(volumes=len(ids), overlap=len(exact), exact_overlap=sum(exact),
                  max_absolute_voxel_difference=max(diffs, default=None),
                  min_spatial_correlation=min(correlations, default=None))
    observed = np.asarray(observed)
    for label, ref in refs.items():
        errors = np.abs(observed[:, None] - ref[None, :]).max(axis=2)
        by_trial = errors[np.arange(len(ids)), np.array(ids)-1]
        unique_correct = (by_trial <= TOL) & ((errors <= TOL).sum(axis=1) == 1)
        result[label] = {"matching_trial_ids": int(unique_correct.sum()),
                         "max_expected_trial_roi_error": float(by_trial.max())}
    result["release_equivalent_complete"] = (
        ids == list(range(1, 73)) and len(exact) == len(known) and all(exact)
        and result["datapaper_2025"]["matching_trial_ids"] == 72)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=Path(os.getenv("SRNDNA_DATASET_ROOT", "/ZPOOL/data/datasets/ds003745-work")))
    parser.add_argument("--lss-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [("release", args.dataset / "derivatives/fsl"), ("analysis", args.repo / "derivatives/fsl")]
    # Search related project and backup directories without walking full FEAT trees.
    for parent in (Path("/ZPOOL/data/projects"), Path("/ZPOOL/data/datasets"), Path("/ZPOOL/data/scratch")):
        folders = sorted(set(parent.glob("srndna*")) | set(parent.glob("ds003745*")))
        for folder in folders:
            for sub in ("derivatives/fsl", "bids/derivatives/fsl", "fsl"):
                roots.append((f"{parent.name}/{folder.name}/{sub}", folder/sub))
    for i, path in enumerate(args.lss_root):
        roots.append((f"extra-{i+1}", path))
    manifest = read_rows(args.repo / VERSION / "openneuro_2.2.0_inputs.tsv")
    report = []
    for run in (1, 2):
        row = next(r for r in manifest if r["subject"]=="sub-104" and int(r["run"])==run)
        pub = release_image(args.dataset / row["image"], row)
        known, _ = archived_trial_ids(pub, args.repo, run)
        data = np.asarray(pub.dataobj, dtype=np.float32)
        refs = references(args.repo, run)
        masks = [nib.load(args.repo / f"masks/roi-{k}.nii.gz").get_fdata()>0 for k in REGIONS]
        seen = set()
        for label, root in roots:
            for kind, source in candidates(root, run):
                representative = source[0][1] if kind == "numbered" else source
                identity = (kind, representative.resolve())
                if identity in seen:
                    continue
                seen.add(identity)
                print(f"Checking run {run}, {kind}: {representative}", flush=True)
                try:
                    record = compare(kind, source, pub, data, known, masks, refs)
                except ValueError as error:
                    record = dict(format=kind, status=str(error))
                report.append(dict(run=run, location=label, **record))
    if not any(r["run"]==2 and r["format"]=="numbered" and r.get("volumes")==72 for r in report):
        print("No complete numbered source for run 2 was evaluated. Set SRNDNA_LSS_ROOT if its location is known.")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
