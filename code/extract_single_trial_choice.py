"""Join the released LSS volumes and corrected, opposite-run network contrasts to choices."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from build_event_corrected_trials import event_trials, TRIAL_PATTERN
from extract_figure4_conditions import read_rows, same_grid, verified

VERSION = Path("results/single_trial_choice/v1")
PREP = Path("results/palm_age_main/v1/preparation")
CLUSTER = Path("results/age_figure_audit/v1/attempt-20261030T161854Z-2NzpRx/outputs/dmn/selected_age_roi_49.nii.gz")
SOURCE_SCRATCH = Path("/ZPOOL/data/scratch")


def trial_order(path, blob):
    raw = path.read_bytes()
    if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != blob:
        raise ValueError(f"Events differ from OpenNeuro 2.2.0: {path}")
    rows = read_rows(path)
    # Reproduce the LSS EV generator's keep-last rule before filtering choices.
    positions = {}
    decisions = []
    for index, row in enumerate(rows):
        key = (float(row["onset"]), float(row["duration"]))
        event = row["trial_type"]
        if "event_RT" not in event and "block_" not in event:
            positions[key] = index
        if TRIAL_PATTERN.match(event):
            decisions.append(key)
    lss = sorted(positions, key=positions.get)
    if lss != decisions or len(lss) != 72 or np.any(np.diff([k[0] for k in lss]) <= 0):
        raise ValueError(f"LSS volume order differs from trial order: {path}")
    return event_trials(path, response_delay=1.0)


def release_image(path, record):
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Released image size differs: {path}")
    h = hashlib.new(record["algorithm"].lower())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    if h.hexdigest() != record["digest"]:
        raise ValueError(f"Released image checksum differs: {path}")
    image = nib.load(path)
    expected = 55 if (record["subject"],record["run"]) == ("sub-104","2") else 72
    if len(image.shape) != 4 or image.shape[-1] != expected:
        raise ValueError(f"Unexpected released volume count: {path}")
    return image


def masks(repo):
    base = nib.load(repo/PREP/"full-rt__nppi-dmn/mask.nii.gz")
    ecn = nib.load(repo/PREP/"full-rt__nppi-ecn/mask.nii.gz")
    same_grid(ecn, base)
    common = (base.get_fdata() > 0) & (ecn.get_fdata() > 0)
    maps = {}
    for name, path in {"pole": VERSION/"right_frontal_pole.nii.gz", "focus": CLUSTER,
                        "vmpfc": Path("masks/roi-VMPFC.nii.gz"), "insula": Path("masks/roi-aINS.nii.gz")}.items():
        image = nib.load(repo/path)
        same_grid(image, base)
        maps[name] = (image.get_fdata() > 0) & common
    if maps["focus"].sum() != 49 or maps["vmpfc"].sum() != 19 or maps["insula"].sum() != 38 or maps["pole"].sum() != 2018:
        raise ValueError("Analysis mask or common coverage changed")
    for name, index in [("dmn", 3), ("ecn", 7)]:
        image = nib.load(repo/f"masks/nan_rPNAS_2mm_net000{index}.nii.gz")
        weights = resample_from_to(image, base, order=1).get_fdata()
        maps[name] = np.where(common, weights, 0.)
    return base, common, maps


def activation(image, base, common, maps):
    # Evaluate every volume on its own spatial grid. Masks use nearest-neighbour interpolation.
    grid = (image.shape[:3], image.affine)
    same = grid[0] == base.shape and np.allclose(grid[1], base.affine, rtol=0, atol=1e-4)
    def on_grid(values, order):
        if same:
            return values
        img = nib.Nifti1Image(values.astype(np.float32), base.affine)
        return resample_from_to(img, grid, order=order).get_fdata()
    region = on_grid(common, 0) > 0
    data = np.asarray(image.dataobj, dtype=np.float32)[region]
    valid = np.isfinite(data).all(axis=1) & np.any(data != 0, axis=1)
    data = data[valid]
    values, fractions = {}, {}
    for key, mask in maps.items():
        network = key in {"dmn", "ecn"}
        w = on_grid(mask, 1 if network else 0)[region]
        if network:
            # Signed spatial regression with intercept, one network at a time.
            v = w[valid].astype(float)
            v -= v.mean()
            weights = v / np.dot(v, v)
            fractions[key] = float(valid.mean())
        else:
            support = w > .5
            n = np.sum(support & valid)
            if n == 0:
                raise ValueError(f"No LSS support in {key}: {image.get_filename()}")
            weights = support[valid] / n
            fractions[key] = float(n / support.sum())
        values[key] = weights @ data
        if not np.isfinite(values[key]).all() or np.ptp(values[key]) == 0:
            raise ValueError(f"Invalid LSS series in {key}: {image.get_filename()}")
    return values, fractions


def network_runs(repo, scratch, base, maps):
    results = {}
    for network, targets in [("dmn", ["pole", "focus"]), ("ecn", ["vmpfc"])]:
        inventory = read_rows(repo/PREP/f"full-rt__nppi-{network}/input_inventory.tsv")
        for row in inventory:
            if row["role"] != "L1 COPE7":
                continue
            original = Path(row["path"])
            path = verified(scratch/original.relative_to(SOURCE_SCRATCH), row["sha256"])
            feat = path.parents[1]
            run = int(feat.name.removeprefix("L1_run-").removesuffix(".feat"))
            subject = feat.parent.name
            image = nib.load(path)
            same_grid(image, base)
            data = image.get_fdata()
            for target in targets:
                x = data[maps[target]]
                key = (subject, run, network + "_" + target)
                if key in results or not np.isfinite(x).all():
                    raise ValueError(f"Invalid or duplicate run contrast: {path}")
                results[key] = float(x.mean())
    return results


def recover_104(image, event_path, roots, audit=None):
    """Only use numbered source volumes with matching EVs and byte-equivalent image values."""
    def voxel_hash(vol):
        return hashlib.sha256(np.ascontiguousarray(vol, dtype="<f4").tobytes()).digest()
    published = np.asarray(image.dataobj)
    hashes = [voxel_hash(published[...,i]) for i in range(55)]
    if len(set(hashes)) != 55:
        raise ValueError("Duplicate public volumes prevent unambiguous recovery")
    events = [r for r in read_rows(event_path) if TRIAL_PATTERN.match(r["trial_type"])]
    if audit is None:
        audit = {}
    audit.update(candidate_roots=len(set(roots)),source_series_found=0,max_numbered_files=0,
                 max_ev_and_grid_matches=0,max_exact_public_matches=0)
    for root in dict.fromkeys(roots):
        folder = root/"sub-104/LSS-images_task-ultimatum_model-01_type-act_run-02"
        evbase = root/"EVfiles/sub-104/SingleTrialEVs"
        evs = next((p for p in [evbase/"task-ultimatum/run02",evbase/"run02"] if p.is_dir()),None)
        if evs is None or not folder.is_dir():
            continue
        paths = sorted(folder.glob("zstat_trial-*.nii.gz"))
        audit["source_series_found"] += 1
        audit["max_numbered_files"] = max(audit["max_numbered_files"],len(paths))
        values = {}
        ids = {}
        for path in paths:
            m = re.fullmatch(r"zstat_trial-(\d+)\.nii\.gz",path.name)
            if not m:
                continue
            j = int(m[1])
            if j in values or not 1 <= j <= 72:
                raise ValueError(f"Ambiguous numbered trial: {path}")
            ev = evs/f"trialmodel-{j}_estimage-single.tsv"
            if not ev.is_file():
                continue
            row = np.loadtxt(ev)
            wanted = [float(events[j-1]["onset"]),float(events[j-1]["duration"]),1.]
            if row.shape != (3,) or not np.allclose(row,wanted,rtol=0,atol=1e-5):
                continue
            single = nib.load(path)
            if single.shape != image.shape[:3] or not np.allclose(single.affine,image.affine,rtol=0,atol=.0001):
                continue
            vol = np.asarray(single.dataobj,dtype=np.float32)
            sig = voxel_hash(vol)
            if sig in ids:
                raise ValueError("Two trial IDs have identical source volumes")
            ids[sig] = j
            values[j] = vol
        audit["max_ev_and_grid_matches"] = max(audit["max_ev_and_grid_matches"],len(values))
        audit["max_exact_public_matches"] = max(audit["max_exact_public_matches"],sum(h in ids for h in hashes))
        if all(h in ids for h in hashes):
            mapped = [ids[h] for h in hashes]
            if mapped != sorted(mapped):
                raise ValueError("Numbered source images disagree with release packing order")
            indices = sorted(values)
            if len(indices) < 72:
                # Include only exact public matches when the full 72-volume series is unavailable.
                indices = mapped
            result = nib.Nifti1Image(np.stack([values[j] for j in indices],axis=3),image.affine)
            return result, indices
    return None, []


def write_trials(path, rows):
    with path.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract(repo, dataset, scratch, output, activation_only=False, lss_roots=()):
    prep = repo/PREP/"full-rt__nppi-dmn"
    roster = read_rows(prep/"participants.tsv")
    manifest = read_rows(repo/VERSION/"openneuro_2.2.0_inputs.tsv")
    keys = [(r["subject"], int(r["run"])) for r in manifest]
    expected = {(r["subject"], i) for r in roster for i in (1,2)}
    if len(keys) != 94 or set(keys) != expected or Counter(r["variance_group"] for r in roster) != {"1":25,"2":22}:
        raise ValueError("Mismatch between the published cohort and the image release")
    design = np.loadtxt((prep/"design.mat").read_text().split("/Matrix")[1].splitlines())
    columns = [r["name"] for r in read_rows(prep/"design_columns.tsv")]
    demographics = {r["subject"]: dict(age=int(r["variance_group"]=="2"),
                      **{c:design[i,columns.index(c)] for c in ("ismale","tsnr","fd_mean")}) for i,r in enumerate(roster)}
    base, common, maps = masks(repo)
    ppi = {} if activation_only else network_runs(repo, scratch, base, maps)
    if not activation_only and len(ppi) != 3*len(keys):
        raise ValueError("Incomplete run-level network contrasts")
    rows, extra_rows, coverage, scheduled = [], [], [], 0
    recovered_ids = []
    recovery_audit = {}
    for idx, (record, (subject, run)) in enumerate(zip(manifest, keys)):
        name = f"{subject}_task-ultimatum_run-{run:02d}_events.tsv"
        event_path = repo/"source_data/bids"/subject/"func"/name
        trials = trial_order(event_path, record["events_git_blob"])
        scheduled += len(trials)
        imaging = release_image(dataset/record["image"], record)
        partial = imaging.shape[-1] != 72
        ids = list(range(1,73))
        if partial:
            if activation_only:
                continue
            imaging, ids = recover_104(imaging, event_path, lss_roots, recovery_audit)
            recovered_ids = ids
            if imaging is None:
                continue
        series, fractions = activation(imaging, base, common, maps)
        coverage.append(fractions)
        for j,trial_id in enumerate(ids):
            trial = trials[trial_id-1]
            if trial["human"] != "1" or trial["missed"] == "1":
                continue
            row = dict(subject=subject,run=run,trial=trial_id,offer=trial["offer"],accept=trial["accept"],
                       partner=int(trial["ingroup"])-.5,fair=int(trial["block_fair"]),**demographics[subject])
            for k, v in series.items():
                row[k+"_within"] = v[j] - v.mean()
                row[k+"_mean"] = v.mean()
            for measure in (() if activation_only else ("dmn_pole","dmn_focus","ecn_vmpfc")):
                row[measure] = ppi[subject, 3-run, measure]
            (extra_rows if partial else rows).append(row)
        if (idx+1) % 10 == 0:
            print(f"Validated {idx+1} / {len(keys)} imaging runs",flush=True)
    if scheduled != 6768 or len(rows) != 4382:
        raise ValueError(f"Trial alignment failed: {scheduled} scheduled, {len(rows)} modeled")
    output.mkdir(parents=True,exist_ok=True)
    write_trials(output/"private_trials.tsv", rows)
    if extra_rows:
        write_trials(output/"private_recovered_trials.tsv",rows+extra_rows)
    report = dict(participants=len(roster),released_runs=len(keys),complete_runs=93,scheduled=scheduled,responded_human=len(rows),
                  incomplete_run="sub-104 run-02: 55 public volumes for 72 scheduled offers",
                  numbered_source_search=recovery_audit,recovered_volume_ids=recovered_ids,additional_recovered_human_choices=len(extra_rows),
                  openneuro="ds003745 2.2.0",ppi_files_checked=0 if activation_only else 2*len(keys),
                  min_fraction_valid={k:min(d[k] for d in coverage) for k in coverage[0]},
                  run_to_run_pearson={k:float(np.corrcoef([[ppi[s["subject"],i,k] for s in roster] for i in (1,2)])[0,1])
                                       for k in (() if activation_only else ("dmn_pole","dmn_focus","ecn_vmpfc"))})
    (output/"validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--dataset",type=Path,default=Path("/ZPOOL/data/datasets/ds003745-work"))
    p.add_argument("--scratch",type=Path,default=SOURCE_SCRATCH)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--activation-only",action="store_true",help="Validate and fit the first two blocks using the public data only")
    p.add_argument("--lss-root",type=Path,action="append",default=[],help="Additional retained numbered-trial source directory")
    a=p.parse_args()
    roots = a.lss_root + [a.dataset/"derivatives/fsl", a.repo/"derivatives/fsl", Path("/ZPOOL/data/projects/srndna-datapaper-code/derivatives/fsl"),Path("/ZPOOL/data/projects/srndna-data/derivatives/fsl")]
    extract(a.repo,a.dataset,a.scratch,a.output,a.activation_only,roots)
