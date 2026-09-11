#!/usr/bin/env python3
"""Export the compact, corrected source data needed for manuscript Figure 4.

This script is intended to run on the Linux analysis host after the corrected
group model has completed.  It reads the exact participant inputs from the
tracked, rendered FSF, isolates the significant cluster from the corrected
younger-minus-older contrast, and extracts the participant-level cope 7 values
entered into that model. It does not modify any FEAT/GFEAT directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPAIR_ROOT = Path(
    os.environ.get(
        "SRNDNA_ULTIMATUM_L3_REPAIR_ROOT",
        "/ZPOOL/data/scratch/srndna-ultimatum-l3-repair-v2",
    )
)
DESIGN_RELATIVE = Path(
    "results/reviewer/l3_repair_designs/dmn-age/"
    "reported-covariates-corrected/design.fsf"
)
MODEL_RELATIVE = Path(
    "outputs/dmn-age_reported-covariates-corrected.gfeat/cope1.feat"
)
OUTPUT_ROOT_RELATIVE = Path("results/manuscript/source_data")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"required FSL executable not found: {name}")


def parse_feat_inputs(path: Path) -> list[Path]:
    pattern = re.compile(r'^set feat_files\((\d+)\) "([^"]+)"$')
    indexed: list[tuple[int, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            indexed.append((int(match.group(1)), Path(match.group(2))))
    indexed.sort()
    if [index for index, _ in indexed] != list(range(1, 48)):
        raise ValueError(f"expected exactly 47 sequential feat_files in {path}")
    return [source for _, source in indexed]


def participant_from_path(path: Path) -> str:
    match = re.search(r"/(sub-[A-Za-z0-9]+)/", str(path))
    if not match:
        raise ValueError(f"cannot parse participant identifier from {path}")
    return match.group(1)


def mean_in_mask(image: Path, mask: Path) -> float:
    if not image.is_file():
        raise FileNotFoundError(f"missing participant input: {image}")
    value = run(["fslstats", str(image), "-k", str(mask), "-M"])
    return float(value)


def load_age_groups(path: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            participant = row["subjID"]
            younger = float(row["younger"])
            older = float(row["older"])
            if (younger, older) == (1.0, 0.0):
                groups[participant] = "younger"
            elif (younger, older) == (0.0, 1.0):
                groups[participant] = "older"
            else:
                raise ValueError(f"invalid age coding for {participant}")
    return groups


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--repair-root", type=Path, default=DEFAULT_REPAIR_ROOT)
    parser.add_argument("--expected-voxels", type=int, default=29)
    parser.add_argument("--cluster-label", type=int, default=1)
    args = parser.parse_args()

    repository = args.repository.resolve()
    design = repository / DESIGN_RELATIVE
    model = args.repair_root.resolve() / MODEL_RELATIVE
    source_cluster_labels = model / "cluster_mask_zstat3.nii.gz"
    source_thresholded_zstat = model / "thresh_zstat3.nii.gz"
    output_root = repository / OUTPUT_ROOT_RELATIVE

    for executable in ("fslmaths", "fslstats"):
        require_executable(executable)
    for required in (design, source_cluster_labels, source_thresholded_zstat):
        if not required.is_file():
            raise FileNotFoundError(f"missing required corrected result: {required}")

    inputs = parse_feat_inputs(design)
    age_groups = load_age_groups(
        repository / "behavioral_analyses/data/participant_L3_47.csv"
    )
    participants = [participant_from_path(path) for path in inputs]
    if len(participants) != len(set(participants)):
        raise ValueError("corrected DMN design contains duplicate participants")
    if set(participants) != set(age_groups):
        raise ValueError("corrected DMN design and participant table do not match")

    output_root.mkdir(parents=True, exist_ok=True)
    mask_output = output_root / "figure4_dmn_corrected_cluster_mask.nii.gz"
    zstat_output = output_root / "figure4_dmn_corrected_cluster_zstat.nii.gz"
    roi_output = output_root / "figure4_dmn_corrected_roi.tsv"
    provenance_output = output_root / "figure4_dmn_provenance.tsv"

    with tempfile.TemporaryDirectory(prefix="srndna-dmn-figure-") as temporary:
        temporary_root = Path(temporary)
        temporary_mask = temporary_root / "cluster_mask.nii.gz"
        temporary_zstat = temporary_root / "cluster_zstat.nii.gz"
        run(
            [
                "fslmaths",
                str(source_cluster_labels),
                "-thr",
                str(args.cluster_label),
                "-uthr",
                str(args.cluster_label),
                "-bin",
                str(temporary_mask),
            ]
        )
        voxels_text = run(["fslstats", str(temporary_mask), "-V"])
        voxels = int(float(voxels_text.split()[0]))
        if voxels != args.expected_voxels:
            raise ValueError(
                f"corrected DMN cluster has {voxels} voxels, expected {args.expected_voxels}"
            )
        run(
            [
                "fslmaths",
                str(source_thresholded_zstat),
                "-mas",
                str(temporary_mask),
                str(temporary_zstat),
            ]
        )

        roi_rows: list[dict[str, object]] = []
        for index, difference_input in enumerate(inputs, start=1):
            participant = participants[index - 1]
            difference = mean_in_mask(difference_input, temporary_mask)
            roi_rows.append(
                {
                    "design_index": index,
                    "participant": participant,
                    "age_group": age_groups[participant],
                    "similar_minus_dissimilar": f"{difference:.9g}",
                }
            )

        shutil.copyfile(temporary_mask, mask_output)
        shutil.copyfile(temporary_zstat, zstat_output)
        write_tsv(
            roi_output,
            [
                "design_index",
                "participant",
                "age_group",
                "similar_minus_dissimilar",
            ],
            roi_rows,
        )

    provenance_rows = [
        {"artifact": "rendered_corrected_design", "value": str(DESIGN_RELATIVE)},
        {"artifact": "rendered_corrected_design_sha256", "value": sha256(design)},
        {"artifact": "corrected_group_model", "value": str(model)},
        {"artifact": "contrast", "value": "zstat3: younger > older"},
        {
            "artifact": "participant_image",
            "value": "L2 cope7: in_p-out_p (similar-minus-dissimilar offer modulation)",
        },
        {"artifact": "cluster_label", "value": args.cluster_label},
        {"artifact": "cluster_voxels", "value": args.expected_voxels},
        {"artifact": "cluster_mask_sha256", "value": sha256(mask_output)},
        {"artifact": "cluster_zstat_sha256", "value": sha256(zstat_output)},
        {"artifact": "participant_rows", "value": len(roi_rows)},
        {
            "artifact": "inference",
            "value": "FLAME 1+2; Z > 3.1; cluster-corrected p < .05",
        },
        {
            "artifact": "display_warning",
            "value": "participant values are descriptive because the cluster is group-selected",
        },
    ]
    write_tsv(provenance_output, ["artifact", "value"], provenance_rows)

    print(
        "PASS: exported corrected DMN Figure 4 source data: "
        f"{len(roi_rows)} participants, {args.expected_voxels} voxels"
    )
    print(f"PASS: output root={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
