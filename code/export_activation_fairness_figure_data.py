#!/usr/bin/env python3
"""Export compact manuscript source data for the task-wide offer-size map."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    result_root = REPO_ROOT / "results/reviewer/activation_fairness_main"
    source_root = REPO_ROOT / "results/manuscript/source_data"
    source_root.mkdir(parents=True, exist_ok=True)

    inputs = {
        "thresholded_zstat": result_root / "focal_contrast1/thresh_zstat1.nii.gz",
        "cluster_table": result_root / "focal_contrast1/cluster_zstat1_std.txt",
        "smoothness": result_root / "poststats/smoothness",
        "mask": result_root / "poststats/mask.nii.gz",
        "design": result_root / "design/design.fsf",
    }
    for path in inputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    outputs = {
        "thresholded_zstat": source_root / "figure4_offer_size_positive_zstat.nii.gz",
        "cluster_table": source_root / "figure4_offer_size_positive_clusters.tsv",
    }
    shutil.copy2(inputs["thresholded_zstat"], outputs["thresholded_zstat"])
    shutil.copy2(inputs["cluster_table"], outputs["cluster_table"])

    rows = [
        ("analysis_status", "revision analysis; not part of submitted workflow"),
        ("sample_size", "47"),
        ("first_level_contrast", "equal-weighted mean offer-size slope across computer, age-similar, and age-dissimilar partners"),
        ("group_model", "FLAME 1+2; intercept, centered age group, centered sex, tSNR, mean FD, event-corrected task-wide mean RT"),
        ("voxel_threshold", "Z > 3.1"),
        ("cluster_threshold", "family-wise corrected p < .05"),
        ("minimum_cluster_extent_voxels", "30"),
        ("dlh", "0.207865"),
        ("search_volume_voxels", "59297"),
    ]
    for name, path in inputs.items():
        rows.append((f"input_{name}_sha256", sha256(path)))
    for name, path in outputs.items():
        rows.append((f"output_{name}_sha256", sha256(path)))

    provenance = source_root / "figure4_offer_size_positive_provenance.tsv"
    with provenance.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("field", "value"))
        writer.writerows(rows)
    print(f"PASS: exported Figure 4 source data to {source_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
