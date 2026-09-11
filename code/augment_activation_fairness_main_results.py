#!/usr/bin/env python3
"""Augment the committed fairness-main bundle with post-statistics provenance.

The initial collector intentionally kept the first result bundle small.  This
script is safe to re-run after an interrupted copy: existing files are never
overwritten unless the incoming bytes are identical.  It collects all four
group contrasts, thresholded/cluster products when FEAT created them, and the
small files needed to document the cluster-thresholding calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results/reviewer/activation_fairness_main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_if_safe(source: Path, destination: Path) -> str:
    """Copy source, refusing to silently replace a different committed file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"refusing to overwrite different file: {destination}")
        return "identical"
    shutil.copy2(source, destination)
    return "copied"


def read_l3_output(work_root: Path) -> Path:
    manifest = work_root / "activation_fairness_main_jobs.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    matches = [row for row in rows if row.get("stage") == "l3"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one L3 row, found {len(matches)}")
    output = Path(matches[0]["output"])
    if not output.is_dir():
        raise FileNotFoundError(output)
    return output


def collect(work_root: Path, output_root: Path) -> Path:
    l3_output = read_l3_output(work_root)
    feat_root = l3_output / "cope1.feat"
    if not feat_root.is_dir():
        raise FileNotFoundError(feat_root)

    candidates: list[tuple[str, Path, Path]] = []
    # Shared group-level geometry and smoothness provenance.
    for name in ("mask.nii.gz", "stats/smoothness"):
        source = feat_root / name
        candidates.append(("shared", source, output_root / "poststats" / Path(name).name))

    labels = {
        1: "adjusted-mean-positive",
        2: "adjusted-mean-negative",
        3: "older-younger",
        4: "younger-older",
    }
    for contrast, label in labels.items():
        # Preserve the directory name already used by the initial collector
        # for contrast 1; use explicit labels for the newly added contrasts.
        destination_dir = (
            output_root / "focal_contrast1"
            if contrast == 1
            else output_root / f"contrast{contrast}_{label}"
        )
        for filename in (
            f"cope{contrast}.nii.gz",
            f"varcope{contrast}.nii.gz",
            f"zstat{contrast}.nii.gz",
        ):
            candidates.append((f"contrast{contrast}", feat_root / "stats" / filename, destination_dir / filename))
        for filename in (
            f"thresh_zstat{contrast}.nii.gz",
            f"cluster_mask_zstat{contrast}.nii.gz",
            f"cluster_zstat{contrast}.txt",
            f"cluster_zstat{contrast}_std.txt",
            f"lmax_zstat{contrast}.txt",
        ):
            candidates.append((f"contrast{contrast}", feat_root / filename, destination_dir / filename))

    records: list[dict[str, str]] = []
    for kind, source, destination in candidates:
        record = {
            "kind": kind,
            "source": str(source),
            "destination": str(destination.relative_to(output_root)),
            "source_exists": "0",
            "status": "missing",
            "bytes": "",
            "sha256": "",
        }
        if source.is_file() and source.stat().st_size > 0:
            record["source_exists"] = "1"
            record["status"] = copy_if_safe(source, destination)
            record["bytes"] = str(source.stat().st_size)
            record["sha256"] = sha256(source)
        records.append(record)

    inventory = output_root / "poststats_inventory.tsv"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    with inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["kind", "source", "destination", "source_exists", "status", "bytes", "sha256"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)

    found = sum(row["source_exists"] == "1" for row in records)
    print(f"PASS: augmented fairness-main bundle with {found}/{len(records)} available post-stat files")
    print(f"Inventory: {inventory}")
    return inventory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    collect(args.work_root.resolve(), args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
