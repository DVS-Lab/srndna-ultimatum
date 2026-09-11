#!/usr/bin/env python3
"""Collect compact outputs from the two social-versus-computer L3 models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Sequence

from run_ultimatum_repair_jobs import complete


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results/reviewer/activation_social_computer"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_available(source: Path, destination: Path) -> bool:
    if not source.is_file() or source.stat().st_size == 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def collect(work_root: Path, output_root: Path) -> Path:
    manifest = work_root / "activation_social_computer_l3_jobs.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"collection output is not empty: {output_root}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 2 or {int(row["source_cope"]) for row in rows} != {8, 10}:
        raise ValueError("expected exactly the cope-8 and cope-10 L3 jobs")
    incomplete = [row["model"] for row in rows if not complete(row)]
    if incomplete:
        raise ValueError(f"incomplete L3 jobs: {', '.join(incomplete)}")

    model_names = {
        8: "social_offer_slope",
        10: "social_task_response",
    }
    inventory: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: int(item["source_cope"])):
        cope = int(row["source_cope"])
        model_root = output_root / model_names[cope]
        gfeat = Path(row["output"])
        feat = gfeat / "cope1.feat"
        if not feat.is_dir():
            raise FileNotFoundError(feat)

        sources: list[tuple[str, Path, Path]] = []
        for name in ("design.fsf", "design.mat", "design.con", "design.grp"):
            sources.append(("design", gfeat / name, model_root / "design" / name))
        for name in ("mask.nii.gz", "stats/smoothness"):
            sources.append(("poststats", feat / name, model_root / "poststats" / Path(name).name))
        for contrast in range(1, 5):
            contrast_root = model_root / f"contrast{contrast}"
            for name in (
                f"cope{contrast}.nii.gz",
                f"varcope{contrast}.nii.gz",
                f"zstat{contrast}.nii.gz",
            ):
                sources.append((f"contrast{contrast}", feat / "stats" / name, contrast_root / name))
            for name in (
                f"thresh_zstat{contrast}.nii.gz",
                f"cluster_mask_zstat{contrast}.nii.gz",
                f"cluster_zstat{contrast}_std.txt",
            ):
                sources.append((f"contrast{contrast}", feat / name, contrast_root / name))

        for kind, source, destination in sources:
            copied = copy_available(source, destination)
            inventory.append(
                {
                    "model": model_names[cope],
                    "source_cope": str(cope),
                    "kind": kind,
                    "source": str(source),
                    "destination": str(destination.relative_to(output_root)),
                    "exists": "1" if copied else "0",
                    "bytes": str(source.stat().st_size) if copied else "",
                    "sha256": sha256(source) if copied else "",
                }
            )

    inventory_path = output_root / "inventory.tsv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(inventory[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(inventory)
    found = sum(row["exists"] == "1" for row in inventory)
    print(
        f"PASS: collected {found}/{len(inventory)} available compact files "
        f"from two complete social-versus-computer L3 models; output={output_root}"
    )
    return inventory_path


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
