#!/usr/bin/env python3
"""Collect compact, auditable outputs from the fairness-main FEAT pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Sequence

from run_ultimatum_repair_jobs import complete


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "results/reviewer/activation_fairness_main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    counts = {
        stage: sum(row["stage"] == stage for row in rows)
        for stage in ("l1", "l2", "l3")
    }
    if counts != {"l1": 94, "l2": 47, "l3": 1}:
        raise ValueError(f"unexpected job counts: {counts}")
    return rows


def collect(work_root: Path, output_root: Path) -> Path:
    manifest = work_root / "activation_fairness_main_jobs.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"collection output is not empty: {output_root}")
    rows = read_manifest(manifest)
    incomplete = [
        f"{row['stage']}:{row.get('subject', '')}:{row['run']}"
        for row in rows
        if not complete(row)
    ]
    if incomplete:
        raise ValueError(f"pipeline is incomplete: {', '.join(incomplete[:10])}")

    provenance: list[dict[str, str]] = []
    for row in rows:
        fsf = Path(row["fsf"])
        source = Path(row["source_fsf"])
        if not fsf.is_file() or not source.is_file():
            raise FileNotFoundError(f"missing FSF provenance for {row}")
        provenance.append(
            {
                "stage": row["stage"],
                "subject": row.get("subject", ""),
                "model": row["model"],
                "run": row["run"],
                "model_policy": row["model_policy"],
                "expected_copes": row.get("expected_copes", ""),
                "expected_zstats": row.get("expected_zstats", ""),
                "source_fsf_sha256": sha256(source),
                "rendered_fsf_sha256": sha256(fsf),
                "complete": "1",
            }
        )

    l3 = next(row for row in rows if row["stage"] == "l3")
    l3_output = Path(l3["output"])
    group_design = output_root / "design"
    focal = output_root / "focal_contrast1"
    group_design.mkdir(parents=True)
    focal.mkdir(parents=True)

    for name in ("design.fsf", "design.mat", "design.con", "design.grp"):
        source = l3_output / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, group_design / name)

    focal_sources = {
        "cope1.nii.gz": l3_output / "cope1.feat/stats/cope1.nii.gz",
        "varcope1.nii.gz": l3_output / "cope1.feat/stats/varcope1.nii.gz",
        "zstat1.nii.gz": l3_output / "cope1.feat/stats/zstat1.nii.gz",
        "thresh_zstat1.nii.gz": l3_output / "cope1.feat/thresh_zstat1.nii.gz",
        "cluster_mask_zstat1.nii.gz": l3_output / "cope1.feat/cluster_mask_zstat1.nii.gz",
        "cluster_zstat1.txt": l3_output / "cope1.feat/cluster_zstat1.txt",
    }
    for name, source in focal_sources.items():
        if source.is_file() and source.stat().st_size > 0:
            shutil.copy2(source, focal / name)

    inventory = output_root / "job_provenance.tsv"
    with inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(provenance[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(provenance)
    print(
        "PASS: collected compiled L3 design, available focal contrast-1 "
        f"outputs, and {len(provenance)} job provenance rows; output={output_root}"
    )
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
