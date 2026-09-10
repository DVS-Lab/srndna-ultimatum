#!/usr/bin/env python3
"""Collect compact, exact L3 repair designs from scratch for Git provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from typing import Sequence


DESIGN_SUFFIXES = (".fsf", ".mat", ".con", ".grp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(manifest: Path, output_dir: Path) -> Path:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream, delimiter="\t")
            if row["stage"] == "l3"
        ]
    if len(rows) != 6:
        raise ValueError(f"expected six L3 rows in {manifest}, found {len(rows)}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"collection directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[dict[str, str]] = []
    for row in rows:
        source_fsf = Path(row["fsf"])
        source_base = source_fsf.with_suffix("")
        destination = output_dir / row["model"] / row["run"]
        destination.mkdir(parents=True, exist_ok=False)
        hashes: dict[str, str] = {}
        for suffix in DESIGN_SUFFIXES:
            source = source_base.with_suffix(suffix)
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"missing compiled L3 design artifact: {source}")
            target = destination / f"design{suffix}"
            shutil.copy2(source, target)
            hashes[f"design{suffix}_sha256"] = sha256(target)

        inventory.append(
            {
                "model": row["model"],
                "variant": row["run"],
                "covariate_policy": row["covariate_policy"],
                "changed_ev_columns": row["changed_ev_columns"],
                "n_evs": row["n_evs"],
                "design_rank": row["design_rank"],
                "expected_zstats": row["expected_zstats"],
                "source_production_fsf": row["source_fsf"],
                "sub144_repaired_l2": row["sub144_repaired_l2"],
                "standard_image": row["standard_image"],
                "collected_design_dir": str(destination.relative_to(output_dir)),
                **hashes,
            }
        )

    inventory_path = output_dir / "inventory.tsv"
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(inventory[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(inventory)
    print(
        f"PASS: collected {len(inventory)} exact L3 FSF/matrix/contrast/group "
        f"bundles; inventory={inventory_path}"
    )
    return inventory_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/reviewer/l3_repair_designs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    collect(args.manifest.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
