#!/usr/bin/env python3
"""Compile and validate prepared Ultimatum L3 designs without fitting images."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from prepare_ultimatum_l3_repair import matrix_rank


def read_header_value(path: Path, key: str) -> int:
    prefix = f"/{key}\t"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return int(line[len(prefix) :])
    raise ValueError(f"missing /{key} in {path}")


def read_matrix(path: Path) -> list[list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("/Matrix") + 1
    except ValueError as error:
        raise ValueError(f"missing /Matrix in {path}") from error
    matrix = [
        [float(value) for value in line.split()]
        for line in lines[start:]
        if line.strip()
    ]
    if not matrix:
        raise ValueError(f"empty matrix in {path}")
    return matrix


def audit(manifest: Path) -> None:
    if shutil.which("feat_model") is None:
        raise SystemExit("ERROR: feat_model was not found on PATH")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream, delimiter="\t")
            if row["stage"] == "l3"
        ]
    if not rows:
        raise ValueError(f"no L3 rows in {manifest}")

    for row in rows:
        fsf = Path(row["fsf"]).resolve()
        if fsf.parent != (manifest.parent / "fsf").resolve():
            raise ValueError(f"FSF is outside the prepared scratch tree: {fsf}")
        basename = fsf.with_suffix("")
        result = subprocess.run(
            ["feat_model", str(basename)],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"feat_model failed for {fsf}: {detail}")

        design_mat = basename.with_suffix(".mat")
        design_con = basename.with_suffix(".con")
        design_grp = basename.with_suffix(".grp")
        matrix = read_matrix(design_mat)
        waves = read_header_value(design_mat, "NumWaves")
        points = read_header_value(design_mat, "NumPoints")
        rank = matrix_rank(matrix)
        expected_waves = int(row["n_evs"])
        expected_rank = int(row["design_rank"])
        expected_contrasts = int(row["expected_zstats"])
        if (waves, points, rank) != (expected_waves, 47, expected_rank):
            raise ValueError(
                f"unexpected compiled design for {fsf}: "
                f"waves={waves}, points={points}, rank={rank}"
            )
        if read_header_value(design_con, "NumContrasts") != expected_contrasts:
            raise ValueError(f"unexpected contrast count in {design_con}")
        if read_header_value(design_grp, "NumPoints") != 47:
            raise ValueError(f"unexpected group row count in {design_grp}")
        print(
            f"PASS: {row['model']} {row['run']}: "
            f"47 rows, {waves} EVs, rank {rank}, {expected_contrasts} contrasts"
        )

    print(f"PASS: compiled and validated {len(rows)} prepared L3 designs")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit(args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
