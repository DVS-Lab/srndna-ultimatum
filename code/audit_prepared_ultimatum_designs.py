#!/usr/bin/env python3
"""Validate prepared repair designs against retained production nuisance columns.

Each prepared L1 FSF is rendered with ``feat_model`` only. Corrected task/PPI
columns are expected to differ, but all columns after the declared original
EVs (the generated confound tail) must match the retained design exactly.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from audit_server_rt_events import parse_vest_matrix
from prepare_sub144_imaging_repair import fsf_value


def block_difference(
    production: list[list[float]],
    candidate: list[list[float]],
    start: int,
    stop: int,
) -> tuple[float, float]:
    if len(production) != len(candidate):
        raise ValueError("design row counts differ")
    if len(production[0]) != len(candidate[0]):
        raise ValueError("design column counts differ")
    differences = [
        production[row][column] - candidate[row][column]
        for row in range(len(production))
        for column in range(start, stop)
    ]
    if not differences:
        return 0.0, 0.0
    return (
        max(abs(value) for value in differences),
        math.sqrt(sum(value * value for value in differences) / len(differences)),
    )


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def audit(manifest: Path, output: Path, feat_model: str) -> int:
    with manifest.open(newline="", encoding="utf-8") as stream:
        jobs = [
            row
            for row in csv.DictReader(stream, delimiter="\t")
            if row["stage"] == "l1"
        ]
    if len(jobs) != 6:
        raise ValueError(f"expected six L1 repair jobs, found {len(jobs)}")

    rows: list[dict[str, object]] = []
    failed = False
    for job in jobs:
        fsf = Path(job["fsf"])
        model_root = fsf.with_suffix("")
        result = subprocess.run(
            [feat_model, str(model_root)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"feat_model failed for {fsf} (exit {result.returncode}):\n{result.stdout}"
            )
        candidate_path = model_root.with_suffix(".mat")
        production_path = Path(job["source_fsf"]).parent / "design.mat"
        production = parse_vest_matrix(production_path)
        candidate = parse_vest_matrix(candidate_path)
        text = fsf.read_text(encoding="utf-8", errors="replace")
        original_evs_value = fsf_value(text, "evs_real")
        if original_evs_value is None:
            raise ValueError(f"missing evs_real in {fsf}")
        original_evs = int(original_evs_value)
        if len(production[0]) != len(candidate[0]):
            task_max = task_rmse = tail_max = tail_rmse = math.inf
        else:
            task_max, task_rmse = block_difference(
                production, candidate, 0, original_evs
            )
            tail_max, tail_rmse = block_difference(
                production, candidate, original_evs, len(production[0])
            )
        passed = (
            len(production) == len(candidate)
            and len(production[0]) == len(candidate[0])
            and tail_max <= 1e-7
            and task_max > 1e-7
        )
        failed = failed or not passed
        rows.append(
            {
                "model": job["model"],
                "run": job["run"],
                "npoints": len(candidate),
                "production_num_waves": len(production[0]),
                "candidate_num_waves": len(candidate[0]),
                "original_ev_columns": original_evs,
                "task_block_max_abs_difference": task_max,
                "task_block_rmse": task_rmse,
                "confound_tail_columns": len(candidate[0]) - original_evs,
                "confound_tail_max_abs_difference": tail_max,
                "confound_tail_rmse": tail_rmse,
                "task_changed_and_confound_tail_exact": int(passed),
            }
        )
        print(
            f"{'PASS' if passed else 'FAIL'}: {job['model']} run-{job['run']}, "
            f"task_max_diff={task_max:.6g}, confound_tail_max_diff={tail_max:.6g}"
        )
    write_tsv(output, rows)
    if failed:
        print(f"ERROR: one or more prepared designs failed; output={output}")
        return 1
    print(f"PASS: all six prepared designs passed; output={output}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feat-model", default="feat_model")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which(args.feat_model) is None:
        raise SystemExit(f"ERROR: feat_model not found: {args.feat_model}")
    manifest = args.manifest.resolve()
    output = (
        args.output.resolve()
        if args.output
        else manifest.parent / "prepared_design_audit.tsv"
    )
    return audit(manifest, output, args.feat_model)


if __name__ == "__main__":
    raise SystemExit(main())
