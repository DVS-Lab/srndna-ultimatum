#!/usr/bin/env python3
"""Run prepared, isolated Ultimatum repair FEAT jobs with bounded concurrency."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence


SAFE_STAGE = {"l1", "l2"}


def read_manifest(path: Path, stage: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t") if row["stage"] == stage]
    if not rows:
        raise ValueError(f"no {stage} jobs in {path}")
    return rows


def complete(row: dict[str, str]) -> bool:
    output = Path(row["output"])
    if row["stage"] == "l1":
        required = (output / "stats" / "cope1.nii.gz", output / "stats" / "cope7.nii.gz")
    else:
        required = (output / "cope7.feat" / "stats" / "cope1.nii.gz",)
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def validate_job(row: dict[str, str], resume: bool) -> str:
    fsf = Path(row["fsf"])
    output = Path(row["output"])
    if not fsf.is_file():
        raise FileNotFoundError(fsf)
    for value in row["inputs"].split("|"):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"missing job input: {path}")
    if output.exists():
        if resume and complete(row):
            return "skip_complete"
        raise FileExistsError(
            f"output already exists and will not be overwritten: {output}"
        )
    return "run"


def ensure_identity_registration(output: Path) -> None:
    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        raise EnvironmentError("FSLDIR is not set")
    identity = Path(fsl_dir) / "etc" / "flirtsch" / "ident.mat"
    mean_func = output / "mean_func.nii.gz"
    if not identity.is_file() or not mean_func.is_file():
        raise FileNotFoundError(f"identity registration inputs missing for {output}")
    reg = output / "reg"
    reg.mkdir(exist_ok=True)
    links = {
        reg / "example_func2standard.mat": identity,
        reg / "standard2example_func.mat": identity,
        reg / "standard.nii.gz": mean_func,
    }
    for link, target in links.items():
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)


def run_one(row: dict[str, str], log_dir: Path) -> tuple[str, int]:
    label = f"{row['stage']}_{row['model']}_run-{row['run'] or 'combined'}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"unsafe job label: {label}")
    log_path = log_dir / f"{label}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            ["feat", row["fsf"]],
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode == 0 and row["stage"] == "l1":
        ensure_identity_registration(Path(row["output"]))
    return label, result.returncode


def run_jobs(
    manifest: Path,
    stage: str,
    jobs: int,
    resume: bool,
    dry_run: bool,
) -> int:
    if stage not in SAFE_STAGE:
        raise ValueError(f"invalid stage: {stage}")
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    rows = read_manifest(manifest, stage)
    runnable: list[dict[str, str]] = []
    for row in rows:
        state = validate_job(row, resume)
        print(f"{state.upper()}: {row['model']} run-{row['run'] or 'combined'} -> {row['output']}")
        if state == "run":
            runnable.append(row)
    if dry_run:
        print(f"PASS: dry run; {len(runnable)} {stage.upper()} jobs ready")
        return 0
    if not runnable:
        print(f"PASS: no {stage.upper()} jobs require execution")
        return 0

    log_dir = manifest.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(jobs, len(runnable))) as executor:
        futures = {executor.submit(run_one, row, log_dir): row for row in runnable}
        for future in as_completed(futures):
            label, returncode = future.result()
            print(f"DONE: {label}, exit={returncode}")
            if returncode != 0:
                failures.append(label)
    if failures:
        print(f"ERROR: failed jobs: {', '.join(failures)}")
        return 1
    for row in runnable:
        if not complete(row):
            print(f"ERROR: expected completion files missing: {row['output']}")
            return 1
    print(f"PASS: completed {len(runnable)} {stage.upper()} jobs")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(SAFE_STAGE), required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and shutil.which("feat") is None:
        raise SystemExit("ERROR: feat was not found on PATH")
    return run_jobs(
        args.manifest.resolve(), args.stage, args.jobs, args.resume, args.dry_run
    )


if __name__ == "__main__":
    raise SystemExit(main())
