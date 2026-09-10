#!/usr/bin/env python3
"""Prepare scratch-only L3 reruns from the recovered production designs.

The submitted focal masks remain untouched. Each rendered design preserves the
production participant order, contrasts, nuisance covariates, inference
settings, and continuous-network L2 inputs. Scientific changes are limited to
the repaired sub-144 cope path and, for explicitly labeled variants, the
event-corrected fairness covariate columns. Output, input-root, and installed
FSL standard-image paths are remapped for portability.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SETTING_RE = re.compile(r"^set (?:(fmri\([^)]*\))|(feat_files\(\d+\))) (.*)$")
INPUT_RE = re.compile(r'^set feat_files\((\d+)\) "(.*)"$')
EV_RE = re.compile(r"^set fmri\(evg(\d+)\.(\d+)\) (.*)$")
SUBJECT_RE = re.compile(r"(sub-\d+)")


@dataclass(frozen=True)
class Model:
    model_id: str
    bundle: str
    l2_model: str
    corrected_prefix: str | None
    corrected_policy: str | None


MODELS = (
    Model("dmn-age", "dmn-age", "nppi-dmn", None, None),
    Model(
        "ecn-sensitivity",
        "ecn-sensitivity",
        "nppi-ecn",
        "corrected_sensitivity",
        "event_corrected_fairness_sensitivity;submitted_group_rt",
    ),
    Model(
        "activation-norm",
        "activation",
        "act",
        "corrected_norm",
        "event_corrected_fairness_norm_proxy;submitted_group_rt",
    ),
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty covariate table: {path}")
    return rows


def parse_inputs(lines: Sequence[str]) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for line in lines:
        match = INPUT_RE.match(line)
        if not match:
            continue
        subject_match = SUBJECT_RE.search(match.group(2))
        if subject_match is None:
            raise ValueError(f"L3 input has no participant label: {line}")
        rows.append((int(match.group(1)), subject_match.group(1), match.group(2)))
    return sorted(rows)


def parse_evs(lines: Sequence[str]) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for line in lines:
        match = EV_RE.match(line)
        if match:
            values[(int(match.group(1)), int(match.group(2)))] = float(match.group(3))
    return values


def matrix_rank(matrix: list[list[float]], tolerance: float = 1e-10) -> int:
    work = [row[:] for row in matrix]
    if not work:
        return 0
    rows, columns = len(work), len(work[0])
    rank = pivot_row = 0
    for column in range(columns):
        pivot = max(range(pivot_row, rows), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= tolerance:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        divisor = work[pivot_row][column]
        work[pivot_row] = [value / divisor for value in work[pivot_row]]
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = work[row][column]
            if abs(factor) > tolerance:
                work[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(work[row], work[pivot_row])
                ]
        rank += 1
        pivot_row += 1
        if pivot_row == rows:
            break
    return rank


def replace_setting(lines: list[str], key: str, value: str) -> None:
    pattern = re.compile(rf"^set {re.escape(key)} .*$")
    matches = [index for index, line in enumerate(lines) if pattern.match(line)]
    if len(matches) != 1:
        raise ValueError(f"expected one {key}, found {len(matches)}")
    lines[matches[0]] = f"set {key} {value}"


def remap_input(path: str, production_fsl_root: Path) -> str:
    match = re.search(r"/derivatives/fsl/(.*)$", path)
    if match is None:
        raise ValueError(f"cannot remap production L2 input: {path}")
    return str(production_fsl_root / match.group(1))


def render(
    source: Path,
    destination: Path,
    output_root: Path,
    production_fsl_root: Path,
    repaired_l2: Path,
    standard_image: Path,
    covariates: dict[str, dict[str, str]],
    corrected_prefix: str | None,
) -> dict[str, object]:
    source_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = source_lines[:]
    inputs = parse_inputs(source_lines)
    participants = [subject for _, subject, _ in inputs]
    if len(inputs) != 47 or len(set(participants)) != 47:
        raise ValueError(f"expected 47 unique production inputs in {source}")
    if participants.count("sub-144") != 1:
        raise ValueError(f"expected one sub-144 input in {source}")
    if set(participants) != set(covariates):
        raise ValueError("covariate participants do not match production inputs")

    replace_setting(lines, "fmri(outputdir)", f'"{output_root}"')
    replace_setting(lines, "fmri(regstandard)", f'"{standard_image}"')
    input_paths: list[str] = []
    for index, subject, path in inputs:
        updated = remap_input(path, production_fsl_root)
        if subject == "sub-144":
            updated = str(repaired_l2)
        replace_setting(lines, f"feat_files({index})", f'"{updated}"')
        input_paths.append(updated)

    changed_ev_columns: tuple[int, ...] = ()
    if corrected_prefix is not None:
        changed_ev_columns = (7, 8)
        for index, subject, _ in inputs:
            row = covariates[subject]
            replace_setting(
                lines,
                f"fmri(evg{index}.7)",
                row[f"{corrected_prefix}_young"],
            )
            replace_setting(
                lines,
                f"fmri(evg{index}.8)",
                row[f"{corrected_prefix}_old"],
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

    source_evs = parse_evs(source_lines)
    rendered_evs = parse_evs(lines)
    ev_columns = sorted({column for _, column in source_evs})
    if source_evs.keys() != rendered_evs.keys():
        raise ValueError("rendered EV structure differs from production")
    for key, source_value in source_evs.items():
        if key[1] not in changed_ev_columns and rendered_evs[key] != source_value:
            raise ValueError(f"unexpected EV change at row {key[0]}, column {key[1]}")
    matrix = [[rendered_evs[(row, column)] for column in ev_columns] for row, _, _ in inputs]
    rank = matrix_rank(matrix)
    if rank != len(ev_columns):
        raise ValueError(f"rank-deficient rendered design: rank={rank}, EVs={len(ev_columns)}")
    return {
        "inputs": "|".join([*input_paths, str(standard_image)]),
        "participants": participants,
        "n_evs": len(ev_columns),
        "rank": rank,
        "changed_ev_columns": ",".join(str(value) for value in changed_ev_columns),
    }


def prepare(
    repository: Path,
    production_fsl_root: Path,
    sub144_repair_root: Path,
    work_root: Path,
    covariate_table: Path,
    standard_image: Path,
) -> Path:
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"work root is not empty: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    rows = read_tsv(covariate_table)
    covariates = {row["subjID"]: row for row in rows}
    if len(covariates) != 47:
        raise ValueError(f"expected 47 covariate rows, found {len(covariates)}")

    manifest_rows: list[dict[str, object]] = []
    for model in MODELS:
        source = (
            repository
            / "results/reviewer/production_audits"
            / model.bundle
            / "design/design.fsf"
        )
        variants = (
            ("image-only",)
            if model.corrected_prefix is None
            else ("image-only", "fairness-covariate-corrected")
        )
        for variant in variants:
            corrected_prefix = (
                model.corrected_prefix
                if variant == "fairness-covariate-corrected"
                else None
            )
            output_root = work_root / "outputs" / f"{model.model_id}_{variant}"
            fsf = work_root / "fsf" / f"{model.model_id}_{variant}.fsf"
            repaired_l2 = (
                sub144_repair_root
                / "derivatives/fsl/sub-144"
                / f"L2_task-ultimatum_model-02_type-{model.l2_model}_sm-6.gfeat"
                / "cope7.feat/stats/cope1.nii.gz"
            )
            audit = render(
                source,
                fsf,
                output_root,
                production_fsl_root,
                repaired_l2,
                standard_image,
                covariates,
                corrected_prefix,
            )
            manifest_rows.append(
                {
                    "stage": "l3",
                    "model": model.model_id,
                    "run": variant,
                    "fsf": fsf,
                    "output": Path(f"{output_root}.gfeat"),
                    "inputs": audit["inputs"],
                    "source_fsf": source,
                    "covariate_policy": (
                        "production_all_group_covariates"
                        if variant == "image-only"
                        else model.corrected_policy
                    ),
                    "expected_zstats": 4 if model.model_id == "dmn-age" else 8,
                    "n_evs": audit["n_evs"],
                    "design_rank": audit["rank"],
                    "changed_ev_columns": audit["changed_ev_columns"],
                    "sub144_repaired_l2": repaired_l2,
                    "standard_image": standard_image,
                }
            )

    manifest = work_root / "l3_repair_jobs.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(
        f"PASS: prepared {len(manifest_rows)} scratch-only L3 jobs; "
        f"all designs full-rank; manifest={manifest}"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=root)
    parser.add_argument("--production-fsl-root", type=Path, required=True)
    parser.add_argument("--sub144-repair-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--standard-image",
        type=Path,
        required=True,
        help="installed FSL MNI152_T1_2mm_brain image (same reference, portable path)",
    )
    parser.add_argument(
        "--covariates",
        type=Path,
        default=root / "results/reviewer/private/corrected_l3_covariates.tsv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepare(
        args.repository.resolve(),
        args.production_fsl_root.resolve(),
        args.sub144_repair_root.resolve(),
        args.work_root.resolve(),
        args.covariates.resolve(),
        args.standard_image.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
