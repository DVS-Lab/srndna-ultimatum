#!/usr/bin/env python3
"""Prepare corrected all-participant activation offer-slope L3 models.

The existing activation model has separate offer-size parametric effects for
the age-similar (cope 4) and age-dissimilar (cope 6) human partners. This
script prepares both simple effects using the corrected 47-participant input
set and standard nuisance covariates. It does not mislabel cope 7, the
similar-minus-dissimilar difference, as a general fairness effect.

Each model uses an intercept plus centered older-group membership, centered
sex, tSNR, mean FD, and event-corrected task-wide mean RT. The positive and
negative adjusted grand means are contrasts 1 and 2; age-group differences
are retained as contrasts 3 and 4 for auditing rather than figure selection.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FSF_RELATIVE = Path(
    "results/reviewer/l3_repair_designs/dmn-age/"
    "reported-covariates-corrected/design.fsf"
)
INPUT_RE = re.compile(r'^set feat_files\((\d+)\) "([^"]+)"$')
EV_RE = re.compile(r"^set fmri\(evg(\d+)\.(\d+)\) (.*)$")
SUBJECT_RE = re.compile(r"/(sub-[A-Za-z0-9]+)/")


def replace_unique(lines: list[str], prefix: str, replacement: str) -> None:
    indices = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(indices) != 1:
        raise ValueError(f"expected one setting beginning {prefix!r}, found {len(indices)}")
    lines[indices[0]] = replacement


def parse_inputs(lines: Sequence[str]) -> list[tuple[int, str, str]]:
    inputs: list[tuple[int, str, str]] = []
    for line in lines:
        match = INPUT_RE.match(line)
        if not match:
            continue
        subject = SUBJECT_RE.search(match.group(2))
        if subject is None:
            raise ValueError(f"cannot identify participant in {line}")
        inputs.append((int(match.group(1)), subject.group(1), match.group(2)))
    inputs.sort()
    if [index for index, _, _ in inputs] != list(range(1, 48)):
        raise ValueError("expected exactly 47 sequential corrected inputs")
    if len({subject for _, subject, _ in inputs}) != 47:
        raise ValueError("corrected inputs are not participant-unique")
    return inputs


def parse_evs(lines: Sequence[str]) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for line in lines:
        match = EV_RE.match(line)
        if match:
            values[(int(match.group(1)), int(match.group(2)))] = float(
                match.group(3)
            )
    return values


def matrix_rank(matrix: list[list[float]], tolerance: float = 1e-10) -> int:
    work = [row[:] for row in matrix]
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


def activation_input(path: str, cope: int) -> str:
    if "type-nppi-dmn" not in path or "/cope7.feat/" not in path:
        raise ValueError(f"unexpected corrected DMN input: {path}")
    return path.replace("type-nppi-dmn", "type-act", 1).replace(
        "/cope7.feat/", f"/cope{cope}.feat/", 1
    )


def render_model(source: Path, destination: Path, output: Path, cope: int) -> list[str]:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    inputs = parse_inputs(lines)
    original_evs = parse_evs(lines)
    if len(original_evs) != 47 * 6:
        raise ValueError("source design must have 47 rows and 6 EVs")

    replace_unique(lines, "set fmri(outputdir) ", f'set fmri(outputdir) "{output}"')
    rendered_inputs: list[str] = []
    for index, _, path in inputs:
        rendered = activation_input(path, cope)
        replace_unique(
            lines,
            f"set feat_files({index}) ",
            f'set feat_files({index}) "{rendered}"',
        )
        rendered_inputs.append(rendered)

    older_proportion = sum(original_evs[(row, 2)] for row in range(1, 48)) / 47
    for row in range(1, 48):
        older = original_evs[(row, 2)]
        replace_unique(lines, f"set fmri(evg{row}.1) ", f"set fmri(evg{row}.1) 1.0")
        replace_unique(
            lines,
            f"set fmri(evg{row}.2) ",
            f"set fmri(evg{row}.2) {older - older_proportion:.12g}",
        )
    replace_unique(lines, "set fmri(evtitle1) ", 'set fmri(evtitle1) "intercept"')
    replace_unique(
        lines,
        "set fmri(evtitle2) ",
        'set fmri(evtitle2) "older_centered"',
    )

    contrast_specs = (
        (1, "adjusted-mean-positive", (1, 0, 0, 0, 0, 0)),
        (2, "adjusted-mean-negative", (-1, 0, 0, 0, 0, 0)),
        (3, "older-greater-than-younger", (0, 1, 0, 0, 0, 0)),
        (4, "younger-greater-than-older", (0, -1, 0, 0, 0, 0)),
    )
    for contrast, title, vector in contrast_specs:
        replace_unique(
            lines,
            f"set fmri(conname_real.{contrast}) ",
            f'set fmri(conname_real.{contrast}) "{title}"',
        )
        for column, value in enumerate(vector, start=1):
            replace_unique(
                lines,
                f"set fmri(con_real{contrast}.{column}) ",
                f"set fmri(con_real{contrast}.{column}) {float(value):.1f}",
            )

    rendered_evs = parse_evs(lines)
    matrix = [
        [rendered_evs[(row, column)] for column in range(1, 7)]
        for row in range(1, 48)
    ]
    if matrix_rank(matrix) != 6:
        raise ValueError("rendered activation main-effect design is not full rank")
    if abs(sum(row[1] for row in matrix)) > 1e-8:
        raise ValueError("rendered age-group covariate is not centered")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rendered_inputs


def prepare(repository: Path, work_root: Path) -> Path:
    source = repository / SOURCE_FSF_RELATIVE
    if not source.is_file():
        raise FileNotFoundError(source)
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"work root is not empty: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for condition, cope in (("similar", 4), ("dissimilar", 6)):
        output = work_root / "outputs" / f"activation-offer-slope_{condition}"
        fsf = work_root / "fsf" / f"activation-offer-slope_{condition}.fsf"
        inputs = render_model(source, fsf, output, cope)
        sub144 = [path for path in inputs if "/sub-144/" in path]
        if len(sub144) != 1 or "srndna-ultimatum-sub144-repair-v2" not in sub144[0]:
            raise ValueError(f"{condition} model does not retain repaired sub-144")
        rows.append(
            {
                "stage": "l3",
                "model": "activation-offer-slope",
                "run": condition,
                "fsf": fsf,
                "output": Path(f"{output}.gfeat"),
                "inputs": "|".join(inputs),
                "source_fsf": source,
                "covariate_policy": (
                    "intercept;centered_age_group;centered_sex;tsnr;mean_fd;"
                    "event_corrected_taskwide_mean_rt"
                ),
                "expected_zstats": 4,
                "source_cope": cope,
            }
        )

    manifest = work_root / "activation_fairness_l3_jobs.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        "PASS: prepared corrected activation offer-slope models for similar "
        f"and dissimilar partners; manifest={manifest}"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepare(args.repository.resolve(), args.work_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
