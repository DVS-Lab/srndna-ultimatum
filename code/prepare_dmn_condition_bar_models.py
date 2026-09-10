#!/usr/bin/env python3
"""Prepare two scratch-only corrected DMN condition-specific FLAME models.

The models use the exact corrected six-column DMN group design. The similar
model substitutes L2 cope 4 for every cope 7 input; the dissimilar model
substitutes L2 cope 6. The repaired sub-144 input and corrected RT covariate
remain unchanged. These models provide variance-weighted group estimates for
the four-bar descriptive decomposition of the confirmed cope-7 result.
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
        raise ValueError("expected exactly 47 sequential corrected DMN inputs")
    if len({subject for _, subject, _ in inputs}) != 47:
        raise ValueError("corrected DMN inputs are not participant-unique")
    return inputs


def condition_input(path: str, cope: int) -> str:
    marker = "/cope7.feat/"
    if marker not in path:
        raise ValueError(f"expected cope-7 group input, found {path}")
    return path.replace(marker, f"/cope{cope}.feat/", 1)


def render_condition(source: Path, destination: Path, output: Path, cope: int) -> list[str]:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    inputs = parse_inputs(lines)
    replace_unique(
        lines,
        "set fmri(outputdir) ",
        f'set fmri(outputdir) "{output}"',
    )
    rendered_inputs: list[str] = []
    for index, _, path in inputs:
        rendered = condition_input(path, cope)
        replace_unique(
            lines,
            f"set feat_files({index}) ",
            f'set feat_files({index}) "{rendered}"',
        )
        rendered_inputs.append(rendered)
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

    model_rows: list[dict[str, object]] = []
    for condition, cope in (("similar", 4), ("dissimilar", 6)):
        output = work_root / "outputs" / f"dmn-age_condition-{condition}"
        fsf = work_root / "fsf" / f"dmn-age_condition-{condition}.fsf"
        inputs = render_condition(source, fsf, output, cope)
        sub144 = [path for path in inputs if "/sub-144/" in path]
        if len(sub144) != 1 or "srndna-ultimatum-sub144-repair-v2" not in sub144[0]:
            raise ValueError(f"{condition} model does not retain the repaired sub-144 input")
        model_rows.append(
            {
                "stage": "l3",
                "model": "dmn-condition-bars",
                "run": condition,
                "fsf": fsf,
                "output": Path(f"{output}.gfeat"),
                "inputs": "|".join(inputs),
                "source_fsf": source,
                "covariate_policy": "corrected_dmn_six_column_design",
                "expected_zstats": 4,
                "source_cope": cope,
            }
        )

    manifest = work_root / "dmn_condition_bar_jobs.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(model_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(model_rows)
    print(
        "PASS: prepared 2 corrected condition-specific DMN FLAME models; "
        f"manifest={manifest}"
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
