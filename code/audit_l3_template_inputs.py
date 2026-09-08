#!/usr/bin/env python3
"""Inventory L3 template inputs, with an explicit trace for sub-143."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


NPTS_RE = re.compile(r"^set fmri\(npts\) (\d+)$")
MULTIPLE_RE = re.compile(r"^set fmri\(multiple\) (\d+)$")
INPUT_RE = re.compile(r'^set feat_files\((\d+)\) "([^"]+)"$')
PARTICIPANT_RE = re.compile(r"/(sub-[A-Za-z0-9]+)/")
TYPE_RE = re.compile(r"model-02_(.+?)_sm-6")
COPE_RE = re.compile(r"/cope([^/]+)\.feat/")
PLACEHOLDER_RE = re.compile(r"(?:REPLACEME|COPENUM|BASEDIR|OUTPUT)")


def template_inventory(
    path: Path, display_path: str | None = None
) -> tuple[dict[str, object], list[dict[str, object]]]:
    template_label = display_path or path.name
    npts: int | None = None
    multiple: int | None = None
    inputs: list[tuple[int, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if match := NPTS_RE.match(line):
            npts = int(match.group(1))
        elif match := MULTIPLE_RE.match(line):
            multiple = int(match.group(1))
        elif match := INPUT_RE.match(line):
            inputs.append((int(match.group(1)), match.group(2)))

    participants = [match.group(1) for _, source in inputs if (match := PARTICIPANT_RE.search(source))]
    sub143_rows: list[dict[str, object]] = []
    for index, source in inputs:
        participant = PARTICIPANT_RE.search(source)
        if participant is None or participant.group(1) != "sub-143":
            continue
        analysis_type = TYPE_RE.search(source)
        cope = COPE_RE.search(source)
        sub143_rows.append(
            {
                "template": template_label,
                "input_index": index,
                "npts_declared": "" if npts is None else npts,
                "analysis_type": "" if analysis_type is None else analysis_type.group(1),
                "cope": "" if cope is None else cope.group(1),
                "source_path": source,
            }
        )

    source_copes = sorted({match.group(1) for _, source in inputs if (match := COPE_RE.search(source))})
    summary = {
        "template": template_label,
        "npts_declared": "" if npts is None else npts,
        "multiple_declared": "" if multiple is None else multiple,
        "input_rows_found": len(inputs),
        "unique_participants": len(set(participants)),
        "sub143_input_rows": len(sub143_rows),
        "sub143_input_indices": ",".join(str(row["input_index"]) for row in sub143_rows),
        "source_copes": ",".join(source_copes),
        "unresolved_placeholders": ",".join(sorted({token for _, source in inputs for token in PLACEHOLDER_RE.findall(source)})),
        "declared_count_matches_inputs": int(npts == len(inputs) and multiple == len(inputs)),
    }
    return summary, sub143_rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(templates_dir: Path, output_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    templates = sorted(templates_dir.rglob("L3*.fsf"))
    if not templates:
        raise FileNotFoundError(f"no L3 templates found in {templates_dir}")
    summaries: list[dict[str, object]] = []
    sub143: list[dict[str, object]] = []
    for template in templates:
        display_path = f"templates/{template.relative_to(templates_dir)}"
        summary, rows = template_inventory(template, display_path)
        summaries.append(summary)
        sub143.extend(rows)
    write_tsv(output_dir / "l3_template_input_inventory.tsv", summaries)
    write_tsv(output_dir / "l3_template_sub143_inputs.tsv", sub143)
    print(
        "PASS: "
        f"templates={len(summaries)}, "
        f"templates_with_matching_input_counts={sum(row['declared_count_matches_inputs'] for row in summaries)}, "
        f"templates_with_sub143={sum(bool(row['sub143_input_rows']) for row in summaries)}, "
        f"sub143_input_rows={len(sub143)}"
    )
    return summaries, sub143


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates-dir", type=Path, default=root / "templates")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "reviewer" / "tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit(args.templates_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
