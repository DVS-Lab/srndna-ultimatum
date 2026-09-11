#!/usr/bin/env python3
"""Repair and refresh only the prepared fairness-main L3 job.

This preserves the completed 94 L1 and 47 L2 outputs. If a failed L3 output
already exists, ``--archive-incomplete-output`` moves it aside with a UTC
timestamp before updating the rendered L3 FSF and manifest atomically.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from prepare_activation_fairness_main_pipeline import (
    L3_TEMPLATE_RELATIVE,
    render_l3,
)
from run_ultimatum_repair_jobs import complete


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows or not fields:
        raise ValueError(f"empty manifest: {path}")
    return rows, fields


def refresh(
    repository: Path,
    work_root: Path,
    archive_incomplete_output: bool,
) -> Path:
    manifest = work_root / "activation_fairness_main_jobs.tsv"
    rows, fields = read_rows(manifest)
    l1_rows = [row for row in rows if row["stage"] == "l1"]
    l2_rows = [row for row in rows if row["stage"] == "l2"]
    l3_rows = [row for row in rows if row["stage"] == "l3"]
    if (len(l1_rows), len(l2_rows), len(l3_rows)) != (94, 47, 1):
        raise ValueError("expected 94 L1, 47 L2, and 1 L3 manifest rows")
    incomplete_l2 = [row["subject"] for row in l2_rows if not complete(row)]
    if incomplete_l2:
        raise ValueError(f"L2 is incomplete for: {', '.join(incomplete_l2)}")

    l3 = l3_rows[0]
    output = Path(l3["output"])
    if output.exists():
        if complete(l3):
            raise FileExistsError(f"refusing to replace complete L3 output: {output}")
        if not archive_incomplete_output:
            raise FileExistsError(
                "incomplete L3 output exists; rerun with "
                f"--archive-incomplete-output to preserve and move it: {output}"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = output.with_name(f"{output.name}.failed-{stamp}")
        if archived.exists():
            raise FileExistsError(archived)
        output.rename(archived)
        print(f"ARCHIVED: {output} -> {archived}")

    l2_outputs = {row["subject"]: Path(row["output"]) for row in l2_rows}
    output_text = str(output)
    if not output_text.endswith(".gfeat"):
        raise ValueError(f"unexpected L3 output suffix: {output}")
    output_base = Path(output_text[:-6])
    template = repository / L3_TEMPLATE_RELATIVE
    if not template.is_file():
        raise FileNotFoundError(template)
    rendered_inputs = render_l3(
        template,
        Path(l3["fsf"]),
        output_base,
        l2_outputs,
    )
    missing = [value for value in rendered_inputs if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"missing corrected L3 image input: {missing[0]}")
    l3["inputs"] = "|".join(rendered_inputs)
    l3["source_fsf"] = str(template)
    l3["revision_template"] = str(template)

    temporary = manifest.with_name(f".{manifest.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        "PASS: refreshed the L3 FSF and manifest with 47 fixed-effects "
        f"cope images; manifest={manifest}"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--archive-incomplete-output", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    refresh(
        args.repository.resolve(),
        args.work_root.resolve(),
        args.archive_incomplete_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
