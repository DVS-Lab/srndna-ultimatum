#!/usr/bin/env python3
"""Prepare corrected social-versus-computer group activation models.

This reuses the 47 completed fixed-effects outputs from the task-wide
offer-size pipeline.  It prepares two L3-only FLAME 1+2 jobs:

* cope 10: (similar + dissimilar) - 2 * computer task response;
* cope 8:  (similar + dissimilar) - 2 * computer offer-size slope.

The factor of two preserves the historical first-level contrasts and affects
COPE scale, not Z inference.  Both group models use the same corrected,
full-rank nuisance design as the task-wide offer-size analysis.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from prepare_activation_fairness_main_pipeline import L3_TEMPLATE_RELATIVE, render_l3
from run_ultimatum_repair_jobs import complete


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_completed_l2(fairness_work_root: Path) -> dict[str, Path]:
    manifest = fairness_work_root / "activation_fairness_main_jobs.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    l2_rows = [row for row in rows if row.get("stage") == "l2"]
    if len(l2_rows) != 47 or len({row.get("subject") for row in l2_rows}) != 47:
        raise ValueError("expected 47 participant-unique L2 jobs")
    incomplete = [row["subject"] for row in l2_rows if not complete(row)]
    if incomplete:
        raise ValueError(f"incomplete L2 outputs: {', '.join(incomplete)}")
    return {row["subject"]: Path(row["output"]) for row in l2_rows}


def prepare(repository: Path, fairness_work_root: Path, work_root: Path) -> Path:
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"work root is not empty: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    template = repository / L3_TEMPLATE_RELATIVE
    if not template.is_file():
        raise FileNotFoundError(template)
    l2_outputs = read_completed_l2(fairness_work_root)

    specifications = (
        (10, "social-task-response", "social-minus-computer"),
        (8, "social-offer-slope", "social-minus-computer-offer-slope"),
    )
    rows: list[dict[str, object]] = []
    for cope, model, run in specifications:
        output_base = work_root / "outputs" / model
        fsf = work_root / "fsf" / f"L3_{model}.fsf"
        inputs = render_l3(template, fsf, output_base, l2_outputs, cope=cope)
        missing = [path for path in inputs if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"missing fixed-effects input: {missing[0]}")
        rows.append(
            {
                "stage": "l3",
                "subject": "all-47",
                "model": model,
                "run": run,
                "fsf": fsf,
                "output": Path(f"{output_base}.gfeat"),
                "inputs": "|".join(inputs),
                "source_fsf": template,
                "revision_template": template,
                "model_policy": (
                    "intercept;centered_age_group;centered_sex;tsnr;mean_fd;"
                    "event_corrected_taskwide_mean_rt;flame1plus2"
                ),
                "expected_copes": "",
                "expected_zstats": 4,
                "source_cope": cope,
            }
        )

    manifest = work_root / "activation_social_computer_l3_jobs.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        "PASS: prepared corrected cope-10 social task-response and cope-8 "
        f"social offer-slope L3 jobs; manifest={manifest}"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--fairness-work-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepare(
        args.repository.resolve(),
        args.fairness_work_root.resolve(),
        args.work_root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
