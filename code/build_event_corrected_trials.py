#!/usr/bin/env python3
"""Overlay corrected BIDS trial metadata without altering historical brain estimates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence


TRIAL_PATTERN = re.compile(
    r"^(?:event_(?P<decision>accept|reject)_(?P<partner>computer|ingroup|outgroup)"
    r"(?:_(?:un)?fair)?|missed_trial)$"
)
BLOCK_PATTERN = re.compile(
    r"^block_(?P<partner>computer|ingroup|outgroup)_(?P<fairness>unfair|fair)$"
)
BEHAVIOR_COLUMNS = (
    "offer",
    "missed",
    "accept",
    "response_time",
    "human",
    "ingroup",
    "block_fair",
    "run",
)


def read_rows(path: Path, delimiter: str = ",") -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        rows = list(reader)
    if not reader.fieldnames:
        raise ValueError(f"missing header: {path}")
    return list(reader.fieldnames), rows


def event_trials(path: Path, response_delay: float) -> list[dict[str, str]]:
    _, rows = read_rows(path, "\t")
    blocks: list[tuple[float, str, str]] = []
    for row in rows:
        match = BLOCK_PATTERN.match(row["trial_type"])
        if match:
            blocks.append(
                (float(row["onset"]), match.group("partner"), match.group("fairness"))
            )

    trials: list[dict[str, str]] = []
    for row in rows:
        match = TRIAL_PATTERN.match(row["trial_type"])
        if not match:
            continue
        onset = float(row["onset"])
        active = [block for block in blocks if block[0] <= onset + 1e-6]
        if not active:
            raise ValueError(f"trial has no block context in {path}: onset={onset}")
        _, block_partner, block_fairness = active[-1]
        missed = match.group(0) == "missed_trial"
        partner = match.group("partner") or block_partner
        if partner != block_partner:
            raise ValueError(
                f"trial/block partner mismatch in {path}: onset={onset}, "
                f"trial={partner}, block={block_partner}"
            )
        fair_value = row.get("IsFairBlock", "")
        if fair_value not in {"0", "1"}:
            fair_value = "1" if block_fairness == "fair" else "0"
        elif int(fair_value) != int(block_fairness == "fair"):
            raise ValueError(f"trial/block fairness mismatch in {path}: onset={onset}")

        response_time = ""
        if not missed:
            raw_rt = float(row["response_time"])
            response_time = f"{raw_rt - response_delay:.12g}"
        trials.append(
            {
                "offer": row["Offer"],
                "missed": "1" if missed else "0",
                "accept": "" if missed else ("1" if match.group("decision") == "accept" else "0"),
                "response_time": response_time,
                "human": "0" if partner == "computer" else "1",
                "ingroup": "1" if partner == "ingroup" else "0",
                "block_fair": fair_value,
            }
        )
    if len(trials) != 72:
        raise ValueError(f"expected 72 trials in {path}, found {len(trials)}")
    return trials


def profile(rows: Iterable[dict[str, str]]) -> str:
    values = [[row[column] for column in BEHAVIOR_COLUMNS] for row in rows]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def build(
    source: Path,
    bids_root: Path,
    output: Path,
    *,
    subject: str = "sub-144",
    response_delay: float = 1.0,
) -> dict[str, object]:
    fieldnames, rows = read_rows(source)
    required = {"subjID", *BEHAVIOR_COLUMNS}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(sorted(missing))}")

    target_indices = [index for index, row in enumerate(rows) if row["subjID"] == subject]
    if len(target_indices) != 144:
        raise ValueError(f"expected 144 historical rows for {subject}, found {len(target_indices)}")
    before = [rows[index].copy() for index in target_indices]

    replacement_by_run: dict[str, list[dict[str, str]]] = {}
    for run in (1, 2):
        event_file = (
            bids_root
            / subject
            / "func"
            / f"{subject}_task-ultimatum_run-{run:02d}_events.tsv"
        )
        if not event_file.is_file():
            raise FileNotFoundError(event_file)
        replacement_by_run[str(run)] = event_trials(event_file, response_delay)

    changed_cells = 0
    for run in ("1", "2"):
        indices = [index for index in target_indices if rows[index]["run"] == run]
        if len(indices) != 72:
            raise ValueError(f"expected 72 historical {subject} run-{run} rows, found {len(indices)}")
        for index, replacement in zip(indices, replacement_by_run[run]):
            replacement["run"] = run
            for column in BEHAVIOR_COLUMNS:
                if rows[index][column] != replacement[column]:
                    changed_cells += 1
                rows[index][column] = replacement[column]

    after = [rows[index] for index in target_indices]
    for index, original in zip(target_indices, before):
        for column in fieldnames:
            if column not in BEHAVIOR_COLUMNS and rows[index][column] != original[column]:
                raise AssertionError(f"nonbehavioral column changed: row={index}, column={column}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)

    summary = {
        "subject": subject,
        "rows_overlaid": len(after),
        "changed_behavior_cells": changed_cells,
        "historical_profile_sha256": profile(before),
        "corrected_profile_sha256": profile(after),
        "historical_missed_trials": sum(int(row["missed"]) for row in before),
        "corrected_missed_trials": sum(int(row["missed"]) for row in after),
        "historical_responded_human_trials": sum(
            row["missed"] == "0" and row["human"] == "1" for row in before
        ),
        "corrected_responded_human_trials": sum(
            row["missed"] == "0" and row["human"] == "1" for row in after
        ),
    }
    return summary


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "behavioral_analyses" / "data" / "all_trials_brains.csv",
    )
    parser.add_argument("--bids-root", type=Path, default=root / "source_data" / "bids")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "reviewer" / "private" / "all_trials_brains_event_corrected.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "results" / "reviewer" / "tables" / "sub144_trial_metadata_correction.tsv",
    )
    parser.add_argument("--subject", default="sub-144")
    parser.add_argument("--response-delay", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build(
        args.source,
        args.bids_root,
        args.output,
        subject=args.subject,
        response_delay=args.response_delay,
    )
    write_summary(args.summary, summary)
    print(
        f"PASS: overlaid {summary['rows_overlaid']} {summary['subject']} trial rows; "
        f"changed {summary['changed_behavior_cells']} behavioral cells; "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
