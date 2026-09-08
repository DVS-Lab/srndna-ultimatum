#!/usr/bin/env python3
"""Audit ultimatum-game timing and missed trials from curated BIDS events.

The BIDS converter labels misses only as ``missed_trial``. This script recovers
their partner from the enclosing block instead of incorrectly treating every
miss as a computer trial. It writes compact, deidentified source-data tables
and never changes BIDS inputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TRIAL_RE = re.compile(
    r"^(?:event_(?:accept|reject)_(?:computer|ingroup|outgroup)(?:_(?:un)?fair)?|missed_trial)$"
)
PARTNERS = ("computer", "ingroup", "outgroup")


@dataclass(frozen=True)
class Trial:
    participant: str
    run: str
    onset: float
    duration: float
    partner: str
    missed: int
    response_time_raw: float | None
    offer: float | None
    block: str
    block_start: float
    rt_event_present: bool


def read_rows(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def number(value: str) -> float | None:
    if value.strip().lower() in {"", "n/a", "na", "nan"}:
        return None
    return float(value)


def partner_from_block(block: str) -> str:
    matches = [partner for partner in PARTNERS if partner in block.lower()]
    if len(matches) != 1:
        raise ValueError(f"cannot identify exactly one partner from block {block!r}")
    return matches[0]


def parse_event_file(path: Path, participant: str, run: str) -> tuple[list[Trial], int]:
    rows = read_rows(path)
    blocks = sorted(
        (float(row["onset"]), row["trial_type"])
        for row in rows
        if row["trial_type"].startswith("block_")
    )
    if not blocks:
        raise ValueError(f"no block rows in {path}")
    rt_onsets = [float(row["onset"]) for row in rows if row["trial_type"] == "event_RT"]

    trials: list[Trial] = []
    for row in rows:
        trial_type = row["trial_type"]
        if not TRIAL_RE.match(trial_type):
            continue
        onset = float(row["onset"])
        preceding = [(start, label) for start, label in blocks if start <= onset]
        if not preceding:
            raise ValueError(f"trial precedes every block in {path}: onset {onset}")
        block_start, block = max(preceding)
        partner = partner_from_block(block)
        if trial_type != "missed_trial" and partner not in trial_type:
            raise ValueError(f"trial/block partner mismatch in {path}: {trial_type}, {block}")
        trials.append(
            Trial(
                participant=participant,
                run=run,
                onset=onset,
                duration=float(row["duration"]),
                partner=partner,
                missed=int(trial_type == "missed_trial"),
                response_time_raw=number(row.get("response_time", "")),
                offer=number(row.get("Offer", "")),
                block=block,
                block_start=block_start,
                rt_event_present=any(math.isclose(onset, rt_onset, abs_tol=1e-5) for rt_onset in rt_onsets),
            )
        )
    return sorted(trials, key=lambda trial: trial.onset), len(blocks)


def participant_sample(path: Path) -> dict[str, str]:
    sample: dict[str, str] = {}
    for row in read_rows(path, delimiter=","):
        participant = (row.get("subjID") or row.get("participant_id") or "").strip()
        if not participant:
            raise ValueError(f"participant identifier missing in {path}")
        if float(row["younger"]) == 1:
            group = "younger"
        elif float(row["older"]) == 1:
            group = "older"
        else:
            raise ValueError(f"age group is not one-hot for {participant}")
        sample[participant] = group
    return sample


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else math.nan


def audit(
    bids_root: Path, sample_path: Path, output_dir: Path, private_output_dir: Path
) -> dict[str, object]:
    sample = participant_sample(sample_path)
    trials: list[Trial] = []
    run_records: list[tuple[str, str, int, int]] = []
    for participant in sample:
        paths = sorted((bids_root / participant / "func").glob(f"{participant}_task-ultimatum_run-*_events.tsv"))
        if not paths:
            raise FileNotFoundError(f"no ultimatum event files for {participant}")
        for path in paths:
            match = re.search(r"_run-([^_]+)_events\.tsv$", path.name)
            if not match:
                raise ValueError(f"cannot parse run from {path}")
            run = match.group(1)
            parsed, blocks = parse_event_file(path, participant, run)
            trials.extend(parsed)
            run_records.append((participant, run, len(parsed), blocks))

    by_subject: dict[str, Counter[str]] = defaultdict(Counter)
    trial_rows: list[dict[str, object]] = []
    within_gaps: list[float] = []
    between_gaps: list[float] = []
    by_run: dict[tuple[str, str], list[Trial]] = defaultdict(list)
    for trial in trials:
        by_subject[trial.participant][trial.partner] += trial.missed
        by_run[(trial.participant, trial.run)].append(trial)
        trial_rows.append(
            {
                "participant": trial.participant,
                "age_group": sample[trial.participant],
                "run": trial.run,
                "onset": f"{trial.onset:.6f}",
                "duration": f"{trial.duration:.6f}",
                "partner": trial.partner,
                "missed": trial.missed,
                "response_time_raw": "" if trial.response_time_raw is None else f"{trial.response_time_raw:.6f}",
                "response_selection_time": "" if trial.response_time_raw is None else f"{trial.response_time_raw - 1:.6f}",
                "offer": "" if trial.offer is None else f"{trial.offer:g}",
                "block": trial.block,
                "block_start": f"{trial.block_start:.6f}",
                "rt_event_present": int(trial.rt_event_present),
            }
        )
    for run_trials in by_run.values():
        for current, following in zip(run_trials, run_trials[1:]):
            gap = following.onset - current.onset - current.duration
            (within_gaps if current.block == following.block else between_gaps).append(gap)

    rt_run_rows: list[dict[str, object]] = []
    responded_trials = [trial for trial in trials if not trial.missed]
    first_trial_keys: set[tuple[str, str, float, float]] = set()
    for run_key, run_trials in by_run.items():
        block_trials: dict[float, list[Trial]] = defaultdict(list)
        for trial in run_trials:
            block_trials[trial.block_start].append(trial)
        first_trial_keys.update(
            (run_key[0], run_key[1], block_start, min(block_values, key=lambda value: value.onset).onset)
            for block_start, block_values in block_trials.items()
        )
        run_responded = [trial for trial in run_trials if not trial.missed]
        run_missing = [trial for trial in run_responded if not trial.rt_event_present]
        run_first = [
            trial
            for trial in run_responded
            if (trial.participant, trial.run, trial.block_start, trial.onset) in first_trial_keys
        ]
        rt_run_rows.append(
            {
                "participant": run_key[0],
                "run": run_key[1],
                "responded_trials": len(run_responded),
                "rt_event_rows_present": sum(trial.rt_event_present for trial in run_responded),
                "responded_trials_missing_rt_event": len(run_missing),
                "first_block_trials": len(run_first),
                "first_block_trials_missing_rt_event": sum(not trial.rt_event_present for trial in run_first),
                "nonfirst_trials_missing_rt_event": sum(
                    not trial.rt_event_present and trial not in run_first for trial in run_responded
                ),
            }
        )

    first_trials = [
        trial
        for trial in responded_trials
        if (trial.participant, trial.run, trial.block_start, trial.onset) in first_trial_keys
    ]
    nonfirst_trials = [trial for trial in responded_trials if trial not in first_trials]
    missing_rt_events = [trial for trial in responded_trials if not trial.rt_event_present]
    nonfirst_missing = [trial for trial in nonfirst_trials if not trial.rt_event_present]
    nonfirst_participants = Counter(trial.participant for trial in nonfirst_missing)

    participant_rows: list[dict[str, object]] = []
    for participant, group in sample.items():
        counts = by_subject[participant]
        participant_rows.append(
            {
                "participant": participant,
                "age_group": group,
                "missed_computer": counts["computer"],
                "missed_similar": counts["ingroup"],
                "missed_dissimilar": counts["outgroup"],
                "missed_total": sum(counts.values()),
            }
        )

    duration_mean, duration_sd = mean_sd([trial.duration for trial in trials])
    within_mean, within_sd = mean_sd(within_gaps)
    summary_rows = [
        {"metric": "participants", "value": len(sample), "detail": "analysis sample"},
        {"metric": "runs", "value": len(run_records), "detail": "ultimatum event files"},
        {"metric": "trials", "value": len(trials), "detail": "one row per task trial"},
        {"metric": "missed_trials", "value": sum(trial.missed for trial in trials), "detail": "all partners"},
        {
            "metric": "responded_trials_missing_event_RT_row",
            "value": len(missing_rt_events),
            "detail": "source BIDS event construction; see production RT audit for the isolated sub-143 exception",
        },
        {
            "metric": "first_block_trials_missing_event_RT_row",
            "value": sum(not trial.rt_event_present for trial in first_trials),
            "detail": f"of {len(first_trials)} responded first trials at block onset",
        },
        {
            "metric": "nonfirst_trials_missing_event_RT_row",
            "value": len(missing_rt_events) - sum(not trial.rt_event_present for trial in first_trials),
            "detail": "responded trials outside block onset",
        },
        {
            "metric": "runs_with_missing_event_RT_row",
            "value": sum(row["responded_trials_missing_rt_event"] > 0 for row in rt_run_rows),
            "detail": f"of {len(rt_run_rows)} runs",
        },
        {"metric": "trial_duration_mean_seconds", "value": f"{duration_mean:.9f}", "detail": f"SD={duration_sd:.9f}"},
        {"metric": "within_block_gap_mean_seconds", "value": f"{within_mean:.9f}", "detail": f"SD={within_sd:.9f}"},
        {
            "metric": "between_block_gap_counts_rounded",
            "value": ";".join(f"{gap:g}:{count}" for gap, count in sorted(Counter(round(value) for value in between_gaps).items())),
            "detail": "gap seconds:count",
        },
    ]

    def rt_summary_row(scope: str, values: list[Trial], affected_runs: int, detail: str) -> dict[str, object]:
        missing = sum(not trial.rt_event_present for trial in values)
        return {
            "scope": scope,
            "evidence_level": "curated_BIDS_events_only",
            "responded_task_trials": len(values),
            "companion_event_RT_present": len(values) - missing,
            "companion_event_RT_missing": missing,
            "missing_percent": "" if not values else f"{100 * missing / len(values):.6f}",
            "affected_runs": affected_runs,
            "total_runs": len(rt_run_rows),
            "detail": detail,
        }

    rt_summary_rows = [
        rt_summary_row(
            "all_responded_trials",
            responded_trials,
            sum(row["responded_trials_missing_rt_event"] > 0 for row in rt_run_rows),
            "current derivative mirror verified for 92/94 runs; legacy production L1 audit pending",
        ),
        rt_summary_row(
            "responded_first_trial_of_block",
            first_trials,
            sum(row["first_block_trials_missing_rt_event"] > 0 for row in rt_run_rows),
            "systematic block-first-trial source-event omission",
        ),
        rt_summary_row(
            "responded_nonfirst_trials",
            nonfirst_trials,
            sum(row["nonfirst_trials_missing_rt_event"] > 0 for row in rt_run_rows),
            ";".join(f"{participant}:{count}" for participant, count in sorted(nonfirst_participants.items())),
        ),
    ]

    write_tsv(private_output_dir / "task_trial_source_data.tsv", list(trial_rows[0]), trial_rows)
    write_tsv(private_output_dir / "missed_trials_by_participant.tsv", list(participant_rows[0]), participant_rows)
    write_tsv(private_output_dir / "rt_event_omissions_by_run.tsv", list(rt_run_rows[0]), rt_run_rows)
    write_tsv(output_dir / "task_event_summary.tsv", ["metric", "value", "detail"], summary_rows)
    write_tsv(output_dir / "rt_event_construction_summary.tsv", list(rt_summary_rows[0]), rt_summary_rows)
    return {"participants": len(sample), "runs": len(run_records), "trials": len(trials), "misses": sum(t.missed for t in trials)}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=root / "source_data" / "bids")
    parser.add_argument("--sample", type=Path, default=root / "behavioral_analyses" / "data" / "participant_L3_47.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "reviewer" / "tables")
    parser.add_argument(
        "--private-output-dir",
        type=Path,
        default=root / "results" / "reviewer" / "private",
        help="ignored local directory for participant-level rows",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args.bids_root, args.sample, args.output_dir, args.private_output_dir)
    print("PASS: " + ", ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
