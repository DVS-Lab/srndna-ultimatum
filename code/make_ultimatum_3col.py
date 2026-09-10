#!/usr/bin/env python3
"""Create the nine Ultimatum Game model-02 FSL EV files from BIDS events."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Sequence


TRIAL_RE = re.compile(
    r"^event_(?:accept|reject)_(?P<partner>computer|ingroup|outgroup)"
    r"(?:_(?:un)?fair)?$"
)
PARTNERS = ("computer", "ingroup", "outgroup")


def read_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"onset", "duration", "trial_type", "response_time", "Offer"}
    if not rows or not required.issubset(rows[0]):
        missing = required.difference(rows[0] if rows else {})
        raise ValueError(f"{path} is empty or missing columns: {', '.join(sorted(missing))}")
    return rows


def numeric(value: str, *, field: str, event: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} for {event}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} for {event}: {value!r}")
    return result


def format_number(value: float) -> str:
    return f"{value:.12g}"


def three_column(rows: list[tuple[float, float, float]]) -> str:
    return "".join(
        f"{format_number(onset)}\t{format_number(duration)}\t{format_number(amplitude)}\n"
        for onset, duration, amplitude in rows
    )


def ev_rows(
    events: list[dict[str, str]], rt_source: str
) -> dict[str, list[tuple[float, float, float]]]:
    if rt_source not in {"companion", "substantive"}:
        raise ValueError(f"unknown RT source: {rt_source}")

    output: dict[str, list[tuple[float, float, float]]] = {}
    for partner in PARTNERS:
        output[f"event_{partner}"] = []
        output[f"event_{partner}_pmod"] = []
    output["missed_trial"] = []
    output["event_RT"] = []
    output["event_RT_pmod"] = []

    substantive_rt: list[tuple[float, float, float]] = []
    for row in events:
        event = row["trial_type"]
        match = TRIAL_RE.match(event)
        if match:
            onset = numeric(row["onset"], field="onset", event=event)
            duration = numeric(row["duration"], field="duration", event=event)
            offer = numeric(row["Offer"], field="Offer", event=event)
            response_time = numeric(
                row["response_time"], field="response_time", event=event
            )
            partner = match.group("partner")
            output[f"event_{partner}"].append((onset, duration, 1.0))
            output[f"event_{partner}_pmod"].append((onset, duration, offer))
            substantive_rt.append((onset, 0.0, response_time))
        elif event == "missed_trial":
            output["missed_trial"].append(
                (
                    numeric(row["onset"], field="onset", event=event),
                    numeric(row["duration"], field="duration", event=event),
                    1.0,
                )
            )
        elif event == "event_RT":
            onset = numeric(row["onset"], field="onset", event=event)
            duration = numeric(row["duration"], field="duration", event=event)
            response_time = numeric(
                row["response_time"], field="response_time", event=event
            )
            output["event_RT"].append((onset, duration, 1.0))
            output["event_RT_pmod"].append((onset, duration, response_time))

    if rt_source == "substantive":
        output["event_RT"] = [(onset, duration, 1.0) for onset, duration, _ in substantive_rt]
        output["event_RT_pmod"] = substantive_rt
    return output


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def generate(events_path: Path, output_prefix: Path, rt_source: str) -> dict[str, int]:
    rows = ev_rows(read_events(events_path), rt_source)
    counts: dict[str, int] = {}
    for name, values in rows.items():
        path = Path(f"{output_prefix}_{name}.txt")
        counts[name] = len(values)
        if values:
            write_atomic(path, three_column(values))
        elif path.exists():
            path.unlink()
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="prefix such as /work/EVfiles/sub-144/run-01",
    )
    parser.add_argument(
        "--rt-source",
        choices=("companion", "substantive"),
        default="companion",
        help="use explicit event_RT rows or derive RT EVs from responded decision rows",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    counts = generate(args.events, args.output_prefix, args.rt_source)
    print(
        "PASS: "
        f"events={args.events}, rt_source={args.rt_source}, "
        f"task_trials={sum(counts[f'event_{partner}'] for partner in PARTNERS)}, "
        f"misses={counts['missed_trial']}, rt_rows={counts['event_RT']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
