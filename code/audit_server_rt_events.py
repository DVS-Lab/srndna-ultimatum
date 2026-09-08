#!/usr/bin/env python3
"""Audit production RT EV files and retained FEAT designs without rerunning FSL."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


TRIAL_RE = re.compile(r"^(?:event_(?:accept|reject)_(?:computer|ingroup|outgroup)(?:_(?:un)?fair)?|missed_trial)$")


def read_table(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def sample_ids(path: Path) -> list[str]:
    rows = read_table(path, ",")
    values = [(row.get("subjID") or row.get("participant_id") or "").strip() for row in rows]
    if not values or any(not value for value in values):
        raise ValueError(f"invalid participant sample: {path}")
    return values


def nonblank_lines(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as stream:
        return sum(bool(line.strip()) for line in stream)


def fsf_value(lines: list[str], key: str) -> str | None:
    pattern = re.compile(rf'^set fmri\({re.escape(key)}\) "?(.*?)"?$')
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            return match.group(1).rstrip('"')
    return None


def parse_vest_matrix(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    in_matrix = False
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if line == "/Matrix":
                in_matrix = True
                continue
            if in_matrix and line:
                rows.append([float(value) for value in line.split()])
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError(f"invalid VEST matrix: {path}")
    return rows


def column_sd(matrix: list[list[float]], index: int) -> float | None:
    if not matrix or index >= len(matrix[0]):
        return None
    values = [row[index] for row in matrix]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def correlation(matrix: list[list[float]], left: int, right: int) -> float | None:
    if not matrix or max(left, right) >= len(matrix[0]):
        return None
    x = [row[left] for row in matrix]
    y = [row[right] for row in matrix]
    sx = statistics.stdev(x)
    sy = statistics.stdev(y)
    if sx == 0 or sy == 0:
        return None
    mx = statistics.mean(x)
    my = statistics.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / ((len(x) - 1) * sx * sy)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def event_counts(path: Path) -> tuple[int, int]:
    rows = read_table(path, "\t")
    responded = sum(bool(TRIAL_RE.match(row["trial_type"])) and row["trial_type"] != "missed_trial" for row in rows)
    rt_rows = sum(row["trial_type"] == "event_RT" for row in rows)
    return responded, rt_rows


def remap_path(path: Path, mappings: list[tuple[Path, Path]]) -> Path:
    for old_root, new_root in mappings:
        try:
            relative = path.relative_to(old_root)
        except ValueError:
            continue
        return new_root / relative
    return path


def audit(
    bids_root: Path,
    ev_root: Path,
    l1_root: Path,
    sample: Path,
    output_dir: Path,
    tracked_summary: Path | None = None,
    path_mappings: list[tuple[Path, Path]] | None = None,
) -> dict[str, object]:
    path_mappings = path_mappings or []
    run_rows: list[dict[str, object]] = []
    for participant in sample_ids(sample):
        for run in ("01", "02"):
            event_file = bids_root / participant / "func" / f"{participant}_task-ultimatum_run-{run}_events.tsv"
            if not event_file.is_file():
                raise FileNotFoundError(event_file)
            responded, source_rt_rows = event_counts(event_file)

            feat_dir = l1_root / participant / f"L1_task-ultimatum_model-02_type-act_run-{run}_sm-6.feat"
            fsf_path = feat_dir / "design.fsf"
            matrix_path = feat_dir / "design.mat"
            fsf_lines = fsf_path.read_text(encoding="utf-8", errors="replace").splitlines() if fsf_path.is_file() else []
            custom = {index: fsf_value(fsf_lines, f"custom{index}") for index in range(1, 10)}
            custom_paths = {
                index: remap_path(Path(value), path_mappings) if value else None for index, value in custom.items()
            }

            default_rt = ev_root / participant / "ultimatum-rt" / f"run-{run}_event_RT.txt"
            default_rt_pmod = ev_root / participant / "ultimatum-rt" / f"run-{run}_event_RT_pmod.txt"
            rt_path = custom_paths[8] or default_rt
            rt_pmod_path = custom_paths[9] or default_rt_pmod
            main_paths = [custom_paths[index] for index in (1, 3, 5)]
            main_counts = [nonblank_lines(path) for path in main_paths]

            matrix = parse_vest_matrix(matrix_path) if matrix_path.is_file() else []
            original_rt_sd = column_sd(matrix, 7)
            original_rt_pmod_sd = column_sd(matrix, 8)
            run_rows.append(
                {
                    "participant": participant,
                    "run": run,
                    "source_responded_trials": responded,
                    "source_event_RT_rows": source_rt_rows,
                    "rt_file_path_from_fsf": str(rt_path),
                    "rt_file_exists_now": int(rt_path.is_file()),
                    "rt_file_rows_now": "" if nonblank_lines(rt_path) is None else nonblank_lines(rt_path),
                    "rt_pmod_file_exists_now": int(rt_pmod_path.is_file()),
                    "rt_pmod_file_rows_now": "" if nonblank_lines(rt_pmod_path) is None else nonblank_lines(rt_pmod_path),
                    "main_task_ev_files_found_now": sum(path is not None and path.is_file() for path in main_paths),
                    "main_task_ev_rows_now": "" if any(value is None for value in main_counts) else sum(main_counts),
                    "design_fsf_exists": int(fsf_path.is_file()),
                    "design_mat_exists": int(matrix_path.is_file()),
                    "design_num_points": len(matrix),
                    "design_num_waves": len(matrix[0]) if matrix else 0,
                    "design_original_rt_column_sd": "" if original_rt_sd is None else f"{original_rt_sd:.12g}",
                    "design_original_rt_pmod_column_sd": "" if original_rt_pmod_sd is None else f"{original_rt_pmod_sd:.12g}",
                    "design_original_rt_vs_pmod_correlation": ""
                    if correlation(matrix, 7, 8) is None
                    else f"{correlation(matrix, 7, 8):.12g}",
                }
            )

    def count(predicate) -> int:
        return sum(bool(predicate(row)) for row in run_rows)

    summary_rows = [
        {"metric": "analysis_sample_runs_expected", "value": len(run_rows), "detail": "47 participants x 2 runs"},
        {"metric": "rendered_design_fsf_found", "value": count(lambda row: row["design_fsf_exists"]), "detail": "activation model-02"},
        {"metric": "rendered_design_mat_found", "value": count(lambda row: row["design_mat_exists"]), "detail": "activation model-02"},
        {"metric": "rt_files_found_now", "value": count(lambda row: row["rt_file_exists_now"]), "detail": "paths recorded by rendered design.fsf"},
        {"metric": "rt_pmod_files_found_now", "value": count(lambda row: row["rt_pmod_file_exists_now"]), "detail": "paths recorded by rendered design.fsf"},
        {
            "metric": "rt_file_rows_match_source_event_RT_rows",
            "value": count(lambda row: row["rt_file_exists_now"] and int(row["rt_file_rows_now"]) == row["source_event_RT_rows"]),
            "detail": "current EV file versus curated source-event count",
        },
        {
            "metric": "designs_with_nonconstant_rt_column",
            "value": count(lambda row: row["design_original_rt_column_sd"] != "" and float(row["design_original_rt_column_sd"]) > 0),
            "detail": "retained design.mat original EV column 8",
        },
        {
            "metric": "designs_with_nonconstant_rt_pmod_column",
            "value": count(lambda row: row["design_original_rt_pmod_column_sd"] != "" and float(row["design_original_rt_pmod_column_sd"]) > 0),
            "detail": "retained design.mat original EV column 9",
        },
        {
            "metric": "runs_with_three_main_task_ev_files_found_now",
            "value": count(lambda row: row["main_task_ev_files_found_now"] == 3),
            "detail": "computer, similar, and dissimilar task EV paths from design.fsf",
        },
    ]
    write_tsv(output_dir / "rt_production_by_run.tsv", run_rows)
    write_tsv(output_dir / "rt_production_summary.tsv", summary_rows)
    if tracked_summary is not None:
        write_tsv(tracked_summary, summary_rows)
    sub143 = [row for row in run_rows if row["participant"] == "sub-143"]
    print("PASS: " + ", ".join(f"{row['metric']}={row['value']}" for row in summary_rows))
    for row in sub143:
        print(
            "SUB143: "
            + ", ".join(
                f"{key}={row[key]}"
                for key in (
                    "run",
                    "source_responded_trials",
                    "source_event_RT_rows",
                    "rt_file_exists_now",
                    "design_mat_exists",
                    "design_original_rt_column_sd",
                    "design_original_rt_pmod_column_sd",
                    "main_task_ev_files_found_now",
                    "main_task_ev_rows_now",
                )
            )
        )
    return {row["metric"]: row["value"] for row in summary_rows}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=root / "source_data" / "bids")
    parser.add_argument("--ev-root", type=Path, required=True)
    parser.add_argument("--l1-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=root / "behavioral_analyses" / "data" / "participant_L3_47.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tracked-summary",
        type=Path,
        help="optional aggregate-only TSV suitable for version control; never contains run rows or server paths",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="remap a stale absolute FSF path prefix while reading; may be repeated",
    )
    return parser.parse_args()


def parse_path_mappings(values: list[str]) -> list[tuple[Path, Path]]:
    mappings: list[tuple[Path, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --path-map {value!r}; expected OLD=NEW")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError(f"invalid --path-map {value!r}; expected nonempty OLD=NEW")
        mappings.append((Path(old), Path(new)))
    return mappings


def main() -> int:
    args = parse_args()
    audit(
        args.bids_root,
        args.ev_root,
        args.l1_root,
        args.sample,
        args.output_dir,
        args.tracked_summary,
        parse_path_mappings(args.path_map),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
