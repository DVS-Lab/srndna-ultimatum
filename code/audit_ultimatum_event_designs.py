#!/usr/bin/env python3
"""Match retained activation designs to historical/current Ultimatum event EVs.

This is a provenance audit, not an image analysis. It runs FSL ``feat_model``
only, generating candidate design matrices without fitting any voxel data or
modifying retained FEAT directories.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Sequence

from audit_server_rt_events import parse_vest_matrix
from make_ultimatum_3col import ev_rows, read_events, three_column, write_atomic


DEFAULT_HISTORICAL_REVISION = "02ba301"
CUSTOM_NAMES = {
    1: "event_computer",
    2: "event_computer_pmod",
    3: "event_ingroup",
    4: "event_ingroup_pmod",
    5: "event_outgroup",
    6: "event_outgroup_pmod",
    7: "missed_trial",
    8: "event_RT",
    9: "event_RT_pmod",
}


def parse_events_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def events_for_revision(
    repository: Path, relative_path: Path, revision: str
) -> list[dict[str, str]]:
    if revision == "WORKTREE":
        return read_events(repository / relative_path)
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{revision}:{relative_path.as_posix()}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rows = parse_events_text(result.stdout)
    if not rows:
        raise ValueError(f"empty event file at {revision}:{relative_path}")
    return rows


def write_candidate_evs(
    events: list[dict[str, str]], prefix: Path, rt_source: str
) -> dict[str, int]:
    values = ev_rows(events, rt_source)
    counts: dict[str, int] = {}
    for name, rows in values.items():
        counts[name] = len(rows)
        path = Path(f"{prefix}_{name}.txt")
        if rows:
            write_atomic(path, three_column(rows))
        elif path.exists():
            path.unlink()
    return counts


def replace_fsf_setting(text: str, key: str, value: str, *, quoted: bool) -> str:
    rendered_value = f'"{value}"' if quoted else value
    pattern = re.compile(rf"^set (?:fmri\()?{re.escape(key)}\)? .*$", re.MULTILINE)
    replacement_key = f"fmri({key})" if key != "confoundev_files(1)" else key
    replacement = f"set {replacement_key} {rendered_value}"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"expected one FSF setting for {key}, found {count}")
    return updated


def render_fsf(
    source: Path,
    destination: Path,
    model_root: Path,
    ev_prefix: Path,
    counts: dict[str, int],
) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    text = replace_fsf_setting(text, "outputdir", str(model_root), quoted=True)
    text = replace_fsf_setting(text, "confoundevs", "0", quoted=False)
    text = replace_fsf_setting(text, "confoundev_files(1)", "", quoted=True)
    text = replace_fsf_setting(
        text, "shape7", "3" if counts["missed_trial"] else "10", quoted=False
    )
    for index, name in CUSTOM_NAMES.items():
        text = replace_fsf_setting(
            text, f"custom{index}", str(Path(f"{ev_prefix}_{name}.txt")), quoted=True
        )
    write_atomic(destination, text)


def matrix_metrics(
    production: list[list[float]], candidate: list[list[float]], columns: int = 9
) -> dict[str, object]:
    if len(production) != len(candidate):
        raise ValueError(
            f"row mismatch: production={len(production)}, candidate={len(candidate)}"
        )
    if min(len(production[0]), len(candidate[0])) < columns:
        raise ValueError(
            "insufficient columns: "
            f"production={len(production[0])}, candidate={len(candidate[0])}"
        )
    differences: list[float] = []
    production_squares: list[float] = []
    correlations: list[float] = []
    for index in range(columns):
        left = [row[index] for row in production]
        right = [row[index] for row in candidate]
        differences.extend(a - b for a, b in zip(left, right))
        production_squares.extend(a * a for a in left)
        left_sd = statistics.stdev(left)
        right_sd = statistics.stdev(right)
        if left_sd == 0 and right_sd == 0:
            correlations.append(1.0 if left == right else 0.0)
        elif left_sd == 0 or right_sd == 0:
            correlations.append(0.0)
        else:
            correlations.append(
                sum(
                    (a - statistics.mean(left)) * (b - statistics.mean(right))
                    for a, b in zip(left, right)
                )
                / ((len(left) - 1) * left_sd * right_sd)
            )
    squared_error = sum(value * value for value in differences)
    denominator = sum(production_squares)
    return {
        "columns_compared": columns,
        "max_absolute_difference": max(abs(value) for value in differences),
        "relative_rmse": math.sqrt(squared_error / denominator) if denominator else math.inf,
        "minimum_column_correlation": min(correlations),
        "mean_column_correlation": statistics.mean(correlations),
        "column_correlations": ",".join(f"{value:.12g}" for value in correlations),
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_revision(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"invalid revision {value!r}; expected LABEL=REVISION")
    label, revision = value.split("=", 1)
    if not label or not revision or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"invalid revision {value!r}; expected safe LABEL=REVISION")
    return label, revision


def audit(
    repository: Path,
    production_l1_root: Path,
    work_root: Path,
    output: Path,
    subjects: list[str],
    revisions: list[tuple[str, str]],
    feat_model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject in subjects:
        for run in ("01", "02"):
            relative = Path(
                f"source_data/bids/{subject}/func/"
                f"{subject}_task-ultimatum_run-{run}_events.tsv"
            )
            feat_dir = production_l1_root / subject / (
                f"L1_task-ultimatum_model-02_type-act_run-{run}_sm-6.feat"
            )
            production_fsf = feat_dir / "design.fsf"
            production_matrix_path = feat_dir / "design.mat"
            if not production_fsf.is_file() or not production_matrix_path.is_file():
                raise FileNotFoundError(
                    f"missing retained activation design for {subject} run-{run}: {feat_dir}"
                )
            production_matrix = parse_vest_matrix(production_matrix_path)

            for label, revision in revisions:
                events = events_for_revision(repository, relative, revision)
                for rt_source in ("companion", "substantive"):
                    candidate_dir = work_root / subject / f"run-{run}" / f"{label}_{rt_source}"
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    ev_prefix = candidate_dir / "ev" / f"run-{run}"
                    counts = write_candidate_evs(events, ev_prefix, rt_source)
                    row: dict[str, object] = {
                        "participant": subject,
                        "run": run,
                        "event_candidate": label,
                        "git_revision": revision,
                        "rt_source": rt_source,
                        "task_trials": sum(
                            counts[f"event_{partner}"]
                            for partner in ("computer", "ingroup", "outgroup")
                        ),
                        "missed_trials": counts["missed_trial"],
                        "rt_rows": counts["event_RT"],
                        "status": "",
                        "production_num_waves": len(production_matrix[0]),
                        "candidate_num_waves": "",
                        "columns_compared": "",
                        "max_absolute_difference": "",
                        "relative_rmse": "",
                        "minimum_column_correlation": "",
                        "mean_column_correlation": "",
                        "column_correlations": "",
                        "best_for_run": 0,
                    }
                    if counts["event_RT"] == 0:
                        row["status"] = "not_estimable_no_rt_rows"
                        rows.append(row)
                        continue
                    model_root = candidate_dir / "candidate"
                    candidate_fsf = Path(f"{model_root}.fsf")
                    render_fsf(
                        production_fsf,
                        candidate_fsf,
                        model_root,
                        ev_prefix,
                        counts,
                    )
                    result = subprocess.run(
                        [feat_model, str(model_root)],
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    if result.returncode != 0:
                        row["status"] = f"feat_model_failed_{result.returncode}"
                        (candidate_dir / "feat_model.log").write_text(
                            result.stdout, encoding="utf-8"
                        )
                        rows.append(row)
                        continue
                    candidate_matrix = parse_vest_matrix(Path(f"{model_root}.mat"))
                    row["candidate_num_waves"] = len(candidate_matrix[0])
                    row.update(matrix_metrics(production_matrix, candidate_matrix))
                    row["status"] = "compared"
                    rows.append(row)

            comparable = [
                row
                for row in rows
                if row["participant"] == subject
                and row["run"] == run
                and row["status"] == "compared"
            ]
            if comparable:
                best = min(comparable, key=lambda item: float(item["relative_rmse"]))
                best["best_for_run"] = 1

    write_tsv(output, rows)
    best_rows = [row for row in rows if row["best_for_run"]]
    for row in best_rows:
        print(
            "BEST: "
            f"{row['participant']} run-{row['run']} "
            f"candidate={row['event_candidate']} rt_source={row['rt_source']} "
            f"relative_rmse={float(row['relative_rmse']):.6g} "
            f"minimum_column_r={float(row['minimum_column_correlation']):.6g}"
        )
    print(f"PASS: compared candidate event designs; output={output}")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=root)
    parser.add_argument("--production-l1-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["sub-143", "sub-144"])
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="LABEL=REVISION",
        help="candidate event source; use REVISION=WORKTREE for current files",
    )
    parser.add_argument("--feat-model", default="feat_model")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which(args.feat_model) is None:
        raise SystemExit(f"ERROR: feat_model not found: {args.feat_model}")
    revisions = (
        [parse_revision(value) for value in args.revision]
        if args.revision
        else [("corrected", "WORKTREE"), ("historical", DEFAULT_HISTORICAL_REVISION)]
    )
    audit(
        args.repository.resolve(),
        args.production_l1_root.resolve(),
        args.work_root.resolve(),
        args.output.resolve(),
        args.subjects,
        revisions,
        args.feat_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
