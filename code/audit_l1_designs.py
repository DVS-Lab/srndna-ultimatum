#!/usr/bin/env python3
"""Audit retained L1 FEAT design matrices without running or modifying FSL."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    import numpy as np
except ImportError:  # Reported clearly by main; permits syntax/static validation.
    np = None


EV_TITLE_RE = re.compile(r'^set fmri\(evtitle(\d+)\) "(.*)"$')
WARNING_RE = re.compile(r"rank|collinear|singular|warning", re.IGNORECASE)


def read_sample(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    values = [(row.get("subjID") or row.get("participant_id") or "").strip() for row in rows]
    if not values or any(not value for value in values):
        raise ValueError(f"invalid participant sample: {path}")
    return values


def read_vest_matrix(path: Path) -> np.ndarray:
    if np is None:
        raise RuntimeError("audit_l1_designs.py requires NumPy")
    rows: list[list[float]] = []
    in_matrix = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "/Matrix":
            in_matrix = True
        elif in_matrix and line:
            rows.append([float(value) for value in line.split()])
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError(f"invalid VEST matrix: {path}")
    return matrix


def ev_titles(path: Path) -> str:
    if not path.is_file():
        return ""
    titles: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := EV_TITLE_RE.match(raw.strip()):
            titles[int(match.group(1))] = match.group(2)
    return "|".join(f"{index}:{titles[index]}" for index in sorted(titles))


def warning_count(feat_dir: Path) -> int:
    count = 0
    candidates = list(feat_dir.glob("*.log")) + list((feat_dir / "logs").glob("*"))
    for path in candidates:
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        count += len(WARNING_RE.findall(text))
    return count


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, object]:
    if np is None:
        raise RuntimeError("audit_l1_designs.py requires NumPy")
    npoints, nwaves = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
    norms = np.linalg.norm(matrix, axis=0)
    nonzero = norms > np.finfo(float).eps
    normalized = matrix[:, nonzero] / norms[nonzero]
    condition = float(np.linalg.cond(normalized)) if normalized.size else float("inf")

    sds = np.std(matrix, axis=0, ddof=1)
    varying = np.flatnonzero(sds > np.finfo(float).eps)
    if len(varying) > 1:
        correlations = np.corrcoef(matrix[:, varying], rowvar=False)
        upper = np.triu_indices_from(correlations, k=1)
        absolute = np.abs(correlations[upper])
        max_position = int(np.argmax(absolute))
        max_abs = float(absolute[max_position])
        pair = (
            int(varying[upper[0][max_position]]) + 1,
            int(varying[upper[1][max_position]]) + 1,
        )
        counts = {threshold: int(np.count_nonzero(absolute > threshold)) for threshold in (0.80, 0.90, 0.95)}
    else:
        max_abs = float("nan")
        pair = (0, 0)
        counts = {threshold: 0 for threshold in (0.80, 0.90, 0.95)}

    return {
        "npoints": npoints,
        "nwaves": nwaves,
        "rank": rank,
        "rank_deficient": int(rank < nwaves),
        "zero_variance_columns": int(nwaves - len(varying)),
        "column_normalized_condition_number": condition,
        "max_abs_column_correlation": max_abs,
        "max_correlation_column_pair": f"{pair[0]},{pair[1]}",
        "pairs_abs_r_gt_0_80": counts[0.80],
        "pairs_abs_r_gt_0_90": counts[0.90],
        "pairs_abs_r_gt_0_95": counts[0.95],
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(l1_root: Path, sample: Path, models: list[str], output_dir: Path, tracked_summary: Path | None) -> None:
    run_rows: list[dict[str, object]] = []
    for participant in read_sample(sample):
        for run in ("01", "02"):
            for model in models:
                feat_dir = l1_root / participant / (
                    f"L1_task-ultimatum_model-02_type-{model}_run-{run}_sm-6.feat"
                )
                matrix_path = feat_dir / "design.mat"
                row: dict[str, object] = {
                    "participant": participant,
                    "run": run,
                    "model": model,
                    "feat_dir_exists": int(feat_dir.is_dir()),
                    "design_mat_exists": int(matrix_path.is_file()),
                    "design_fsf_exists": int((feat_dir / "design.fsf").is_file()),
                    "ev_titles": ev_titles(feat_dir / "design.fsf"),
                    "warning_keyword_count": warning_count(feat_dir) if feat_dir.is_dir() else 0,
                }
                if matrix_path.is_file():
                    row.update(matrix_diagnostics(read_vest_matrix(matrix_path)))
                else:
                    row.update(
                        {
                            "npoints": 0,
                            "nwaves": 0,
                            "rank": 0,
                            "rank_deficient": "",
                            "zero_variance_columns": "",
                            "column_normalized_condition_number": "",
                            "max_abs_column_correlation": "",
                            "max_correlation_column_pair": "",
                            "pairs_abs_r_gt_0_80": "",
                            "pairs_abs_r_gt_0_90": "",
                            "pairs_abs_r_gt_0_95": "",
                        }
                    )
                run_rows.append(row)

    summary_rows: list[dict[str, object]] = []
    for model in models:
        rows = [row for row in run_rows if row["model"] == model]
        available = [row for row in rows if row["design_mat_exists"]]
        conditions = [float(row["column_normalized_condition_number"]) for row in available]
        correlations = [float(row["max_abs_column_correlation"]) for row in available]
        worst_condition = max(available, key=lambda row: float(row["column_normalized_condition_number"])) if available else None
        worst_correlation = max(available, key=lambda row: float(row["max_abs_column_correlation"])) if available else None
        summary_rows.append(
            {
                "model": model,
                "runs_expected": len(rows),
                "design_matrices_found": len(available),
                "rank_deficient_runs": sum(int(row["rank_deficient"]) for row in available),
                "runs_with_warning_keywords": sum(int(row["warning_keyword_count"]) > 0 for row in rows),
                "condition_number_median": "" if not conditions else f"{np.median(conditions):.12g}",
                "condition_number_range": "" if not conditions else f"{min(conditions):.12g},{max(conditions):.12g}",
                "worst_condition_run": "" if worst_condition is None else f"{worst_condition['participant']}_run-{worst_condition['run']}",
                "max_abs_correlation_median": "" if not correlations else f"{np.median(correlations):.12g}",
                "max_abs_correlation_range": "" if not correlations else f"{min(correlations):.12g},{max(correlations):.12g}",
                "worst_correlation_run": "" if worst_correlation is None else f"{worst_correlation['participant']}_run-{worst_correlation['run']}",
                "runs_with_pair_abs_r_gt_0_80": sum(int(row["pairs_abs_r_gt_0_80"]) > 0 for row in available),
                "runs_with_pair_abs_r_gt_0_90": sum(int(row["pairs_abs_r_gt_0_90"]) > 0 for row in available),
                "runs_with_pair_abs_r_gt_0_95": sum(int(row["pairs_abs_r_gt_0_95"]) > 0 for row in available),
            }
        )

    write_tsv(output_dir / "l1_design_by_run.tsv", run_rows)
    write_tsv(output_dir / "l1_design_summary.tsv", summary_rows)
    if tracked_summary is not None:
        write_tsv(tracked_summary, summary_rows)
    print(
        "PASS: "
        + ", ".join(
            f"{row['model']}={row['design_matrices_found']}/{row['runs_expected']} matrices, "
            f"rank_deficient={row['rank_deficient_runs']}"
            for row in summary_rows
        )
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=root / "behavioral_analyses/data/participant_L3_47.csv")
    parser.add_argument("--models", nargs="+", default=["act", "nppi-dmn", "nppi-ecn"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracked-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    if np is None:
        raise SystemExit("ERROR: audit_l1_designs.py requires NumPy")
    args = parse_args()
    audit(args.l1_root, args.sample, args.models, args.output_dir, args.tracked_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
