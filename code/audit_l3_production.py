#!/usr/bin/env python3
"""Trace tracked paper masks to retained L3 outputs and collect smoothness.

The audit is read-only. It matches tracked masks to retained cluster masks by
SHA-256, then records the enclosing GFEAT design, participant inputs, inference
settings, cluster table, and any FSL residual-smoothness record. It also writes
an inventory of every rendered group design found beneath the search roots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SETTING_RE = re.compile(r'^set (?:fmri\(([^)]+)\)|([^ ]+)) "?(.*?)"?$')
PARTICIPANT_RE = re.compile(r"/(sub-[A-Za-z0-9]+)/")
COPE_RE = re.compile(r"/cope([^/]+)\.feat/")
ZSTAT_RE = re.compile(r"cluster_mask_zstat(\d+)\.nii\.gz$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def modified_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=/absolute/path")
    label, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"unsafe label: {label!r}")
    path = Path(raw_path).expanduser().resolve()
    return label, path


def fsf_settings(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SETTING_RE.match(raw.strip())
        if match:
            settings[match.group(1) or match.group(2)] = match.group(3).rstrip('"')
    return settings


def group_inputs(settings: dict[str, str]) -> list[tuple[int, str]]:
    inputs: list[tuple[int, str]] = []
    for key, value in settings.items():
        match = re.fullmatch(r"feat_files\((\d+)\)", key)
        if match:
            inputs.append((int(match.group(1)), value))
    return sorted(inputs)


def smoothness_values(cope_dir: Path) -> tuple[str, str, str, str]:
    candidates = (cope_dir / "stats" / "smoothness", cope_dir / "smoothness")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return "", "", "", ""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if len(fields) >= 2:
            values[fields[0].upper()] = fields[1]
    return str(path), values.get("DLH", ""), values.get("VOLUME", ""), values.get("RESELS", "")


def design_record(label: str, root: Path, fsf: Path) -> dict[str, object]:
    settings = fsf_settings(fsf)
    inputs = group_inputs(settings)
    participants = [
        match.group(1)
        for _, value in inputs
        if (match := PARTICIPANT_RE.search(value))
    ]
    sub144 = [(index, value) for index, value in inputs if "/sub-144/" in value]
    source_copes = sorted(
        {match.group(1) for _, value in inputs if (match := COPE_RE.search(value))}
    )
    gfeat = fsf.parent
    cope_dirs = sorted(path for path in gfeat.glob("cope*.feat") if path.is_dir())
    return {
        "search_root": label,
        "gfeat_path": str(gfeat),
        "relative_gfeat_path": str(gfeat.relative_to(root)),
        "modified_utc": modified_utc(fsf),
        "npts": settings.get("npts", ""),
        "multiple": settings.get("multiple", ""),
        "mixed_yn": settings.get("mixed_yn", ""),
        "robust_yn": settings.get("robust_yn", ""),
        "z_thresh": settings.get("z_thresh", ""),
        "prob_thresh": settings.get("prob_thresh", ""),
        "thresh": settings.get("thresh", ""),
        "poststats_yn": settings.get("poststats_yn", ""),
        "input_count": len(inputs),
        "unique_participants": len(set(participants)),
        "sub144_input_indices": ",".join(str(index) for index, _ in sub144),
        "sub144_input_paths": "|".join(value for _, value in sub144),
        "source_copes": ",".join(source_copes),
        "cope_level_outputs": len(cope_dirs),
        "cluster_masks": sum(1 for path in cope_dirs for _ in path.glob("cluster_mask_zstat*.nii.gz")),
        "smoothness_files": sum(
            int((path / "stats/smoothness").is_file() or (path / "smoothness").is_file())
            for path in cope_dirs
        ),
        "design_fsf_sha256": sha256(fsf),
        "design_mat_sha256": sha256(gfeat / "design.mat") if (gfeat / "design.mat").is_file() else "",
        "design_con_sha256": sha256(gfeat / "design.con") if (gfeat / "design.con").is_file() else "",
        "design_grp_sha256": sha256(gfeat / "design.grp") if (gfeat / "design.grp").is_file() else "",
    }


def enclosing_cope_dir(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name.endswith(".feat") and parent.parent.name.endswith(".gfeat"):
            return parent
    return None


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def audit(
    search_roots: list[tuple[str, Path]],
    masks: list[tuple[str, Path]],
    output_dir: Path,
    tracked_summary: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    for label, path in [*search_roots, *masks]:
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")

    group_rows: list[dict[str, object]] = []
    cluster_by_hash: dict[str, list[tuple[str, Path]]] = {}
    for label, root in search_roots:
        for fsf in sorted(root.rglob("*.gfeat/design.fsf")):
            if not fsf.is_file():
                continue
            group_rows.append(design_record(label, root, fsf))
        for cluster_mask in sorted(root.rglob("cluster_mask_zstat*.nii.gz")):
            if not cluster_mask.is_file():
                continue
            cluster_by_hash.setdefault(sha256(cluster_mask), []).append(
                (label, cluster_mask)
            )
    if not group_rows:
        raise FileNotFoundError("no rendered GFEAT design.fsf files found")

    trace_rows: list[dict[str, object]] = []
    for mask_id, mask in masks:
        mask_hash = sha256(mask)
        matches = cluster_by_hash.get(mask_hash, [])
        if not matches:
            trace_rows.append(
                {
                    "mask_id": mask_id,
                    "mask_path": str(mask),
                    "mask_sha256": mask_hash,
                    "exact_match_count": 0,
                    "search_root": "",
                    "matched_cluster_mask": "",
                    "cluster_mask_modified_utc": "",
                    "zstat_index": "",
                    "gfeat_path": "",
                    "cope_dir": "",
                    "npts": "",
                    "mixed_yn": "",
                    "robust_yn": "",
                    "z_thresh": "",
                    "prob_thresh": "",
                    "input_count": "",
                    "unique_participants": "",
                    "sub144_input_indices": "",
                    "sub144_input_paths": "",
                    "source_copes": "",
                    "design_fsf_sha256": "",
                    "design_mat_sha256": "",
                    "design_con_sha256": "",
                    "design_grp_sha256": "",
                    "smoothness_path": "",
                    "dlh": "",
                    "volume": "",
                    "resels": "",
                    "cluster_table_path": "",
                    "cluster_table_sha256": "",
                }
            )
            continue

        for root_label, cluster_mask in matches:
            cope_dir = enclosing_cope_dir(cluster_mask)
            if cope_dir is None:
                raise ValueError(f"cluster mask is not inside GFEAT cope output: {cluster_mask}")
            gfeat = cope_dir.parent
            group_fsf = gfeat / "design.fsf"
            settings = fsf_settings(group_fsf)
            inputs = group_inputs(settings)
            participants = [
                match.group(1)
                for _, value in inputs
                if (match := PARTICIPANT_RE.search(value))
            ]
            sub144 = [(index, value) for index, value in inputs if "/sub-144/" in value]
            source_copes = sorted(
                {match.group(1) for _, value in inputs if (match := COPE_RE.search(value))}
            )
            zstat_match = ZSTAT_RE.search(cluster_mask.name)
            zstat = zstat_match.group(1) if zstat_match else ""
            cluster_table = cope_dir / f"cluster_zstat{zstat}.txt"
            smoothness_path, dlh, volume, resels = smoothness_values(cope_dir)
            trace_rows.append(
                {
                    "mask_id": mask_id,
                    "mask_path": str(mask),
                    "mask_sha256": mask_hash,
                    "exact_match_count": len(matches),
                    "search_root": root_label,
                    "matched_cluster_mask": str(cluster_mask),
                    "cluster_mask_modified_utc": modified_utc(cluster_mask),
                    "zstat_index": zstat,
                    "gfeat_path": str(gfeat),
                    "cope_dir": str(cope_dir),
                    "npts": settings.get("npts", ""),
                    "mixed_yn": settings.get("mixed_yn", ""),
                    "robust_yn": settings.get("robust_yn", ""),
                    "z_thresh": settings.get("z_thresh", ""),
                    "prob_thresh": settings.get("prob_thresh", ""),
                    "input_count": len(inputs),
                    "unique_participants": len(set(participants)),
                    "sub144_input_indices": ",".join(str(index) for index, _ in sub144),
                    "sub144_input_paths": "|".join(value for _, value in sub144),
                    "source_copes": ",".join(source_copes),
                    "design_fsf_sha256": sha256(group_fsf),
                    "design_mat_sha256": sha256(gfeat / "design.mat") if (gfeat / "design.mat").is_file() else "",
                    "design_con_sha256": sha256(gfeat / "design.con") if (gfeat / "design.con").is_file() else "",
                    "design_grp_sha256": sha256(gfeat / "design.grp") if (gfeat / "design.grp").is_file() else "",
                    "smoothness_path": smoothness_path,
                    "dlh": dlh,
                    "volume": volume,
                    "resels": resels,
                    "cluster_table_path": str(cluster_table) if cluster_table.is_file() else "",
                    "cluster_table_sha256": sha256(cluster_table) if cluster_table.is_file() else "",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "l3_group_inventory.tsv", group_rows)
    write_tsv(output_dir / "focal_l3_trace.tsv", trace_rows)
    if tracked_summary is not None:
        write_tsv(tracked_summary, trace_rows)
    unmatched = sum(int(row["exact_match_count"] == 0) for row in trace_rows)
    print(
        f"PASS: group_designs={len(group_rows)}, focal_trace_rows={len(trace_rows)}, "
        f"unmatched_focal_masks={unmatched}; output={output_dir}"
    )
    return group_rows, trace_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root", action="append", type=parse_labeled_path, required=True
    )
    parser.add_argument("--mask", action="append", type=parse_labeled_path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracked-summary", type=Path)
    parser.set_defaults(
        default_masks=[
            ("dmn_age", root / "masks_SANS/dmn_p_in-out_y-o.nii.gz"),
            (
                "ecn_sensitivity",
                root / "masks_SANS/ecn_p_in-out_o-y_sens-logit.nii.gz",
            ),
            (
                "activation",
                root / "masks_SANS/act_p_in-out_o-y_norm-logit.nii.gz",
            ),
        ]
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit(
        args.search_root,
        args.mask or args.default_masks,
        args.output_dir.resolve(),
        args.tracked_summary.resolve() if args.tracked_summary else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
