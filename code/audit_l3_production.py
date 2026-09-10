#!/usr/bin/env python3
"""Trace tracked paper masks to retained L3 outputs and collect smoothness.

The audit is read-only. It first matches tracked masks to retained cluster masks
by SHA-256. With ``--match-mode support``, it also compares the uncompressed
voxel support of a binary focal mask with each labeled FSL cluster mask. This
distinguishes a copied/recompressed file from the common case in which one
cluster label was selected and binarized. The enclosing GFEAT design,
participant inputs, inference settings, cluster table, and any FSL
residual-smoothness record are recorded for every match. An inventory of every
rendered group design found beneath the search roots is also written.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SETTING_RE = re.compile(r'^set (?:fmri\(([^)]+)\)|([^ ]+)) "?(.*?)"?$')
PARTICIPANT_RE = re.compile(r"/(sub-[A-Za-z0-9]+)/")
COPE_RE = re.compile(r"/cope([^/]+)\.feat/")
ZSTAT_RE = re.compile(r"cluster_mask_zstat(\d+)\.nii\.gz$")
NIFTI_DTYPES = {
    2: ("B", 1),
    4: ("h", 2),
    8: ("i", 4),
    16: ("f", 4),
    64: ("d", 8),
    256: ("b", 1),
    512: ("H", 2),
    768: ("I", 4),
    1024: ("q", 8),
    1280: ("Q", 8),
}


@dataclass(frozen=True)
class NiftiSupport:
    grid: tuple[object, ...]
    nonzero: dict[int, int | float]
    label_counts: Counter[int | float]


def _open_nifti(path: Path) -> bytes:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return path.read_bytes()


def _rounded(values: tuple[float, ...], digits: int = 5) -> tuple[float, ...]:
    return tuple(round(value, digits) for value in values)


def _label(value: float) -> int | float:
    nearest = round(value)
    if math.isclose(value, nearest, abs_tol=1e-5):
        return int(nearest)
    return round(value, 6)


def nifti_support(path: Path) -> NiftiSupport:
    """Read the grid and nonzero values of a NIfTI-1 image using stdlib only."""

    blob = _open_nifti(path)
    if len(blob) < 352:
        raise ValueError(f"not a complete NIfTI-1 file: {path}")
    little = struct.unpack_from("<i", blob, 0)[0]
    big = struct.unpack_from(">i", blob, 0)[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise ValueError(f"unsupported NIfTI header size in {path}")

    dims = struct.unpack_from(f"{endian}8h", blob, 40)
    ndim = dims[0]
    if not 1 <= ndim <= 7 or any(value < 1 for value in dims[1 : ndim + 1]):
        raise ValueError(f"invalid NIfTI dimensions in {path}: {dims}")
    shape = tuple(dims[1 : ndim + 1])
    datatype = struct.unpack_from(f"{endian}h", blob, 70)[0]
    if datatype not in NIFTI_DTYPES:
        raise ValueError(f"unsupported NIfTI datatype {datatype} in {path}")
    code, width = NIFTI_DTYPES[datatype]
    vox_offset = max(352, int(round(struct.unpack_from(f"{endian}f", blob, 108)[0])))
    slope = struct.unpack_from(f"{endian}f", blob, 112)[0]
    intercept = struct.unpack_from(f"{endian}f", blob, 116)[0]
    if slope == 0 or not math.isfinite(slope):
        slope = 1.0
    if not math.isfinite(intercept):
        intercept = 0.0
    count = math.prod(shape)
    required = vox_offset + count * width
    if len(blob) < required:
        raise ValueError(f"truncated NIfTI payload in {path}")

    pixdim = struct.unpack_from(f"{endian}8f", blob, 76)
    qform_code, sform_code = struct.unpack_from(f"{endian}2h", blob, 252)
    if sform_code > 0:
        transform = _rounded(struct.unpack_from(f"{endian}12f", blob, 280))
        transform_kind = "sform"
    else:
        quaternion = struct.unpack_from(f"{endian}6f", blob, 256)
        transform = _rounded((*pixdim[1:4], *quaternion))
        transform_kind = "qform"
    grid: tuple[object, ...] = (
        shape,
        _rounded(tuple(abs(value) for value in pixdim[1 : ndim + 1])),
        transform_kind,
        sform_code if sform_code > 0 else qform_code,
        transform,
    )

    payload = memoryview(blob)[vox_offset:required]
    nonzero: dict[int, int | float] = {}
    counts: Counter[int | float] = Counter()
    for index, (raw,) in enumerate(struct.iter_unpack(f"{endian}{code}", payload)):
        value = float(raw) * slope + intercept
        if math.isfinite(value) and not math.isclose(value, 0.0, abs_tol=1e-12):
            label = _label(value)
            nonzero[index] = label
            counts[label] += 1
    return NiftiSupport(grid=grid, nonzero=nonzero, label_counts=counts)


def support_match(
    focal: NiftiSupport, candidate: NiftiSupport
) -> dict[str, object] | None:
    """Return a strong focal-to-cluster support match, or ``None``."""

    if focal.grid != candidate.grid or not focal.nonzero or not candidate.nonzero:
        return None
    overlap_labels = Counter(
        candidate.nonzero[index]
        for index in focal.nonzero
        if index in candidate.nonzero
    )
    if not overlap_labels:
        return None
    label, overlap = max(
        overlap_labels.items(), key=lambda item: (item[1], -float(item[0]))
    )
    focal_voxels = len(focal.nonzero)
    cluster_voxels = candidate.label_counts[label]
    union = focal_voxels + cluster_voxels - overlap
    focal_fraction = overlap / focal_voxels
    cluster_fraction = overlap / cluster_voxels
    dice = 2 * overlap / (focal_voxels + cluster_voxels)
    jaccard = overlap / union

    if overlap == focal_voxels == cluster_voxels:
        method = "exact_cluster_support"
    elif overlap == focal_voxels:
        method = "focal_contained_in_cluster"
    elif overlap == cluster_voxels:
        method = "cluster_contained_in_focal"
    elif focal_fraction >= 0.5 or cluster_fraction >= 0.5:
        method = "partial_cluster_overlap"
    else:
        return None
    return {
        "match_method": method,
        "focal_voxels": focal_voxels,
        "cluster_label": label,
        "cluster_voxels": cluster_voxels,
        "overlap_voxels": overlap,
        "focal_overlap_fraction": f"{focal_fraction:.6f}",
        "cluster_overlap_fraction": f"{cluster_fraction:.6f}",
        "dice": f"{dice:.6f}",
        "jaccard": f"{jaccard:.6f}",
    }


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


def trace_record(
    mask_id: str,
    mask: Path,
    mask_hash: str,
    exact_match_count: int,
    support_match_count: int,
    root_label: str = "",
    cluster_mask: Path | None = None,
    spatial: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "mask_id": mask_id,
        "mask_path": str(mask),
        "mask_sha256": mask_hash,
        "exact_match_count": exact_match_count,
        "support_match_count": support_match_count,
        "match_method": "",
        "focal_voxels": "",
        "cluster_label": "",
        "cluster_voxels": "",
        "overlap_voxels": "",
        "focal_overlap_fraction": "",
        "cluster_overlap_fraction": "",
        "dice": "",
        "jaccard": "",
        "search_root": root_label,
        "matched_cluster_mask": str(cluster_mask) if cluster_mask else "",
        "cluster_mask_modified_utc": modified_utc(cluster_mask) if cluster_mask else "",
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
    if cluster_mask is None:
        return row

    row.update(spatial or {"match_method": "exact_sha256"})
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
    row.update(
        {
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
            "design_mat_sha256": sha256(gfeat / "design.mat")
            if (gfeat / "design.mat").is_file()
            else "",
            "design_con_sha256": sha256(gfeat / "design.con")
            if (gfeat / "design.con").is_file()
            else "",
            "design_grp_sha256": sha256(gfeat / "design.grp")
            if (gfeat / "design.grp").is_file()
            else "",
            "smoothness_path": smoothness_path,
            "dlh": dlh,
            "volume": volume,
            "resels": resels,
            "cluster_table_path": str(cluster_table) if cluster_table.is_file() else "",
            "cluster_table_sha256": sha256(cluster_table)
            if cluster_table.is_file()
            else "",
        }
    )
    return row


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
    match_mode: str = "exact",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    for label, path in [*search_roots, *masks]:
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")

    group_rows: list[dict[str, object]] = []
    cluster_by_hash: dict[str, list[tuple[str, Path]]] = {}
    cluster_masks: list[tuple[str, Path]] = []
    for label, root in search_roots:
        for fsf in sorted(root.rglob("*.gfeat/design.fsf")):
            if not fsf.is_file():
                continue
            group_rows.append(design_record(label, root, fsf))
        for cluster_mask in sorted(root.rglob("cluster_mask_zstat*.nii.gz")):
            if not cluster_mask.is_file():
                continue
            cluster_masks.append((label, cluster_mask))
            cluster_by_hash.setdefault(sha256(cluster_mask), []).append(
                (label, cluster_mask)
            )
    if not group_rows:
        raise FileNotFoundError("no rendered GFEAT design.fsf files found")

    mask_hashes = {mask_id: sha256(mask) for mask_id, mask in masks}
    focal_supports: dict[str, NiftiSupport] = {}
    support_matches: dict[str, list[tuple[str, Path, dict[str, object]]]] = {
        mask_id: [] for mask_id, _ in masks
    }
    unreadable_candidates = 0
    if match_mode == "support":
        focal_supports = {mask_id: nifti_support(mask) for mask_id, mask in masks}
        for root_label, cluster_mask in cluster_masks:
            try:
                candidate_support = nifti_support(cluster_mask)
            except (OSError, EOFError, ValueError, struct.error):
                unreadable_candidates += 1
                continue
            for mask_id, focal_support in focal_supports.items():
                match = support_match(focal_support, candidate_support)
                if match is not None:
                    support_matches[mask_id].append((root_label, cluster_mask, match))

    trace_rows: list[dict[str, object]] = []
    for mask_id, mask in masks:
        mask_hash = mask_hashes[mask_id]
        exact_matches = cluster_by_hash.get(mask_hash, [])
        spatial_matches = support_matches[mask_id]
        if exact_matches:
            matches = [
                (root_label, cluster_mask, None)
                for root_label, cluster_mask in exact_matches
            ]
        else:
            ordered_spatial = sorted(
                spatial_matches,
                key=lambda item: (
                    item[2]["match_method"] != "exact_cluster_support",
                    -float(item[2]["dice"]),
                    str(item[1]),
                ),
            )
            exact_support = [
                item
                for item in ordered_spatial
                if item[2]["match_method"] == "exact_cluster_support"
            ]
            contained = [
                item
                for item in ordered_spatial
                if item[2]["match_method"]
                in {"focal_contained_in_cluster", "cluster_contained_in_focal"}
            ]
            matches = exact_support or contained or ordered_spatial[:10]
        if not matches:
            trace_rows.append(
                trace_record(
                    mask_id,
                    mask,
                    mask_hash,
                    len(exact_matches),
                    len(spatial_matches),
                )
            )
            continue

        for root_label, cluster_mask, spatial in matches:
            trace_rows.append(
                trace_record(
                    mask_id,
                    mask,
                    mask_hash,
                    len(exact_matches),
                    len(spatial_matches),
                    root_label,
                    cluster_mask,
                    spatial,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "l3_group_inventory.tsv", group_rows)
    write_tsv(output_dir / "focal_l3_trace.tsv", trace_rows)
    if tracked_summary is not None:
        write_tsv(tracked_summary, trace_rows)
    unmatched = sum(
        1
        for mask_id, _ in masks
        if not cluster_by_hash.get(mask_hashes[mask_id], []) and not support_matches[mask_id]
    )
    print(
        f"PASS: group_designs={len(group_rows)}, focal_trace_rows={len(trace_rows)}, "
        f"unmatched_focal_masks={unmatched}, match_mode={match_mode}, "
        f"cluster_masks={len(cluster_masks)}, "
        f"unreadable_cluster_masks={unreadable_candidates}; output={output_dir}"
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
    parser.add_argument(
        "--match-mode",
        choices=("exact", "support"),
        default="exact",
        help=(
            "exact compares compressed-file SHA-256 only; support additionally "
            "tests whether a binary focal mask is one labeled FSL cluster"
        ),
    )
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
        args.match_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
