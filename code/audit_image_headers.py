#!/usr/bin/env python3
"""Create a read-only inventory of focal NIfTI masks using FSL tools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_RESULTS = (
    ("dmn_age", "masks_SANS/dmn_p_in-out_y-o.nii.gz"),
    ("ecn_sensitivity", "masks_SANS/ecn_p_in-out_o-y_sens-logit.nii.gz"),
    ("activation", "masks_SANS/act_p_in-out_o-y_norm-logit.nii.gz"),
)


def parse_header(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    wanted = {f"dim{i}" for i in range(1, 5)} | {f"pixdim{i}" for i in range(1, 5)} | {
        "datatype",
        "intent",
        "qform_name",
        "sform_name",
    }
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in wanted:
            parsed[fields[0]] = " ".join(fields[1:])
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_fsl_tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    raise FileNotFoundError(f"required FSL command not found: {name}")


def inspect(label: str, path: Path, root: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = parse_header(subprocess.run([resolve_fsl_tool("fslhd"), str(path)], check=True, text=True, capture_output=True).stdout)
    volume_fields = subprocess.run(
        [resolve_fsl_tool("fslstats"), str(path), "-V"], check=True, text=True, capture_output=True
    ).stdout.split()
    if len(volume_fields) != 2:
        raise ValueError(f"unexpected fslstats -V output for {path}: {volume_fields}")
    try:
        display_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        display_path = str(path.resolve())
    return {
        "result_id": label,
        "path": display_path,
        "sha256": sha256(path),
        "dim_x": header.get("dim1", ""),
        "dim_y": header.get("dim2", ""),
        "dim_z": header.get("dim3", ""),
        "dim_t": header.get("dim4", ""),
        "voxel_x_mm": header.get("pixdim1", ""),
        "voxel_y_mm": header.get("pixdim2", ""),
        "voxel_z_mm": header.get("pixdim3", ""),
        "tr_seconds": header.get("pixdim4", ""),
        "nonzero_voxels": int(volume_fields[0]),
        "nonzero_volume_mm3": float(volume_fields[1]),
        "intent": header.get("intent", ""),
        "qform": header.get("qform_name", ""),
        "sform": header.get("sform_name", ""),
    }


def parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("result must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError("result label contains unsupported characters")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--result", action="append", type=parse_result, help="LABEL=PATH; repeatable")
    parser.add_argument("--output", type=Path, default=root / "results" / "reviewer" / "tables" / "focal_cluster_inventory.tsv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = args.result or [(label, args.root / path) for label, path in DEFAULT_RESULTS]
    rows = [inspect(label, path if path.is_absolute() else args.root / path, args.root) for label, path in requested]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: inspected {len(rows)} NIfTI results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
