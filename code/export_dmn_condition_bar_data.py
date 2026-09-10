#!/usr/bin/env python3
"""Export corrected DMN FLAME bar estimates and input COPE/VARCOPE values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_RELATIVE = Path(
    "results/manuscript/source_data/figure3_dmn_corrected_cluster_mask.nii.gz"
)
OUTPUT_RELATIVE = Path("results/manuscript/source_data")
INPUT_RE = re.compile(r'^set feat_files\((\d+)\) "([^"]+)"$')
SUBJECT_RE = re.compile(r"/(sub-[A-Za-z0-9]+)/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def mean_in_mask(image: Path, mask: Path) -> float:
    if not image.is_file():
        raise FileNotFoundError(image)
    return float(run(["fslstats", str(image), "-k", str(mask), "-M"]))


def standard_error_in_mask(varcope: Path, mask: Path) -> float:
    with tempfile.TemporaryDirectory(prefix="srndna-dmn-se-") as temporary:
        standard_error = Path(temporary) / "se.nii.gz"
        run(["fslmaths", str(varcope), "-sqrt", str(standard_error)])
        return mean_in_mask(standard_error, mask)


def parse_inputs(path: Path) -> list[tuple[int, str, Path]]:
    rows: list[tuple[int, str, Path]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = INPUT_RE.match(line)
        if not match:
            continue
        subject = SUBJECT_RE.search(match.group(2))
        if subject is None:
            raise ValueError(f"cannot identify participant in {line}")
        rows.append((int(match.group(1)), subject.group(1), Path(match.group(2))))
    rows.sort()
    if len(rows) != 47 or len({subject for _, subject, _ in rows}) != 47:
        raise ValueError(f"expected 47 participant-unique inputs in {path}")
    return rows


def input_varcope(cope: Path) -> Path:
    marker = "/stats/cope1.nii.gz"
    if marker not in str(cope):
        raise ValueError(f"unexpected L2 cope path: {cope}")
    return Path(str(cope).replace(marker, "/stats/varcope1.nii.gz", 1))


def load_age_groups(path: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if float(row["younger"]) == 1.0:
                groups[row["subjID"]] = "younger"
            elif float(row["older"]) == 1.0:
                groups[row["subjID"]] = "older"
            else:
                raise ValueError(f"invalid age coding for {row['subjID']}")
    return groups


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def export(repository: Path, work_root: Path) -> None:
    for executable in ("fslstats", "fslmaths"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"required FSL executable not found: {executable}")
    mask = repository / MASK_RELATIVE
    if not mask.is_file():
        raise FileNotFoundError(mask)
    output_root = repository / OUTPUT_RELATIVE
    age_groups = load_age_groups(
        repository / "behavioral_analyses/data/participant_L3_47.csv"
    )

    bar_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = [
        {"artifact": "corrected_cluster_mask_sha256", "value": sha256(mask)},
        {
            "artifact": "bar_height",
            "value": "mean within cluster of condition-specific FLAME group COPE map",
        },
        {
            "artifact": "error_bar",
            "value": "mean within cluster of voxelwise sqrt(FLAME group VARCOPE)",
        },
    ]

    for condition in ("similar", "dissimilar"):
        fsf = work_root / "fsf" / f"dmn-age_condition-{condition}.fsf"
        gfeat = work_root / "outputs" / f"dmn-age_condition-{condition}.gfeat"
        if not fsf.is_file():
            raise FileNotFoundError(fsf)
        provenance_rows.append(
            {"artifact": f"{condition}_rendered_fsf_sha256", "value": sha256(fsf)}
        )
        inputs = parse_inputs(fsf)
        for design_index, participant, cope in inputs:
            varcope = input_varcope(cope)
            mean_cope = mean_in_mask(cope, mask)
            mean_varcope = mean_in_mask(varcope, mask)
            input_rows.append(
                {
                    "condition": condition,
                    "design_index": design_index,
                    "participant": participant,
                    "age_group": age_groups[participant],
                    "cluster_mean_cope": f"{mean_cope:.9g}",
                    "cluster_mean_varcope": f"{mean_varcope:.9g}",
                    "inverse_varcope_weight": (
                        f"{1.0 / mean_varcope:.9g}" if mean_varcope > 0 else ""
                    ),
                }
            )

        cope_dir = gfeat / "cope1.feat" / "stats"
        for contrast_index, age_group in ((1, "younger"), (2, "older")):
            cope = cope_dir / f"cope{contrast_index}.nii.gz"
            varcope = cope_dir / f"varcope{contrast_index}.nii.gz"
            estimate = mean_in_mask(cope, mask)
            mean_se = standard_error_in_mask(varcope, mask)
            bar_rows.append(
                {
                    "age_group": age_group,
                    "condition": condition,
                    "flame_cluster_mean_estimate": f"{estimate:.9g}",
                    "mean_voxelwise_standard_error": f"{mean_se:.9g}",
                    "display_conf_low": f"{estimate - 1.96 * mean_se:.9g}",
                    "display_conf_high": f"{estimate + 1.96 * mean_se:.9g}",
                }
            )
            provenance_rows.extend(
                [
                    {
                        "artifact": f"{condition}_{age_group}_cope_sha256",
                        "value": sha256(cope),
                    },
                    {
                        "artifact": f"{condition}_{age_group}_varcope_sha256",
                        "value": sha256(varcope),
                    },
                ]
            )

    if len(bar_rows) != 4 or len(input_rows) != 94:
        raise ValueError("expected four group bars and 94 participant-condition inputs")
    write_tsv(
        output_root / "figure3_dmn_flame_bar_summary.tsv",
        [
            "age_group",
            "condition",
            "flame_cluster_mean_estimate",
            "mean_voxelwise_standard_error",
            "display_conf_low",
            "display_conf_high",
        ],
        bar_rows,
    )
    write_tsv(
        output_root / "figure3_dmn_condition_input_roi.tsv",
        [
            "condition",
            "design_index",
            "participant",
            "age_group",
            "cluster_mean_cope",
            "cluster_mean_varcope",
            "inverse_varcope_weight",
        ],
        input_rows,
    )
    write_tsv(
        output_root / "figure3_dmn_flame_bar_provenance.tsv",
        ["artifact", "value"],
        provenance_rows,
    )
    print("PASS: exported 4 FLAME bar estimates and 94 COPE/VARCOPE input rows")
    print(f"PASS: output root={output_root}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export(args.repository.resolve(), args.work_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
