#!/usr/bin/env python3
"""Build the final manuscript figure set from tracked source data."""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("results/manuscript/source_data")
FIGURE_ROOT = Path("results/manuscript/figures")
COLORS = {"younger": "#007C83", "older": "#D55E00"}


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    margin = 1.96 * standard_error
    return mean, mean - margin, mean + margin


def read_vest_matrix(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        matrix_start = lines.index("/Matrix") + 1
    except ValueError as error:
        raise ValueError(f"missing /Matrix section in {path}") from error
    return np.loadtxt(io.StringIO("\n".join(lines[matrix_start:])))


def build_figure3(repository: Path, source_root: Path, figure_root: Path) -> None:
    table_path = source_root / "figure3_dmn_corrected_roi.tsv"
    zstat_path = source_root / "figure3_dmn_corrected_cluster_zstat.nii.gz"
    network_path = repository / "masks/nan_rPNAS_2mm_net0003.nii.gz"
    design_path = repository / (
        "results/reviewer/l3_repair_designs/dmn-age/"
        "reported-covariates-corrected/design.mat"
    )
    for required in (table_path, zstat_path, network_path, design_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing Figure 3 source: {required}")

    data = pd.read_csv(table_path, sep="\t")
    if len(data) != 47 or set(data["age_group"]) != {"younger", "older"}:
        raise ValueError("Figure 3 source table must contain both groups and 47 rows")
    design = read_vest_matrix(design_path)
    if design.shape != (47, 6):
        raise ValueError("corrected DMN design matrix must be 47 x 6")
    younger_rows = data["age_group"].eq("younger").to_numpy()
    older_rows = data["age_group"].eq("older").to_numpy()
    if not (
        np.array_equal(design[:, 0], younger_rows.astype(float))
        and np.array_equal(design[:, 1], older_rows.astype(float))
    ):
        raise ValueError("Figure 3 source rows do not match the corrected design matrix")
    raw_values = data["similar_minus_dissimilar"].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design, raw_values, rcond=None)[0]
    data["nuisance_adjusted_for_display"] = (
        raw_values - design[:, 2:] @ coefficients[2:]
    )
    data.to_csv(
        source_root / "figure3_dmn_plot_data.tsv",
        sep="\t",
        index=False,
        float_format="%.9g",
    )
    zstat = nib.load(zstat_path)
    if int(np.count_nonzero(np.asanyarray(zstat.dataobj))) != 29:
        raise ValueError("Figure 3 corrected Z-stat image must contain 29 nonzero voxels")

    figure = plt.figure(figsize=(10.0, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.1, 1.0))
    network_axis = figure.add_subplot(grid[0, 0])
    cluster_axis = figure.add_subplot(grid[0, 1])
    roi_axis = figure.add_subplot(grid[1, :])

    plotting.plot_stat_map(
        network_path,
        display_mode="ortho",
        cut_coords=(0, -54, 27),
        threshold=0.5,
        cmap="Greens",
        colorbar=False,
        annotate=True,
        draw_cross=False,
        axes=network_axis,
    )
    plotting.plot_stat_map(
        zstat_path,
        display_mode="ortho",
        cut_coords=(-13.3, 34.0, 27.8),
        threshold=3.1,
        cmap="YlOrRd",
        vmin=3.1,
        vmax=4.1,
        colorbar=True,
        annotate=True,
        draw_cross=False,
        axes=cluster_axis,
    )

    random = np.random.default_rng(20260910)
    group_positions = {"younger": 0.25, "older": 0.75}
    for group in ("younger", "older"):
        x_position = group_positions[group]
        group_values = data.loc[
            data["age_group"] == group, "nuisance_adjusted_for_display"
        ].to_numpy(dtype=float)
        jitter = random.uniform(-0.045, 0.045, len(group_values))
        roi_axis.scatter(
            np.full(len(group_values), x_position) + jitter,
            group_values,
            s=44,
            alpha=0.72,
            facecolor=COLORS[group],
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        mean, lower, upper = mean_ci(group_values)
        roi_axis.errorbar(
            x_position,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="o",
            markersize=9,
            color="black",
            capsize=5,
            linewidth=2,
            zorder=3,
        )

    roi_axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--", zorder=1)
    roi_axis.set_xlim(0, 1)
    roi_axis.set_xticks([0.25, 0.75], ["Younger", "Older"])
    roi_axis.set_ylabel("Nuisance-adjusted DMN connectivity contrast")
    roi_axis.set_xlabel("Age group")
    roi_axis.spines[["top", "right"]].set_visible(False)
    roi_axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    roi_axis.set_axisbelow(True)

    figure.text(0.015, 0.965, "A", fontsize=16, fontweight="bold", va="top")
    figure.text(0.26, 0.965, "Default mode network", fontsize=13, ha="center", va="top")
    figure.text(0.505, 0.965, "B", fontsize=16, fontweight="bold", va="top")
    figure.text(0.75, 0.965, "Anterior cingulate cluster", fontsize=13, ha="center", va="top")
    figure.text(0.015, 0.48, "C", fontsize=16, fontweight="bold", va="top")
    output = figure_root / "figure3_corrected_dmn.png"
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repository = args.repository.resolve()
    source_root = repository / SOURCE_ROOT
    figure_root = repository / FIGURE_ROOT
    figure_root.mkdir(parents=True, exist_ok=True)

    task_source = source_root / "figure1_task_schematic.png"
    behavior_source = repository / "results/reviewer/figures/acceptance_curves.png"
    for required in (task_source, behavior_source):
        if not required.is_file():
            raise FileNotFoundError(f"missing manuscript figure source: {required}")
    shutil.copyfile(task_source, figure_root / "figure1_task_schematic.png")
    shutil.copyfile(behavior_source, figure_root / "figure2_corrected_acceptance.png")
    build_figure3(repository, source_root, figure_root)
    print(f"PASS: wrote three final manuscript figures to {figure_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
