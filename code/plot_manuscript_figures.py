#!/usr/bin/env python3
"""Build the final manuscript figure set from tracked source data."""

from __future__ import annotations

import argparse
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


def build_figure3(repository: Path, source_root: Path, figure_root: Path) -> None:
    table_path = source_root / "figure3_dmn_corrected_roi.tsv"
    zstat_path = source_root / "figure3_dmn_corrected_cluster_zstat.nii.gz"
    network_path = repository / "masks/nan_rPNAS_2mm_net0003.nii.gz"
    for required in (table_path, zstat_path, network_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing Figure 3 source: {required}")

    data = pd.read_csv(table_path, sep="\t")
    if len(data) != 47 or set(data["age_group"]) != {"younger", "older"}:
        raise ValueError("Figure 3 source table must contain both groups and 47 rows")
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
        display_mode="z",
        cut_coords=[-6, 18, 42],
        threshold=0.5,
        cmap="Greens",
        colorbar=False,
        annotate=False,
        draw_cross=False,
        axes=network_axis,
        title="Default mode network",
    )
    plotting.plot_stat_map(
        zstat_path,
        display_mode="ortho",
        cut_coords=(-13.3, 34.0, 27.8),
        threshold=3.1,
        cmap="YlOrRd",
        colorbar=True,
        annotate=True,
        draw_cross=False,
        axes=cluster_axis,
        title="Anterior cingulate cluster",
    )

    random = np.random.default_rng(20260910)
    for x_position, group in enumerate(("younger", "older")):
        group_values = data.loc[
            data["age_group"] == group, "similar_minus_dissimilar"
        ].to_numpy(dtype=float)
        jitter = random.uniform(-0.085, 0.085, len(group_values))
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
    roi_axis.set_xticks([0, 1], ["Younger", "Older"])
    roi_axis.set_ylabel("DMN connectivity\n(similar − dissimilar offer modulation)")
    roi_axis.set_xlabel("Age group")
    roi_axis.spines[["top", "right"]].set_visible(False)
    roi_axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    roi_axis.set_axisbelow(True)

    figure.text(0.01, 0.98, "A", fontsize=16, fontweight="bold", va="top")
    figure.text(0.505, 0.98, "B", fontsize=16, fontweight="bold", va="top")
    figure.text(0.01, 0.47, "C", fontsize=16, fontweight="bold", va="top")
    figure.suptitle(
        "Age difference in task-dependent DMN connectivity",
        fontsize=15,
        fontweight="bold",
    )
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
