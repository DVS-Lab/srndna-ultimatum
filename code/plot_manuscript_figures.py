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
PARTNER_COLORS = {"similar": "#28666E", "dissimilar": "#B55239"}


def build_figure3(repository: Path, source_root: Path, figure_root: Path) -> None:
    bar_path = source_root / "figure3_dmn_flame_bar_summary.tsv"
    zstat_path = source_root / "figure3_dmn_corrected_cluster_zstat.nii.gz"
    network_path = repository / "masks/nan_rPNAS_2mm_net0003.nii.gz"
    for required in (bar_path, zstat_path, network_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing Figure 3 source: {required}")

    bars = pd.read_csv(bar_path, sep="\t")
    expected_cells = {
        ("younger", "similar"),
        ("younger", "dissimilar"),
        ("older", "similar"),
        ("older", "dissimilar"),
    }
    observed_cells = set(zip(bars["age_group"], bars["condition"]))
    if len(bars) != 4 or observed_cells != expected_cells:
        raise ValueError("Figure 3 bar table must contain the four age-by-partner cells")
    numeric_columns = [
        "flame_cluster_mean_estimate",
        "mean_voxelwise_standard_error",
        "display_conf_low",
        "display_conf_high",
    ]
    if not np.isfinite(bars[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Figure 3 bar estimates and intervals must be finite")
    if not (bars["mean_voxelwise_standard_error"] > 0).all():
        raise ValueError("Figure 3 bar standard errors must be positive")
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
        cmap="RdBu_r",
        vmax=4.0,
        symmetric_cbar=True,
        colorbar=True,
        annotate=True,
        draw_cross=False,
        axes=network_axis,
    )
    plotting.plot_stat_map(
        zstat_path,
        display_mode="x",
        cut_coords=(-13.3,),
        threshold=3.1,
        cmap="YlOrRd",
        vmin=3.1,
        vmax=4.1,
        colorbar=False,
        annotate=False,
        draw_cross=False,
        axes=cluster_axis,
    )

    age_groups = ("younger", "older")
    conditions = ("similar", "dissimilar")
    group_positions = np.arange(len(age_groups), dtype=float)
    bar_width = 0.32
    offsets = {"similar": -bar_width / 2, "dissimilar": bar_width / 2}
    for condition in conditions:
        condition_rows = (
            bars.loc[bars["condition"] == condition]
            .set_index("age_group")
            .loc[list(age_groups)]
        )
        estimates = condition_rows["flame_cluster_mean_estimate"].to_numpy(
            dtype=float
        )
        standard_errors = condition_rows[
            "mean_voxelwise_standard_error"
        ].to_numpy(dtype=float)
        roi_axis.bar(
            group_positions + offsets[condition],
            estimates,
            width=bar_width,
            color=PARTNER_COLORS[condition],
            edgecolor="white",
            linewidth=0.8,
            label=condition.capitalize(),
            zorder=2,
        )
        roi_axis.errorbar(
            group_positions + offsets[condition],
            estimates,
            yerr=standard_errors,
            fmt="none",
            ecolor="black",
            elinewidth=1.4,
            capsize=4,
            capthick=1.4,
            zorder=3,
        )

    roi_axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--", zorder=1)
    roi_axis.set_xticks(group_positions, ["Younger", "Older"])
    roi_axis.set_ylabel("Covariate-adjusted FLAME group estimate")
    roi_axis.set_xlabel("Age group")
    roi_axis.set_title("Condition estimates in the corrected cluster", pad=12)
    roi_axis.legend(
        title="Partner",
        frameon=False,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
    )
    roi_axis.spines[["top", "right"]].set_visible(False)
    roi_axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    roi_axis.set_axisbelow(True)

    figure.text(0.015, 0.965, "A", fontsize=16, fontweight="bold", va="top")
    figure.text(0.26, 0.965, "Default mode network", fontsize=13, ha="center", va="top")
    figure.text(0.505, 0.965, "B", fontsize=16, fontweight="bold", va="top")
    figure.text(0.75, 0.965, "Anterior cingulate cluster", fontsize=13, ha="center", va="top")
    roi_axis.text(
        -0.04,
        1.02,
        "C",
        transform=roi_axis.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    output = figure_root / "figure3_corrected_dmn.png"
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_figure4(source_root: Path, figure_root: Path) -> None:
    zstat_path = source_root / "figure4_offer_size_positive_zstat.nii.gz"
    if not zstat_path.is_file():
        raise FileNotFoundError(f"missing Figure 4 source: {zstat_path}")
    zstat = nib.load(zstat_path)
    data = np.asanyarray(zstat.dataobj)
    if int(np.count_nonzero(data)) != 348:
        raise ValueError("Figure 4 thresholded Z-stat image must contain 348 nonzero voxels")

    figure, axis = plt.subplots(figsize=(10.0, 3.7), constrained_layout=True)
    plotting.plot_stat_map(
        zstat_path,
        display_mode="z",
        cut_coords=(-14, -5, 5),
        threshold=3.1,
        cmap="YlOrRd",
        vmin=3.1,
        vmax=4.8,
        colorbar=True,
        annotate=True,
        draw_cross=False,
        axes=axis,
    )
    axis.set_title(
        "Positive task-wide offer-size modulation",
        fontsize=13,
        pad=12,
    )
    output = figure_root / "figure4_offer_size_activation.png"
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
    build_figure4(source_root, figure_root)
    print(f"PASS: wrote four final manuscript figures to {figure_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
