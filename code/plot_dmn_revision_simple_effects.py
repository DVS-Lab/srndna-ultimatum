#!/usr/bin/env python3
"""Render both directions of age-specific DMN effects, including empty panels."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/srndna-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting


def plot(root, output):
    with (root/'summary.tsv').open() as stream:
        summary = {(r['model'], int(r['contrast'])):r for r in csv.DictReader(stream, delimiter='\t')}
    panels = [(1, 'Younger: similar > dissimilar'), (5, 'Younger: dissimilar > similar'),
              (2, 'Older: similar > dissimilar'), (6, 'Older: dissimilar > similar')]
    figure, axes = plt.subplots(2, 2, figsize=(16, 8))
    for axis, (contrast, title) in zip(axes.flat, panels):
        row = summary[('simple-effects', contrast)]
        image = root/'simple-effects'/f'contrast{contrast}'/f'thresh_zstat{contrast}.nii.gz'
        data = nib.load(image).get_fdata()
        nonzero = int(np.count_nonzero(data))
        if nonzero != int(row['voxels']):
            raise ValueError(f'map/table voxel mismatch: {image}')
        if nonzero:
            plotting.plot_stat_map(image, display_mode='x', cut_coords=(-50,-22,-13,0,28),
                                   threshold=3.1, vmax=5, cmap='YlOrRd', colorbar=False,
                                   axes=axis, draw_cross=False, annotate=True)
        else:
            plotting.plot_anat(display_mode='x', cut_coords=(-50,-22,-13,0,28), axes=axis,
                               draw_cross=False, annotate=True)
            axis.text(.5, .08, 'No surviving clusters', transform=axis.transAxes,
                      ha='center', color='black', bbox={'facecolor':'white', 'edgecolor':'none'})
        axis.set_title(f"{title}\n{row['clusters']} corrected clusters", fontsize=11)
    figure.suptitle('Offer-modulated DMN coupling: age-specific simple effects', fontsize=14)
    figure.text(.5, .015, 'Whole-brain Z > 3.1; cluster-corrected p < .05 within each contrast. Matched sagittal slices.', ha='center', fontsize=9)
    figure.subplots_adjust(top=.87, bottom=.08, hspace=.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    print(f'PASS: four simple-effect panels, including nulls: {output}')


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, default=repo/'results/reviewer/dmn_revision_checks')
    p.add_argument('--output', type=Path, default=repo/'results/reviewer/figures/dmn_simple_effects.png')
    a = p.parse_args()
    plot(a.results_root, a.output)


if __name__ == '__main__':
    main()
