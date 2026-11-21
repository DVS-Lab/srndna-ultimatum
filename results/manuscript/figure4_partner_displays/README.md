# Figure 4. Posterior midline responses to partner displays

Panel A shows the direct younger-minus-older comparison of the full-display human-minus-computer contrast in the 47-participant PALM model. At Z > 3.1, the selected posterior midline cluster contains 25 voxels (spatial FWE p = .0163). The sagittal view passes through MNI x = −2 mm. This figure accompanies the revised Neuropsychologia manuscript. The earlier DMN figure is a separate historical analysis.

Panel B summarizes those same 25 voxels. The six rows of `condition_estimates.tsv` are the unadjusted group means and between-participant standard errors for computer, age-similar human and age-dissimilar human displays. Linux1 validated 47 participants and 94 runs. The maximum absolute difference between the reconstructed and retained participant contrast was 2.7725702e-05 model units. The aggregate output was transcribed at full printed precision. The source analysis and checksums are recorded in `linux1_validation.json`.

For each run the extractor reads computer (COPE1), similar (COPE3) and dissimilar (COPE5) display estimates. The FEAT contrast is similar + dissimilar − 2 × computer. For each participant, the extractor applies the voxelwise inverse-VARCOPE10 weights from both runs to each condition. It verifies the contrast before and after combining runs, averages each condition across the selected voxels, and computes each group's mean and SEM. The descriptive contrasts in the manuscript use the average of the two human conditions minus computer (one half of COPE10), which gives 14.45 model units in younger adults and −5.63 in older adults. The inferential age comparison comes from the original covariate-adjusted whole-brain PALM analysis. Condition-wise SEMs do not give an SEM for a within-person contrast.

## Reproduce the estimates on Linux1

Run from the repository root in the project Python environment. The extraction reads the retained FEAT inputs under `/ZPOOL/data/scratch` and writes to a fresh directory.

```bash
run_dir=$(mktemp -d)
python3 code/extract_figure4_conditions.py --repo "$PWD" --output "$run_dir/condition_estimates.tsv"
```

The Linux1 writer uses the CSV module's default TSV line endings. The committed TSV uses LF. Compare numeric fields or normalize line endings when verifying the output.

## Regenerate the publication figure

Run from the repository root with NumPy, SciPy, NiBabel, Matplotlib and the FSL 2-mm brain template available. The plotting entry point rebuilds Figures 2–4 from retained aggregate results, including the six Figure 4 estimates committed here.

```bash
plot_dir=$(mktemp -d)
python3 code/plot_neuropsychologia_figures.py "$PWD" "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" "$plot_dir"
```

The complete panel is written to `Figure4.pdf` and `Figure4.png`. To use a fresh Linux1 extraction, add `"$run_dir/condition_estimates.tsv"` as the last argument.
