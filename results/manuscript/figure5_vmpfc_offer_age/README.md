# Offer slopes in vmPFC and the whole-brain age comparison

Panel A shows the direct younger-minus-older whole-brain PALM contrast for the mean offer slope across all three partners in the full-display model. At the primary cluster-forming threshold (Z > 3.1) the minimum spatial FWE p was .1017 and no voxels survived. At Z > 2.6, two clusters survived. Medial frontal cortex had 81 voxels (cluster p = .0352), and left precuneus had 68 (p = .0495). No clusters survived the reverse direction at either threshold. The TSV gives native-grid peak MNI coordinates and the overlap with the fixed vmPFC mask. Cluster labels were checked with the Harvard–Oxford atlas. Each signed map had its own spatial correction.

Panel B uses the 19-voxel fixed mask at MNI [6, 46, −12] from the Feng, Luo and Krueger (2015) fair > unfair meta-analysis. The full-display estimates were extracted independently of the PALM threshold from the original vmPFC ROI analysis. Ten of the 19 ROI voxels overlapped the medial frontal sensitivity cluster. The two analyses use the same 47 participants. The ROI means retain age and adjust for sex, temporal signal-to-noise ratio, mean framewise displacement and mean reaction time. Each error bar gives one conditional participant SEM. The full-display older-minus-younger contrast averaged across partners was −4.46 model units (HC3 95% CI [−7.66, −1.25], nominal p = .0076, Holm p = .0335 across six timing/adjustment specifications). The age-by-partner omnibus had nominal p = .768 and Holm p = 1.00.

The saved ROI inputs, condition estimates, tests and correction families are under `results/vmpfc_ratings/v1/vmpfc/`. The signed statistic and cluster-probability maps are under `results/palm_age_main/v1/maps/taskwide-offer/`. The publication entry point exports the validated Figure 3 and builds Figures 2, 4 and 5.

```bash
python code/plot_neuropsychologia_figures.py "$PWD" "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" /tmp/srndna-publication-plots
```

Original meta-analysis: https://doi.org/10.1002/hbm.22649
