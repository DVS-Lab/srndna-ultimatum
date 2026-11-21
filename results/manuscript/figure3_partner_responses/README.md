# Participant responses for a focused Figure 3

This analysis addresses two questions about the full offer display. Does the bilateral anterior insula respond during the display relative to the implicit baseline? Does its response vary linearly with the offered amount? The region is the existing 38-voxel bilateral mask in `masks/roi-aINS.nii.gz`. Its centers come from the Feng et al. unfair-versus-fair meta-analysis listed in `masks/README.md`. This exploratory ROI analysis uses the fixed mask regardless of the estimates or their direction. The previously reported right-insula FLAME cluster was defined separately.

For each of the 47 participants, the extraction reads the two-run fixed-effects estimates for computer, age-similar human and age-dissimilar human displays (COPEs 1, 3 and 5) and their corresponding offer slopes (COPEs 2, 4 and 6). The two regional tests use the equal-weight average of the three display estimates and the equal-weight average of the three slopes, respectively. The six-column model uses the retained participant order and centered design. Each intercept has a two-sided HC3 test and a 95% confidence interval. Holm adjustment covers these two exploratory tests. The display estimate includes the full modeled trial (offer, decision and brief postresponse period). Offer amount covaries with both the responder’s payoff and split equity.

A second two-panel figure describes the already reported sample-wide human > computer display contrast (PALM, cluster Z > 3.1, spatial FWE p = .0002). Its mask includes all 270 selected voxels. The three condition estimates use the voxelwise inverse-variance weights for COPE10 and must reconstruct the retained run and participant contrast. The section shows the selected mask at MNI z = −10 mm (nearest 2-mm template slice to the native peak). The bars describe the sample that defined the mask. Inference comes from the original PALM map.

Each figure contains one axial section and three partner bars pooled across the 47 participants. Each bar uses one value per participant, adjusted for sex, tSNR, mean displacement and mean reaction time, retaining age in the model. Error bars show the sample SD of those adjusted values divided by the square root of 47. They describe individual condition means. The additional three insula offer slopes are included in the accompanying table. The complete results for both masks are retained regardless of the two ROI tests.

From the current main branch on Linux1, run the following command in the repository.

```bash
bash code/run_figure3_responses.sh
```

The runner validates the two-run roster, coverage, mask hashes, design, contrast coding and COPE10 reconstruction. It stages and pushes the two aggregate tables, validation record, and both figures in PDF and PNG to main. Participant-level images and statistics stay on Linux1. The article figure and its caption can then be selected by the authors. To render from an existing aggregate table, use `code/plot_figure3_responses.py --source TABLE --template MNI152_T1_2mm_brain.nii.gz --output DIRECTORY`.
