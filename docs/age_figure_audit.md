# Age figures and anatomical/influence audit

`bash code/run_age_figure_audit.sh` creates a separate, timestamped report in
`results/age_figure_audit/v1`. It reads existing estimates and maps; it does not
run FEAT, PALM, randomise, modify their scratch directories, or exclude anyone.
Earlier figures and extraction reports remain unchanged. Independent phases
continue if another phase fails; the overall exit status and per-phase tables
record the failure. Push the logs as well as successful figures.

## DMN three-partner estimates

The selected region is the 49-voxel PALM cluster-Z2.6 age result from the strict
coverage check. It must equal the original 51 voxels intersected with valid
coverage. The randomise cluster-t3.1 region also has 49 voxels, but **46 overlap**
with PALM (52 in the union); it is not an identical region. Map checksums and
all participant COPE7 checksums are verified before extraction.

Native L2 computer/similar/dissimilar estimates and common-COPE7-run-weight
estimates are kept separate. The latter are released only if the reconstructed
similar-minus-dissimilar estimate matches retained L2 COPE7 for every participant.
Both show age-only, no-group-FD and full-covariate bars. Error bars describe
participant SEM conditional on fitted nuisance slopes, not coefficient SE or
independent ROI inference. The computer condition is contextual, not in COPE7.

## Supplemental effects and full-sample foundation

Separate figures and participant tables cover all significant voxels in five
signed randomise cluster-t3.1 results: positive task-wide offer response,
human-greater-than-computer response, computer-greater-than-human response,
younger-greater-than-older task-wide offer slope, and younger-greater-than-older
human-minus-computer response. Full-sample and age-split bars use the original
six-column design. The negative response contrast is sign-flipped for its named
display, with the direction recorded. These are same-data, selected-region
descriptions, not independent effect estimates. The corrected p-values belong
to the existing whole-brain tests, not to the bar plots. These supplemental
fixed-display models are distinct from the three main timing-model families.

Full-sample cluster-t3.1 spatial-FWE results are: offer-positive p=.0364 (95
voxels), human-greater-than-computer p=.0016 (633), and the reverse p=.0150
(262). No negative offer main effect survives. Anatomy and visual/attention
confounding must be considered before a fairness-specific interpretation.

## Parietal influence

Preserve the existing scatterplot extraction region: **125 voxels**, the
intersection of randomise cluster-t3.1 maps across all three timings. The earlier
61-voxel summary additionally required FLAME overlap; it is not the region used
to extract the displayed scatterplots. This audit does not switch regions.

For each timing model, export every leave-one-out and every leave-two-out fit
(47 and 1,081 combinations), refitting all retained nuisance terms. Record rank
loss, contrast changes, sign reversals, leverage, Cook's distance, studentized
residuals and Huber weights. No deletion defines a replacement primary analysis.
Compare original full-design OLS with fixed-scale Huber IRLS (tuning 1.345) and
within-group Theil–Sen slopes on OLS-nuisance-adjusted outcomes. Huber does not
solve predictor leverage; Theil–Sen retains an OLS nuisance adjustment. No new
selected-ROI p-values are computed.

Slope bars show the younger/older behavioral-regression coefficients. Their OLS
error bars are HC3 coefficient SE, **not participant SEM**; robust coefficients
have no uncertainty bars. Three-partner activation bars separately show native
condition means with conditional participant SEM. Those means do not represent
the continuous brain–behavior interaction, and their native L2 run weights differ.
Original source COPEs, native contrasts, VARCOPEs and FEAT-mask support are
recorded. Coverage flags are saved without trimming the region or dropping people.
Stable deletion signs alone do not establish stable corrected whole-brain p-values.

## Anatomical localization

Requires the existing FSL Harvard-Oxford cortical and subcortical probability
atlases and their XML dictionaries. Supply `--fsl-root` and `--standard` if the
installation is elsewhere. No atlas is downloaded automatically.

Linear probability sampling at native statistical voxel centres preserves each
atlas's original 0–100 percentages. Summaries include every significant voxel,
every connected component, cortical/subcortical labels, and gray-matter,
white-matter, ventricular and brainstem atlas probabilities. Probability means
and relative dominant labels are both reported; probabilities are not forced to
sum to 100. Zero atlas support is unassigned. The ventricular labels do not
represent all CSF. These are template atlas descriptions, not participant tissue
fractions or proof of accurate registration. No gray-matter restriction is added.

DMN and parietal selected regions receive the same atlas audit. Overlays display
the largest-component centroid, with nearest-neighbour mask resampling only.
The compact source tables and provenance accompany every figure.
