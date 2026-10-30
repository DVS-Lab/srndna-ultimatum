# Human/computer and parietal coverage sensitivity

Run `bash code/run_activation_coverage.sh --jobs 8` on Linux1. Scratch is
`/ZPOOL/data/scratch/srndna-ultimatum-activation-coverage-v1`; compact outputs,
including unsuccessful attempts, go to `results/activation_coverage/v1`.
The initial dry run audits/prepares inputs but launches no statistical jobs.
Repeating the command verifies receipts and skips completed jobs; interrupted
attempts stay intact and retry in a new directory. The work root has its own
lock and does not touch the original PALM batch or previous analysis outputs.

## Fixed scope

The human–computer task-response model retains all four original contrasts.
Its input contrast keeps the original `(similar + dissimilar) - 2 * computer`
weights; it is not rescaled to an average-human-minus-computer contrast.
The parietal activation-versus-acceptance-intercept models retain all eight
contrasts under full-RT, pooled-post and partner-post timing. The 47 participants,
subject ordering, designs, contrasts, covariates and input estimates are fixed.
No participants, runs or signal values are replaced or imputed.

Coverage is determined over the whole original search mask before looking at
selected regions. Require finite COPE, finite positive VARCOPE and finite
positive FEAT-mask support in both L1 runs and L2 for every participant. Follow
the retained L2 FSF declarations to the actual L1 directories; do not guess
paths. Compiled L1 contrasts must match the FSFs, including zero external
nuisance weights. For parietal partner bars, also check COPEs 2/4/6 as well as
the tested COPE7. Use one validity intersection across all three timing models.
Remove between-participant constants; never expand the original search mask.

## Inference and calibration

Randomise uses the original signed contrasts, seeds, 10,000 resamples and -R -N
options. Rerun cluster t=2.6 and t=3.1 only; no new TFCE or PALM jobs. There are
56 randomise jobs across 28 signed contrasts, retaining null results too.

The FLAME step is **spatial cluster re-inference of the retained Z maps**, not a
FLAME coefficient refit. Use FSL smoothest on the Z field and the explicit
testing mask, followed by FSL fsl-cluster at Z=3.1, spatial-FWE p<.05. The
probability/search-volume calculation uses the native mask voxel count.
Recompute both original-mask and coverage-mask versions with this identical
procedure (56 short jobs). `flame_original_mask_calibration.tsv` compares the
original-mask result with the historical significant map. A mismatch is
reported, not hidden: differences in historical smoothness estimation or
rounding must then be distinguished from coverage effects. Compare the paired
new original-mask/coverage-mask runs for the isolated mask effect. Do not claim
that the underlying FLAME coefficients or variance model were re-estimated.

This follows the documented FSL cluster/smoothness interface:
https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/cluster.html

`summary.tsv` includes every completed signed test; absent jobs remain pending,
not null. The 112 total inference jobs are checksum-verified on resume. Software,
input sources, mask policies and seeds are frozen in a separate work contract.
Large merged input images remain in scratch. Only compact maps, designs,
coverage counts, logs, null distributions and source tables are collected.

## Descriptive figures

Use the previous human>computer region and the existing 125-voxel parietal
scatterplot region intersected with the independently derived validity masks.
These are not newly significant region selections. Export pooled/age-split
human–computer bars, parietal slope bars, native three-partner bars, robust
scatterplots and every one-/two-participant deletion. Participant means use
conditional participant SEM; OLS coefficient bars use HC3 coefficient SE.
These selected-region summaries supply no independent inferential p-values.
All primary tests remain whole-brain tests with spatial correction per signed
contrast, not correction across the model/method search.
