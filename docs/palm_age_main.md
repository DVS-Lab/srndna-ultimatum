# PALM full-sample means and age differences

This isolated batch adds four signed tests per retained participant contrast:
positive and negative adjusted full-sample mean, younger minus older, and older
minus younger. It uses all 47 participants (25 younger, 22 older), with no
participant exclusions, L1/L2 refits, VARCOPE substitutions or imputation.
Historical reports and active PALM jobs are never changed or signalled.

## Relation to the submitted Methods

The submitted paper describes younger/older age groups and control covariates
of gender, temporal SNR, mean framewise displacement and mean task response
time. It reports FLAME 1+2 with Z > 3.1 and whole-brain cluster-extent FWE .05
for the similar-minus-dissimilar offer-slope contrast, with a separate model
adding behavioral sensitivity-by-age regressors.

The new PALM batch retains those four controls and categorical age. It repairs
motion/tSNR values using the independently verified participant-table columns
already adopted for standardized DMN. It preserves the event-corrected group
RT column from that retained DMN design: the historical participant CSV's RT
column is **not** substituted. Sex and RT must agree across retained models
after centering, or preparation stops. Every source-to-final covariate change
is exported. Previously retained FLAME/randomise outputs remain historical
comparisons and do not silently become covariate-matched new fits.

PALM is an unequal-variance robustness analysis, not the submitted FLAME
estimator. Full-sample means and human/computer contrasts are additional
analyses. Z = 3.1 is the paper-matched primary cluster threshold; Z = 2.6 is a
sensitivity check. This batch has no TFCE, within-age simple effects, behavioral
interactions, Gu-derived parameters or vmPFC ROI analyses.

## Scope and design

There are 12 models, 24 PALM jobs and 96 signed contrast/method results:

- Activation, DMN PPI and ECN PPI similar-minus-dissimilar offer slopes under
  full-RT, pooled-post and partner-post timing (9 models; retained COPE7).
- Task-wide offer slope (COPE11), human-minus-computer offer slope (COPE8),
  and human-minus-computer task response (COPE10), using their existing
  supplemental fixed-display estimates (3 models).

Human/computer scaling is retained exactly; no silent factor-of-two change.
An offer slope indexes both offer size and fairness in this task. Full-sample
means of a similar-minus-dissimilar contrast describe that partner difference,
not a task-wide fairness response. The task-wide offer contrast supplies the
latter analysis. PPI effects index offer-modulated network coupling, not
regional activation.

Every design has six columns: an explicit all-ones mean, older-group coding
centered across the actual 47-person sample, and the four centered controls.
Contrasts test +mean, -mean, -age and +age. The intercept is the adjusted
sample-weighted mean (25/47 younger, 22/47 older), not an equally weighted
average of age groups. Behavioral columns from earlier activation/ECN
brain-behavior models are deliberately absent from these direct age/mean tests.

Coverage requires finite COPE, positive finite VARCOPE and FEAT-mask support
in both retained L1 runs and L2 for every participant. L2 FSFs are followed to
actual L1 folders; compiled L1 contrasts must agree with their FSFs. Remove
between-participant constants. Within each activation/DMN/ECN family, use a
single coverage intersection across the three timing variants, restricted to
their original whole-brain masks. The three supplemental contrasts use their
own common-participant masks. No significant map enters mask construction.
The new common timing masks can differ from older masks; therefore these jobs
are not replacements for, or repetitions of, historical completion receipts.

## Inference and execution

Use the existing clean pinned PALM commit
`9086eff76dcacf4afde12c9af5400a1fe0033b9e` and the successfully validated
headless MATLAB runtime. Its original contract/smoke/canary receipts are read
and verified without acquiring its batch lock. No MATLAB/Octave substitution
or changes to the original statistical/runtime scripts are made.

PALM uses separate age variance groups, independent symmetric-error
sign-flips (`-ise`), Freedman–Lane nuisance handling, 10,000 resamples, seed
20261101 and cluster extent at equivalent Z = 3.1 and 2.6. PALM group inference
does not use FLAME's participant VARCOPE weighting. Symmetry/independence are
assumptions, not established by the small engine smoke. FWE is spatial per
signed contrast; the model/direction/threshold family is not jointly corrected.
Threshold agreement is not independent replication.

Default scratch: `/ZPOOL/data/scratch/srndna-ultimatum-palm-age-main-v1`.
Compact outputs: `results/palm_age_main/v1`. Large merged COPEs stay in scratch.
`summary.tsv` includes null findings as well as significant findings; absent
jobs remain unfinished, not null. `significant_components.tsv` describes
connected components of thresholded corrected-p maps; its coordinates are
representative minimum-p locations, not necessarily statistic peaks.

Run `bash code/run_palm_age_main.sh --palm-root "$PALM_ROOT" --jobs 24 --execute`
in tmux on Linux1. Omit `--execute` for preparation/validation only, or replace
it with `--smoke-only` for the two small new-design MATLAB checks. Resume with
the same command. Concurrency is capped by available RAM and CPUs, leaving
headroom for the original PALM batch. A failed job stops further scheduling;
already active jobs finish and keep receipts. Each failed attempt is retained.
Source/derived input hashes and contracts are checked before completed jobs
are reused. Summaries are updated after completions and at exit.

`--collect-only` validates and exports finished jobs when this batch is idle.
Its own lock prevents racing this batch's live writer. While active, inspect
its log and `status.json` instead. All logs remain small and trackable.

PALM variance groups, sign-flips and Z conversion:
https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/palm/user_guide.html
