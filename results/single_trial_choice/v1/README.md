# Does network coupling improve prediction of offer acceptance?

This exploratory follow-up tests whether network coupling explains choices beyond offer characteristics and trial activation. All model specifications below were fixed before examining the prediction scores. The cohort and the earlier imaging findings were known when the analysis was designed.

## Inputs and measurements

The outcome is acceptance of each responded human offer. In the 47-participant manuscript cohort, each of two runs has 72 scheduled offers. Across both runs, 4,438 human offers have a recorded choice. The corrected BIDS event files identify choice, offer size, proposer and block. The extractor rebuilds the event lists used for least-squares-separate (LSS) estimation, checks trial order including missed and computer trials, and then selects the human choices. Each image and event file is verified against [OpenNeuro ds003745 version 2.2.0](https://doi.org/10.18112/openneuro.ds003745.v2.2.0), including the two repaired runs for sub-144. One released image, sub-104 run 2, has 55 volumes for 72 scheduled offers. The primary comparison keeps all 47 participants and uses the 93 complete runs (4,382 choices). The other run’s 56 responded human offers are excluded from every primary block. The corrected run-level gPPI estimate from that run remains available to predict choices in sub-104 run 1. The accompanying manifest gives the released git-annex image checksums and the Git blob IDs for the event files.

The LSS volumes are trial-level **z statistics**. Local activation is their mean within a fixed mask. For network activation, each trial's voxel values are spatially regressed on the signed continuous network map with an intercept. This gives a trial-level measure of network expression. It differs from the run-level generalized psychophysiological interaction (gPPI) estimate. The analysis includes the fixed literature vmPFC and anterior-insula regions and the relevant network and target. The target and common coverage masks use the retained native grid. Network maps are interpolated linearly and binary masks by nearest neighbor as needed. Each run's LSS features use voxels with finite support across its 72 trials. The extraction report records the minimum supported fraction for each feature.

The connectivity predictor is mean **COPE7**, similar-minus-dissimilar modulation of coupling by offer amount, from the corrected full-display gPPI fit in the participant's **other run**. Run 1 supplies coupling for choices in run 2 and vice versa. Each of the 188 retained run images (two networks × 94 runs) is matched to its original SHA-256 checksum. Within a run, the connectivity value is identical for all choices. The run-to-run correlation accompanies the predictions because a noisy connectivity measurement can limit incremental prediction.

## Nested comparisons

1. The behavioral baseline includes offer amount × partner similarity × age group, common quadratic and cubic offer terms, block fairness, run, trial position, sex, temporal SNR and framewise displacement. The logistic mixed model has a participant-specific intercept and offer slope. Current-trial response time and scores computed from the participant's choices do not enter the predictor set.
2. The activation block adds each trial's deviation from its run mean and the run mean itself. In the primary analysis, run means use all 72 scheduled offers before trials are selected on partner or response. This separates changes from trial to trial from differences in average activation. The principal activation comparison uses vmPFC, anterior insula, DMN expression and right frontal pole.
3. The connectivity block adds the opposite-run coupling estimate, its interactions with offer amount and partner similarity, and its three-way coupling × offer × similarity interaction. The last term directly asks whether coupling moderates the partner difference in behavioral offer sensitivity. The four-term block can also detect a gain through the lower-order terms. Age moderation of this neural association is outside the primary test.

The **primary comparison** uses the atlas-defined right frontal pole for DMN coupling. The choice of region was informed by the earlier DMN result. Its voxel boundary comes from the Harvard–Oxford cortical maximum-probability atlas (25%, 2 mm), label 1, nearest-neighbor resampled onto the native DMN grid, restricted to MNI x > 0 and the original coverage mask (2,018 voxels). The source atlas SHA-256 is 7397ffdbae7559e0f0aa6237f998dc0f6aca3f9db9009f6ffa637c0c62b686ca. A **descriptive sensitivity comparison** uses the original 49-voxel lower-threshold DMN age cluster. Its location was selected using the full sample, including participants later held out of the choice regression. A second exploratory model uses ECN coupling to the literature-defined vmPFC, with ECN expression, vmPFC and anterior-insula activation in the matching baseline. This explores the earlier ECN question using a target defined independently of the association with behavior.

### Recovery of the partial public image

When numbered Linux1 LSS files and their EVs are available for sub-104 run 2, the extractor checks trial onset, duration and grid and requires unique, exact matches to all 55 published volumes in the correct order. If all 72 numbered volumes are present and verified by trial EV, it restores the complete series. If only a subset can be matched, it uses just the 55 exact matches. A separate DMN atlas sensitivity comparison then includes the recovered trials, with run means computed across available verified volumes. If the numbered sources cannot establish alignment, the primary 93-run analysis still completes, and the validation record states how many trials could be recovered. SRNDNA_LSS_ROOT can specify an additional retained LSS work root containing sub-104 and EVfiles.

## Evaluation

Five age-stratified folds hold out both runs and every choice of each test participant. The same folds and trials are used throughout. Regression scaling is learned within each training fold. Predictions integrate over the fitted distribution of random intercepts and offer slopes for a new participant by integrating the normal linear predictor, checking convergence of each probability to 10⁻⁸. The primary endpoint is the change in held-out log loss, averaged first over trials within each participant and then across participants. **Positive reduction means better predictions.** Brier-score change and the number of participants whose log loss improves provide additional descriptions. The paired, age-stratified participant bootstrap interval resamples the 47 sets of held-out scores conditional on the fitted folds. It does not include uncertainty from refitting the entire pipeline. The full-sample nested likelihood-ratio tests and focal coefficients describe model fit and direction; exploratory comparisons do not provide separate confirmatory tests.

A gain in the primary comparison would show that coupling measured in a separate run carries information about choices beyond task variables and contemporaneous activation. The same-trial image spans the offer/response period, and the design uses both runs retrospectively. The result cannot establish a pre-choice forecast, momentary trial-by-trial coupling, or a causal mechanism. The smaller independent sample for generalization is 47 people.

## Activation results from the public images

The [observed held-out scores](activation_only/heldout_scores.tsv) use 4,382 choices from all 47 participants. Mean held-out log loss was 0.60511 for the behavioral model and 0.60865 after adding the activation block. The paired reduction was −0.00354, with a conditional 95% bootstrap interval of [−0.01538, 0.00960]. Brier scores were 0.20672 and 0.20886. Fourteen of 47 participants had lower log loss with activation. The [full-sample likelihood-ratio comparison](activation_only/nested_tests.tsv) was χ²(8) = 6.27, p = .617. These estimates address the activation block. The connectivity comparisons require the run-level FEAT images on Linux1.

## Run on Linux1

From the main branch of DVS-Lab/srndna-ultimatum run:

```bash
git pull --ff-only origin main
bash code/run_single_trial_choice.sh
```

The defaults are /ZPOOL/data/datasets/ds003745-work for the released images and /ZPOOL/data/scratch for the corrected FEAT images. SRNDNA_DATASET_ROOT and SRNDNA_SCRATCH can override them. The script uses Python with nibabel, NumPy and SciPy, and R with lme4. It writes the trial-level working table to a private temporary directory. Only the aggregate validation, prediction scores, model diagnostics, coefficients, nested tests and software versions are committed and pushed to main. A public-data-only run can use the two underlying scripts with --activation-only. The full integration is intended for Linux1.

The design extends the incremental block comparison used by [Smith et al. (2014)](https://doi.org/10.1093/scan/nsu005) to a binary choice, participant-level validation, and a measured connectivity contrast from a separate run.
