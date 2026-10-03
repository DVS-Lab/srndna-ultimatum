# Hierarchical norm-learning analysis

This exploratory behavioral analysis uses the corrected 47-participant sample
and the 48 observed offers per partner. It does not change any imaging model or
release imaging covariates. Independent bounded-MLE screening is implemented in
`code/run_gu_norm_learning.py`; the hierarchical analysis is a separate estimator.

## Models and estimands

`code/stan/gu_hierarchical.stan` implements four models:

| Name | Choice model | Parameters per participant/partner |
| --- | --- | --- |
| `rw_free` | Gu-style norm updating; updated norm enters current utility | alpha, gamma, initial norm, epsilon |
| `fs_free` | Static, freely estimated norm | alpha, gamma, norm |
| `logistic` | Acceptance depends on offer amount | intercept at $5, offer slope |
| `rw_free_prior` | Sensitivity: pre-update norm enters current utility | alpha, gamma, initial norm, epsilon |

For learning models, `norm_after = norm_before + epsilon*(offer - norm_before)`.
Norm-based acceptance has log odds
`gamma*(offer - alpha*max(norm_at_choice - offer, 0))`.
Dollar units, the $20 stake, and bounds alpha/gamma/epsilon in (0,1) and norm in
(0,20) preserve the independent-MLE model's support apart from excluded exact
endpoints. The printed Gu update convention is the primary specification;
the pre-update convention is explicitly a sensitivity analysis.
See [Gu et al. (2015)](https://doi.org/10.1523/JNEUROSCI.2906-14.2015).

Each participant has a separate norm trajectory for each partner. Norms carry
across blocks and runs. A missed response contributes no choice likelihood, but
the displayed offer still updates its partner's norm. That is an observation
assumption, not proof that every missed-trial offer was attended. No partner's
offers update another partner's state. The task events and sub-143/sub-144
duplication guard are shared with the independent-MLE implementation.

The non-centered hierarchy operates on latent parameter scales:

```
latent[s,p,k] = mu[k] + subject_deviation[s,k]
             + human_computer[p] * (effect[1,k] + subject_contrast[s,1,k])
             + similar_dissimilar[p] * (effect[2,k] + subject_contrast[s,2,k])
human_computer   = [-2/3, 1/3, 1/3]
similar_dissimilar = [0, 1/2, -1/2]
partner order: computer, similar, dissimilar
```

Thus the coefficients are human-minus-computer and similar-minus-dissimilar on
the latent scale. Subject and contrast deviations have independent Gaussian
distributions with estimated scales; this deliberately avoids a large covariance
matrix. The parameters are transformed with inverse-logit (initial norm times
20). Logistic comparator parameters use `30*tanh(latent/30)` and
`10*tanh(latent/10)`, preserving the earlier bounds but behaving approximately
linearly near zero.

Norm-model latent means have Normal(0,1.5) priors, partner effects Normal(0,.75),
participant SDs half-Normal(0,.75), and participant-contrast SDs
half-Normal(0,.5). For the logistic intercept/slope respectively, these scales are
(2,1), (1,.5), (1,.5), and (1,.5). `--prior-scales` multiplies all these scales;
it does not alter support. Prior predictive summaries must be inspected, not
treated as automatically plausible. The default hierarchy is age-blind. The
optional age-group extension below adds age to the hierarchy, never brain data.

Posterior partner contrasts are calculated after transformation to the natural
parameter scale, paired within participant. `sample_mean` is the average of
these paired differences across the fitted sample, not a population-marginal
effect. The primary contrast is similar-minus-dissimilar; human-minus-computer
is secondary. Summaries use 95% posterior credible intervals.

## Validation and prediction

Each model is fitted to all choices (`full`) and to run 1 only (`run1`). Run-1
fits forecast run-2 choices while updating on the offers actually presented;
they do not condition on run-2 choices. Outputs include both summed trialwise
posterior-predictive log densities and joint run log densities per
participant/partner. These are distinct scores. This is held-out prediction for
the same participants, not generalization to unseen participants. Ordinary
trialwise LOO is not used to assess this sequential learning process.

The sampler diagnostic screen requires no divergences or maximum-tree-depth
hits, all monitored R-hat <=1.01, bulk/tail ESS >=400, and per-chain E-BFMI >=.3.
Generated quantities are excluded from that screen; all free parameters and
individual transformed parameters are included. A completed fit can fail this
screen. Flags produce exit code 2, and `inference_eligible` remains false.
Short smoke runs always remain ineligible, regardless of their diagnostics.

Posterior predictive outputs compare acceptance by partner, run, and offer.
Recovery simulations generate independent datasets from the specified hierarchy
on the actual offers with the actual missing-response pattern. They refit the
generating model and summarize individual-parameter and paired-contrast recovery
and interval coverage. Ten replications are an initial recovery audit, not an
exhaustive calibration study, model-selection recovery, or completed SBC test.
Prior sensitivity and recovery must be reviewed before using individual
parameters in imaging; shrinkage alone does not establish identifiability.

## Linux1 setup and execution

### Longer fits and optional age-group extension

`bash code/run_gu_followup.sh baseline` runs the same four scientific models,
each on full data and run 1, with 3,000 warmup and 6,000 retained iterations per
chain. Four chains and two threads per chain allow five simultaneous fits within
the 40-CPU budget. All eight models are refitted with matched settings, including
the previously passing full-data logistic comparator. Existing results under
`stan-v1` and scratch `srndna-gu-stan-v1` remain unchanged. Longer chains are a
diagnostic experiment, not proof that the learning-rate regions mix adequately.
Do not drop inconvenient chains or retrospectively trim draws to obtain a pass.

The optional `--age-mode group` adds **one older-minus-younger coefficient per
latent parameter**, common to the three partners. The centered indicator is
computed once per unique participant, not once per trial or partner:

```
age_x[s] = I(older[s]) - mean(I(older))
latent[s,p,k] = baseline_latent[s,p,k] + age_x[s] * age_beta[k]
age_beta[k] ~ Normal(0, 0.5 * prior_scale)
```

In the full 47-person sample the indicator is -22/47 for the 25 younger adults
and 25/47 for the 22 older adults. Recorded ages are 20–34 and 63–80, respectively
(`derivatives/imaging_plots/participants.tsv`, joined by ID to the fixed sample).
The actual fitted indicator uses the existing, source-hashed sample group
assignment. It does not fit a continuous lifespan slope across the unsampled
middle ages. The age prior is configurable with `--age-prior-sd`; 0.5 is an
explicit exploratory regularization choice, not a prior learned from imaging.
No age-by-partner interactions are fitted. Transformations can nevertheless make
natural-scale older-minus-younger differences differ across partner baselines.

The age model preserves offers, choices, likelihood masks, partner histories,
and parameter bounds. All age coefficients enter the convergence screen.
`age_effects.tsv` reports older-minus-younger effects on the **latent** scale,
with posterior intervals, MCSE, R-hat, and ESS. `parameters.tsv` contains
age-conditional individual estimates, **not age-residualized parameters**.
Age effects are incorporated into prior prediction. Age-effect recovery is not
implemented; `--phase recovery --age-mode group` therefore fails explicitly.
Baseline parameter recovery must not be presented as validation of age effects.

Use a tmux session and run these as separate steps:

```bash
cd /ZPOOL/data/projects/srndna-ultimatum
git pull --ff-only
conda activate srndna-ultimatum
bash code/run_gu_followup.sh smoke-age
bash code/run_gu_followup.sh baseline
```

The smoke run checks all eight model/stage combinations on four participants,
not scientific convergence. Inspect the longer baseline diagnostics before
launching the age extension:

```bash
bash code/run_gu_followup.sh age
```

The wrapper automatically exports tables, nine diagnostic PNGs, and persistent
logs even when sampling finishes with diagnostic flags (exit 2). Execution or
export errors stop it. It never commits automatically. Run only one wrapper at
a time to respect the shared 40-CPU budget. Each command can be repeated to skip
verified completed fits. Changes to code or settings require a new work root.
Updated source hashes mean the old runner must not be used to resume `stan-v1`;
its completed outputs remain valid historical records and are still readable by
the standalone diagnostic exporter.

| Command | Scratch directory under `/ZPOOL/data/scratch` | Tracked output under `results/norm_learning` |
| --- | --- | --- |
| `smoke-age` | `srndna-gu-stan-agegroup-smoke-v1` | `stan-agegroup-smoke-v1` |
| `baseline` | `srndna-gu-stan-long-v2` | `stan-long-v2` |
| `age` | `srndna-gu-stan-agegroup-v1` | `stan-agegroup-v1` |

After each completed command, add only its output directory, commit, and push:

```bash
git add results/norm_learning/stan-long-v2 &&
git commit -m "Add longer baseline norm-model fits and diagnostics" &&
git push origin main
```

For the age run, use `results/norm_learning/stan-agegroup-v1` instead. Raw chains,
compiler products, and interrupted attempts remain in scratch.

#### Implications for imaging

Neither extension exports imaging covariates or modifies any FEAT model.
At the trial level, participant/partner parameters may generate expected norms
or prediction errors. Using behavior and age without brain data is not selection
on an imaging outcome, but age-informed shrinkage can change regressor shape and
scale differently across groups. Compare age-blind and age-informed trajectories,
check their relationship to existing offer/task regressors, and assess parameter
and regressor uncertainty before fitting imaging models. Plug-in posterior means
do not propagate behavioral uncertainty through FSL.

At the group level, a parameter estimated using age still contains age-related
information. Its association with an age-related imaging outcome is not evidence
of a link independent of age. A model containing age and that parameter asks for
a conditional association; the age coefficient then differs in meaning from a
model without the parameter. Check collinearity and retain an age-blind behavioral
estimator as a sensitivity comparison. Do not automatically residualize parameter
means, describe them as independent age evidence, or infer mediation. Joint
behavior–imaging modeling or a justified uncertainty-propagation analysis would
be separate work. See [Katahira and Toyama (2021)](https://doi.org/10.1371/journal.pcbi.1008738)
on parameter estimation for model-based fMRI.

### Validation before model-based imaging

`code/run_gu_validation.sh` adds logged, resumable launches without changing
the fitting code, Stan model, or completed baseline/age fits. It defaults to a
dry run and uses the matched longer-chain settings: four chains, 3,000 warmup,
6,000 retained draws, two threads per chain, and a 40-CPU ceiling.

Run each batch separately inside tmux; do not overlap the CPU budgets:

```bash
bash code/run_gu_validation.sh priors &&
bash code/run_gu_validation.sh priors --execute
```

This fits the age-blind primary `rw_free` model at prior scales 0.5, 1, and 2,
each on full data and run 1 (six fits). Scaling changes all hierarchical prior
scales, not parameter support. It is not an isolated sensitivity test of the
age coefficient's prior.

```bash
bash code/run_gu_validation.sh recovery &&
bash code/run_gu_validation.sh recovery --execute
```

This refits ten independent prior-generated full-sample datasets under the
same age-blind primary model. It reports individual-parameter and paired-partner
contrast recovery and interval coverage. This is an initial audit, not age-effect
recovery, model-selection recovery, or proof of empirical identifiability. If
the observed posterior occupies a difficult region poorly represented in these
simulations, targeted recovery in that region is also required. In particular,
inspect high alpha and low learning-rate values rather than judging recovery
only by a single correlation across the entire prior range.

| Batch | Scratch directory under `/ZPOOL/data/scratch` | Output under `results/norm_learning` |
| --- | --- | --- |
| priors | `srndna-gu-stan-priors-long-v1` | `stan-priors-long-v1` |
| recovery | `srndna-gu-stan-recovery-long-v1` | `stan-recovery-long-v1` |

Both batches export diagnostic figures after completed sampling, including
flagged fits. `preflight.txt`, `run.txt`, and `export.txt` are intentionally
trackable logs; raw chains remain in scratch. Exit 2 means diagnostic review is
needed, not that outputs should be deleted. Execution/export failures stop the
wrapper. No imaging jobs or covariate files are produced.

### Proposed imaging extensions (not implemented or run)

These extensions remain exploratory. Successful behavioral convergence alone
does not establish that a norm-learning account outperforms the offer-based
comparator, that a null age contrast is equivalence, or that individual parameters
are suitable imaging covariates.

**Trial-level signals.** For a particular participant/partner, define the
pre-offer expectation as `E_t = norm_before_t`, the signed norm prediction error
as `PE_t = offer_t - E_t`, and the updated norm as
`norm_after_t = E_t + epsilon * PE_t`. Positive PE denotes a more generous offer
than expected, not a received reward. Preserve the fitted model's separate
partner histories and carryover across runs. The primary behavioral model uses
the updated norm in choice utility; that must not silently redefine the pre-offer
expectation or PE. Its one-sided choice shortfall is
`max(norm_after_t - offer_t, 0)`, a different signal from signed PE.

Use the age-blind full-data posterior as the primary behavioral estimator. Draw-
wise trajectories are needed for posterior mean signals and uncertainty; the
trajectory evaluated at mean parameter values is generally not the posterior
mean trajectory. Fitting behavior from both runs does not use imaging data, but
these full-fit trajectories must not be called held-out behavioral predictions.

Prepare separate offer-replacement models for expectation and signed PE, retaining
the same onsets, durations, missed-trial exclusion, post-response nuisance terms,
and run/partner amplitude-centering policy as the chosen comparison design.
Missed offers continue to update the behavioral state under the fitted observation
assumption, but contribute only to the missed-trial imaging nuisance regressor.
Do not change timing and the computational modulator simultaneously.

The identity `PE = offer - expectation` means all three same-timing columns
cannot be estimated together. A PE-only result also does not establish PE
encoding over offer encoding. Before any imaging launch, audit posterior signal
uncertainty, offer/expectation/PE correlations, variance remaining after HRF
convolution and high-pass filtering, design rank, and focal contrast variance.
A joint offer-plus-expectation diagnostic is possible if estimable, but its
question differs from either replacement model. Do not apply automatic serial
orthogonalization to manufacture separability. Start with activation; network-PPI
extensions require rebuilding and auditing the matching interaction regressors,
not just swapping the psychological EV files.

**Group-level norm sensitivity.** Alpha measures the penalty for offers below
the fitted norm. It is distinct from the learning rate epsilon, the initial norm
f0, the logistic offer slope, and the legacy acceptance-intercept proxy. Retain
those labels and estimands rather than replacing an old column under its name.

Two scientifically distinct covariates are possible: mean alpha across partners
for a general sensitivity association, and alpha(similar)-alpha(dissimilar) for
a differential-sensitivity association with the matching partner contrast.
Human-minus-computer is secondary. Posterior contrasts must be computed within
draws before summarizing; differences of marginal interval limits are invalid.
Review recovery and between-participant spread for the exact chosen covariate,
particularly because current alpha estimates cluster near their upper bound.

First group-level tests can reuse existing corrected L2 maps; they do not require
the computational-modulator L1 reruns. Retain age and the established nuisance
covariates, add a centered norm covariate, and audit rank/collinearity. A pooled
association adjusted for age and an age-by-norm interaction are different tests;
do not introduce the interaction merely to search for an age effect. With age-
informed behavioral parameters, the brain association is not independent evidence
for age. Use age-blind estimates primarily and age-informed estimates as a
sensitivity check. Posterior-mean plug-in FSL covariates do not propagate behavioral
uncertainty and must be described accordingly.

Define the limited contrast family before looking at maps. Retain the planned
FLAME and permutation checks (TFCE and cluster thresholds 2.6/3.1), use the same
predefined group design and appropriate exchangeability assumptions, and report
all planned results, not only the thresholding method that produces clusters.
Whole-brain correction within a map does not address multiplicity across models,
covariates, networks, and inference methods.

### Original workflow (retained for provenance)

Run in the existing tmux session. No new branch is required. Dependencies are
pinned to CmdStanPy 1.3.0 and CmdStan 2.40.0. The toolchain, compiler products,
raw draws, and failed attempts stay outside Git.

```bash
cd /ZPOOL/data/projects/srndna-ultimatum
git pull --ff-only
conda activate srndna-ultimatum
python -m pip install -r code/requirements-gu-stan.txt
python code/setup_gu_stan.py \
  --install-root /ZPOOL/data/scratch/srndna-stan-toolchain --cores 8
```

First compile and check the pipeline on four participants, including sub-143
and sub-144. This is not an inferential run; diagnostic flags are expected with
100 samples. Compilation or execution errors stop the subsequent launch.

```bash
python -u code/run_gu_hierarchical.py \
  --cmdstan /ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0 \
  --work-root /ZPOOL/data/scratch/srndna-gu-stan-smoke-v1 \
  --phase smoke --jobs 40 --warmup 100 --samples 100 --execute
```

Full fits: eight model/stage combinations. With four chains per fit and two
likelihood threads per chain, `--jobs 40` permits at most **five simultaneous
fits**. BLAS/OpenMP pools are capped at one. The 40-CPU budget is a ceiling,
not a promise of full utilization: serial model calculations and a partly full
last batch reduce utilization. Likelihood threading is over complete
participant/partner trajectories using Stan `reduce_sum`, never over dependent
trials within a trajectory.

```bash
python -u code/run_gu_hierarchical.py \
  --cmdstan /ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0 \
  --work-root /ZPOOL/data/scratch/srndna-gu-stan-v1 \
  --phase fit --jobs 40 --warmup 1500 --samples 1500 --adapt-delta .99 \
  --collect-to results/norm_learning/stan-v1 --execute
```

Repeat that exact command to resume. Completed fits with verified output hashes
are skipped. Interrupted fits restart their chains in new attempt directories;
Stan chains do not resume midway through warmup/sampling. A per-root lock prevents
two runners using the same root. Changes to a job's code, inputs, priors, seed,
tool versions or sampling settings require a new work root. The total CPU budget
can change. Do not launch the following runs concurrently with the first one.

Prior-scale sensitivity for the primary learning model, after the full run:

```bash
python -u code/run_gu_hierarchical.py \
  --cmdstan /ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0 \
  --work-root /ZPOOL/data/scratch/srndna-gu-stan-priors-v1 \
  --phase fit --models rw_free --prior-scales .5 1 2 \
  --jobs 40 --warmup 1500 --samples 1500 --adapt-delta .99 \
  --collect-to results/norm_learning/stan-priors-v1 --execute
```

Initial hierarchical recovery audit (ten new full-sample datasets):

```bash
python -u code/run_gu_hierarchical.py \
  --cmdstan /ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0 \
  --work-root /ZPOOL/data/scratch/srndna-gu-stan-recovery-v1 \
  --phase recovery --models rw_free --recovery-reps 10 \
  --jobs 40 --warmup 1500 --samples 1500 --adapt-delta .99 \
  --collect-to results/norm_learning/stan-recovery-v1 --execute
```

Compact exports are explicitly suitable for Git; posterior draws are not. Every
export includes a provenance record and diagnostic eligibility labels. Exit 1
means execution failure; exit 2 means sampling finished but requires diagnostic
review. Neither is a reason to delete output or suppress a flag.

References: [CmdStanPy sampling](https://mc-stan.org/cmdstanpy/users-guide/examples/MCMC%20Sampling.html),
[Stan likelihood parallelization](https://mc-stan.org/docs/stan-users-guide/parallelization.html).

## Export diagnostic tables and figures from existing fits

`code/export_gu_stan_diagnostics.py` is separate from the fitting runner. Adding
or updating this exporter does not change the recorded fitting-code hashes or
invalidate completed fits. It reads the original scratch outputs without
sampling, recompiling, or modifying them. Run in the repository on Linux1:

```bash
python -u code/export_gu_stan_diagnostics.py \
  --work-root /ZPOOL/data/scratch/srndna-gu-stan-v1 \
  --output-dir results/norm_learning/stan-v1/diagnostics-v1
```

The default expects all eight completed fits, including the one passing all
diagnostics. It verifies recorded input and posterior-chain hashes, draw counts,
and chain IDs. It saves:

- All monitored parameter diagnostics, a flagged-only table, and the parameters
  selected for plotting. Labels map parameter indices to the recorded model,
  participant, partner, and contrast; e.g. parameter 4 is epsilon in RW fits.
- Per-chain means, intervals, first-/second-half means, and selected within-chain
  pair correlations. These are sampling checks, not population-effect estimates.
- One overview PNG and one trace/rank-histogram PNG per fit. Each fit displays
  four variables, covering the worst R-hat, bulk ESS, and tail ESS before filling
  by R-hat. Traces display at most 750 points per chain, while rank histograms
  and summaries use every retained draw.
- A manifest with source and export checksums, options, and exporter version.

The output is small enough for Git. Raw draws stay in scratch. An identical
repeat verifies and reuses the export; changed inputs/options or an edited
export require a new output directory. For other batches, set `--phase` and
`--expected-jobs` explicitly. Diagnostic flags do not block export or count as
execution failure. The exporter never upgrades the fit's inference eligibility.

```bash
git add results/norm_learning/stan-v1/diagnostics-v1
git commit -m "Add parameter-level Stan diagnostics and chain plots"
git push origin main
```
