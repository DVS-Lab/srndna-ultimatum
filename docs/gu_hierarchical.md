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
treated as automatically plausible. No age or brain measurements enter this
parameter hierarchy.

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
