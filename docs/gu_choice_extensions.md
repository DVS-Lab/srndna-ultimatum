# Positive-support norm-choice extensions

Exploratory age-blind behavioral models on the corrected 47-person sample.
These are distinct from the archived baseline and do not replace it. No imaging
covariates, event files, or FEAT designs are generated. The separate runner is
derived from `run_gu_hierarchical.py` to preserve baseline source fingerprints.

## Models

Both retain separate partner histories, carryover between runs, updating on
displayed offers even if the response is missing, and choice utility evaluated
using the updated norm:

```
norm_after = norm_before + epsilon * (offer - norm_before)
logit(P(accept)) = bias + gamma * (offer - alpha * max(norm_after - offer, 0))
```

- `rw_positive`: bias is fixed at zero. Alpha and gamma are positive without
  an upper bound; epsilon remains in (0,1), initial norm in (0,20) dollars.
- `rw_positive_bias`: same model plus one partially pooled acceptance-bias
  parameter per participant, shared across the three partners. Bias is on the
  log-odds scale, outside the multiplication by gamma. There are no partner
  effects or age effects on bias. Positive bias favors acceptance.

Alpha and gamma use exponential transforms. Their log-scale population means
have Normal(0,1) priors, partner effects Normal(0,.5), participant SDs
half-Normal(0,.5), and participant contrast SDs half-Normal(0,.35). Initial
norm and epsilon keep the baseline logit-scale priors: mean SD 1.5, partner
effect SD .75, participant SD prior .75, contrast SD prior .5. Bias has a
Normal(0,1.5) population mean and half-Normal(0,1) participant SD, with standard
normal non-centered participant deviations. Prior scales are recorded in each
fit's data and configuration. Relaxing support necessarily changes the prior;
this is not a prior-matched numerical replication of the bounded model.

Changing bounds can accommodate highly consistent acceptance of low offers.
Adding bias can also produce trade-offs with initial norm. Neither change is
assumed to improve fit or identification. Compare held-out prediction, low-offer
acceptance, posterior dependencies, and recovery before interpreting parameters
or constructing brain covariates. Parameter support is not an empirical result.

## Linux1 launch inside tmux

```
cd /ZPOOL/data/projects/srndna-ultimatum
git pull --ff-only
conda activate srndna-ultimatum
bash code/run_gu_choice_extensions.sh smoke --execute &&
bash code/run_gu_choice_extensions.sh fit &&
bash code/run_gu_choice_extensions.sh fit --execute
```

The smoke test checks four participants (including sub-143 and sub-144), not
scientific convergence. Every executable launch first validates compiled Stan
against Python, including shared bias, train/held-out generated likelihoods and
the actual reduce_sum likelihood target. An equation mismatch stops execution.
Smoke diagnostic flags are retained but do not prevent the full launch after a
successful plumbing check. Execution failures do stop the launch.

The full batch consists of four fits: two models, each on full data and run 1.
Run-1 fits predict run-2 choices without conditioning on run-2 choices. All use
four chains, 3,000 warmup and 6,000 retained samples per chain, adapt_delta .99,
and two likelihood threads per chain. The CPU ceiling is 40; four simultaneous
fits use up to 32 threads. No existing baseline or age model is rerun.

Full scratch: `/ZPOOL/data/scratch/srndna-gu-stan-positive-bias-v2`.
Smoke scratch: `/ZPOOL/data/scratch/srndna-gu-stan-positive-bias-smoke-v2`.
Override the scratch parent with `SRNDNA_STAN_SCRATCH_BASE`, and the existing
CmdStan 2.40.0 location with `SRNDNA_CMDSTAN_ROOT` if needed.

Repeat the same fit command to resume. Verified completed fits are skipped,
including fits with diagnostic flags. Interrupted fits restart their chains in
new retained attempt directories; individual chains do not resume midway.
Code/data/settings changes require a new work root, not deletion of old results.
A scratch-root lock prevents concurrent runners using the same root. Keep this
batch separate from other CPU-heavy launches.

## Outputs and interpretation

Small results go to `results/norm_learning/stan-positive-bias-v2` (and the
corresponding `stan-positive-bias-smoke-v2` for smoke). Raw chains stay in scratch.
Trackable `preflight.txt`, `run.txt`, and `export.txt` preserve terminal output.
Exit 2 means sampling completed with diagnostic flags, not execution failure.
All flags and ineligibility labels are preserved in summaries and trace plots.

In addition to standard parameter, partner-contrast, held-out score, and
posterior-predictive tables, exports include:

- `participant_predictive.tsv`: observed acceptance and posterior expected
  acceptance by participant/partner/run, for all offers, $1 offers, and $1–3
  offers. Intervals describe uncertainty in expected rates, not predictive
  intervals for replicated counts. The standard aggregated PPC table contains
  replicated-count intervals. Run-2 checks from full-data fits are in-sample.
- `posterior_parameter_correlations.tsv`: within-unit posterior dependencies,
  including bias versus initial norm, to diagnose trade-offs rather than infer
  between-person relationships. Chains must mix before interpreting these.
- `diagnostics-v1/`: trace/rank plots, overview, parameter flags and provenance.

Bias appears once per participant with partner `all` in `parameters.tsv`.
There are no bias partner contrasts because equality is imposed by construction.
The Stan theta array repeats the same bias across partners for likelihood
evaluation; those copies are not separate estimates.

```
git add results/norm_learning/stan-positive-bias-smoke-v2 results/norm_learning/stan-positive-bias-v2
git commit -m "Add positive-support norm-model fits and diagnostics"
git push origin main
```

Recovery is available explicitly through the Python runner's `--phase recovery`
but is not part of this overnight launch. These fits alone do not validate
individual parameters. Archive all comparisons, including unfavorable results.

### Initialization repair (v2)

The v1 smoke launch stopped before HMC: the equation-check initializer wrote
`age_beta=[]`, which JSON represents as a vector rather than the declared
zero-row matrix. The strictly age-blind extension no longer declares this
unused parameter. Likelihoods, priors, and sampling settings are unchanged.
New v2 roots preserve the failed v1 attempt and its code fingerprints.

## Diagnostic-only divergence audit

Before extending sampling, inspect retained divergent transitions in the
existing fits. This command reads and verifies completed chains; it does not
compile, sample, change priors, exclude participants, or upgrade eligibility:

```bash
python3 -u code/audit_gu_divergences.py \
  --work-root /ZPOOL/data/scratch/srndna-gu-stan-positive-bias-v2 \
  --output-dir results/norm_learning/stan-positive-bias-v2/divergence-audit-v1
```

Four completed extension fits are required. Processing is sequential, so no
`--jobs` flag is needed. All posterior-only draws contribute to the numerical
summaries. The output contains sampler diagnostics by chain, descriptive
divergent/nondivergent parameter comparisons (within chain and pooled), console
warning counts, and paired-parameter plots for sub-104, sub-107, sub-126,
sub-128, sub-155, and sub-156. These illustrate always/near-always acceptors,
strong rejectors, and the participant flagged in the no-bias full fit.
Use `--subjects` to request another set in a new output directory.

Plots retain every divergent endpoint and subsample other points for display.
An endpoint association is not evidence that a participant caused divergence.
No independent-draw significance tests are performed. Console warnings may
include warmup and cannot be interpreted as counts of retained divergences.
Unchanged exports are verified and skipped; altered inputs or outputs require
a new output directory. Raw chains remain in scratch; only small audit files
and figures should be committed.
