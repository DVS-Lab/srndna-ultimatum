# Decision and post-response model

This is a separate temporal model, not a replacement for the archived full-display analysis. It changes the estimand from an offer-modulated full-display effect to an offer-modulated pre-response effect conditional on the modeled post-response activity.

For responded trials, the decision epoch starts at recorded offer onset and ends at the button press. Recorded RT is already referenced to offer onset; the initial second is not subtracted. Both the condition mean and its offer modulator have RT as duration. Post-response epochs cover button press to recorded display offset with unit amplitude. Missed trials are excluded from all decision and post-response epochs and retain a separate full-display nuisance regressor. Dedicated RT onset/height regressors are removed.

Two specifications are supported; they have separate templates, scratch trees, reports and recorded model identities:

- `--post-model pooled` (unchanged default): one shared post-response mean, 8 activation EVs and 26 nPPI EVs before confounds.
- `--post-model partner`: separate computer, similar (`ingroup`) and dissimilar (`outgroup`) post-response means, 10 activation EVs and 30 nPPI EVs before confounds. No offer or choice subdivisions are added.

For the partner model, psychological EVs 1–6 are the original three decision/offer pairs; EV 7 is missed trials and EVs 8–10 are the three post-response means. nPPI adds the main network at EV 11, its ten psychological interactions at EVs 12–21, and the nine retained network nuisance series at EVs 22–30. Contrast numbers are preserved: COPE 7 is activation EV 4 minus EV 6, or nPPI EV 15 minus EV 17. Post-response and missed-trial components are nuisance terms in those contrasts. Original offer amplitudes, orthogonalization, network series, preprocessing, confounds, sample, smoothing and group covariates are retained.

The pooled model assumes a shared post-response component. The partner model relaxes that assumption but **neither** separately models offer- or choice-dependent post-response effects. Adjacent phases may be difficult to separate after HRF convolution. Statistical estimability therefore does not establish physiological specificity. No significance-driven model selection is built into this workflow. Building the partner model does not replace or endorse the pooled results.

## Prepare and inspect on Linux1

The commands below prepare the **partner model**. Use the same environment and exact input locations as the completed RT-coverage run. The configuration file supplies only its input roots and relocation maps; its model choice is not inherited and the old derivatives are not overwritten or treated as new-model outputs.

```bash
tmux new-session -A -s ug-decision-post-partner
```

Inside tmux:

```bash
cd /ZPOOL/data/projects/srndna-ultimatum
conda activate srndna-ultimatum

python3 code/run_decision_postresponse.py \
  --post-model partner \
  --reuse-input-config /ZPOOL/data/scratch/srndna-ultimatum-rt-correction-v1/preparation_config.json \
  --jobs 40
```

The default is preparation and design checking only; no FEAT fits are launched. It prepares 282 L1, 141 L2 and five L3 jobs under `/ZPOOL/data/scratch/srndna-ultimatum-decision-postresponse-partner-v1`. All 282 retained baseline matrices must reproduce before the new L1 designs pass their rank and focal-contrast checks. L2/L3 compilation waits for their new inputs to exist.

Small reports are written to `results/decision_postresponse_partner/design_audit/`:

- `phase_construction.tsv`: responded-trial and post-response counts and timing policy.
- `baseline_preflight.tsv` and `l1_preflight.tsv`: baseline agreement, rank and focal estimability.
- `contrast_diagnostics.tsv`: contrast variance and variance ratios in the compiled basis.
- `phase_correlations.tsv`: partner-matched decision/post-response correlations (activity EVs 1/8, 3/9 and 5/10; nPPI EVs 12/19, 14/20 and 16/21).
- `analysis_plan.json` and `diagnostic_scope.json`: model specification and interpretation limits.

These supplementary metrics are conditional on the compiled, orthogonalized FEAT design and IID noise. They are **not** original-stimulus contrast VIFs, estimates of FILM-prewhitened precision, or an automatic endorsement of the model. Inspect them before fitting. They can be committed and reviewed before any imaging results are generated.

## Fit only after reviewing diagnostics

Keep the same environment, code and scratch root between preparation and execution. Do not pull code updates during an active batch; code, template, input and software checks guard against mixed specifications.

```bash
python3 code/run_decision_postresponse.py \
  --post-model partner \
  --reuse-input-config /ZPOOL/data/scratch/srndna-ultimatum-rt-correction-v1/preparation_config.json \
  --jobs 40 --execute --accept-design-diagnostics
```

Execution requires an existing prepared tree and explicit acknowledgement of the diagnostic review. Up to 40 L1 jobs run concurrently; L2 and L3 concurrency is capped at two. All five group models retain their numerical covariates and contrasts; the 32 group contrasts, including null results, are collected in the separate `results/decision_postresponse_partner/imaging/` directory. Existing primary figures and results are not replaced.

Detach with **Ctrl-b**, then **d**; reconnect with `tmux attach -t ug-decision-post-partner`. Completed outputs are skipped on rerun. An existing incomplete FEAT directory is not overwritten: first verify that no surviving process is writing to it, then preserve and review the interrupted output before restarting it. Do not delete the scratch tree to resume.

The pooled mode still defaults to `/ZPOOL/data/scratch/srndna-ultimatum-decision-postresponse-v1` and `results/decision_postresponse/`. Do not point the partner mode at those existing trees. Explicit mode checks reject pooled/partner collisions, including report trees with a conflicting analysis plan. Code updates also invalidate an existing preparation configuration; they do not silently update a completed fit.

For a deliberate configuration change, use a new `--work-root` and a separate `--output-root`. `--reuse-input-config` cannot be combined with conflicting roots or additional `--input-map` arguments. No Git operations, remote uploads, permutation testing or manuscript changes are performed by the runner.
