# Revision analysis templates

UPDATE 1 OCTOBER 2026 — SUPERSEDED PRIMARY RT COVERAGE: the historical ACC
age result/figure below is not current publication evidence. Exact input hashes
confirm 786 omitted block-first RT events in the corrected baseline. The
completed all-trial RT DMN analysis has no significant age-difference clusters.
See docs/RT_COVERAGE_CORRECTION.txt (repository root) for the new complete-RT
activation/DMN/ECN batch and publication gates. Content below records earlier
work; do not submit its unupdated imaging claims or associated response DOCX.

The templates in the parent `templates/` directory document the historical
production analysis. They remain unchanged so the originally submitted model
can be reconstructed.

This directory is the canonical template source for analyses added during the
resubmission:

- `L1_task-ultimatum_model-02_type-act_fairness-main.fsf` preserves all nine
  historical EVs and contrasts 1–10, then adds contrast 11,
  `all_partner_offer_pmod`. Contrast 11 is the equal-weighted mean of the
  computer, age-similar, and age-dissimilar offer-size parametric effects:
  `[0, 1/3, 0, 1/3, 0, 1/3, 0, 0, 0]`. The three conditions were each
  scheduled for 48 trials; occasional missed trials are modeled separately.
  Dividing by three makes the cope an equal-weighted average without changing
  its Z statistic.
- `L2_task-ultimatum_model-02_type-act_fairness-main.fsf` carries all eleven
  L1 contrasts through the two-run fixed-effects model.
- `L3_task-ultimatum_model-02_type-act-fairness-main.fsf` is the portable
  47-participant template for the new task-wide offer-size effect. Its first
  EV is an intercept (all ones); the remaining EVs are centered age group,
  centered sex, tSNR, mean FD, and corrected task-wide mean RT. Its four
  contrasts test positive and negative adjusted means and both directions of
  the age-group effect. `OUTPUTDIR` and `L2_ROOT` are renderer placeholders.
- The three `L3_task-ultimatum_type-*_reported-covariates-corrected.fsf`
  files are exact reference copies of the corrected manuscript models stored
  with their compiled design bundles under `results/reviewer/l3_repair_designs/`.
  They intentionally retain the rendered paths and covariate rows used by the
  repair; they are provenance templates, not portable execution entry points.

`code/prepare_activation_fairness_main_pipeline.py` applies these declared
changes to each participant's rendered historical FSFs, preserving all other
run-specific settings. Current BOLD and confound inputs are resolved from the
OpenNeuro dataset, and the nine task EVs are regenerated from the tracked BIDS
events using substantive decision rows for RT; duplicated `event_RT` rows are
therefore not required. For `sub-144`, the builder deliberately uses the
repaired L1/L2 designs rather than the historical production designs. It then
prepares a 47-participant FLAME 1+2 model for cope 11 with an intercept and the
same corrected nuisance covariates used by the resubmission analysis:
centered age group, centered sex, tSNR, mean framewise displacement, and
event-corrected task-wide mean RT.

All generated FEAT outputs are written to a user-specified scratch directory.
Historical production results are never overwritten.

## Decision/post-response model

`L1_task-ultimatum_model-decision-postresponse_type-act.fsf` and the corresponding
`type-nppi.fsf` are explicit reference templates for a separate temporal model.
The activation template has eight EVs: three decision/offer pairs, missed trials,
and one pooled post-response epoch. The nPPI template has 26 EVs, including the
main network, all eight psychological interactions, and the nine original
network nuisance series. There are no post-response partner, offer, or choice
subdivisions. Contrast numbers remain unchanged.

The separate `L1_task-ultimatum_model-decision-postresponse-partner_type-act.fsf`
and `type-nppi.fsf` templates add three partner-specific post-response means
instead of the pooled mean. They contain 10 activation and 30 nPPI EVs before
confounds. Psychological EVs 1–6 remain decision/offer pairs; EV 7 is missed
trials and EVs 8–10 are computer, similar and dissimilar post-response epochs.
nPPI EV 11 is the main network, EVs 12–21 its ten psychological interactions,
and EVs 22–30 the nine retained nuisance network series. Existing contrast
numbers are preserved; COPE 7 is EV 4 minus EV 6 for activation and EV 15 minus
EV 17 for nPPI. No post-response offer or choice regressors are added. These
templates do not modify or replace the pooled templates or completed outputs.

`code/decision_postresponse_model.py` generates these reference templates and
applies the same structural transformation to retained run-specific FSFs.
`PHASE_EVDIR` identifies the newly generated three-column files; it is not a
replacement for the runner's input checks. Use `code/run_decision_postresponse.py`
to preserve each run's original processing settings and inputs, not `feat`
directly on these placeholder templates. See [the model runbook](../../docs/decision_postresponse.md).
Select `--post-model partner` to prepare the new specification. Its default
scratch and report roots are separate from the pooled model; fitting still
requires a completed design audit and explicit opt-in.

## Final reviewer sensitivity models

`L3_task-ultimatum_type-nppi-dmn_age_simple-effects.fsf` and
`L3_task-ultimatum_type-nppi-dmn_age_outlier-deweighted.fsf` retain all 47
corrected inputs and the exact six-column corrected DMN design. They preserve
contrasts 1–4 and add younger-negative and older-negative simple contrasts
as contrasts 5–6. The second template enables full-sample FLAME outlier
deweighting; neither excludes participants. Both request poststatistics.
`OUTPUTDIR` and `STANDARD_IMAGE` are placeholders; use the preparer, not
`feat` directly on these reference files. The preparer verifies they match
its renderer, preventing a disconnected template copy.

The RT preparer applies two explicitly separate changes to retained run-specific
DMN FSFs: all responded trials contribute to RT EVs; the second sensitivity
also changes task/offer EV durations to offer-onset-to-response intervals.
It preserves task amplitudes, contrasts, interaction structure, network signals,
confounds, and smoothing. See `docs/FINAL_REVIEW_RUNBOOK.txt`. Generated run-level
FSFs stay in scratch; compact compiled diagnostics and L3 designs return to Git.
