# Revision analysis templates

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
