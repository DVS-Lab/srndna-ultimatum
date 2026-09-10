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
