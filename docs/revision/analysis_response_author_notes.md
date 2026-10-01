# Evidence and author decisions for the response draft

UPDATE 1 OCTOBER 2026 — SUPERSEDED PRIMARY RT COVERAGE: the historical ACC
age result/figure below is not current publication evidence. Exact input hashes
confirm 786 omitted block-first RT events in the corrected baseline. The
completed all-trial RT DMN analysis has no significant age-difference clusters.
See docs/RT_COVERAGE_CORRECTION.txt (repository root) for the new complete-RT
activation/DMN/ECN batch and publication gates. Content below records earlier
work; do not submit its unupdated imaging claims or associated response DOCX.

The companion `analysis_responses.md` is a first writing pass, not a completed
response letter. The manuscript remains unchanged. Concern summaries are
drawn from the original review text recovered from the attachment to
**Jen paper -- SRNDNA-ug**, not from the earlier generated response.
The source attachment is named `SRNDNA-UG-ResponseToReviewers.docx` and contains
the Reviewer 1 September 3 and Reviewer 2 August 25 reviews.

## Decisions preserved from the earlier conversation

- Do not prepare permutation/randomise inference merely because a cluster is
  small. Document the recorded FSL inference.
- Do not remove tSNR or mean FD simply because they correlate.
- Do not exclude a participant because of a selected-ROI influence screen.
  Complete-sample FLAME outlier deweighting was discussed as a possible
  sensitivity analysis but was not authorized or demonstrated as completed.
- Do not substitute the unified behavioral score to try to recover the ECN
  finding. The corrected original-definition analysis determines its outcome.
- Historical behavioral estimates from that conversation have been superseded
  by the event-corrected model (p=.230, 4,438 trials), not copied into this draft.

## Responses not yet ready for a final letter

### First level design rank and collinearity

`results/reviewer/tables/production_l1_design_summary.tsv` records all 94
matrices for each of act, nppi-dmn, and nppi-ecn, but flags 50 rank-deficient
runs in each model family. The current audit drops zero-norm columns when
computing the normalized condition number but compares raw rank with all
columns when flagging deficiency. Consequently, empty nuisance EVs are a
possible explanation; the aggregate table does not establish that explanation
or demonstrate estimability of the focal contrasts. Its log keyword count is
also a screening measure, not a classification of substantive FSL failures.

The run-level diagnostic file was not found in this local checkout. A bounded,
read-only follow-up should identify zero columns, rank after their removal,
EV labels, and whether focal contrast vectors lie in the design row space.
Do not say all first-level models were full rank or that the observed
activation main effect proves the focal PPI contrasts were unaffected.
Responses on short ITIs and regressor count are provisional on this point.
No new imaging fit has been launched or implied by the draft.

Follow-up prepared on 30 September: `code/audit_l1_estimability.py` reads the
282 production designs and the six repaired sub-144 designs, identifies exact
zero columns, and tests every saved contrast against the design row space.
Its corrected summary substitutes repaired sub-144 runs, not the old runs.
The full matrices are not available in the local clone; the Linux1 run is
still pending. Instructions are in `docs/SERVER_IMAGING_AUDIT.md`.

A local source-event check found 49/94 corrected runs with no `missed_trial`
rows. At historical revision `02ba301`, sub-144 run-01 had no missed rows,
whereas its corrected file has one; run-02 has none in either version. Thus
the historical event set predicts 50 empty missed-trial EVs. `L1stats.sh`
selects shape 10 when that EV file is absent. This numerical match supports
the empty-column explanation, but does not yet prove the actual matrix
columns or contrast estimability. Do not convert it into a conclusive
reviewer-response claim before the Linux audit returns.

### Participant influence

`code/audit_roi_influence.R` reads historical figure inputs and original
covariates. Its numerical results are deliberately omitted from the proposed
response. The proposed response explains why there was no selected-plot-driven
exclusion, but it cannot claim to demonstrate robustness to participant
removal. Decide whether that explanation is sufficient or whether a separately
authorized, complete-sample outlier-deweighting sensitivity analysis is needed.

### Simple effect displays

The corrected group tables contain the younger and older positive simple
contrasts, and Figure 4 has four condition bars. The review specifically asks
for separate whole-brain renderings of the age-specific contrasts. The draft
does not claim those additional renderings already exist. Add the younger map
and a clearly labeled no-surviving-clusters older panel if agreeing to this
presentation request; do not turn an empty corrected map into an unthresholded
positive result.

## Evidence map

Paths below are relative to the repository root.

| Response topic | Primary evidence |
| --- | --- |
| Choice model and coding | `code/analyze_reviewer_behavior.R`; `results/reviewer/tables/primary_acceptance_models.tsv`; `primary_optimizer_diagnostics.tsv` in the same directory |
| Unified score and age distribution | `results/reviewer/tables/unified_sensitivity_model.tsv`; `fairness_sensitivity_diagnostics.tsv`; `fairness_sensitivity_age_test.tsv` |
| Corrected DMN and ECN | `results/reviewer/l3_repair_results/*/reported-covariates-corrected/`; `results/reviewer/l3_repair_designs/` |
| RT construction and display | `docs/REVISION_FINDINGS.md`; `results/reviewer/tables/production_rt_ev_audit.tsv`; retained task scripts |
| Network signals | `code/L1stats.sh`; `templates/L1_task-ultimatum_model-02_type-nppi.fsf`; original network maps in `masks/` |
| Missed trials | `results/reviewer/tables/missed_trials_age_summary.tsv`; `missed_trials_age_test.tsv` |
| Main and social activation | `results/reviewer/activation_fairness_main/`; `results/reviewer/activation_social_computer/` |
| Ratings and choice moderation | `results/reviewer/ultimatum_ratings/`; source selection verified against the data-paper BIDS exports |
| Figure 4 estimates and error bars | `results/manuscript/source_data/figure4_dmn_flame_bar_summary.tsv`; `code/plot_manuscript_figures.py` |

## Editorial choices

Each response opens with brief, specific thanks, then explains the answer in
two or three sentences. Proposed manuscript text follows separately. Longer
responses close only when a short synthesis helps distinguish the separate
parts of the answer. Manuscript text uses direct claims tied to estimates and
limitations, rather than defending software by reputation or interpreting a
nonsignificant result as equivalence.

No manuscript locations are invented. Add page and line references only after
the actual manuscript revision. Keep exploratory additions identified as such,
and report the corrected ECN null rather than removing the tested question
from the record. The OpenNeuro publication statement is outside this
analysis-only draft and must await a verified snapshot.
