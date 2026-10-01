# Revision readiness — 30 September 2026

## Final reviewer checks now prepared

`code/run_revision_completion.py` is the Linux1 entry point for the remaining
work; see `FINAL_REVIEW_RUNBOOK.txt`. The quick phase checks saved L1 contrast
estimability and runs six-contrast DMN simple-effect/deweighting models. The
separate RT phase prepares 188 L1, 94 L2, and two L3 sensitivity fits. These
server fits are **pending**, not completed by local script validation.

All 116 repository tests pass. The 94 source events files also pass the new
RT-construction checks: 6,654 responded trials, unique substantive onsets, and
finite positive RTs within their recorded display epochs. This does not replace
the Linux comparison against the retained three-column EVs and compiled designs.

The returned Linux audit (b380f6a) now resolves the saved-design rank concern:
all production/corrected runs were audited, raw deficiencies were accounted for
by zero columns, and all tested contrasts were estimable. The new RT fits have
not started: preparation stopped on a missing historical `srndna-data` BOLD path.
`code/audit_dmn_rt_inputs.py` inventories all required inputs and surviving BOLD
copies before any explicit relocation. Neither substitute preprocessing nor
whole-sample checksum equivalence was assumed from the sub-144 comparison.

The subsequently returned input audit (9f8d8c6) establishes byte equality for
all 94 BOLD pairs between the surviving `srndna-ug` tree and the downloaded
OpenNeuro derivative. Both report fMRIPrep 21.0.2; no copy differences were found.
The original `srndna-data` location is unavailable. Network time-series paths
resolve, but task-EV/confound relocations still need the mapped preflight check.
The standard-image resolver now accepts the extensionless basename used in all
94 saved L1 FSFs, while requiring the same template name and resolution.

The proposed EV/confound relocation to the production FSL tree failed on
Linux1: 688 small-input references remain missing, and the RT fits have not
started. The runner now offers `--use-feat-input-copies`, which checks each run's
own saved EV/confound copies. Recovery requires recompilation of the original
design and cellwise numerical agreement with the retained design.mat for all
94 runs before any sensitivity fitting. These recovery checks have not yet run
on Linux1. If copies are unavailable, `code/locate_dmn_rt_text_inputs.py` performs
a bounded search of known input directories without choosing replacements.
Do not repeat the failed directory mapping as a launch recipe.

The new corrected selected-ROI influence diagnostic retains all 47 participants.
Three observations exceed Cook's 4/N screen; the diagnostic age contrast stays
positive in all leave-one-out fits (range 14.55–18.05). These are descriptive
OLS estimates in a selected cluster, not whole-brain FLAME robustness results.
See `results/reviewer/dmn_corrected_influence/`. The new participant-SEM plot is
an explicitly labeled candidate, not a silent replacement of the FLAME display.

The supplied task script measures raw RT from offer onset and uses a three-second
response timer plus 0.5-s additional display/feedback; check the manuscript's
"2.5-s response window" wording against acquisition versions. The new duration
sensitivity uses actual recorded RT and does not add a second.

The corrected primary analyses and revision-added activation analyses have
completed. The author confirms that no manuscript edits have yet been made
and the analytical response now has a first draft covering 18 concerns. Its text
is preparation, not completed revision work. OpenNeuro release publication is
a separate blocker; it does not prevent the writing pass. See
`ANALYSIS_SUMMARY.md` for the consolidated handoff.

## Verified in the preceding audit

- The 78 repository tests pass (including strengthened ratings-availability
  assertions). These are software/result-contract tests, not fresh FSL fits.
- Re-running `analyze_ultimatum_ratings.py` from the data-paper normalized
  ratings reproduced all five tracked tables to numeric tolerance (relative
  1e-6, absolute 1e-8). Choice GLMMs were not refitted in this pass; their
  tracked coefficients and diagnostics were inspected and contract-tested.
- All 84 selected Ultimatum ratings administrations in the paper sample
  (756 responses) match the data-paper BIDS behavior exports exactly by
  participant, timepoint, partner, and dimension. The final complete attempt
  is the approved rule; there is no outstanding repeat-selection decision.
- Sub-143 has no ratings file. Sub-144's ratings are its own and are not
  duplicate sub-143 records. The response must not confuse the event repair
  with ratings availability or treat missing ratings as complete-case data.
- Corrected imaging statistics are supported by the collected FSL outputs
  and `results/manuscript/tables/final_result_set.tsv`, not by image-only
  diagnostic runs or the submitted cluster masks.

## Settled reporting endpoint

| Result | What to report |
| --- | --- |
| Primary choice model | N=47; 4,438 human-partner trials; Offer × Age × Similarity beta=.11358, SE=.09468, p=.230. No equivalence claim. |
| Corrected DMN | 29 voxels, cluster p=.0138, Zmax=4.07; retained, Figure 4. |
| Corrected ECN | No surviving clusters in any of eight contrasts. Remove the positive claim/figure, retain a transparent account of the tested model and corrected null. |
| Task-wide offer-size activation | Five positive clusters, 348 voxels total; none negative. Figure 3; explicitly added during revision. |
| Ratings | 42 pre, 42 post, 41 paired. Human-versus-computer fairness/likeability effects do not establish a similar-versus-dissimilar effect. Exploratory supplementary context. |
| Other imaging follow-ups | Keep complete audit outputs. Do not promote the norm-proxy or periventricular social-offer cluster to a new headline claim. |

## Corrections to the writing, not new imaging results

1. FSL `smoothest` reports resel **size** under `RESELS`, not total resel
   count. The corrected DMN record is 15.5628 voxels per resel and 56,907
   search voxels; FWHM is 7.72104 × 7.7501 × 7.40201 mm. Do not describe this
   as a search space of 15.5628 resels. Preserve raw FSL records unchanged.
   See [FMRIB technical report, footnote 1](https://www.fmrib.ox.ac.uk/datasets/techrep/tr08tn1/tr08tn1.pdf).
2. `audit_roi_influence.R` uses the submitted figure's ROI values and original
   covariates. Its Cook's distance and leave-one-out numbers are historical,
   not corrected-model robustness evidence. Neither a selected-ROI p-value
   nor the new descriptive bars independently validates the whole-brain effect.
3. Figure 4 error bars are the cluster mean of voxelwise FLAME standard
   errors, not empirical participant SEM or the SE of a spatial ROI average.
   Retain this explicit caption; do not relabel them as participant SEM.
4. Minimal focal-model repairs preserved the submitted companion-RT row
   construction. The revision activation workflow uses substantive decision
   rows. Distinguish these in Methods; do not claim an all-participant focal
   RT-EV reconstruction or latency-duration model was run.

## Remaining work before submission

- Obtain the current editable manuscript, apply the revision text and checklist,
  and cross-check every response against the actual comment. The original
  reviewer letter was recovered from the old ChatGPT task attachment during
  response drafting. The local response uses paraphrased concern summaries,
  not verbatim quotations or original comment numbering.
- Add manuscript page/line references only after editing and typesetting.
  Proposed rebuttal language saying edits were made is not evidence that
  they have been made in an editable manuscript.
- Address the reviewer influence concern honestly: no corrected full
  image-level leave-one-out robustness result is established in this audit.
  A new robustness analysis, if requested, must retain the full-sample primary
  endpoint rather than use post hoc participant exclusion.
- Incorporate the completed first-level estimability audit into the analytical
  response. All 94 runs per family were checked; zero columns account for the
  raw deficiencies and no tested contrast was nonestimable. These are saved,
  unwhitened-design checks, not proof of new RT-model validity or imaging robustness.
- Finalize the scope of exploratory supplementary ratings/age analyses and
  check citations, acquisition wording, captions, tables, and figure numbering.
  Do not erase the submitted ECN question because its corrected result is null.
- Once OpenNeuro resolves the server checkout rejection, verify the published
  revision and cite its actual snapshot/version. Do not yet claim 2.2.0 is
  publicly released. Freeze the code commit used for the submission.

Writing can proceed independently, but the new sensitivity checks above require
Linux execution before the analytical response can be finalized.

The first analysis-focused response draft is now in
`revision/analysis_responses.md`. It is proposed prose, not a completed
manuscript revision. Its author notes identify the responses that still need
evidence or an author decision.
