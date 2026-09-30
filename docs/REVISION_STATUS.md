# Revision readiness — 30 September 2026

The corrected primary analyses and revision-added activation analyses have
completed. The author confirms that no manuscript edits have yet been made
and the response to reviewers still needs to be drafted. Existing draft text
is preparation, not completed revision work. OpenNeuro release publication is
a separate blocker; it does not prevent the writing pass. See
`ANALYSIS_SUMMARY.md` for the consolidated handoff.

## Verified in this pass

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

- Obtain the current editable manuscript and original reviewer letter, apply
  the revision text and checklist, and cross-check every response against
  the actual comment. The old attachments are no longer at their supplied
  Downloads paths. The local response uses comment summaries, not verified
  verbatim quotations.
- Add manuscript page/line references only after editing and typesetting.
  Proposed rebuttal language saying edits were made is not evidence that
  they have been made in an editable manuscript.
- Address the reviewer influence concern honestly: no corrected full
  image-level leave-one-out robustness result is established in this audit.
  A new robustness analysis, if requested, must retain the full-sample primary
  endpoint rather than use post hoc participant exclusion.
- Finalize the scope of exploratory supplementary ratings/age analyses and
  check citations, acquisition wording, captions, tables, and figure numbering.
  Do not erase the submitted ECN question because its corrected result is null.
- Once OpenNeuro resolves the server checkout rejection, verify the published
  revision and cite its actual snapshot/version. Do not yet claim 2.2.0 is
  publicly released. Freeze the code commit used for the submission.

No new Linux launch is required for the writing and reconciliation above.
