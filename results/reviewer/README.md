# Reviewer-analysis outputs

These are compact, reproducible outputs for the revision. They do not replace
or silently relabel the analyses in the submitted manuscript.

Run from the repository root:

```bash
make reviewer-behavior
```

## Headline behavioral results

- The intended logistic mixed model with participant-specific intercepts and
  offer-size slopes converges without singularity in R 4.5.2 / lme4 2.0.1.
  The Offer x Age Group x Partner Similarity coefficient
  is 0.06665 (SE 0.09452, z = 0.705, p = .481, Wald 95% CI [-0.11861,
  0.25191]; 47 participants, 4,439 trials). Five `allFit` optimizers agree on
  the estimate and all converge without singularity.
- A random-intercept robustness model also yields no three-way interaction:
  beta = 0.02579, SE = 0.09227, z = 0.279, p = .780, 95% CI [-0.15506,
  0.20663]. It should not replace the better-supported intended slope model.
- The submitted two-model fairness-sensitivity score is exactly reproducible
  up to numerical tolerance (r > .999999999; maximum absolute discrepancy
  4.2e-6). A unified uncorrelated random-effects model is nonsingular; its
  fixed Offer x Similarity estimate is -0.05299 (SE 0.06223, z = -0.851,
  p = .395, 95% CI [-0.17496, 0.06899]). The participant interaction slopes
  correlate r = .672 with the submitted score. The corresponding correlated
  random-effects fit is singular.
- The submitted fairness-sensitivity score does not differ detectably by age
  group (younger-minus-older mean difference -0.0621, 95% CI [-0.3882,
  0.2641], Welch p = .702). Positive scores mean that acceptance changes more
  steeply with offer for similar than dissimilar partners; negative scores mean
  the reverse. A score of zero means no partner difference in offer slope. The
  standardized difference is Hedges' g = -0.11 (approximate 95% CI [-0.68,
  0.46]).
- There are 113 missed trials among 6,768 trials (1.67%). Older participants
  missed 3.45 trials on average versus 1.48 for younger participants; the
  participant-level difference is imprecise (younger-minus-older -1.97, 95%
  CI [-4.38, 0.43], Welch p = .105).
- Mean response-selection latency is 0.392 s after the response choices appear.
  The older-group coefficient is -0.1215 s (SE 0.0560, p = .035) in the
  prespecified reviewer model; neither the similarity main effect nor the age
  interaction is detectable. This secondary analysis should be presented as
  descriptive/exploratory rather than a new central claim.
- The curated BIDS files contain 6,655 responded task trials but only 5,724
  companion `event_RT` rows. All 805 responded first trials of blocks lack the
  RT row; 126 additional omissions are the non-first trials in both sub-143
  runs, which contain no `event_RT` rows. Production 3-column EV and FEAT
  verification is required before interpreting the modeled consequence.
- Explicit ratings offer no evidence that similar and dissimilar human
  partners were rated differently on the recorded traits (all paired p > .38).
  Forty-one pre-task and 39 post-task records are complete and unique. Four
  participant-session files contain appended duplicate administrations and are
  flagged rather than resolved post hoc; five files are missing at each time.
  These ratings do not directly measure belief in the cover story.

## Output map

- `tables/primary_acceptance_models.tsv`: fixed effects, uncertainty, sample
  sizes, convergence, and singularity for the primary and robustness models.
- `tables/primary_optimizer_diagnostics.tsv`: primary-model agreement across
  optimizers.
- `tables/fairness_sensitivity_*.tsv`: unified model,
  interpretation-relevant diagnostics, and aggregate age comparison.
- `tables/task_event_summary.tsv` and `missed_trials_*.tsv`: aggregate
  event/timing and miss results.
- `tables/rt_event_construction_summary.tsv`: aggregate source-BIDS evidence
  for missing companion RT-event rows. Run-level details remain in the ignored
  `private/rt_event_omissions_by_run.tsv`.
- `tables/production_rt_ev_audit.tsv`: reserved for the corresponding audit of
  the authoritative production root. The full 94-run table remains under
  ignored `logs/audits/`; narrowly relevant paths may be retained in compact
  provenance inventories.
- `tables/l3_template_input_inventory.tsv` and
  `tables/l3_template_sub143_inputs.tsv`: declared input counts and exact
  sub-143 entries across every tracked L3 template, including unresolved
  substitution placeholders.
- `tables/response_time_*.tsv`: exact response-time model and group summaries.
- `tables/partner_ratings_human_contrasts.tsv`: aggregate human-partner rating
  contrasts.
- `tables/acceptance_*_source_data.tsv`: observed and model-implied data behind
  the acceptance figure.
- `figures/acceptance_curves.png` and
  `figures/fairness_sensitivity_distribution.png`: reviewer-ready plots with
  their source data in `tables/`.
- `tables/focal_cluster_inventory.tsv`: checksums, grids, voxel counts, and
  physical volumes for the three tracked focal masks.
- `tables/l3_design_*.tsv` and `l3_contrasts.tsv`: parsed group-design rank,
  conditioning, per-EV diagnostics, input counts, group membership, and exact
  contrasts for the original DMN template and later ECN reconstruction. Both
  are full rank with 47 unique inputs; production identity remains server
  pending.
- `tables/dmn_roi_influence_summary.tsv`: descriptive selected-ROI diagnostics
  and leave-one-out coefficient range. Participant rows remain in `private/`.
- `production_audits/`: compact Linux production-design, cluster-table,
  provenance, checksum, and image-header captures. The collector never copies
  NIfTI payloads; inspect the bundle before committing it.

Production imaging provenance remains server-gated. The source-event audit has
isolated the systematic first-trial omissions and both sub-143 runs for
production tracing; this does not authorize an L1 rerun. See
`docs/SERVER_IMAGING_AUDIT.md`; the collection script is strictly read-only and
does not run FEAT, permutation inference, reduced-nuisance models, or
participant-deletion analyses. Robust FLAME deweighting and any genuinely
missing L3 contrast remain separate author-pending decisions.

Participant-level event, sensitivity, rating, and completeness tables are
written to the ignored `private/` directory. They use study identifiers but no
direct identifiers and remain local unless their release is separately
approved under the journal's source-data policy.
