# Reviewer-analysis outputs

These are compact, reproducible outputs for the revision. They do not replace
or silently relabel the analyses in the submitted manuscript.

Run from the repository root:

```bash
make reviewer-behavior
```

## Headline behavioral results

- The event-corrected intended logistic mixed model with participant-specific
  intercepts and offer-size slopes converges without singularity in R 4.5.2 /
  lme4 2.0.1. The Offer x Age Group x Partner Similarity coefficient is
  0.11358 (SE 0.09468, z = 1.200, p = .230, Wald 95% CI [-0.07198,
  0.29915]; 47 participants, 4,438 trials). Three of five `allFit` alternatives
  converge cleanly and agree; two NLopt alternatives retain warnings.
- A random-intercept robustness model also yields no three-way interaction:
  beta = 0.06840, SE = 0.09150, z = 0.748, p = .455, 95% CI [-0.11093,
  0.24773]. It should not replace the better-supported intended slope model.
- Only sub-144's source rows change, but refitting the shared mixed models
  updates the conditional estimates for all 47 participants. The submitted and
  corrected centered fairness-sensitivity vectors correlate r = .981; sub-144
  changes from -0.19364 to 0.46813 in the L3 design scale. Refitting the
  historical rows reproduces the production vector to within 3.5e-6, confirming
  the recovered formula. A unified uncorrelated random-effects model is
  nonsingular; its fixed Offer x Similarity estimate is -0.03423
  (SE 0.06219, z = -0.550, p = .582, 95% CI [-0.15612, 0.08766]). The
  participant interaction slopes correlate r = .669 with the corrected score.
- The corrected fairness-sensitivity score does not differ detectably by age
  group (younger-minus-older mean difference -0.0979, 95% CI [-0.4564,
  0.2607], Welch p = .584). Positive scores mean that acceptance changes more
  steeply with offer for similar than dissimilar partners; negative scores mean
  the reverse. A score of zero means no partner difference in offer slope. The
  standardized difference is Hedges' g = -0.16 (approximate 95% CI [-0.73,
  0.40]).
- There are 114 missed trials among 6,768 trials (1.68%). Older participants
  missed 3.50 trials on average versus 1.48 for younger participants; the
  participant-level difference is imprecise (younger-minus-older -2.02, 95%
  CI [-4.41, 0.37], Welch p = .096).
- Mean response-selection latency is 0.393 s after the response choices appear.
  The older-group coefficient is -0.1187 s (SE 0.0556, p = .038) in the
  prespecified reviewer model; neither the similarity main effect nor the age
  interaction is detectable. This secondary analysis should be presented as
  descriptive/exploratory rather than a new central claim.
- The curated BIDS files contain 6,654 responded task trials but only 5,724
  companion `event_RT` rows. All 804 responded first trials of blocks lack the
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

- `tables/sub144_trial_metadata_correction.tsv`: checksums and aggregate counts
  for the non-destructive overlay of recovered sub-144 labels onto the
  historical trial-order brain estimates. The corrected row-level file remains
  ignored under `private/`.
- `tables/primary_acceptance_models.tsv`: fixed effects, uncertainty, sample
  sizes, convergence, and singularity for the primary and robustness models.
- `tables/primary_optimizer_diagnostics.tsv`: primary-model agreement across
  optimizers.
- `tables/fairness_sensitivity_*.tsv`: unified model,
  interpretation-relevant diagnostics, and aggregate age comparison.
- `tables/l3_covariate_correction_summary.tsv` and
  `l3_covariate_model_diagnostics.tsv`: exact production-to-historical-refit
  calibration and event-corrected fairness-sensitivity/norm-proxy diagnostics.
  `l3_event_corrected_covariates.tsv` contains only the six pseudonymous
  identifier/design columns required to reproduce the corrected group models,
  including event-corrected task-wide mean RT;
  the fuller participant-level diagnostic table remains ignored under
  `private/`.
- `tables/l3_group_rt_provenance.tsv`: sub-144 RT summaries, correlations with
  the recovered production RT vector, and the explicit manuscript-aligned
  policy to use event-corrected task-wide mean RT.
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
- `tables/production_l3_trace.tsv`: generated on Linux by SHA-256 and labeled
  voxel-support matching of the three tracked binary paper masks to retained
  cluster masks in both analysis repositories; includes overlap metrics, the
  enclosing L3 path, design hashes, sub-144 input, inference settings, and
  available FSL smoothness values. A containment or partial-overlap row is a
  provenance lead rather than proof of mask identity.

Production imaging provenance and the sub-143/sub-144 source-event match are
now traced. Only sub-144 requires the identity repair. See
`docs/SERVER_IMAGING_AUDIT.md`; the collection scripts remain read-only, while
the separately documented repair writes new outputs only under versioned
scratch roots. Robust FLAME deweighting, altered first-level companion-RT
construction, reduced-nuisance models, participant deletion, and any genuinely
missing L3 contrast remain outside this repair.

The guarded group-repair workflow in `code/prepare_ultimatum_l3_repair.py`
creates scratch-only image-only and reported-covariates-corrected designs from
the exact recovered production FSFs. It never edits the original GFEAT trees or
uses the submitted binary focal masks as model inputs. Historical templates
remain unchanged; the exact rendered replacements are collected below.
`l3_repair_designs/` is populated on Linux after `feat_model` validation and
contains the exact rendered FSF/matrix/contrast/group bundles plus checksums;
the much larger GFEAT outputs remain under scratch and untracked.

Participant-level event, sensitivity, rating, and completeness tables are
written to the ignored `private/` directory. They use study identifiers but no
direct identifiers and remain local unless their release is separately
approved under the journal's source-data policy.
