# Revision findings and response document updates

This file translates the completed repository audit into submission-facing
changes. It is a working author aid, not a reviewer response ready for upload.
Items labeled **server pending** must remain bracketed in the response until the
authoritative production derivatives have been collected and checked.

## Highest priority correction to the current draft

The draft response to Reviewer 1 Comment 1 says that the full fixed-effects
model will be reported with a participant random intercept because the random
offer slope did not converge. That is no longer the best-supported response.
With the event-corrected manuscript dataset, R 4.5.2, lme4 2.0.1, centered offer, and
`bobyqa`, the intended `(1 + Offer | participant)` model converges without a
singular fit. Three of five `allFit` alternatives converge cleanly and agree;
two NLopt alternatives retain convergence warnings.

Suggested replacement:

> We agree that nonconvergence is an estimation problem rather than evidence
> against the interaction. We re-fit the intended model after centering offer
> size and increasing the optimizer iteration budget. The model retained fixed
> effects of Offer Size, Age Group, Partner Similarity, and all interactions,
> with participant-specific intercepts and offer-size slopes. It converged
> without a singular fit. Three of five alternative optimizers also converged
> cleanly and produced nearly identical estimates; two NLopt alternatives
> retained convergence warnings. The Offer Size × Age Group × Partner
> Similarity coefficient was 0.1136 (SE = 0.0947, z = 1.200, p = .230,
> 95% CI [-0.0720, 0.2991]; 47 participants, 4,438 trials). A
> random-intercept-only robustness model led to the same inferential conclusion
> (β = 0.0684, SE = 0.0915, z = 0.748, p = .455, 95% CI [-0.1109,
> 0.2477]). We now report the intended
> random-intercept and random-offer-slope model as primary and avoid
> interpreting the nonsignificant interaction as evidence of equivalence.

## Completed reviewer analyses

### Behavioral fairness sensitivity

The tracked submitted score is reproducible from the historical duplicated
labels, but correcting sub-144 changes that participant's score from -0.2065
to 0.4931. The tracked and corrected vectors correlate r = .981. A unified
model with correlated participant interaction slopes is singular; an explicit
uncorrelated random-effects model is nonsingular. Its fixed Offer × Similarity
coefficient is -0.0342 (SE = 0.0622, z = -0.550, p = .582, 95% CI [-0.1561,
0.0877]). Unified participant slopes correlate r = .669 with the corrected
separate-model score. This supports reporting the unified quantity as a
robustness check, not silently replacing the submitted imaging covariate.

The corrected score does not differ detectably by age group: younger mean
-0.0209 (SD = 0.4930), older mean 0.0770 (SD = 0.6896), younger-minus-older
difference -0.0979, 95% CI [-0.4564, 0.2607], Welch p = .584. Positive values
mean a steeper offer-acceptance slope for similar than dissimilar partners;
negative values mean the reverse. The standardized difference is Hedges'
g = -0.16 (approximate 95% CI [-0.73, 0.40]).

**Author decision pending:** First compare the estimands, scaling, shrinkage,
distribution, age association, and r = .669 correspondence of the corrected
and unified participant measures. Do not construct or run a new ECN group model
unless that comparison supports it and the author decides it is scientifically
useful. The behavioral correlation is not evidence of equivalence or neural
robustness.

### Missed trials and response time

The 47-person sample contributes 6,768 trials. Participants missed 114 trials
(1.68%): 37 among younger adults and 77 among older adults. Mean misses were
1.48 (SD = 3.55) for younger and 3.50 (SD = 4.45) for older participants;
younger-minus-older difference -2.02 trials, 95% CI [-4.41, 0.37], Welch
p = .096. Partner identity for misses was reconstructed from the enclosing
block because the BIDS miss row omits it.

Mean response-selection latency was 0.393 s after choices appeared. The older
group coefficient was -0.1187 s (SE = 0.0556, p = .038); the similarity effect
and age-by-similarity interaction were not detected. Treat this secondary
result as descriptive/exploratory. The task source confirms that the partner,
offer, and selected response remained visible through the approximately 3.5-s
epoch, so first-level task regressors do not isolate deliberation.

### RT nuisance-event construction: production audit pending

The curated BIDS event files contain 6,654 responded task trials, but only
5,724 matching `event_RT` rows. All 804 responded first trials of blocks lack
the companion row. An additional 126 non-first omissions occur in sub-143;
both of that participant's runs contain no `event_RT` rows at all. Thus 930
responded trials (13.98%) lack the source row used by the RT 3-column
conversion. The substantive task-event rows remain present for these trials;
the retained sub-143 designs contain nonconstant RT regressors, so candidate
EV-to-design matching is required before considering sub-143 affected.

The first audit was accidentally run against the separate `srndna-ug` checkout
and is not production evidence. Its 92/94 result is therefore excluded from
this repository's aggregate outputs.

The authoritative production root is
`/ZPOOL/data/projects/srndna-ultimatum`, which is also a clean checkout of this
repository. A targeted read-only check there found all nine queried sub-143 L2
outputs: activation, DMN nPPI, and ECN nPPI copes 4, 6, and 7, dated August
2021. The retained L2 FSFs declare the two corresponding L1 runs as inputs.
This establishes that sub-143 contributed the cope 7 files required by the
focal 47-input designs and the cope 4/6 files required by condition-stacked
templates.

It does not yet establish what the sub-143 L1 RT columns contained. The
production L1 `design.fsf`, `design.mat`, and EV files must be audited directly,
and the rendered submitted L3 output directory must be located. Do not silently
regenerate EVs or rerun L1. The public BIDS TSVs are source data; the FSL
3-column EVs are generated derivatives and should be rebuilt only if a later
scientific decision authorizes a rerun.

### Partner ratings

There are 41 complete unique pre-task administrations and 39 complete unique
post-task administrations. Four participant-session files contain appended
duplicate administrations and five are missing at each session. Rather than
choosing among duplicates post hoc, the primary rating summary excludes those
sessions and records them in the local completeness audit.

No similar-versus-dissimilar human-partner contrast is detected for pre-task
fairness or likeability or post-task fairness, likeability, anger, or
satisfaction (all paired p > .38). These ratings do not directly measure
perceived similarity or belief that partners were real, so the design
limitation remains.

### Tracked group-design and influence checks

Both focal tracked L3 templates contain 47 unique participant inputs, are full
rank, and use one group-membership value. Scaled condition numbers are 4.54 and
4.56. The maximum no-intercept design variance factor is 4.62. tSNR and mean FD
are correlated (r = -0.796); this is a descriptive design fact, not a
methodological defect or a reason to remove either prespecified nuisance
covariate. No reduced-nuisance model is planned.

The selected DMN ROI diagnostic has one participant above Cook's 4/n screening
threshold. Across descriptive leave-one-participant-out fits, the older-group
coefficient ranges from -16.55 to -13.66 and remains small-p in every fit. This
does not answer the inferential concern because the ROI was selected from the
group result. It must not be used to exclude sub-138 or any other participant.
After production provenance is established, the audit should explain whether
the installed FSL version supports robust FLAME outlier deweighting for this
FLAME 1+2 model. Such a complete-sample sensitivity analysis remains
author-pending and must not redefine the primary result.

## Established task and model details

- Two 72-trial runs yielded 144 trials per participant: 48 computer, 48
  age-similar human, and 48 age-dissimilar human trials.
- Trial duration averages 3.5176 s. Within-block gaps average 0.7650 s;
  between-block gaps are approximately 8, 10, or 12 s.
- The first-level activation and nPPI templates retain computer constants and
  offer parametric modulators. Contrasts 8 and 10 encode social-versus-computer
  offer modulation and constant effects, respectively.
- The submitted DMN group template includes younger, older, younger-minus-older,
  and older-minus-younger contrasts on each participant's similar-minus-
  dissimilar cope. The ECN sensitivity template includes between-group and
  within-group positive/negative sensitivity contrasts.
- The focal tracked masks contain 26 and 23 voxels and are on a 2.973 × 2.973 ×
  3.220 mm grid. The cited OpenNeuro task sidecar records 2.80-mm acquired slice
  thickness and 3.22-mm spacing between slices, an exact 15% spacing increment.
  Manuscript-facing language should distinguish approximately 2.97 × 2.97 mm
  in-plane resolution, 2.80-mm slice thickness, and 3.22-mm through-plane
  spacing/analyzed grid. Production headers must still confirm whether
  normalization changed any other dimension.
- All three focal files are 0/1 binary masks whose copied headers retain FSL's
  Z-score intent and `2203.12` build description. Although none is
  byte-identical to a labeled `cluster_mask_zstat` image, voxel-support tracing
  establishes that each is exactly one complete labeled FSL cluster. The
  26-voxel DMN mask is cluster 1 of zstat 3 in the original 47-participant
  cope-7 DMN group model, with identical cluster support retained in both
  analysis repositories. The 23-voxel ECN mask is cluster 1 of zstat 1 in the
  47-participant cope-7 group-specific sensitivity model. The 161-voxel
  activation mask is cluster 1 of zstat 1 in the 47-participant cope-7
  `norm2_logit` model.
- Every traced design uses FLAME 1+2 (`mixed_yn=1` in these FSFs), no automatic
  outlier deweighting, Z > 3.1, and cluster-corrected p = .05. Each contains 47
  unique participants and places sub-144 at input 35. Retained smoothness
  records report DLH 0.300799 and 15.5111 resels for DMN, DLH 0.331876 and
  14.0378 resels for ECN, and DLH 0.205028 and 22.7228 resels for activation,
  each over a 56,872-voxel search volume.
- The ECN output predates the later filename suffix and is stored under a
  `sensitivity2` directory. Git subsequently renamed its source template to
  `sensitivity2_logit` without changing the file. Its production design matrix,
  contrast, and group files exactly match the committed historical companions;
  the group-specific sensitivity EVs match the submitted logit covariate to
  numerical rounding. The activation production design files likewise exactly
  match the committed `norm2_logit` companions.

## Server pending items that must not be filled by inference

1. Retain the exact fMRIPrep 21.0.2 preprocessing boilerplate. The two affected
   sub-144 OpenNeuro BOLD files are byte-identical to their surviving
   `srndna-ug` copies, closing input identity for the repair. A whole-sample
   checksum inventory would strengthen general provenance but is not required
   to rerun sub-144.
2. Exact production BOLD and group-output headers.
3. A compact copy of the rendered production `design.fsf` files. ECN and
   activation `design.mat`, `design.con`, and `design.grp` identity is already
   established by exact historical checksums; the two retained DMN copies have
   identical matrix, contrast, and group checksums.
4. Residual smoothness, search volume/resels, cluster tables, peaks, corrected
   probabilities, and confirmation that no post-statistics ROI mask was used.
5. Corrected main effects, within-age simple effects, and social-versus-computer
   results from existing contrasts.
6. Remaining first-level design correlations and RT/offer-modulator
   estimability, including downstream provenance for both sub-143 runs. The
   source-event omissions are quantified, but no production-run subset is
   considered verified until the authoritative Linux audit is run.
7. Exact behavior of robust FLAME outlier deweighting in the production FSL
   version, including compatibility, settings, and diagnostic outputs; do not
   run it without author approval.
8. A statistical comparison of the submitted and unified sensitivity measures;
   do not build a new ECN L3 model without a subsequent scientific decision.
9. An inventory classifying main/simple/computer-effect requests as existing,
   descriptive, genuinely new, or not scientifically recommended.

Use `docs/SERVER_IMAGING_AUDIT.md` to collect the evidence. Do not run the
tracked `L3stats_SANS.sh`; its current FSF redirection is defective.

## Interpretation changes supported now

- Replace equivalence-like null language with failure-to-detect language and
  report estimates and confidence intervals.
- Replace effective connectivity with task-dependent functional connectivity.
- Describe the partner manipulation as ostensible human partners varying in
  age similarity; do not claim demonstrated closeness, ingroup identification,
  or belief.
- Replace compensation, reorganization, and adaptive-recalibration claims with
  descriptive age-related connectivity differences. Compensation may be named
  only as a future hypothesis.
- Describe DMN and ECN effects as small corrected regional clusters and include
  the full inference details once recovered.
- Present comparable behavior and differing connectivity as co-occurring
  observations, not evidence that connectivity preserved behavior.
