# Ultimatum Game revision: work completed and findings

**Status: 30 September 2026.** The core reanalyses are complete and organized.
The manuscript has **not** been revised, and the response to reviewers still
needs to be drafted. Existing response/checklist material is preparatory only.
OpenNeuro publication is separately awaiting resolution of its server-side
checkout failure.

## What we did

### Recovered the analysis provenance and organized the repository

Established `DVS-Lab/srndna-ultimatum` as the paper's repository of record.
Recovered relevant work from the other SRNDNA repositories, separated legacy
material from active revision scripts, and retained original templates as
historical evidence. The revision templates, compact design outputs, results,
figure source data, and reproducible entry points now live together. Large
imaging derivatives remain outside Git.

Traced the reported group results back to the retained first-/second-level
inputs, participant lists, model matrices, contrasts, network maps, and
preprocessing metadata. The paper sample is 47 participants: 25 younger and
22 older adults. The retained derivative metadata identify fMRIPrep 21.0.2;
the audited FSL environment is 6.0.7.17. The network inputs are the original
signed continuous maps, not newly substituted binary masks.

### Corrected the participant-level problem and propagated it

Recovered sub-144's correct Ultimatum task information and corrected its two
event files. Verified that missing duplicate `event_RT` rows did not mean
sub-143's substantive task/RT information was missing, and that its retained
production task design matched its own trials. Sub-143 did not require the
same event-identity repair.

Regenerated sub-144's affected activation and DMN/ECN first- and second-level
outputs. Recomputed the behavioral quantities used as group covariates,
including event-corrected task-wide mean RT, and reran the focal group models.
Models with only corrected images but old covariates remain provenance checks,
not alternative reportable endpoints.

### Refit behavior and completed the revision-added analyses

Recovered a converged, nonsingular choice model with participant-specific
intercepts and offer-size slopes; checked optimizer agreement, random-effects
sensitivity, missed trials, response times, and the fairness-sensitivity score.

Built and ran the task-wide offer-size activation analysis, including its
positive/negative mean and age contrasts. Also ran human-versus-computer
activation and offer-slope follow-ups. Prepared the four-figure set and compact
source tables. These analysis additions must be labeled as exploratory work
added during revision, not retrospectively presented as original hypotheses.

Audited the ratings scripts, response scales, participant provenance,
pre-/post-task timing, and repeated attempts. Applied the author-approved
final-complete-attempt rule across all tasks. Analyzed the Ultimatum ratings,
their associations with behavioral fairness sensitivity, and their direct
moderation of trial-level choices.

## What we learned

| Question | Answer supported by the corrected analyses |
| --- | --- |
| Does offer size predict acceptance? | Strongly: beta=1.8013, SE=.2373, p=3.17e-14. |
| Is the behavioral Offer × Age × Similarity interaction established? | No: beta=.1136, SE=.0947, p=.230, 95% CI [-.0720,.2991]; 47 participants and 4,438 human-partner trials. This is not evidence of equivalence. |
| Does the DMN finding survive? | Yes under the corrected specified model: younger>older similar-minus-dissimilar offer-modulated connectivity, 29 ACC voxels, cluster p=.0138, Zmax=4.07. |
| Does the ECN fairness-sensitivity finding survive? | No clusters survive in any of eight corrected contrasts. Withdraw the positive claim and figure, but report the corrected null transparently. |
| Is there a task-wide offer-size activation result? | Five positive clusters, 348 voxels total, maximum Z=4.75; no negative clusters. This supports new Figure 3, preceding the DMN figure. |
| Did the additional imaging contrasts yield a clean replacement social effect? | No. Human>computer task responses are confounded with partner-image differences; the 32-voxel social offer-slope cluster is periventricular and not a clean mechanistic finding. Keep the audit, not a new headline claim. |
| Do ratings distinguish the partners? | Humans were rated fairer and more likeable than the computer before and after the task (FDR q<.0062). Similar versus dissimilar humans were not detectably different on the six measured trait/timepoint contrasts (unadjusted p>=.243). |
| Do ratings explain the similar-versus-dissimilar choice effect? | Direct moderation was not detected for pre-task fairness (p=.868) or likeability (p=.834), nor their age interactions (p=.203/.622); 42 participants, 3,969 trials. |
| Are ratings related to the behavioral sensitivity score? | Not detectably for fairness ratings. Exploratory rank associations occur for pre-task likeability and post-task anger (both q=.037), but Pearson tests do not survive correction (both q=.098). These do not establish choice moderation or an age-specific mechanism. |

Ratings coverage within the paper sample is 42 pre-task and 42 post-task
administrations, with 41 paired participants. Sub-143 has no ratings source;
sub-144's ratings have separate, participant-consistent provenance. Missing
ratings have not been imputed.

The task-wide offer-size age contrast also yielded three younger>older
clusters (125 voxels total), with no reverse-direction clusters. It remains an
exploratory supplementary candidate, not a replacement for a failed submitted
claim. The activation/norm-proxy result remains in the audit record.

## Figures already prepared

1. Task schematic.
2. Corrected behavioral acceptance predictions.
3. Task-wide positive offer-size activation (added during revision).
4. Corrected DMN–ACC result and descriptive condition estimates.

Figure 4 bars are covariate-adjusted, variance-aware FLAME estimates. Their
error bars are the mean of voxelwise model standard errors within the selected
cluster, **not empirical participant SEM**. They describe the selected effect;
inference comes from the whole-brain model, not an independent test of the bars.

## What this final audit established—and what it did not

- All 78 repository tests pass. All five ratings-analysis tables reproduced
  from the source audit. All 84 selected paper-sample administrations (756
  responses) exactly match the prepared BIDS behavior files.
- Updated the stale documentation that still called the repeat-selection rule
  provisional and still used the earlier exclusion-only sample counts.
- Corrected a smoothness-reporting unit error: FSL's `RESELS` field denotes
  resel size, not total search-volume resel count. The underlying FSL results
  did not change. Details and the FMRIB reference are in `REVISION_STATUS.md`.
- Identified the earlier Cook's-distance/leave-one-out calculations as
  diagnostics of the **submitted** ROI plot, not the corrected analysis.
  Corrected image-level leave-one-out robustness has not been established.
- The minimal focal repair preserved the submitted companion-RT construction;
  the new activation workflow derives RT regressors from substantive trials.
  An all-participant alternative focal RT model was not run. This distinction
  needs explicit Methods wording rather than an assertion that every model
  was rebuilt identically from scratch.
- No new FSL fits or trial-level GLMM fits were run during this final audit.
  Completed imaging/GLMM outputs were inspected and tested; the descriptive
  ratings analyses were rerun locally.

## Parallel data-paper/OpenNeuro work

The separate data-paper workflow prepared four corrected sub-144 event files
(two Ultimatum and two Shared Reward), rebuilt 218 unsmoothed Trust single-trial
run images plus four affected sub-144 images, and exported 220 ratings
acquisitions with their JSON dictionaries and provenance. Metadata and
acknowledgment updates are included in the release preparation. Trust event
rows are not part of the four-file event repair.

All 222 replacement annex objects were verified remotely in the reported
transfer audit, but the OpenNeuro `main` update was rejected by its server
working-tree checkout. Thus the prepared update is not yet a verified public
2.2.0 snapshot. Do not describe it as released in the manuscript.

## Next phase

Start the manuscript and response together from this result set, using the
current editable manuscript and original reviewer comments. Retain the DMN
finding with corrected numbers; remove the positive ECN interpretation but
acknowledge its corrected null; include the revision-added activation result;
and use ratings as qualified exploratory context. Clearly distinguish null
findings from equivalence and avoid compensation or causal preservation claims.

The main remaining work is writing and author review, not a wholesale imaging
rerun. Specific limitations above must be addressed honestly; any additional
robustness model should be a named supplementary analysis. Figure aesthetics,
supplement scope, page/line references, final code freeze, and the published
OpenNeuro version remain to be finalized.

Evidence entry points: `REVISION_STATUS.md`, `REVISION_FINDINGS.md`,
`results/manuscript/tables/final_result_set.tsv`,
`results/reviewer/ultimatum_ratings/README.md`, and
`results/manuscript/README.md` (paths relative to the repository root except
the adjacent documents in this directory).
