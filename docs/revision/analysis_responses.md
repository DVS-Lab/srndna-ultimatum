# Responses to analytical and results concerns

UPDATE 1 OCTOBER 2026 — SUPERSEDED PRIMARY RT COVERAGE: the historical ACC
age result/figure below is not current publication evidence. Exact input hashes
confirm 786 omitted block-first RT events in the corrected baseline. The
completed all-trial RT DMN analysis has no significant age-difference clusters.
See docs/RT_COVERAGE_CORRECTION.txt (repository root) for the new complete-RT
activation/DMN/ECN batch and publication gates. Content below records earlier
work; do not submit its unupdated imaging claims or associated response DOCX.

First author review pass — 30 September 2026

These responses address the analytical and results concerns in the original reviews. The concern summaries below are paraphrases, not quotations or original comment numbers. Each response is followed by specific proposed manuscript text; none of these passages has yet been inserted into the manuscript. Broader framing, literature, acquisition-description, and editorial comments are reserved for a later pass.

## Correction identified during revision

The reviewer responses should be preceded by a short account of the correction that affects several results.

During the revision, we identified incorrectly assigned trial information in both Ultimatum Game event files for one participant, sub-144. We recovered that participant's task information, regenerated the affected first- and second-level imaging estimates, recalculated the behavioral covariates, and reran the group analyses. The corrected analysis retains the default mode network finding but does not retain the executive control network fairness-sensitivity finding. We report the corrected results throughout the responses below.

Proposed addition to Methods

> A provenance audit during revision identified incorrectly assigned trial information in the two Ultimatum Game event files for sub-144. The affected events were corrected using the recovered task records, and the corresponding activation and network-PPI estimates were regenerated. Behavioral covariates were recalculated before refitting the group models. The analysis sample remained unchanged at 47 participants.

## Reviewer 1

### Convergence of the behavioral interaction model

Concern summary: A failed fit does not test the three-way interaction. Could the random-effects structure be simplified while retaining that interaction?

Thank you for identifying this distinction. We refit the full interaction model after centering offer size and increasing the optimizer iteration limit, and the model converged without a singular fit while retaining participant-specific intercepts and offer-size slopes. A random-intercept-only model also converged and led to the same conclusion, so simplifying the random-effects structure was not necessary to test the interaction.

Proposed replacement in Behavioral Analyses and Results

> Acceptance was analyzed with a logistic mixed-effects model containing centered offer size, age group, partner similarity, and all interactions, with participant-specific intercepts and offer-size slopes. The model converged without a singular fit in 47 participants contributing 4,438 responded human-partner trials. The Offer Size × Age Group × Partner Similarity interaction was not detected (β = 0.1136, SE = 0.0947, z = 1.20, p = .230, 95% CI [−0.0720, 0.2991]). Three of five tested optimizers converged cleanly and gave closely agreeing estimates; two retained convergence warnings. A random-intercept-only sensitivity model yielded the same inferential conclusion (β = 0.0684, SE = 0.0915, p = .455).

### Power and interpretation of null results

Concern summary: Given the limited sensitivity to smaller effects, can the manuscript conclude that the groups did not differ?

We appreciate the reviewer's emphasis on what a null result can establish. The sensitivity calculation does not demonstrate equivalence, and the corrected interaction estimate remains too imprecise to exclude smaller age-related differences. We will replace language implying an absence of differences with the estimate, its uncertainty, and a statement that the interaction was not detected.

Proposed replacement in Results and Discussion

> The behavioral three-way interaction was not detected (β = 0.1136, 95% CI [−0.0720, 0.2991], p = .230). This result does not establish equivalent behavioral responses across age groups. The modest sample limits precision, particularly for interactions, and effects smaller than those detectable with the available sample remain possible.

### Whole brain correction and the small clusters

Concern summary: Were the reported clusters corrected across the whole brain, and what smoothness and search volume entered that correction?

Thank you for asking us to document the inference more fully. We traced the analyses to their retained FSL designs, search masks, and cluster tables, which confirm whole-brain FLAME 1+2 inference rather than correction within a selected ROI. After correcting the participant-level events and group covariates, the DMN cluster survives, whereas the ECN finding does not.

Proposed replacement in Imaging Analyses

> Group analyses used FSL FLAME 1+2, a cluster-forming threshold of Z > 3.1, and whole-brain cluster-level family-wise error correction at p < .05. No selected ROI was used to restrict cluster inference. For the corrected DMN age-group contrast, the search mask contained 56,907 voxels and estimated smoothness was 7.72 × 7.75 × 7.40 mm FWHM (DLH = 0.299801). FSL's RESELS value was 15.5628 voxels per resel; this value describes resel size, not the total number of resels in the search volume. The analyzed grid was approximately 2.973 × 2.973 × 3.220 mm, distinct from the acquired 2.80-mm slice thickness with a 15% interslice gap.

Proposed replacement in Results

> Younger adults showed greater similar-minus-dissimilar offer-modulated DMN connectivity than older adults in a 29-voxel anterior cingulate cluster (cluster-corrected p = .0138, Zmax = 4.07, peak MNI [−13.3, 34.0, 27.8]). No clusters survived correction in any of the eight contrasts of the corrected ECN fairness-sensitivity model. The previously reported positive ECN finding is therefore withdrawn.

These additions specify the search space, smoothness, and corrected probability supporting the DMN result. They also distinguish that retained finding from the ECN result that no longer survives correction.

### Separate and unified fairness sensitivity models

Concern summary: Why estimate partner-specific slopes separately instead of extracting participant Offer × Similarity slopes from a unified model?

Thank you for suggesting this comparison. The separate models estimated the offer–acceptance slope within each partner condition, whereas the unified model estimates the partner difference within one jointly fitted model; their participant estimates need not coincide because the models pool information differently. We fit the unified model as a sensitivity analysis and found related, but not interchangeable, participant scores.

Proposed addition to the supplementary Behavioral Analyses

> The original fairness-sensitivity score was the participant's estimated offer–acceptance slope for similar partners minus the corresponding slope for dissimilar partners, obtained from separate condition-specific mixed models. As a sensitivity analysis, we also fitted a unified Offer Size × Partner Similarity model. A correlated random-effects specification was singular; an uncorrelated specification converged without singularity. Its fixed interaction was β = −0.0342 (SE = 0.0622, p = .582, 95% CI [−0.1561, 0.0877]). Participant interaction slopes correlated r = .669 with the corrected separate-model score. The corrected ECN analysis retained the original score definition and did not yield significant clusters; the unified score was not substituted to recover that finding.

### Trial duration and postresponse processing

Concern summary: Does the fixed trial duration conflate decision processing with the period after the response, and can RT adjustment rule out this explanation of the DMN effect?

We thank the reviewer for highlighting this ambiguity. The offer, partner, and selected response remained visible until the end of the trial, so the modeled epoch includes both decision-related and postresponse processing. RT adjustment does not separate those components, and we will narrow the interpretation accordingly rather than claim that it rules out a contribution from postresponse activity.

Proposed clarification in Imaging Analyses

> Each task epoch included approximately 1 s of offer presentation followed by a 2.5-s response display. After a response, the offer, partner, and selected choice remained visible until the epoch ended. Task regressors therefore modeled the full display epoch rather than the participant's deliberation duration. The focal model repair preserved the submitted RT-regressor construction, whose companion-event series omits first trials of blocks. Corrected task-wide mean RT was included at the group level. The separately added activation workflow derived RT regressors from substantive decision rows.

Proposed addition to Discussion

> The fixed-duration model cannot distinguish decision-related connectivity from connectivity during the remainder of the display. Although RT was included as a covariate, these adjustments do not establish that the DMN difference reflects deliberation rather than postresponse processing.

### Inclusion of the computer condition

Concern summary: Why were the computer trials excluded, and can the analyses distinguish social processing from general task engagement?

Thank you for pointing out that our description obscured the role of the computer trials. Those trials were included in the first-level imaging models as separate task and offer-size regressors; the submitted group questions focused on the two human partners. We also examined direct human-versus-computer contrasts during revision, but their interpretation is limited by differences in the partner images.

Proposed replacement in Imaging Analyses and Results

> First-level models included separate task and offer-size regressors for computer, age-similar human, and age-dissimilar human trials. The principal partner-similarity contrast compared the two human conditions. During revision, we additionally tested the equal-weighted mean of the three offer-size slopes and direct human-versus-computer contrasts. The unmodulated human-versus-computer comparisons yielded five positive and two reverse-direction clusters, predominantly involving posterior visual and ventral-temporal regions. The human-greater-than-computer offer-slope contrast yielded one 32-voxel periventricular cluster (cluster-corrected p = .023, Zmax = 3.86). These effects are reported as exploratory; neither the stimulus differences nor the boundary localization supports a specific social-cognitive interpretation.

### Ratings and the partner manipulation

Concern summary: Did participants perceive the two human partners as different in similarity or believe the partner cover story?

We appreciate the distinction between presenting different partners and demonstrating perceived similarity. We recovered and audited the partner ratings, which distinguish humans from the computer on fairness and likeability but do not establish a similar-versus-dissimilar difference. Because the ratings did not ask about perceived similarity or belief in the cover story, they cannot serve as direct validation of either assumption.

Proposed addition to Behavioral Results and limitations

> Exploratory ratings analyses retained the final complete attempt when an administration was repeated, applying the same curation rule across tasks. Ratings were available for 42 participants before the task and 42 afterward, with 41 assessed at both timepoints. The mean of the two human partners was rated fairer and more likeable than the computer at both timepoints (all four within-family FDR q < .0062). No similar-versus-dissimilar difference was detected in pre-task fairness or likeability, or post-task fairness, likeability, anger, or satisfaction (all six unadjusted p ≥ .243). These measures assessed partner evaluations, not perceived similarity or belief that the partners were real.

### Short intertrial intervals and model separability

Concern summary: How did the fixed 0.75-s interval permit separation of the modeled responses?

Thank you for drawing attention to the distinction between within-block and between-block timing. The 0.75-s interval occurred within eight-trial blocks, whereas blocks were separated by randomized intervals of approximately 8, 10, or 12 s. This timing supports a mixed block and event design, but it does not by itself establish that all first-level regressors were well separated.

Proposed clarification in Experimental Design

> Trials were separated by 0.75 s within each eight-trial block; blocks were separated by randomized intervals of approximately 8, 10, or 12 s. Partner identity was constant within a block, whereas offer size varied across trials and was modeled parametrically. The design therefore combined block-level partner context with trial-level offer variation, rather than using a widely jittered interval between every pair of trials.

## Reviewer 2

### Behavioral coding and missed trials

Concern summary: How were age and similarity coded, how many trials were analyzed, and did missed trials differ by age?

Thank you for requesting these details. Age was modeled as a categorical group, not age in years, and partner similarity compared age-similar with age-dissimilar human partners. We also quantified missed trials in the full imaging sample so that their frequency and age distribution can be reported explicitly.

Proposed replacement in Behavioral Analyses and Results

> The sample contained 25 younger and 22 older adults. Age group was coded categorically, with younger adults as the reference, and partner similarity used age-dissimilar partners as the reference; offer size was centered. Each participant completed two runs of 72 trials, with 48 trials per partner across runs. Of 6,768 trials, 114 were missed (1.68%): 37 among younger adults and 77 among older adults. Mean missed-trial counts were 1.48 (SD = 3.55) and 3.50 (SD = 4.45), respectively; the younger-minus-older difference was −2.02 trials (95% CI [−4.41, 0.37], Welch p = .096). Missed trials were excluded from responded-trial behavioral analyses and represented separately in the imaging models.

### Network PPI and adjustment for other networks

Concern summary: What does a distributed network PPI estimate, and why include the other networks as covariates?

We appreciate the request to make the target of the analysis more explicit. Network PPI was used to test coupling associated with a distributed network signal, not to identify a particular causal node or assume that every region has a fixed network affiliation. Including the other network time courses adjusts for their shared signal, but it does not remove the limitations of a spatially distributed seed.

Proposed replacement in Imaging Analyses

> Network time courses were extracted by spatial regression against the original signed, continuous network maps. Each first-level network-PPI model included the target network time course and the remaining nine network time courses, together with task and interaction regressors. These terms account for signal shared with the other modeled networks. The resulting PPI describes task-dependent statistical coupling with the modeled network signal; it does not establish directional or effective connectivity. Because the maps describe canonical networks, the estimate also does not establish that every constituent region contributes equally or maintains the same affiliation during the task.

### RT and image quality covariates

Concern summary: Why adjust for RT at both levels, and why include both tSNR and framewise displacement?

Thank you for asking us to distinguish the roles of these covariates. The first-level RT terms address trial-related variation, whereas the group RT covariate addresses between-participant differences; these are different adjustments, although neither separates deliberation from postresponse processing. Temporal signal-to-noise ratio and framewise displacement measure different aspects of image quality, so we retained both rather than change the nuisance model after inspecting the results.

Proposed clarification in Imaging Analyses

> The corrected group models included event-derived task-wide mean RT to account for between-participant latency differences, separately from the first-level RT regressors described above. The group models also retained sex, temporal signal-to-noise ratio, and mean framewise displacement. Temporal signal-to-noise ratio characterizes temporal signal stability, whereas framewise displacement measures head motion. The corrected DMN group design had six columns and full rank; the corrected ECN model had eight columns and full rank. Retaining both image-quality covariates therefore did not create exact group-level collinearity.

### Main effects of activation

Concern summary: Were there activity main effects, and can these be reported even though the original focus was the interaction?

Thank you for encouraging us to report the broader task response. We added a group analysis of the mean offer-size slope across all three partner conditions, which identifies activation that increases with offer size after covariate adjustment. This analysis was added during revision and will be distinguished from the submitted interaction tests.

Proposed addition to Imaging Analyses and Results

> A revision analysis tested the equal-weighted mean of the computer, age-similar, and age-dissimilar offer-size slopes. The group model included an intercept, centered age group and sex, temporal signal-to-noise ratio, mean framewise displacement, and corrected task-wide mean RT. Five clusters showed positive modulation at Z > 3.1 and cluster-corrected p < .05 (348 voxels in total; maximum Z = 4.75; cluster p values from 2.56 × 10⁻⁶ to .0453; Figure 3). No negative clusters survived. The exploratory younger-greater-than-older contrast yielded three clusters totaling 125 voxels; no clusters survived in the reverse direction. These offer-size effects do not separate perceived fairness from the economic value of the offer.

### Number of regressors and alternative models

Concern summary: Could the number of regressors obscure the effects, and were models without parametric modulators tested?

We thank the reviewer for raising the question of model estimability. Removing the offer-size modulators would remove the effect under study rather than provide another estimate of it, so we did not use that reduced model as a replacement. We inspected the retained first-level matrices, but the aggregate rank flags require further attribution before they can support a claim that the relevant contrasts were unaffected.

Proposed clarification in Imaging Analyses

> Task-response regressors modeled each partner condition, and separate parametric regressors modeled variation with offer size across the full offer range. Thus, the imaging analyses were not restricted to unfair offers. The offer-modulated contrasts test different quantities from the unmodulated task-response contrasts; a model omitting the offer modulators was not substituted for the primary analysis.

### Simple effects underlying the DMN interaction

Concern summary: Please show the within-age effects that underlie the interaction and provide an interpretable figure.

Thank you for suggesting this decomposition. We examined the within-age contrasts from the corrected group model and extracted condition-specific estimates for the revised DMN figure. The younger-adult positive contrast survived whole-brain correction, but the difference between one significant simple effect and one nonsignificant simple effect is not itself the test of the age interaction.

Proposed addition to Results and the Figure 4 caption

> The younger-adult similar-minus-dissimilar contrast yielded five whole-brain corrected clusters; the corresponding older-adult positive contrast yielded none. The age-group difference was tested directly and retained the 29-voxel anterior cingulate cluster reported above. Figure 4 displays separate similar- and dissimilar-partner estimates for each age group from condition-specific FLAME models using the corrected group design. Bar heights are cluster means of the group estimates, and error bars show the cluster mean of voxelwise model standard errors. The bars are descriptive because the cluster was selected by the age-group contrast; they are not independent ROI tests or empirical participant SEMs. No ECN simple-effect interpretation is advanced because its corrected model yielded no surviving clusters.

This decomposition makes the condition estimates visible while preserving the direct group contrast as the inferential test. It does not use the selected-cluster display to provide a second test of the same finding.

### A potentially influential older participant

Concern summary: Is the older participant visible in the original ROI plot an outlier, and does the result survive that participant's removal?

Thank you for pointing out the potentially influential observation. We have not excluded a participant on the basis of a plot extracted from the same selected cluster, and our earlier descriptive influence calculations used the submitted ROI values rather than the corrected result. Those calculations therefore do not establish that the corrected whole-brain finding survives participant removal, and we will not describe them as such.

Proposed clarification in Results

> The corrected group analysis retained all 47 participants. No participant was excluded because of their position in the selected-cluster plot. The condition estimates in Figure 4 are descriptive, and the reported whole-brain result should not be interpreted as demonstrating robustness to removal of every individual participant.

### Fairness sensitivity across age groups

Concern summary: Please show the sensitivity distribution separately from the imaging result and test the age-group difference.

We appreciate this suggestion because the behavioral measure should be interpretable independently of the imaging result. We summarized the corrected score by age group and prepared its distribution separately from the selected neural clusters. The age-group difference was not detected, but its interval remains compatible with differences in either direction.

Proposed addition to supplementary Behavioral Results

> The fairness-sensitivity score was the similar-minus-dissimilar difference in estimated offer–acceptance slopes. Positive scores indicate a steeper offer–acceptance relationship for similar partners; negative scores indicate the reverse. The mean score was −0.0209 (SD = 0.4930) in younger adults and 0.0770 (SD = 0.6896) in older adults. The younger-minus-older difference was −0.0979 (95% CI [−0.4564, 0.2607], Welch p = .584; Hedges' g = −0.16, approximate 95% CI [−0.73, 0.40]).

### Fairness sensitivity and acceptance probability

Concern summary: How does the sensitivity score affect acceptance, and do independent fairness evaluations help explain the choices?

Thank you for asking us to connect the measure more clearly to behavior. Because the sensitivity score is derived from the offer–acceptance slopes in these same choices, an association with those choices would not provide independent validation of the score. We instead tested whether pre-task partner ratings moderated the offer-by-similarity effect, but did not detect the predicted moderation.

Proposed addition to supplementary Behavioral Results

> Exploratory choice models tested whether pre-task similar-minus-dissimilar ratings moderated the Offer Size × Partner Similarity effect. The random-intercept and random-offer-slope models included 42 participants and 3,969 responded human-partner trials. Moderation was not detected for fairness ratings (β = 0.0097, SE = 0.0582, p = .868, 95% CI [−0.1045, 0.1238]) or likeability ratings (β = −0.0099, SE = 0.0475, p = .834, 95% CI [−0.1031, 0.0832]). Their corresponding age interactions were also not detected (p = .203 and .622). The behavioral sensitivity score itself was not detectably correlated with the similar-minus-dissimilar fairness-rating difference before (r = .093, p = .559) or after the task (r = .032, p = .840).

### Behavioral relevance of the connectivity result

Concern summary: Does the neural difference explain acceptance behavior or support an interpretation of preserved behavior?

We thank the reviewer for emphasizing that a connectivity difference does not by itself explain behavior. The corrected ECN brain–behavior finding does not survive, and the retained DMN group contrast is not a trial-level test of whether connectivity predicts acceptance. We will therefore remove the claim that the connectivity difference preserves behavior rather than infer a mechanism from the coexistence of the two results.

Proposed replacement in Discussion

> The age-group difference in offer-modulated DMN connectivity and the behavioral findings are separate observations. The present analyses do not establish that this connectivity difference predicts acceptance, compensates for declining efficiency, or preserves behavior in older adults. Establishing such a relationship would require a specifically designed brain–behavior analysis with appropriate separation between effect selection and evaluation. The corrected ECN result provides no support for the previously proposed fairness-sensitivity account.
