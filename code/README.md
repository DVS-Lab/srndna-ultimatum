# Analysis code

## Active revision entry points

- `validate_workflow.sh`: syntax, unit, contract, and path-portability checks.
- `audit_task_events.py`: task timing, missing trials, and source RT-event audit.
- `build_event_corrected_trials.py`: non-destructive overlay of recovered
  sub-144 trial labels onto the historical trial-order brain estimates; every
  nonbehavioral column is required to remain unchanged.
- `analyze_reviewer_behavior.R`: reviewer-requested mixed models, aggregate
  tables, and figures.
- `analyze_rating_choice_moderation.R`: exploratory associations between
  explicit ratings, the event-corrected behavioral fairness-sensitivity score,
  and trial-level choices; includes repeated-block sensitivity analyses and
  figure generation.
- `audit_image_headers.py`: focal mask checksums, grids, and volumes.
- `audit_l3_designs.R`: diagnostics for the focal original/candidate L3
  templates.
- `audit_l3_template_inputs.py`: recursive inventory of original and later L3
  template inputs, including sub-143.
- `audit_l3_production.py`: compare the three tracked paper masks with cluster
  masks in retained GFEAT trees, first by file hash and optionally by labeled
  voxel support, then trace matches back to their group design, participant
  inputs, inference settings, cluster table, and FSL residual-smoothness
  record.
- `audit_roi_influence.R`: descriptive selected-ROI influence check; it does
  not authorize participant deletion.
- `audit_server_rt_events.py`: read-only comparison of BIDS RT rows, 3-column
  EVs, and retained activation designs on Linux.
- `make_ultimatum_3col.py`: portable construction of the nine model-02 task,
  offer-modulator, miss, and RT EV files from either historical or corrected
  BIDS event layouts.
- `audit_ultimatum_event_designs.py`: lightweight `feat_model`-only matching of
  retained sub-143/sub-144 activation matrices to current and historical event
  candidates. It does not fit image data or modify retained FEAT directories.
- `make_fsl_confounds.py`: dependency-light, tested reproduction of the
  historical FSL nuisance-matrix selection and column order from fMRIPrep TSVs.
- `prepare_sub144_imaging_repair.py`: render six corrected sub-144 L1 FSFs and
  three dependent L2 FSFs in a new scratch tree using the retained production
  FSFs, existing nPPI time series, and companion-RT policy. If the historical
  headerless FSL confound files are absent, it regenerates them under scratch.
- `audit_prepared_ultimatum_designs.py`: run `feat_model` only on the six
  prepared L1 FSFs and require the nuisance-regressor tail to match the
  retained production design exactly while the corrected task block changes.
- `run_ultimatum_repair_jobs.py`: guarded, bounded-concurrency execution of a
  prepared L1, L2, or L3 repair manifest. It refuses to overwrite an existing
  output and verifies every expected cope or group Z statistic.
- `prepare_ultimatum_l3_repair.py`: render six scratch-only group jobs from
  the recovered production designs: three image-only identity repairs and
  three manuscript-aligned corrected-covariate variants. The latter replace
  task-wide mean RT in every focal model and also replace the fairness-
  sensitivity/norm-proxy EVs in the ECN/activation models. It consumes the
  calibrated minimal covariate table tracked under
  `results/reviewer/tables/`, so Linux does not need a working R installation
  to render the designs. The focal submitted masks are never inputs or
  overwrite targets.
- `audit_prepared_ultimatum_l3_designs.py`: compile the six scratch FSFs with
  `feat_model` only and require 47 rows, the expected EV rank, group rows, and
  contrast count before any group image fit.
- `collect_ultimatum_l3_repair_designs.py`: copy only the exact rendered FSF,
  design matrix, contrasts, and group files from the validated scratch manifest
  into a compact Git provenance bundle. It refuses a nonempty destination and
  never copies image payloads.
- `audit_l1_designs.py`: aggregate rank, conditioning, and column-correlation
  diagnostics for retained activation and nPPI design matrices.
- `audit_server_imaging.sh`: read-only collection of compact production
  provenance.
- `run_logged.sh`: timestamped execution wrapper for local revision workflows.

Use the root `Makefile` for local entry points and
`docs/SERVER_IMAGING_AUDIT.md` for the Linux handoff.
Repository-specific code provenance is recorded in
`docs/ANALYSIS_CODE_PROVENANCE.md`.

## Original production workflow

The historical `run_*` scripts loop over subjects and invoke their corresponding
preprocessing or FSL scripts. They preserve the original workflow and often
contain lab-specific paths. Do not assume they are safe revision entry points
or silently rewrite their paths; first match them to retained production
outputs.

The original sequence was:

1. convert/source BIDS data and run fMRIPrep;
2. generate confounds and FSL 3-column EV files;
3. run FSL L1, L2, and L3 analyses.

The completed production audit was non-mutating. Its collectors remain
non-mutating if repeated. The event-provenance matcher may run `feat_model` in
scratch space; do not run FEAT, FLAME, `randomise`, or a replacement image
model as part of validation.
