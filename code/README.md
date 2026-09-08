# Analysis code

## Active revision entry points

- `validate_workflow.sh`: syntax, unit, contract, and path-portability checks.
- `audit_task_events.py`: task timing, missing trials, and source RT-event audit.
- `analyze_reviewer_behavior.R`: reviewer-requested mixed models, aggregate
  tables, and figures.
- `audit_image_headers.py`: focal mask checksums, grids, and volumes.
- `audit_l3_designs.R`: diagnostics for the focal original/candidate L3
  templates.
- `audit_l3_template_inputs.py`: recursive inventory of original and later L3
  template inputs, including sub-143.
- `audit_roi_influence.R`: descriptive selected-ROI influence check; it does
  not authorize participant deletion.
- `audit_server_rt_events.py`: read-only comparison of BIDS RT rows, 3-column
  EVs, and retained activation designs on Linux.
- `audit_l1_designs.py`: aggregate rank, conditioning, and column-correlation
  diagnostics for retained activation and nPPI design matrices.
- `audit_server_imaging.sh`: read-only collection of compact production
  provenance.
- `run_logged.sh`: timestamped execution wrapper for local revision workflows.

Use the root `Makefile` for local entry points and
`docs/SERVER_IMAGING_AUDIT.md` for the Linux handoff.

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

The production audit must remain non-mutating. Do not run FEAT, FLAME,
`randomise`, or a replacement model as part of validation.
