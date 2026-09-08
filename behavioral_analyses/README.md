# Behavioral revision inputs

This directory contains only the compact cleaned inputs needed by the active
reviewer-response workflow. The submitted per-trial behavioral/neural table
remains in `code/MLM/` for historical provenance.

- `data/all_trials_brains.csv`: cleaned trial-level behavioral table used by
  `code/analyze_reviewer_behavior.R`.
- `data/participant_L3_47.csv`: the 47-participant analysis sample and group
  assignments.
- `data/in_out_sensitivity_indiv_logit.csv`: submitted separate-model
  participant sensitivity score used for exact reproduction and comparison.

Active revision code lives in `code/`. Generated participant-level reviewer
tables are written to the ignored `results/reviewer/private/` directory.
