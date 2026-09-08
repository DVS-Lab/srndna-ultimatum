# Curated source data

This directory contains the small, coded inputs needed by the active
resubmission analyses. It is intentionally **not** a complete BIDS dataset.

- `bids/` mirrors the participant/functional layout for the 47-person analysis
  sample's 94 Ultimatum Game `events.tsv` files only. It contains no MRI images.
- `bids/dataset_description.json` and `bids/task-ultimatum_bold.json` are the
  dataset/task sidecars from the cited OpenNeuro 2.0.2 snapshot (commit
  `88d8d458`). In particular, the task sidecar distinguishes 2.8-mm slice
  thickness from 3.22-mm spacing between slices.
- `partner_ratings/` contains only the available Bargaining partner-rating CSVs
  for that same analysis sample consumed by `code/analyze_reviewer_behavior.R`.

The full BIDS dataset belongs outside Git (locally under `bids/`, which is
ignored, or in the authoritative data repository). If analysis-relevant BIDS
JSON sidecars are later needed, copy only those small sidecars into this
curated tree and document them here. Do not copy NIfTI, DICOM, derivatives,
stimulus logs from unrelated tasks, or acquisition metadata with identifiers.

The retained participant identifiers are study codes. Participant-level
derived reviewer outputs remain ignored under `results/reviewer/private/`.
