# Repository structure audit

This inventory reflects the corrected repository of record. No top-level move
should occur until the production imaging audit has resolved exact template,
mask, and extraction dependencies.

| Directory | Purpose | Manuscript/revision dependency | Status | Referenced by active code? | Recommended action |
| --- | --- | --- | --- | --- | --- |
| `behavioral_analyses/` | Compact cleaned inputs for reviewer models | Required for behavioral models and 47-person sample | Active | Yes | Keep; do not import the unrelated RL/DDM trees from `srndna-ug` |
| `code/` | Original preprocessing/FEAT code plus portable reviewer audits | Required | Mixed active/historical | Yes | Keep; classify legacy scripts after production provenance is complete |
| `derivatives/` | Original compact single-trial and plotting text derivatives | Potential provenance and figure inputs | Historical/production support | Some original code and analyses | Keep in place during revision; assess participant-level release policy later |
| `docs/` | Scientific audit and handoff documentation | Required for revision | Active | Yes | Keep |
| `imaging_plots_SANS/` | Compact submitted ROI/plot extracts recovered from later working tree | Supports focal-result checks and Figure 3 diagnostic | Active, provenance pending | Yes | Keep until production comparison; then consider `results/submitted/roi_extracts/` |
| `logs/` | Durable aggregate provenance records; ignored runtime captures | Required for audit traceability | Active | Yes | Keep |
| `masks/` | Original network, ROI, and seed masks | Production workflow dependency | Active/historical | Yes | Keep unchanged pending path audit |
| `masks_SANS/` | Three focal submitted-result masks recovered from later working tree | Required for low-cost cluster/grid verification | Active, provenance pending | Yes | Keep until matched to production outputs; then consider `masks/submitted/` |
| `results/` | Aggregate reviewer tables and figures | Required for response preparation | Active | Yes | Keep; participant-level rows remain ignored |
| `source_data/` | Public events TSVs and coded partner ratings used by reviewer analyses | Required | Active | Yes | Keep narrow; never add MRI payloads |
| `templates/` | Original production templates plus isolated later candidates | Required for production comparison | Active/provenance | Yes | Preserve originals; do not promote later candidates until rendered designs match |
| `tests/` | Fast static and synthetic checks | Required for safe maintenance | Active | Yes | Keep |

Directories discussed in the earlier cleanup brief (`SANS2025/`,
`imaging_plots_reg1/`, `masks_network/`, `masks_neurosynth/`, and `masks_reg1/`)
belong to the separate `srndna-ug` history and are not present here. They should
not be imported unless a documented manuscript dependency is found.

## Before resubmission

1. Complete the read-only production imaging audit.
2. Match original and later candidate L3 templates to rendered production
   designs.
3. Update the manuscript code-availability link to this repository.
4. Decide, with author approval, whether the compact single-trial text
   derivatives are appropriate for public retention.

## Post-publication cleanup only

- Consolidate focal masks/extracts after all active paths are known.
- Archive or label historical production wrappers that are not safe entry
  points.
- Add folder-level licensing clarification if code and data require different
  terms. The existing root license is preserved; no new license is inferred.
