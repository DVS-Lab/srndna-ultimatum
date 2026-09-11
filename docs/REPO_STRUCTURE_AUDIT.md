# Repository structure audit

This inventory reflects the corrected repository of record after completion
of the production imaging audit. Exact template, design, focal-mask, and
extraction dependencies are now traced. Historical material remains in place
where moving it would obscure provenance or break the original workflow.

| Directory | Purpose | Manuscript/revision dependency | Status | Referenced by active code? | Recommended action |
| --- | --- | --- | --- | --- | --- |
| `behavioral_analyses/` | Compact cleaned inputs for reviewer models | Required for behavioral models and 47-person sample | Active | Yes | Keep; do not import the unrelated RL/DDM trees from `srndna-ug` |
| `code/` | Original preprocessing/FEAT code plus portable reviewer audits | Required | Mixed active/historical | Yes | Keep; active revision entry points are classified in `code/README.md` |
| `derivatives/` | Original compact single-trial and plotting text derivatives | Potential provenance and figure inputs | Historical/production support | Some original code and analyses | Keep in place during revision; assess participant-level release policy later |
| `docs/` | Scientific audit and handoff documentation | Required for revision | Active | Yes | Keep |
| `imaging_plots_SANS/` | Compact submitted ROI/plot extracts recovered from later working tree | Supports focal-result and influence checks | Historical/submitted provenance | Yes | Keep in place because checksums and documented paths now trace these extracts to the submitted results |
| `logs/` | Durable aggregate provenance records; ignored runtime captures | Required for audit traceability | Active | Yes | Keep |
| `masks/` | Original continuous network maps plus ROI and seed masks | Production workflow dependency | Active/historical | Yes | Keep unchanged; the production script's network-number mapping is established |
| `masks_SANS/` | Three focal submitted-result masks recovered from later working tree | Required for submitted-result provenance | Historical/submitted provenance | Yes | Keep; all three masks exactly match one labeled cluster's voxel support in retained production outputs |
| `results/` | Aggregate reviewer tables and figures | Required for response preparation | Active | Yes | Keep; participant-level rows remain ignored |
| `source_data/` | Public events TSVs and coded partner ratings used by reviewer analyses | Required | Active | Yes | Keep narrow; never add MRI payloads |
| `templates/` | Original production templates plus canonical resubmission templates | Required | Active/provenance | Yes | Preserve originals; use only documented templates under `templates/revision/` for corrected analyses |
| `tests/` | Fast static and synthetic checks | Required for safe maintenance | Active | Yes | Keep |

Directories discussed in the earlier cleanup brief (`SANS2025/`,
`imaging_plots_reg1/`, `masks_network/`, `masks_neurosynth/`, and `masks_reg1/`)
belong to the separate `srndna-ug` history and are not present here. They should
not be imported unless a documented manuscript dependency is found.

## Before archival release

1. Publish the corrected OpenNeuro snapshot and record its identifier in the
   repository and manuscript.
2. Update the manuscript code-availability link to this repository.
3. Create a versioned software release and archive it after manuscript metadata
   are final.

## Post-publication cleanup only

- Add folder-level licensing clarification if code and data require different
  terms. The existing root license is preserved; no new license is inferred.
