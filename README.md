# SRNDNA Ultimatum Game

Repository of record for **Older Adults Show Altered Default Mode and Executive
Control Network Connectivity during Fairness Decisions**
([bioRxiv DOI 10.1101/2025.08.13.670194](https://doi.org/10.1101/2025.08.13.670194)).

This repository preserves the original production analysis history and now
also contains the compact reviewer-response workflow. MRI source data are
available from OpenNeuro dataset `ds003745`; MRI images and FEAT directories
are not duplicated in Git.

## Reproducible entry points

Run from the repository root:

```bash
make test
make reviewer-behavior
make reviewer-imaging-audit
```

The behavioral workflow requires R with `lme4`, `lmerTest`, `ggplot2`, and
`scales`. Image-header checks require FSL's `fslhd` and `fslstats`; validation
skips them cleanly when FSL is unavailable.

The production imaging audit is read-only and must run on Linux against
`/ZPOOL/data/projects/srndna-ultimatum`. Follow
`docs/SERVER_IMAGING_AUDIT.md`; do not launch FEAT, FLAME, or `randomise` as
part of that audit.

## Repository map

- `code/`: original preprocessing/FEAT scripts and active revision audits.
- `templates/`: original production templates. Later files recovered from the
  separate working repository are isolated under `templates/later_working_tree/`
  until compared with rendered production designs.
- `derivatives/`: compact historical text derivatives retained by the original
  repository; large local FSL/fMRIPrep outputs remain ignored.
- `masks/`: original network, ROI, and seed masks.
- `masks_SANS/` and `imaging_plots_SANS/`: focal submitted masks and compact
  plotting extracts used by the reviewer audit.
- `source_data/`: curated public events TSVs and partner-rating inputs only.
- `behavioral_analyses/data/`: compact cleaned inputs used by the active
  reviewer models.
- `results/reviewer/`: aggregate revision tables and figures.
- `logs/records/`: durable provenance records; participant-level audit output
  and server captures are ignored.
- `tests/`: fast static and synthetic workflow checks.

Start with `code/WORKFLOW_AUDIT.md`. The machine-readable result index is
`logs/records/manuscript-result-manifest.tsv`.

## Data and provenance boundaries

OpenNeuro `ds003745` is the source for the public BIDS dataset. The manuscript
cites snapshot 2.0.2. The local root-level `bids/` directory is intentionally
ignored; only small analysis-support TSV/JSON files belong under
`source_data/`.

The Linux checkout at `/ZPOOL/data/projects/srndna-ultimatum` is both a clean
checkout of this repository and the location of the production derivative
tree. Historical absolute paths inside FSFs are provenance and should not be
silently rewritten. New audit scripts accept explicit roots or derive paths
from the checkout.

The separate `DVS-Lab/srndna-ug` repository contains later working material
but is not the repository of record. Its unrelated history is not merged here;
only paper-relevant, compact artifacts are imported with their status made
explicit. See `docs/REPOSITORY_OF_RECORD.md`.

The analyses were not preregistered. Reviewer-requested work is labeled as
revision analysis and is not presented as part of the submitted workflow.

The revised Figure 3 decomposes the retained default-mode-network contrast
into four age-group-by-partner bars. Bar heights are cluster-mean group COPE
estimates from corrected condition-specific FLAME 1+2 models, so the display
retains FLAME's within- and between-participant variance modeling. The
participant-level condition COPE and VARCOPE extracts and all image hashes are
tracked under `results/manuscript/source_data/` for auditability. Figure error
bars show one model standard error rather than 95% confidence intervals.

## Acknowledgments

This work was supported in part by NIH awards R21-MH113917 and R03-DA046733 to
David V. Smith, R15-MH122927 to Dominic S. Fareri, and a Scientific Research
Network on Decision Neuroscience and Aging pilot award (NIH R24-AG054355,
Gregory Samanez-Larkin, PI). See the manuscript for the full contributor list.
