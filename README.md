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
conda env create -f code/environment.yml
conda activate srndna-ultimatum
make test
make reviewer-behavior
make reviewer-imaging-audit
```

The portable Conda environment supplies the Python dependencies used by the
audits and manuscript figures. The behavioral workflow additionally requires
R with `lme4`, `lmerTest`, `ggplot2`, and `scales`; the exact versions used for
the revision are recorded in
`results/reviewer/tables/behavior_software_versions.tsv`. Image-header checks
require FSL's `fslhd` and `fslstats`; validation skips them cleanly when FSL is
unavailable. Production and revision imaging provenance records FSL 6.0.7.17
and fMRIPrep 21.0.2.

The production imaging audit was completed read-only on Linux against
`/ZPOOL/data/projects/srndna-ultimatum`; its compact records are tracked under
`results/reviewer/production_audits/`. `docs/SERVER_IMAGING_AUDIT.md` preserves
the collection procedure. FEAT, FLAME, and `randomise` are not part of that
read-only audit.

## Repository map

- `code/`: original preprocessing/FEAT scripts and active revision audits.
- `templates/`: historical production templates plus canonical, documented
  resubmission templates under `templates/revision/`.
- `legacy/jen_working_tree/`: checksummed quarantine for compact files
  recovered from Jen's separate working repository; nothing there is an
  active execution entry point.
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
cites snapshot 2.0.2. The corrected public snapshot identifier will be added
after the corresponding OpenNeuro update is released. The local root-level
`bids/` directory is intentionally ignored; only small analysis-support
TSV/JSON files belong under `source_data/`.

`derivatives/imaging_plots/participants.tsv` is intentionally retained in its
historical location because `code/plotROIdata.m` reads it alongside the compact
ROI extracts. It contains pseudonymous study IDs, age, and sex, but no direct
identifiers. The broader `participants.tsv` ignore rule prevents accidental
addition of other participant tables.

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

## Resubmission imaging models

Historical production templates remain unchanged in `templates/`. Canonical
resubmission templates are under `templates/revision/`, with their intended
differences documented in `templates/revision/README.md`. The new task-wide
fairness model preserves the submitted nine-EV activation GLM and adds an
equal-weighted mean of its three partner-specific offer-size slopes as
contrast 11. `code/prepare_activation_fairness_main_pipeline.py` prepares the
complete 47-participant L1, L2, and covariate-adjusted L3 workflow in an
external scratch directory; it uses current OpenNeuro inputs, substantive
decision rows for the RT nuisance EVs, and the repaired `sub-144` source model.
The shared manifest runner executes those jobs without replacing historical
FEAT outputs.
See `docs/ACTIVATION_FAIRNESS_MAIN.md` for the exact Linux1 preparation,
preflight, execution, resume, and completion commands.

The revised Figure 4 decomposes the retained default-mode-network contrast
into four age-group-by-partner bars. Bar heights are cluster-mean group COPE
estimates from corrected condition-specific FLAME 1+2 models, so the display
retains FLAME's within- and between-participant variance modeling. The
participant-level condition COPE and VARCOPE extracts and all image hashes are
tracked under `results/manuscript/source_data/` for auditability. Figure error
bars show one model standard error rather than 95% confidence intervals.

## Citation

Citation metadata and the preferred article citation are provided in
`CITATION.cff`. A versioned archival DOI should be added there when the final
software release is deposited.

## Acknowledgments

This work was supported in part by NIH awards R21-MH113917 and R03-DA046733 to
David V. Smith, R15-MH122927 to Dominic S. Fareri, and a Scientific Research
Network on Decision Neuroscience and Aging pilot award (NIH R24-AG054355,
Gregory Samanez-Larkin, PI). See the manuscript for the full contributor list.
