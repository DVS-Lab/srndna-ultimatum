# Repository of record

The repository of record for **Older Adults Show Altered Default Mode and
Executive Control Network Connectivity during Fairness Decisions** is:

`https://github.com/DVS-Lab/srndna-ultimatum`, branch `main`

This is the original 2021 repository and the Git root present at the production
Linux path `/ZPOOL/data/projects/srndna-ultimatum`. It owns the paper-specific
production history, original FSL templates and scripts, compact source tables,
revision analyses, and submission-facing provenance documentation.

The relevant storage boundaries are:

- OpenNeuro `ds003745` owns the public BIDS dataset. The analyzed snapshot must
  be identified explicitly; the manuscript cites version 2.0.2.
- `/ZPOOL/data/projects/srndna-ultimatum/derivatives/` holds the production
  imaging derivatives alongside the clean Git checkout. Large NIfTI and FEAT
  payloads remain outside Git; compact designs, logs, paths, and checksums may
  be committed when needed for provenance.
- `DVS-Lab/srndna-ug` is a separate, unrelated Git history created later. It
  contains useful reviewer-response work and later template candidates, but it
  is not authoritative by itself. Relevant compact artifacts are imported
  selectively rather than by merging histories or copying its whole tree.
- `DVS-Lab/srndna` is the historical grant-wide repository and
  `DVS-Lab/srndna-datapaper` is the all-task data-management/preprocessing
  repository. Neither is the revision target for this paper.

Original templates remain in `templates/`. Files recovered from the later
working repository are kept under `templates/later_working_tree/` until the
read-only audit compares them with rendered production `design.fsf`,
`design.mat`, `design.con`, and `design.grp` files. Their presence does not
establish that they generated a submitted result.

Hard-coded paths in historical FSFs are preserved as provenance. New audit and
analysis entry points must accept roots as arguments or derive paths from the
checkout. If a rerun is approved later, render a new versioned FSF; never edit
an old production file and relabel it as the submitted analysis.
