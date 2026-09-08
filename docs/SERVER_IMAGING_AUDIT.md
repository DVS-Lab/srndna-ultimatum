# Imaging revision audit on the Linux analysis server

The production checkout and derivative tree are both rooted at
`/ZPOOL/data/projects/srndna-ultimatum`. The audit is read-only with respect to
scientific outputs: it may write compact inventories under `logs/audits/` and
aggregate tables under `results/reviewer/tables/`, but it must not invoke FEAT,
FLAME, `randomise`, or replacement cluster inference.

## Synchronize and validate

```bash
cd /ZPOOL/data/projects/srndna-ultimatum
git pull --ff-only origin main
git status --short --branch
make test
```

## Audit RT events and retained activation designs

```bash
python3 code/audit_server_rt_events.py \
  --bids-root source_data/bids \
  --ev-root /ZPOOL/data/projects/srndna-ultimatum/derivatives/fsl/EVfiles \
  --l1-root /ZPOOL/data/projects/srndna-ultimatum/derivatives/fsl \
  --path-map /data/projects/srndna-ultimatum=/ZPOOL/data/projects/srndna-ultimatum \
  --output-dir logs/audits/server/rt-production \
  --tracked-summary results/reviewer/tables/production_rt_ev_audit.tsv
```

The tracked summary contains aggregate counts only. The 94-run table remains
under ignored `logs/audits/`. `--path-map` affects only how retained absolute
FSF paths are resolved; it does not alter an FSF or create a symlink.

Audit estimability across the three focal first-level model types:

```bash
python3 code/audit_l1_designs.py \
  --l1-root /ZPOOL/data/projects/srndna-ultimatum/derivatives/fsl \
  --output-dir logs/audits/server/l1-designs \
  --tracked-summary results/reviewer/tables/production_l1_design_summary.tsv
```

This writes participant/run rows only under ignored `logs/audits/` and an
aggregate model-level summary suitable for Git. Correlation thresholds are
descriptive flags, not automatic failure criteria.

## Locate rendered 47-participant group designs

```bash
mkdir -p logs/audits/server
find derivatives/fsl -type f -name design.fsf -print0 |
  while IFS= read -r -d '' fsf; do
    grep -q 'set fmri(npts) 47' "$fsf" && printf '%s\n' "$fsf"
  done |
  sort |
  tee logs/audits/server/l3-design-fsf-paths.txt
```

Inspect the resulting path list for the focal DMN age-group and ECN
fairness-sensitivity analyses. Do not choose a directory from its name alone:
confirm its input paths, EV titles, and contrast names. For each confirmed
cope-level `.feat` directory, run:

```bash
bash code/audit_server_imaging.sh \
  --fmriprep-root /ZPOOL/data/projects/srndna-ultimatum/derivatives/fmriprep \
  --group-dir /ABSOLUTE/PATH/TO/CONFIRMED/COPE.feat \
  --output-dir logs/audits/server/RESULT-ID
```

Repeat for the focal DMN and ECN results. The collector copies small design,
log, cluster-table, and preprocessing-provenance files and records image
headers/volumes; it does not copy NIfTI payloads.

## Required production evidence

The returned bundles must establish:

1. exact FSL and fMRIPrep versions;
2. production `design.fsf`, `design.mat`, `design.con`, and `design.grp`;
3. participant order, cope mapping, rank, conditioning, covariates, and
   contrasts relative to both the original templates and
   `templates/later_working_tree/` candidates;
4. voxel threshold, corrected cluster threshold, search mask, any threshold
   mask, FLAME mode, and post-statistics setting;
5. DLH, search volume, residual degrees of freedom, resel/smoothness records,
   corrected cluster probabilities, peaks, and minimum significant extent;
6. raw acquisition geometry versus analyzed output-grid geometry;
7. existing main, simple, social/computer, and sensitivity contrasts before
   any new L3 analysis is proposed;
8. actual L1 design estimability and the downstream role of both sub-143 runs.

The two focal tracked masks contain 26 and 23 voxels, but their statistical
meaning cannot be finalized from mask size alone. The authoritative production
search mask, smoothness, and cluster tables are required.

## Known RT source-data issue

The curated BIDS events contain 6,655 responded trials and 5,724 companion
`event_RT` rows. All 805 responded first trials of blocks lack a companion RT
row, and sub-143 has another 126 omissions because neither run contains any
`event_RT` rows. The substantive task events remain present. The Linux command
above determines what entered the actual retained FEAT designs; it does not
authorize regeneration or rerunning L1.

Earlier counts obtained from `/ZPOOL/data/projects/srndna-ug` were from the
wrong repository and are not production evidence. They are intentionally not
carried into this repository's aggregate results.

## Analyses excluded pending a scientific decision

Do not run permutation/TFCE inference, a model dropping tSNR, participant
deletion/leave-one-out FLAME models, or an automatic ECN rerun with the unified
behavioral slope. Possible robust FLAME outlier deweighting with all 47
participants and genuinely missing L3 contrasts remain author-pending after
the production audit.
