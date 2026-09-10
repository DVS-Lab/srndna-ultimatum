# Imaging revision audit on the Linux analysis server

The production checkout and derivative tree are both rooted at
`/ZPOOL/data/projects/srndna-ultimatum`. The audit is read-only with respect to
scientific outputs: it may write compact inventories under `logs/audits/` and
aggregate tables under `results/reviewer/tables/`, but it must not invoke FEAT,
FLAME, `randomise`, or replacement cluster inference. The one documented
exception is `feat_model`, which creates design matrices without fitting image
data and is run only under scratch space.

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

## Match sub-143 and sub-144 to their actual event source

Run this while image-fitting jobs are active; `feat_model` is brief and does
not fit voxel data. The command tests the corrected working-tree events and the
pre-correction event files at Git revision `02ba301`, each with both the
explicit companion-`event_RT` and substantive-response RT constructions.

```bash
REPOSITORY=/ZPOOL/data/projects/srndna-ultimatum
DESIGN_AUDIT_ROOT=/ZPOOL/data/scratch/srndna-ultimatum-event-design-audit

cd "$REPOSITORY"
python3 code/audit_ultimatum_event_designs.py \
  --production-l1-root "$REPOSITORY/derivatives/fsl" \
  --work-root "$DESIGN_AUDIT_ROOT" \
  --output results/reviewer/tables/affected_event_design_matches.tsv
```

The script never writes inside a retained `.feat` directory. It compares the
first nine task-design columns against each retained activation `design.mat`
and marks the lowest relative RMSE for each participant-run. A genuine source
match should have correlations essentially equal to 1 and relative RMSE near
zero. Commit and push the resulting compact TSV; the generated candidate FSFs,
EVs, and matrices remain in scratch. This result decides whether sub-143 needs
any refit and which RT construction preserves the submitted model.

The completed audit establishes that both sub-143 runs exactly match their own
substantive trial rows, whereas both sub-144 runs exactly match the historical
duplicated events with the companion-RT construction. Therefore sub-143 is not
part of the identity repair. The minimal corrected analysis uses the recovered
sub-144 events but retains the companion-RT policy; changing RT policy is a
separate whole-sample sensitivity analysis.

## Prepare the isolated sub-144 L1/L2 repair

Preparation is noncomputational and refuses a nonempty work root. It reuses
the rendered production FSFs as model definitions, changes only required
paths and corrected EVs, and leaves every retained FEAT output untouched. If
the historical headerless FSL confound matrices are not available, the script
reconstructs them under the repair root from the downloaded fMRIPrep confound
TSVs using the historical column selection and order.

```bash
REPOSITORY=/ZPOOL/data/projects/srndna-ultimatum
DATASET_ROOT=/ZPOOL/data/datasets/ds003745-work
# v1 was left intentionally untouched after the missing-confound preflight
# stopped. Use a new root because preparation refuses nonempty destinations.
REPAIR_ROOT=/ZPOOL/data/scratch/srndna-ultimatum-sub144-repair-v2

cd "$REPOSITORY"
python3 code/prepare_sub144_imaging_repair.py \
  --dataset-root "$DATASET_ROOT" \
  --production-fsl-root "$REPOSITORY/derivatives/fsl" \
  --work-root "$REPAIR_ROOT"

python3 - "$DATASET_ROOT/derivatives/fmriprep/dataset_description.json" <<'PY'
import json
import sys
from pathlib import Path

description = Path(sys.argv[1])
data = json.loads(description.read_text())
print("OpenNeuro fMRIPrep GeneratedBy:", data.get("GeneratedBy"))
PY

cut -f1-4,7-12 "$REPAIR_ROOT/repair_jobs.tsv" | sed -n '1,8p'

python3 code/audit_prepared_ultimatum_designs.py \
  --manifest "$REPAIR_ROOT/repair_jobs.tsv"

python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$REPAIR_ROOT/repair_jobs.tsv" \
  --stage l1 --jobs 2 --dry-run
```

Do not run FEAT unless `audit_prepared_ultimatum_designs.py` reports six passes.
That audit runs only `feat_model`: the corrected task/PPI columns must differ,
while every nuisance column after `evs_real` must match the retained production
matrix to numerical tolerance. Then inspect the selected BOLD, confound, event,
production FSF, and retained nPPI time-series paths in the manifest and dry-run
output. The eventual L1 execution uses six jobs (two runs each of activation,
DMN nPPI, and ECN nPPI):

The L1 manifest records the fMRIPrep version, candidate BOLD checksum, and the
original path written into each production FSF. If an original BOLD copy still
exists under `srndna-data`, `srndna-ug`, or the archive location, compare its
`sha256sum` with `candidate_bold_sha256`. Exact equality closes the remaining
bitwise preprocessing-input provenance check.

For sub-144, that comparison is complete: both OpenNeuro files exactly match
the surviving `srndna-ug` copies (run-1 SHA-256 begins `2055f273`; run-2 begins
`73139354`). Do not rerun fMRIPrep for the minimal identity repair.

```bash
python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$REPAIR_ROOT/repair_jobs.tsv" \
  --stage l1 --jobs 2
```

Only after all L1 completion checks pass should the three L2 jobs run:

```bash
python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$REPAIR_ROOT/repair_jobs.tsv" \
  --stage l2 --jobs 2
```

Reissuing a completed stage without `--resume` is expected to fail rather than
overwrite data. To verify an already completed L1 stage and proceed safely:

```bash
python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$REPAIR_ROOT/repair_jobs.tsv" \
  --stage l1 --jobs 2 --resume

python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$REPAIR_ROOT/repair_jobs.tsv" \
  --stage l2 --jobs 2 --dry-run
```

The runner writes logs inside the repair root, creates only the identity
registration links expected by the historical standard-space workflow, and
never deletes or replaces an output. Use `--resume` only to skip jobs whose
required completion images already exist; an incomplete existing output is a
hard stop.

## Trace the paper masks to Jen's rendered L3 outputs

Search both repositories because the original broad L3 tree is under
`srndna-ultimatum`, while Jen's later paper-specific SANS work is expected under
`srndna-ug`. The first checksum-only run established that none of the three
small binary focal masks is an untouched compressed `cluster_mask_zstat` file.
That does not establish a mismatch in voxel content: selecting one integer
cluster label and binarizing it changes the file hash. Run the support-aware
audit:

```bash
python3 code/audit_l3_production.py \
  --search-root ultimatum="$REPOSITORY/derivatives/fsl" \
  --search-root srndna-ug=/ZPOOL/data/projects/srndna-ug/derivatives/fsl \
  --match-mode support \
  --output-dir logs/audits/server/l3-production \
  --tracked-summary results/reviewer/tables/production_l3_trace.tsv
```

The support matcher reads NIfTI-1 files with Python's standard library and does
not alter image data. `exact_cluster_support` means the focal mask contains
exactly every voxel carrying one cluster label in the retained FSL output.
Containment or partial-overlap rows are leads, not proof of identity. The
tracked table also contains the GFEAT path, design hashes, sub-144 input
position, threshold settings, cluster table, and available `DLH`, `VOLUME`,
and `RESELS` values. The larger all-design inventory remains under ignored
`logs/audits/`. Commit and push the compact tracked TSV.

## Locate all rendered 47-participant group designs manually

```bash
mkdir -p results/reviewer/production_audits
find derivatives/fsl -type f -name design.fsf -print0 |
  while IFS= read -r -d '' fsf; do
    grep -q 'set fmri(npts) 47' "$fsf" && printf '%s\n' "$fsf"
  done |
  sort |
  tee results/reviewer/production_audits/l3-design-fsf-paths.txt
```

Inspect the resulting path list for the focal DMN age-group and ECN
fairness-sensitivity analyses. Do not choose a directory from its name alone:
confirm its input paths, EV titles, and contrast names. For each confirmed
cope-level `.feat` directory, run:

```bash
bash code/audit_server_imaging.sh \
  --fmriprep-root /ZPOOL/data/projects/srndna-ultimatum/derivatives/fmriprep \
  --group-dir /ABSOLUTE/PATH/TO/CONFIRMED/COPE.feat \
  --output-dir results/reviewer/production_audits/RESULT-ID
```

Repeat for the focal DMN and ECN results. The collector copies small design,
log, cluster-table, and preprocessing-provenance files and records image
headers/volumes; it does not copy NIfTI payloads. These compact bundles and
paths are intended to be committed after inspection.

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

The corrected curated BIDS events contain 6,654 responded trials and 5,724
companion `event_RT` rows. All 804 responded first trials of blocks lack an RT
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
