# Task-wide activation response to offer size

This revision analysis tests the task-wide activation response to increasingly
fair offers. It preserves the historical nine-EV first-level model and adds
one contrast rather than collapsing partner conditions into a new fitted EV.
That isolates the scientific question while preserving the original nuisance
partition and condition-specific estimates.

## Model specification

- L1 contrast 11: equal-weighted mean of the computer, age-similar, and
  age-dissimilar offer-size slopes.
- L2: fixed effects across the two task runs, carrying all 11 copes.
- L3: 47-participant FLAME 1+2 model with an intercept, centered age group,
  centered sex, tSNR, mean framewise displacement, and event-corrected
  task-wide mean RT.
- L3 contrast 1 is the positive adjusted task-wide fairness effect. Contrast 2
  is its negative direction; contrasts 3 and 4 audit age-group differences.

The pipeline rebuilds all nine task EV files from substantive decision rows.
Consequently, the redundant `event_RT` rows in the BIDS files are not an input
requirement. The tracked corrected events are used for `sub-144`, and its
repaired rendered model is the source of all other run-specific settings.

## Prepare and validate on Linux1

Run from a clean, current checkout. The work directory must be new or empty.

```bash
REPOSITORY=/ZPOOL/data/projects/srndna-ultimatum
DATASET_ROOT=/ZPOOL/data/datasets/ds003745-work
SUB144_REPAIR_ROOT=/ZPOOL/data/scratch/srndna-ultimatum-sub144-repair-v2
FAIRNESS_ROOT=/ZPOOL/data/scratch/srndna-ultimatum-activation-fairness-main-v1

cd "$REPOSITORY"
git pull --ff-only

python3 code/prepare_activation_fairness_main_pipeline.py \
  --repository "$REPOSITORY" \
  --dataset-root "$DATASET_ROOT" \
  --production-fsl-root "$REPOSITORY/derivatives/fsl" \
  --sub144-repair-root "$SUB144_REPAIR_ROOT" \
  --work-root "$FAIRNESS_ROOT"

cut -f1 "$FAIRNESS_ROOT/activation_fairness_main_jobs.tsv" | \
  sort | uniq -c

python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$FAIRNESS_ROOT/activation_fairness_main_jobs.tsv" \
  --stage l1 --jobs 1 --dry-run
```

The expected manifest counts are 94 L1 jobs, 47 L2 jobs, and one L3 job. The
dry run must report 94 runnable L1 jobs and no missing inputs.

Compile one representative first-level design before the full launch:

```bash
SMOKE_FSF="$FAIRNESS_ROOT/fsf/sub-104/L1_act-fairness-main_run-01.fsf"
feat_model "${SMOKE_FSF%.fsf}"

grep '^set fmri(ncon_\|^set fmri(conname_.*11\|^set fmri(con_.*11\.' \
  "$SMOKE_FSF"
```

Both contrast counts must be 11, and original and real contrast 11 must be
`0, 1/3, 0, 1/3, 0, 1/3, 0, 0, 0`.

## Execute

Do not start 40 concurrent FEAT jobs while the trust LSS queue is still using
nearly all Linux1 CPUs. After that queue finishes, the intended launch is:

```bash
python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$FAIRNESS_ROOT/activation_fairness_main_jobs.tsv" \
  --stage l1 --jobs 40 --resume

python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$FAIRNESS_ROOT/activation_fairness_main_jobs.tsv" \
  --stage l2 --jobs 12 --resume

python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$FAIRNESS_ROOT/activation_fairness_main_jobs.tsv" \
  --stage l3 --jobs 1 --resume
```

Re-running any line with `--resume` skips complete jobs. Existing incomplete
output directories are rejected rather than silently overwritten.

Final completion check:

```bash
for STAGE in l1 l2 l3; do
  python3 code/run_ultimatum_repair_jobs.py \
    --manifest "$FAIRNESS_ROOT/activation_fairness_main_jobs.tsv" \
    --stage "$STAGE" --jobs 1 --resume --dry-run
done
```

Each stage must report zero jobs ready. The focal group result is:

```text
/ZPOOL/data/scratch/srndna-ultimatum-activation-fairness-main-v1/
  outputs/activation-fairness-main_all-participants.gfeat/
  cope1.feat/stats/zstat1.nii.gz
```

After all three stages pass, collect the compact design and focal outputs into
the repository of record:

```bash
python3 code/collect_activation_fairness_main_results.py \
  --work-root "$FAIRNESS_ROOT"

git add results/reviewer/activation_fairness_main
git commit -m "results: add task-wide activation fairness model"
git push origin main
```

All outputs are isolated in scratch; historical production FEAT directories
are not modified.
