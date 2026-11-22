#!/usr/bin/env bash
# Run the complete choice comparison and return only aggregate results to GitHub.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
if [[ $(git branch --show-current) != main ]]; then
    echo "Run this command from the main branch of srndna-ultimatum." >&2
    exit 1
fi
dest=results/single_trial_choice/v1/full
files=(validation.json heldout_scores.tsv heldout_gains.tsv nested_tests.tsv diagnostics.tsv coefficients.tsv software.txt)
paths=()
optional=(heldout_scores.tsv heldout_gains.tsv nested_tests.tsv diagnostics.tsv coefficients.tsv software.txt)
for name in "${files[@]}"; do paths+=("$dest/$name"); done
git fetch origin main
git merge --ff-only FETCH_HEAD
# Permit a retry after an interrupted results push, while keeping unrelated local commits private.
if ! git diff --quiet FETCH_HEAD..HEAD -- . ":(exclude)$dest"; then
    echo "Local main contains unpublished changes outside $dest. Use a checkout synced with origin/main." >&2
    exit 1
fi
if [[ -n $(git status --porcelain -- "${paths[@]}" "$dest/recovered" code/extract_single_trial_choice.py code/fit_single_trial_choice.R code/choice_r_runtime.sh code/run_single_trial_choice.sh results/single_trial_choice/v1/{openneuro_2.2.0_inputs.tsv,sub104_roi_reference.tsv,right_frontal_pole.nii.gz}) ]]; then
    echo "There are existing edits to a model input, script or result path. Preserve them before regenerating." >&2
    exit 1
fi
dataset=${SRNDNA_DATASET_ROOT:-/ZPOOL/data/datasets/ds003745-work}
scratch=${SRNDNA_SCRATCH:-/ZPOOL/data/scratch}
py=(python3)
if ! python3 -c 'import nibabel,numpy,scipy' 2>/dev/null; then
    py=(conda run --no-capture-output -n srndna-ultimatum python3)
fi
source "$repo/code/choice_r_runtime.sh"
select_choice_r
"${rr[@]}" -e 'cat(R.version.string, "with lme4", as.character(packageVersion("lme4")), "\n")'
umask 077
stage=$(mktemp -d "${TMPDIR:-/tmp}/srndna-choice.XXXXXX")
echo "Private trial table and working records: $stage"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
extra=()
if [[ -n ${SRNDNA_LSS_ROOT:-} ]]; then extra+=(--lss-root "$SRNDNA_LSS_ROOT"); fi
"${py[@]}" code/extract_single_trial_choice.py --dataset "$dataset" --scratch "$scratch" --output "$stage" "${extra[@]}"
"${rr[@]}" code/fit_single_trial_choice.R "$stage/private_trials.tsv" "$stage"
if [[ -s $stage/private_recovered_trials.tsv ]]; then
    "${rr[@]}" code/fit_single_trial_choice.R "$stage/private_recovered_trials.tsv" "$stage/recovered" --recovered
    for name in "${optional[@]}"; do test -s "$stage/recovered/$name"; done
fi
for name in "${files[@]}"; do test -s "$stage/$name"; done
if [[ -d $dest/recovered && ! -d $stage/recovered ]]; then
    echo "An earlier run recovered the partial series. Supply its SRNDNA_LSS_ROOT to reproduce those results." >&2
    exit 1
fi
if [[ -s $dest/validation.json ]]; then
    python3 - "$dest/validation.json" "$stage/validation.json" <<'PY_CHECK'
import json, sys
with open(sys.argv[1]) as old, open(sys.argv[2]) as new:
    if json.load(old)["additional_recovered_human_choices"] > json.load(new)["additional_recovered_human_choices"]:
        sys.exit("A prior sensitivity analysis verified more trials. Set SRNDNA_LSS_ROOT to its retained source directory.")
PY_CHECK
fi
mkdir -p "$dest"
for name in "${files[@]}"; do install -m 644 "$stage/$name" "$dest/$name"; done
if [[ -d $stage/recovered ]]; then
    mkdir -p "$dest/recovered"
    for name in "${optional[@]}"; do
        install -m 644 "$stage/recovered/$name" "$dest/recovered/$name"
        paths+=("$dest/recovered/$name")
    done
fi
git add -- "${paths[@]}"
if ! git diff --cached --quiet -- "${paths[@]}"; then
    git commit --only -m 'Publish held-out single-trial choice and opposite-run connectivity comparisons' -- "${paths[@]}"
fi
git push origin HEAD:refs/heads/main
git ls-remote origin refs/heads/main
