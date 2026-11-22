#!/usr/bin/env bash
# Compute the revised model and cluster plots from verified Linux1 inputs.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
model=results/single_trial_choice/v2/full
fig=results/manuscript/figure5_wholebrain_offer_age
paths=()
for n in same_trial_by_age.tsv random_effects.tsv validation.json coefficients.tsv diagnostics.tsv heldout_by_age.tsv heldout_gains.tsv heldout_scores.tsv nested_tests.tsv software.txt; do paths+=("$model/$n"); done
for n in Figure5.pdf Figure5.png figure5_clusters.tsv figure5_cluster_estimates.tsv; do paths+=("$fig/$n"); done
[[ $(git branch --show-current) == main ]] || { echo "Use the main checkout of srndna-ultimatum."; exit 1; }
git fetch origin main
git merge --ff-only FETCH_HEAD
git diff --quiet FETCH_HEAD..HEAD -- . ":(exclude)$model" ":(exclude)$fig" || { echo "Local main has unrelated unpublished commits."; exit 1; }
[[ -z $(git status --porcelain -- "${paths[@]}" code/{wholebrain_choice.py,fit_single_trial_choice.R,choice_age_descriptives.R,figure5_wholebrain.py,run_wholebrain_choice.sh}) ]] || { echo "Preserve existing edits to the scripts or their outputs."; exit 1; }
dataset=${SRNDNA_DATASET_ROOT:-/ZPOOL/data/datasets/ds003745-work}
scratch=${SRNDNA_SCRATCH:-/ZPOOL/data/scratch}
bg=${SRNDNA_MNI_BACKGROUND:-${FSLDIR:-/usr/local/fsl}/data/standard/MNI152_T1_2mm_brain.nii.gz}
[[ -s "$bg" ]] || bg="$repo/derivatives/bg_image.nii.gz"
py=(python3)
if ! python3 -c 'import nibabel,numpy,scipy,matplotlib' 2>/dev/null; then py=(conda run --no-capture-output -n srndna-ultimatum python3); fi
source "$repo/code/choice_r_runtime.sh"
select_choice_r
umask 077
work=$(mktemp -d "${TMPDIR:-/tmp}/srndna-choice-v2.XXXXXX")
echo "Private working tables: $work"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
"${py[@]}" code/figure5_wholebrain.py --scratch "$scratch" --background "$bg" --output "$work/figure"
"${py[@]}" code/wholebrain_choice.py --dataset "$dataset" --scratch "$scratch" --output "$work/model"
"${rr[@]}" code/fit_single_trial_choice.R "$work/model/private_trials.tsv" "$work/model" --whole-brain
for path in "${paths[@]}"; do
    [[ $path == "$model/"* ]] && src="$work/model/${path##*/}" || src="$work/figure/${path##*/}"
    test -s "$src"
done
mkdir -p "$model" "$fig"
for path in "${paths[@]}"; do
    [[ $path == "$model/"* ]] && src="$work/model/${path##*/}" || src="$work/figure/${path##*/}"
    install -m 644 "$src" "$path"
done
git add -- "${paths[@]}"
if ! git diff --cached --quiet -- "${paths[@]}"; then git commit --only -m 'Return age moderation, opposite-run DMN and whole-brain cluster estimates' -- "${paths[@]}"; fi
git push origin HEAD:refs/heads/main
git ls-remote origin refs/heads/main
