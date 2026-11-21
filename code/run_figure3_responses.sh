#!/usr/bin/env bash
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
cd "$repo"
if [[ $(git branch --show-current) != main ]]; then
    echo "Run from the main branch of srndna-ultimatum." >&2
    exit 1
fi
dest=results/manuscript/figure3_partner_responses
files=(condition_estimates.tsv insula_tests.tsv validation.json anterior_insula.pdf anterior_insula.png human_gt_computer.pdf human_gt_computer.png)
paths=()
for name in "${files[@]}"; do paths+=("$dest/$name"); done
git fetch origin main
if [[ $(git rev-parse HEAD) != $(git rev-parse FETCH_HEAD) ]]; then
    echo "Synchronize your main branch with origin/main before running." >&2
    exit 1
fi
if [[ -n $(git status --porcelain -- "${paths[@]}" code/extract_figure3_responses.py code/plot_figure3_responses.py code/run_figure3_responses.sh) ]]; then
    echo "The Figure 3 inputs or output paths have local edits. Preserve them before regenerating." >&2
    exit 1
fi
py=(python3)
if ! python3 -c 'import numpy, scipy, nibabel, matplotlib' 2>/dev/null; then
    py=(conda run -n srndna-ultimatum python3)
fi
template="${SRNDNA_MNI_TEMPLATE:-${FSLDIR:-/usr/local/fsl}/data/standard/MNI152_T1_2mm_brain.nii.gz}"
if [[ ! -r $template && -z ${SRNDNA_MNI_TEMPLATE:-} ]]; then
    for prefix in "$HOME/fsl" /opt/fsl /usr/share/fsl/*; do
        if [[ -r "$prefix/data/standard/MNI152_T1_2mm_brain.nii.gz" ]]; then
            template="$prefix/data/standard/MNI152_T1_2mm_brain.nii.gz"
            break
        fi
    done
fi
if [[ ! -r $template ]]; then
    echo "Set SRNDNA_MNI_TEMPLATE to the FSL MNI152_T1_2mm_brain.nii.gz file." >&2
    exit 1
fi
stage=$(mktemp -d)
echo "Writing validated outputs to $stage"
export SOURCE_DATE_EPOCH=1795219200
"${py[@]}" code/extract_figure3_responses.py --repo "$repo" --scratch-root "${SRNDNA_SCRATCH:-/ZPOOL/data/scratch}" --output "$stage"
"${py[@]}" code/plot_figure3_responses.py --repo "$repo" --source "$stage/condition_estimates.tsv" --template "$template" --output "$stage"
for name in "${files[@]}"; do test -s "$stage/$name"; done
cat "$stage/insula_tests.tsv"
mkdir -p "$dest"
for name in "${files[@]}"; do cp "$stage/$name" "$dest/$name"; done
git add -- "${paths[@]}"
if ! git diff --cached --quiet -- "${paths[@]}"; then
    git commit --only -m 'Publish validated participant-based Figure 3 alternatives' -- "${paths[@]}"
fi
git push origin HEAD:refs/heads/main
git ls-remote origin refs/heads/main
