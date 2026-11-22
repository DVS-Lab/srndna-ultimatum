#!/usr/bin/env bash
# Compare retained derivatives read-only. Return only the aggregate comparison.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
dest=results/single_trial_choice/v1/sub104_source_audit.json
[[ $(git branch --show-current) == main ]] || { echo "Use the srndna-ultimatum main checkout."; exit 1; }
git fetch origin main
git merge --ff-only FETCH_HEAD
git diff --quiet FETCH_HEAD..HEAD -- . ":(exclude)$dest" || { echo "Unrelated unpublished commits on main."; exit 1; }
[[ -z $(git status --porcelain -- "$dest" code/{audit_sub104_sources.py,run_sub104_source_audit.sh,extract_single_trial_choice.py}) ]] || { echo "Preserve existing edits to the audit or its result first."; exit 1; }
py=(python3)
if ! python3 -c 'import nibabel,numpy,scipy' 2>/dev/null; then py=(conda run --no-capture-output -n srndna-ultimatum python3); fi
work=$(mktemp -d)
trap 'rm -f "$work/report.json"; rmdir "$work"' EXIT
extra=()
if [[ -n ${SRNDNA_LSS_ROOT:-} ]]; then extra+=(--lss-root "$SRNDNA_LSS_ROOT"); fi
"${py[@]}" code/audit_sub104_sources.py --output "$work/report.json" "${extra[@]}"
# A checkout containing only the public files cannot settle the Linux1 question.
python3 - "$work/report.json" <<'CHECK'
import json, sys
with open(sys.argv[1]) as stream:
    rows = json.load(stream)
if not any(r["run"]==2 and r["format"]=="numbered" and r.get("volumes")==72 for r in rows):
    sys.exit("The 72 numbered inputs were not found. Set SRNDNA_LSS_ROOT to their derivatives/fsl directory.")
CHECK
install -m 644 "$work/report.json" "$dest"
git add -- "$dest"
if ! git diff --cached --quiet -- "$dest"; then git commit --only -m 'Record comparison of sub-104 sources with both archived series' -- "$dest"; fi
git push origin HEAD:refs/heads/main
