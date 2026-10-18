#!/usr/bin/env bash
# Resume only PALM; completed overlays/influence checks are not regenerated.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
work=/ZPOOL/data/scratch/srndna-ultimatum-dmn-followup-v1/palm
report="$repo/results/dmn_followup/v1"
palm=${1:-/ZPOOL/data/scratch/srndna-ultimatum-dmn-followup-v1/tools/PALM}
jobs=${2:-40}
mkdir -p "$report" || exit 1
attempt=$(mktemp -d "$report/resume-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
resume_batch() {
    git rev-parse HEAD || return $?
    python3 -m unittest discover -s tests -p test_palm_resume.py -v || return $?
    python3 -m unittest discover -s tests -p test_dmn_followup.py -v || return $?
    python3 code/prepare_palm_resume.py --palm-root "$palm" \
        --work-root "$work" --report-root "$report/palm" || return $?
    local args=(--work-root "$work" --palm-root "$palm" --jobs "$jobs")
    python3 -u code/run_dmn_heteroscedasticity.py "${args[@]}" || return $?
    python3 -u code/run_dmn_heteroscedasticity.py "${args[@]}" --execute || return $?
}
resume_batch 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved resume log: $attempt (exit $status)"
echo 'Commit/push results/dmn_followup/v1, including logs if a check failed.'
exit "$status"
