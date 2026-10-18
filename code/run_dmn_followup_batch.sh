#!/usr/bin/env bash
# Run inside tmux. All outcomes produce a small committable attempt log.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
report="$repo/results/dmn_followup/v1"
work=/ZPOOL/data/scratch/srndna-ultimatum-dmn-followup-v1
palm=${1:-/ZPOOL/data/tools/PALM-srndna-9086eff}
jobs=${2:-15}
mkdir -p "$report" || exit 1
attempt="$report/attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$attempt" || exit 1
run_batch() {
    git rev-parse HEAD || return $?
    python3 --version || return $?
    python3 -m unittest discover -s tests -p test_dmn_followup.py -v || return $?
    python3 -u code/finish_dmn_diagnostics.py || return $?
    python3 code/prepare_palm_resume.py --palm-root "$palm" \
        --work-root "$work/palm" --report-root "$report/palm" || return $?
    local args=(--work-root "$work/palm" --palm-root "$palm" --jobs "$jobs")
    python3 -u code/run_dmn_heteroscedasticity.py "${args[@]}" || return $?
    python3 -u code/run_dmn_heteroscedasticity.py "${args[@]}" --execute || return $?
}
run_batch 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved diagnostics: $attempt (exit $status)"
echo 'Commit/push results/dmn_followup/v1 even if a check failed; completed PALM jobs are resumable.'
exit "$status"
