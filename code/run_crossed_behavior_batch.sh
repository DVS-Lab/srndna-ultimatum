#!/usr/bin/env bash
# Logged Linux1 handoff. Does not alter the calling shell's error options.
# Usage: bash code/run_crossed_behavior_batch.sh [WORK_ROOT] [REPORT_ROOT] [JOBS]

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
work=${1:-/ZPOOL/data/scratch/srndna-ultimatum-crossed-behavior-v1}
report=${2:-$repo/results/crossed_behavior_l3/v1}
jobs=${3:-2}
cd "$repo" || exit 1
mkdir -p "$report" || exit 1
attempt="$report/attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$attempt" || exit 1

run_batch() {
    printf 'Started UTC: %s\n' "$(date -u +%FT%TZ)"
    git rev-parse HEAD || return 1
    git status --short
    python3 --version || return 1
    for tool in feat feat_model; do
        command -v "$tool" || return 1
    done
    if [[ -z ${FSLDIR:-} || ! -f "$FSLDIR/etc/fslversion" ]]; then
        echo 'ERROR: FSLDIR or its version file is missing'
        return 1
    fi
    printf 'FSL: '
    cat "$FSLDIR/etc/fslversion"
    printf '\nWork: %s\nReport: %s\nConcurrent group jobs: %s\n' "$work" "$report" "$jobs"
    local args=(--timing all --work-root "$work" --jobs "$jobs")
    python3 -u code/run_crossed_behavior_l3.py "${args[@]}" || return $?
    python3 -u code/run_crossed_behavior_l3.py "${args[@]}" \
        --execute --collect-to "$report/imaging" || return $?
    echo 'PASS: all 18 group jobs and all 144 contrast exports completed'
}

run_batch 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"

# Preserve per-job FEAT logs on failures as well as successful runs.
if [[ -d "$work/logs" ]]; then
    mkdir -p "$attempt/feat_logs"
    for file in "$work"/logs/*.log; do
        [[ -f "$file" ]] || continue
        cp "$file" "$attempt/feat_logs/$(basename "${file%.log}").txt" || status=1
    done
fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved run record: $attempt (exit $status)"
if [[ $status -ne 0 ]]; then
    echo 'STOP: push the saved diagnostic files for inspection; do not delete scratch outputs.'
fi
exit "$status"
