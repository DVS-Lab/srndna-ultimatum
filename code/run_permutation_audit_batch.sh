#!/usr/bin/env bash
# Run inside tmux. No changes to the calling interactive shell's error settings.
# Optional arguments: WORK_ROOT REPORT_ROOT JOBS
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
work=${1:-/ZPOOL/data/scratch/srndna-ultimatum-permutation-audit-v1}
report=${2:-$repo/results/permutation_audit/v1}
jobs=${3:-40}
cd "$repo" || exit 1
mkdir -p "$report" || exit 1
attempt="$report/attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$attempt" || exit 1

run_batch() {
    printf 'Started UTC: %s\n' "$(date -u +%FT%TZ)"
    git rev-parse HEAD || return 1
    git status --short
    python3 --version || return 1
    local binary
    binary=$(command -v randomise) || return 1
    # Exercise the installed Linux FSL build on synthetic data before touching
    # the research inputs or launching the long batch. No project fits here.
    SRNDNA_TEST_RANDOMISE="$binary" python3 -m unittest discover \
        -s tests -p test_permutation_audit.py -v || return $?
    python3 -u code/run_permutation_audit.py \
        --work-root "$work" --report-root "$report" \
        --jobs "$jobs" --permutations 10000 --seed 20261014 --execute || return $?
    echo 'PASS: permutation audit completed and results collected'
}

run_batch 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved run record: $attempt (exit $status)"
if [[ $status -ne 0 ]]; then
    echo 'STOP: push the saved diagnostics; do not delete or overwrite scratch outputs.'
fi
exit "$status"
