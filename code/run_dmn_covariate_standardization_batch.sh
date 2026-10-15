#!/usr/bin/env bash
# Run inside tmux. Optional arguments: WORK_ROOT REPORT_ROOT RANDOMISE_JOBS GROUP_JOBS
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
work=${1:-/ZPOOL/data/scratch/srndna-ultimatum-dmn-covariates-v1}
report=${2:-$repo/results/dmn_covariate_standardized/v1}
jobs=${3:-40}
group_jobs=${4:-4}
cd "$repo" || exit 1
mkdir -p "$report" || exit 1
attempt="$report/attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$attempt" || exit 1
run_batch() {
    printf 'Started UTC: %s\n' "$(date -u +%FT%TZ)"
    git rev-parse HEAD || return 1
    python3 --version || return 1
    local binary
    binary=$(command -v randomise) || return 1
    local compiler
    compiler=$(command -v feat_model) || return 1
    SRNDNA_TEST_FEAT_MODEL="$compiler" python3 -m unittest discover \
        -s tests -p test_dmn_covariate_standardization.py -v || return $?
    SRNDNA_TEST_RANDOMISE="$binary" python3 -m unittest discover \
        -s tests -p test_permutation_audit.py -v || return $?
    local args=(--work-root "$work" --report-root "$report" --jobs "$jobs" --group-jobs "$group_jobs")
    python3 -u code/run_dmn_covariate_standardization.py "${args[@]}" || return $?
    python3 -u code/run_dmn_covariate_standardization.py "${args[@]}" --execute || return $?
}
run_batch 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
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
    echo 'STOP: push diagnostics. Completed work is preserved; incomplete FEAT folders are never overwritten.'
fi
exit "$status"
