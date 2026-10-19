#!/usr/bin/env bash
# Leave completed jobs and all previous attempts intact. Run inside tmux.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
palm=${1:-/ZPOOL/data/scratch/srndna-ultimatum-dmn-followup-v1/tools/PALM}
jobs=${2:-2}
report="$repo/results/dmn_followup/v1"
mkdir -p "$report" || exit 1
attempt=$(mktemp -d "$report/recovery-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_recovery() {
    git rev-parse HEAD || return $?
    python3 -m unittest discover -s tests -p test_palm_conservative.py -v || return $?
    python3 -m unittest discover -s tests -p test_dmn_followup.py -v || return $?
    python3 -u code/resume_palm_conservative.py --palm-root "$palm" --jobs "$jobs" --execute
}
run_recovery 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved recovery logs: $attempt (exit $status)"
echo 'Commit/push results/dmn_followup/v1, including evidence if a check failed.'
exit "$status"
