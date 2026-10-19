#!/usr/bin/env bash
# Lightweight CPU-only extraction/statistics; never reads/writes PALM scratch.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
report="$repo/results/vmpfc_ratings/v1"
mkdir -p "$report" || exit 1
attempt=$(mktemp -d "$report/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_batch() {
    git rev-parse HEAD || return $?
    python3 --version || return $?
    python3 -m unittest discover -s tests -p test_vmpfc_ratings.py -v || return $?
    python3 -u code/plot_three_partner_ratings.py || return $?
    python3 -u code/analyze_corrected_vmpfc.py --execute || return $?
}
run_batch 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved vmPFC/ratings outputs: $report (exit $status)"
echo 'Commit/push results/vmpfc_ratings/v1, including logs if a check failed.'
exit "$status"
