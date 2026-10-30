#!/usr/bin/env bash
# Read retained estimates. Leave every FEAT/PALM worker and prior export alone.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="$repo/results/age_figure_audit/v1"
mkdir -p "$parent" || exit 1
attempt=$(mktemp -d "$parent/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_audit() {
    git rev-parse HEAD || return $?
    python3 --version || return $?
    python3 -m unittest discover -s tests -p test_age_figure_audit.py -v || return $?
    python3 -u code/audit_age_figures.py --output-root "$attempt/outputs" "$@"
}
run_audit "$@" 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved age-figure audit: $attempt (exit $status)"
echo 'Commit/push results/age_figure_audit/v1, including partial outputs and failure logs.'
exit "$status"
