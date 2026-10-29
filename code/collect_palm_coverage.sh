#!/usr/bin/env bash
# Read-only ROI coverage/variance audit; never signal or restart PALM.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="$repo/results/dmn_followup/v1/coverage_audit"
mkdir -p "$parent" || exit 1
attempt=$(mktemp -d "$parent/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_audit() {
    git rev-parse HEAD || return $?
    python3 -m unittest discover -s tests -p test_palm_coverage.py -v || return $?
    python3 -u code/audit_palm_varcope_coverage.py --output-root "$attempt/outputs" "$@"
}
run_audit "$@" 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}"); status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
python3 -u code/collect_palm_progress.py --metadata-only --summarize 2>&1 | tee "$attempt/palm_progress.txt"
collection_statuses=("${PIPESTATUS[@]}")
if [[ $status -eq 0 && ${collection_statuses[0]} -ne 0 ]]; then status=${collection_statuses[0]}; fi
if [[ $status -eq 0 && ${collection_statuses[1]} -ne 0 ]]; then status=${collection_statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved read-only coverage audit: $attempt (exit $status)"
echo 'Commit/push results/dmn_followup/v1/coverage_audit and results/dmn_followup/v1/palm/collections, including logs if a check failed.'
exit "$status"
