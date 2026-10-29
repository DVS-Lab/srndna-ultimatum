#!/usr/bin/env bash
# Read existing L1/L2 images; preserve PALM workers and every previous report.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="$repo/results/dmn_followup/v1/partner_motion"
mkdir -p "$parent" || exit 1
attempt=$(mktemp -d "$parent/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_audit() {
    git rev-parse HEAD || return $?
    python3 -m unittest discover -s tests -p test_palm_partner_motion.py -v || return $?
    python3 -u code/audit_palm_partner_motion.py --output-root "$attempt/outputs" "$@"
}
run_audit "$@" 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}"); status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
python3 -u code/collect_palm_progress.py --metadata-only --summarize 2>&1 | tee "$attempt/palm_progress.txt"
collection_statuses=("${PIPESTATUS[@]}")
if [[ $status -eq 0 && ${collection_statuses[0]} -ne 0 ]]; then status=${collection_statuses[0]}; fi
if [[ $status -eq 0 && ${collection_statuses[1]} -ne 0 ]]; then status=${collection_statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved partner/motion diagnostic: $attempt (exit $status)"
echo 'Commit/push results/dmn_followup/v1/partner_motion and results/dmn_followup/v1/palm/collections, including logs if validation failed.'
exit "$status"
