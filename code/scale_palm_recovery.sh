#!/usr/bin/env bash
# Run in a NEW tmux window while the existing batch drains.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
palm=${1:-/ZPOOL/data/scratch/srndna-ultimatum-dmn-followup-v1/tools/PALM}
jobs=${2:-8}
report="$repo/results/dmn_followup/v1"
mkdir -p "$report" || exit 1
attempt=$(mktemp -d "$report/scale-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_scale() {
    git rev-parse HEAD || return $?
    python3 -m unittest discover -s tests -p test_palm_scale.py -v || return $?
    python3 -u code/scale_palm_recovery.py --palm-root "$palm" --jobs "$jobs"
}
run_scale 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
python3 -u code/collect_palm_progress.py --metadata-only 2>&1 | tee "$attempt/collection.txt"
collection_statuses=("${PIPESTATUS[@]}")
if [[ $status -eq 0 && ${collection_statuses[0]} -ne 0 ]]; then status=${collection_statuses[0]}; fi
if [[ $status -eq 0 && ${collection_statuses[1]} -ne 0 ]]; then status=${collection_statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved parallel PALM recovery: $attempt (exit $status)"
exit "$status"
