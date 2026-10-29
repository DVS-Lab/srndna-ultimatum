#!/usr/bin/env bash
# Read existing participant COPE7 images; leave PALM running unchanged.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="$repo/results/dmn_followup/v1/age_bars"
mkdir -p "$parent" || exit 1
attempt=$(mktemp -d "$parent/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_export() {
    git rev-parse HEAD || return $?
    python3 -m unittest discover -s tests -p test_palm_age_bars.py -v || return $?
    python3 -u code/plot_palm_age_bars.py --output-root "$attempt/outputs" "$@"
}
run_export "$@" 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved PALM age-bar extraction: $attempt (exit $status)"
echo 'Commit/push results/dmn_followup/v1/age_bars, including logs if a check failed.'
exit "$status"
