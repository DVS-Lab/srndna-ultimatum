#!/usr/bin/env bash
# Scoped logging/status capture; no interactive-shell errexit or pipefail.
if test "$#" -gt 1 || { test "$#" -eq 1 && test "$1" != --execute; }; then
    echo 'Usage: bash code/run_gu_bias_stable.sh [--execute]' >&2
    exit 1
fi
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$CODE_ROOT" || exit 1
OUTPUT_ROOT="$CODE_ROOT/results/norm_learning/stan-bias-stable-v1"
mkdir -p "$OUTPUT_ROOT" || exit 1
python3 -u code/run_gu_bias_stable.py "$@" 2>&1 | tee -a "$OUTPUT_ROOT/launch.txt"
STATUSES=("${PIPESTATUS[@]}")
if test "${STATUSES[1]}" -ne 0; then echo 'Log write failed' >&2; exit 1; fi
exit "${STATUSES[0]}"
