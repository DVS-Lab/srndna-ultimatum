#!/usr/bin/env bash
# Does not edit completed empirical fits or start imaging models.
MODE=${1:-}
ACTION=${2:-}
if test "$#" -gt 2 || { test -n "$ACTION" && test "$ACTION" != --execute; }; then
  echo 'Usage: bash code/run_gu_signal_validation.sh {signals|targeted} [--execute]' >&2
  exit 1
fi
case "$MODE" in signals|targeted) ;; *) echo 'Expected signals or targeted' >&2; exit 1 ;; esac
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$CODE_ROOT" || exit 1
SCRATCH_BASE=${SRNDNA_STAN_SCRATCH_BASE:-/ZPOOL/data/scratch}
CMDSTAN_ROOT=${SRNDNA_CMDSTAN_ROOT:-/ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0}
OUTPUT_ROOT="$CODE_ROOT/results/norm_learning/stan-${MODE}-v1"
mkdir -p "$OUTPUT_ROOT" || exit 1
logged() {
  local log=$1
  shift
  "$@" 2>&1 | tee -a "$log"
  local codes=("${PIPESTATUS[@]}")
  if test "${codes[1]}" -ne 0; then return 1; fi
  return "${codes[0]}"
}
if test "$MODE" = signals; then
  if test "$ACTION" != --execute; then
    echo 'Read-only posterior signal audit: four full-data fits, all draws; outputs only, no Stan/FEAT.'
    echo 'Repeat with --execute to export the audit.'
    exit 0
  fi
  logged "$OUTPUT_ROOT/run.txt" python -u code/audit_gu_trial_signals.py signals \
    --scratch-base "$SCRATCH_BASE" --output-dir "$OUTPUT_ROOT/audit"
  exit $?
fi
ARGS=(--scratch-base "$SCRATCH_BASE" --cmdstan "$CMDSTAN_ROOT" --jobs 40 --collect-to "$OUTPUT_ROOT")
RUN_LOG="$OUTPUT_ROOT/preflight.txt"
if test "$ACTION" = --execute; then ARGS+=(--execute); RUN_LOG="$OUTPUT_ROOT/run.txt"; fi
logged "$RUN_LOG" python -u code/run_gu_targeted_recovery.py "${ARGS[@]}"
STATUS=$?
if test "$STATUS" -ne 0 && test "$STATUS" -ne 2; then exit "$STATUS"; fi
if test "$ACTION" != --execute; then exit "$STATUS"; fi
logged "$OUTPUT_ROOT/export.txt" python -u code/export_gu_stan_diagnostics.py \
  --work-root "$SCRATCH_BASE/srndna-gu-stan-targeted-v1" --phase recovery --expected-jobs 16 \
  --output-dir "$OUTPUT_ROOT/diagnostics-v1" || exit 1
logged "$OUTPUT_ROOT/export.txt" python -u code/audit_gu_trial_signals.py targeted \
  --scratch-base "$SCRATCH_BASE" --output-dir "$OUTPUT_ROOT/audit" || exit 1
echo 'Saved diagnostics and participant-average/partner-difference recovery, including flagged fits.'
echo 'No imaging covariates or EVs released. Exit 2 means sampling completed with diagnostic flags.'
exit "$STATUS"
