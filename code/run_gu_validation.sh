#!/usr/bin/env bash
# Norm-model validation only: never launch FEAT or replace completed fits.
# Default is a dry run. Exit 2 means completed fits need diagnostic review.

MODE=${1:-}
ACTION=${2:-}
if test "$#" -gt 2 || { test -n "$ACTION" && test "$ACTION" != --execute; }; then
  echo "Usage: bash code/run_gu_validation.sh {priors|recovery} [--execute]" >&2
  exit 1
fi
case "$MODE" in
  priors|recovery) ;;
  *) echo "Usage: bash code/run_gu_validation.sh {priors|recovery} [--execute]" >&2; exit 1 ;;
esac
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$CODE_ROOT" || exit 1
SCRATCH_BASE=${SRNDNA_STAN_SCRATCH_BASE:-/ZPOOL/data/scratch}
CMDSTAN_ROOT=${SRNDNA_CMDSTAN_ROOT:-/ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0}
case "$MODE" in
  priors)
    WORK_ROOT="$SCRATCH_BASE/srndna-gu-stan-priors-long-v1"
    OUTPUT_ROOT="$CODE_ROOT/results/norm_learning/stan-priors-long-v1"
    PHASE=fit; EXPECTED_JOBS=6
    EXTRA=(--prior-scales .5 1 2 --stages full run1) ;;
  recovery)
    WORK_ROOT="$SCRATCH_BASE/srndna-gu-stan-recovery-long-v1"
    OUTPUT_ROOT="$CODE_ROOT/results/norm_learning/stan-recovery-long-v1"
    PHASE=recovery; EXPECTED_JOBS=10
    EXTRA=(--prior-scales 1 --recovery-reps 10 --stages full) ;;
esac
mkdir -p "$OUTPUT_ROOT" || exit 1
RUN_ARGS=(--cmdstan "$CMDSTAN_ROOT" --work-root "$WORK_ROOT"
  --phase "$PHASE" --models rw_free --age-mode none
  --jobs 40 --chains 4 --threads-per-chain 2
  --warmup 3000 --samples 6000 --adapt-delta .99
  --collect-to "$OUTPUT_ROOT" "${EXTRA[@]}")
if test "$ACTION" = --execute; then
  RUN_ARGS+=(--execute)
  RUN_LOG="$OUTPUT_ROOT/run.txt"
else
  RUN_LOG="$OUTPUT_ROOT/preflight.txt"
fi
python -u code/run_gu_hierarchical.py "${RUN_ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"
PIPE_CODES=("${PIPESTATUS[@]}")
STATUS=${PIPE_CODES[0]}
if test "${PIPE_CODES[1]}" -ne 0; then
  echo "Could not save the run log: $RUN_LOG" >&2
  exit 1
fi
if test "$STATUS" -ne 0 && test "$STATUS" -ne 2; then
  echo "Validation failed (exit $STATUS); inspect $RUN_LOG" >&2
  exit "$STATUS"
fi
if test "$ACTION" != --execute; then
  echo "Dry run only. Repeat with --execute inside tmux to sample."
  exit "$STATUS"
fi
python -u code/export_gu_stan_diagnostics.py \
  --work-root "$WORK_ROOT" --phase "$PHASE" --expected-jobs "$EXPECTED_JOBS" \
  --output-dir "$OUTPUT_ROOT/diagnostics-v1" 2>&1 | tee -a "$OUTPUT_ROOT/export.txt"
PIPE_CODES=("${PIPESTATUS[@]}")
if test "${PIPE_CODES[0]}" -ne 0; then exit "${PIPE_CODES[0]}"; fi
if test "${PIPE_CODES[1]}" -ne 0; then exit 1; fi
echo "Saved validation summaries, diagnostic figures, and logs: $OUTPUT_ROOT"
if test "$STATUS" -eq 2; then
  echo "Sampling completed with diagnostic flags. Preserve these outputs for review."
fi
echo "This initial audit does not validate age effects, model selection, or imaging regressors."
exit "$STATUS"
