#!/usr/bin/env bash
# Separate, resumable exploratory models; existing fits remain unchanged.
MODE=${1:-}
case "$MODE" in
  smoke|fit) ;;
  *) echo "Usage: bash code/run_gu_choice_extensions.sh {smoke|fit} [--execute]" >&2; exit 1 ;;
esac
if test "$#" -gt 2 || { test "$#" -eq 2 && test "$2" != --execute; }; then
  echo "Only optional --execute is accepted" >&2; exit 1
fi
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$CODE_ROOT" || exit 1
SCRATCH_BASE=${SRNDNA_STAN_SCRATCH_BASE:-/ZPOOL/data/scratch}
CMDSTAN_ROOT=${SRNDNA_CMDSTAN_ROOT:-/ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0}
PHASE=fit; WARMUP=3000; SAMPLES=6000
SUFFIX=positive-bias-v2
if test "$MODE" = smoke; then
  PHASE=smoke; WARMUP=100; SAMPLES=100; SUFFIX=positive-bias-smoke-v2
fi
WORK_ROOT="$SCRATCH_BASE/srndna-gu-stan-$SUFFIX"
OUTPUT_ROOT="$CODE_ROOT/results/norm_learning/stan-$SUFFIX"
mkdir -p "$OUTPUT_ROOT" || exit 1
ARGS=(--cmdstan "$CMDSTAN_ROOT" --work-root "$WORK_ROOT"
      --phase "$PHASE" --models rw_positive rw_positive_bias --stages full run1
      --jobs 40 --chains 4 --threads-per-chain 2
      --warmup "$WARMUP" --samples "$SAMPLES" --adapt-delta .99
      --collect-to "$OUTPUT_ROOT")
LOG=preflight.txt
if test "${2:-}" = --execute; then ARGS+=(--execute); LOG=run.txt; fi
# No shell-wide errexit/pipefail: explicitly check both fitting and logging.
python -u code/run_gu_choice_extensions.py "${ARGS[@]}" 2>&1 | tee -a "$OUTPUT_ROOT/$LOG"
STATUSES=("${PIPESTATUS[@]}")
STATUS=${STATUSES[0]}
if test "${STATUSES[1]}" -ne 0; then echo "Log write failed" >&2; exit 1; fi
if test "$STATUS" -ne 0 && test "$STATUS" -ne 2; then
  echo "Fitting failed: exit $STATUS. Rerun the same command to resume completed fits." >&2
  exit "$STATUS"
fi
if test "${2:-}" != --execute; then exit "$STATUS"; fi
python -u code/export_gu_stan_diagnostics.py \
  --work-root "$WORK_ROOT" --phase "$PHASE" --expected-jobs 4 \
  --output-dir "$OUTPUT_ROOT/diagnostics-v1" 2>&1 | tee -a "$OUTPUT_ROOT/export.txt"
STATUSES=("${PIPESTATUS[@]}")
if test "${STATUSES[0]}" -ne 0; then exit "${STATUSES[0]}"; fi
if test "${STATUSES[1]}" -ne 0; then exit 1; fi
echo "Saved summaries, participant prediction checks, parameter trade-offs, and diagnostic plots: $OUTPUT_ROOT"
if test "$STATUS" -eq 2; then
  echo "All fitting jobs finished; diagnostic flags retained. Do not treat this as a clean scientific pass."
fi
exit "$STATUS"
