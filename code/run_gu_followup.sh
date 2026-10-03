#!/usr/bin/env bash
# Persistent Linux1 fits and diagnostic exports. No imaging jobs are launched.
# Run inside tmux. Exit 2 means completed sampling with diagnostic flags.

MODE=${1:-}
case "$MODE" in
  baseline|age|smoke-age) ;;
  *) echo "Usage: bash code/run_gu_followup.sh {baseline|age|smoke-age}" >&2; exit 1 ;;
esac
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$CODE_ROOT" || exit 1
SCRATCH_BASE=${SRNDNA_STAN_SCRATCH_BASE:-/ZPOOL/data/scratch}
CMDSTAN_ROOT=${SRNDNA_CMDSTAN_ROOT:-/ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0}
case "$MODE" in
  baseline)
    WORK_ROOT="$SCRATCH_BASE/srndna-gu-stan-long-v2"
    OUTPUT_ROOT="$CODE_ROOT/results/norm_learning/stan-long-v2"
    PHASE=fit; AGE_MODE=none; WARMUP=3000; SAMPLES=6000 ;;
  age)
    WORK_ROOT="$SCRATCH_BASE/srndna-gu-stan-agegroup-v1"
    OUTPUT_ROOT="$CODE_ROOT/results/norm_learning/stan-agegroup-v1"
    PHASE=fit; AGE_MODE=group; WARMUP=3000; SAMPLES=6000 ;;
  smoke-age)
    WORK_ROOT="$SCRATCH_BASE/srndna-gu-stan-agegroup-smoke-v1"
    OUTPUT_ROOT="$CODE_ROOT/results/norm_learning/stan-agegroup-smoke-v1"
    PHASE=smoke; AGE_MODE=group; WARMUP=100; SAMPLES=100 ;;
esac
mkdir -p "$OUTPUT_ROOT" || exit 1
# Keep logs even when the terminal disconnects. The fit runner also locks scratch.
python -u code/run_gu_hierarchical.py \
  --cmdstan "$CMDSTAN_ROOT" --work-root "$WORK_ROOT" \
  --phase "$PHASE" --age-mode "$AGE_MODE" --age-prior-sd .5 \
  --jobs 40 --chains 4 --threads-per-chain 2 \
  --warmup "$WARMUP" --samples "$SAMPLES" --adapt-delta .99 \
  --collect-to "$OUTPUT_ROOT" --execute 2>&1 | tee -a "$OUTPUT_ROOT/run.log"
STATUS=${PIPESTATUS[0]}
if test "$STATUS" -ne 0 && test "$STATUS" -ne 2; then
  echo "Fitting failed (exit $STATUS); inspect $OUTPUT_ROOT/run.log" >&2
  exit "$STATUS"
fi
python -u code/export_gu_stan_diagnostics.py \
  --work-root "$WORK_ROOT" --phase "$PHASE" --expected-jobs 8 \
  --output-dir "$OUTPUT_ROOT/diagnostics-v1" 2>&1 | tee -a "$OUTPUT_ROOT/export.log"
EXPORT_STATUS=${PIPESTATUS[0]}
if test "$EXPORT_STATUS" -ne 0; then exit "$EXPORT_STATUS"; fi
echo "Saved summaries, diagnostic figures, and logs: $OUTPUT_ROOT"
if test "$STATUS" -eq 2; then
  echo "Sampling completed with diagnostic flags. Preserve and review outputs; do not discard chains."
fi
exit "$STATUS"
