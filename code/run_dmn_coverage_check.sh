#!/usr/bin/env bash
# Use a separate tmux pane. Does not signal, drain or restart the original PALM batch.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="$repo/results/dmn_followup/v1/coverage_check"
mkdir -p "$parent" || exit 1
attempt=$(mktemp -d "$parent/attempt-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_check() {
    git rev-parse HEAD || return $?
    python3 -m unittest discover -s tests -p test_dmn_coverage_check.py -v || return $?
    python3 -u code/run_dmn_coverage_check.py "$@" || return $?
    python3 -u code/run_dmn_coverage_check.py "$@" --execute
}
run_check "$@" 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}"); status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
# Always retain any FILM preflight evidence, including from an incomplete preparation.
python3 - "$attempt" "$@" <<'PY'
from pathlib import Path
import argparse,shutil,sys
p=argparse.ArgumentParser();p.add_argument('--work-root',default='/ZPOOL/data/scratch/srndna-ultimatum-dmn-coverage-v1')
args,_=p.parse_known_args(sys.argv[2:]);work=Path(args.work_root)
out=Path(sys.argv[1])/'film_preflight'
for root in work.glob('preparation-*'):
    for name in ('film_thresholds.tsv','film_command_evidence.json'):
        p=root/name
        if p.is_file():
            target=out/root.name/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
PY
copy_status=$?
if [[ $status -eq 0 && $copy_status -ne 0 ]]; then status=$copy_status; fi
python3 -u code/collect_palm_progress.py --metadata-only --summarize 2>&1 | tee "$attempt/original_palm_progress.txt"
collection_statuses=("${PIPESTATUS[@]}")
if [[ $status -eq 0 && ${collection_statuses[0]} -ne 0 ]]; then status=${collection_statuses[0]}; fi
if [[ $status -eq 0 && ${collection_statuses[1]} -ne 0 ]]; then status=${collection_statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved separate coverage-check results/logs: $parent (exit $status)"
echo 'Commit/push coverage_check and palm/collections, including logs on failure. Rerun the same command to resume verified jobs.'
exit "$status"
