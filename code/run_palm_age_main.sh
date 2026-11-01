#!/usr/bin/env bash
# New batch only; leave every historical output and existing PALM worker alone.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="$repo/results/palm_age_main/v1"
mkdir -p "$parent" || exit 1
attempt=$(mktemp -d "$parent/launch-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_batch() {
    git rev-parse HEAD || return $?
    python3 --version || return $?
    python3 -m unittest discover -s tests -p test_palm_age_main.py -v || return $?
    # --execute prepares, audits, runs the small MATLAB smoke, then launches.
    # No execution flag means preparation/validation only; --collect-only and
    # --smoke-only are explicit alternative modes handled by the Python runner.
    python3 -u code/run_palm_age_main.py "$@"
}
run_batch "$@" 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
if [[ $status -ne 0 ]]; then
    # Keep small preflight tables even if failure precedes normal export.
    python3 - "$attempt" "$@" <<'PY'
import argparse, shutil, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--work-root',type=Path,default=Path('/ZPOOL/data/scratch/srndna-ultimatum-palm-age-main-v1'))
a,_=p.parse_known_args(sys.argv[2:])
for root in a.work_root.glob('preparation-*'):
    for source in root.rglob('*'):
        if source.is_file() and source.suffix in ('.json','.tsv') and source.stat().st_size < 20_000_000:
            target=Path(sys.argv[1])/'preflight'/root.name/source.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
PY
fi
echo "Saved PALM age/main-effect outputs: $parent (exit $status)"
echo 'Repeat the same command to resume verified jobs. Commit/push results/palm_age_main/v1, including failure logs.'
exit "$status"
