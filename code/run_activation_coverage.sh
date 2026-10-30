#!/usr/bin/env bash
# New scratch only; retain existing models, reports and PALM workers.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$repo" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
parent="$repo/results/activation_coverage/v1"
mkdir -p "$parent" || exit 1
attempt=$(mktemp -d "$parent/launch-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX") || exit 1
run_check() {
    git rev-parse HEAD || return $?
    python3 --version || return $?
    python3 -m unittest discover -s tests -p test_activation_coverage.py -v || return $?
    python3 -u code/run_activation_coverage.py "$@" || return $?
    python3 -u code/run_activation_coverage.py "$@" --execute
}
run_check "$@" 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
if [[ $status -ne 0 ]]; then
    # Retain completed preparation tables even when preflight failed before
    # the compact-report export. Never copy merged imaging data to Git.
    python3 - "$attempt" "$@" <<'PY'
from pathlib import Path
import argparse, shutil, sys
p=argparse.ArgumentParser();p.add_argument('--work-root',default='/ZPOOL/data/scratch/srndna-ultimatum-activation-coverage-v1')
a,_=p.parse_known_args(sys.argv[2:]);work=Path(a.work_root);out=Path(sys.argv[1])/'preflight'
for root in work.glob('preparation-*'):
    for source in root.rglob('*'):
        if source.is_file() and source.suffix in ('.json','.tsv') and source.stat().st_size < 20_000_000:
            target=out/root.name/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
PY
fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved activation coverage report: $parent (exit $status)"
echo 'Repeat this command to resume verified jobs. Commit/push results/activation_coverage/v1, including failure logs.'
exit "$status"
