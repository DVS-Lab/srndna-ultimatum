#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
project_root="$(dirname "$script_dir")"

bash -n "$script_dir/validate_workflow.sh" "$script_dir/audit_server_imaging.sh" \
    "$script_dir/run_logged.sh"
echo "PASS: active shell syntax"

PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/srndna-ug-pycache" \
    python3 -m py_compile "$script_dir/audit_task_events.py" "$script_dir/audit_image_headers.py" \
    "$script_dir/audit_l1_designs.py" "$script_dir/audit_l3_template_inputs.py" \
    "$script_dir/audit_server_rt_events.py" "$script_dir/audit_ultimatum_event_designs.py" \
    "$script_dir/build_event_corrected_trials.py" "$script_dir/make_ultimatum_3col.py" \
    "$script_dir/prepare_sub144_imaging_repair.py" \
    "$script_dir/run_ultimatum_repair_jobs.py"
echo "PASS: active Python syntax"

if command -v Rscript >/dev/null 2>&1; then
    Rscript -e "parse(file='$script_dir/analyze_reviewer_behavior.R'); parse(file='$script_dir/audit_l3_designs.R'); parse(file='$script_dir/audit_roi_influence.R')" >/dev/null
    echo "PASS: active R syntax"
else
    echo "SKIP: Rscript is not installed"
fi

cd "$project_root"
python3 -m unittest discover -s tests -p 'test_*.py' -v

python3 "$script_dir/audit_task_events.py"
python3 "$script_dir/build_event_corrected_trials.py"

if command -v fslhd >/dev/null 2>&1 && command -v fslstats >/dev/null 2>&1; then
    python3 "$script_dir/audit_image_headers.py"
else
    echo "SKIP: FSL header tools are not installed"
fi

if grep -En '/Users/[^/]+|/home/[^/]+' \
    "$script_dir/audit_task_events.py" "$script_dir/audit_image_headers.py" \
    "$script_dir/audit_server_rt_events.py" "$script_dir/audit_ultimatum_event_designs.py" \
    "$script_dir/make_ultimatum_3col.py" "$script_dir/prepare_sub144_imaging_repair.py" \
    "$script_dir/run_ultimatum_repair_jobs.py" "$script_dir/audit_l3_designs.R" \
    "$script_dir/audit_roi_influence.R"; then
    echo "ERROR: active reviewer workflow contains a personal absolute path" >&2
    exit 1
fi
echo "PASS: active reviewer workflow is path-portable"
