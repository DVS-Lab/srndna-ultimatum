#!/usr/bin/env bash
# Read-only input audit, descriptive plots and extraction. No imaging refits.
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
report=${1:-$repo/results/permutation_followup/v1}
cd "$repo" || exit 1
mkdir -p "$report" || exit 1
attempt="$report/attempt-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir "$attempt" || exit 1
run_batch() {
    printf 'Started UTC: %s\n' "$(date -u +%FT%TZ)"
    git rev-parse HEAD || return 1
    python3 --version || return 1
    python3 -m unittest discover -s tests -p test_permutation_followup.py -v || return $?
    local audit_status plot_status
    python3 -u code/audit_dmn_covariate_sources.py \
        --search-root /ZPOOL/data/projects/srndna-ug \
        --search-root /ZPOOL/data/projects/srndna-data \
        --search-root /ZPOOL/data/projects/srndna-datapaper-code/derivatives/mriqc \
        --search-root /ZPOOL/data/datasets/ds003745-work/derivatives/mriqc \
        --output "$report/covariate_audit"
    audit_status=$?
    python3 -u code/plot_permutation_followup.py \
        --report "$repo/results/permutation_audit/v1" \
        --output "$report/figures_and_extractions"
    plot_status=$?
    if [[ $audit_status -ne 0 || $plot_status -ne 0 ]]; then
        echo "STOP: audit exit=$audit_status; plots/extraction exit=$plot_status. Push the diagnostic output."
        return 1
    fi
    echo 'PASS: overlays, brain-behavior plots, influence diagnostics and covariate search completed'
}
run_batch 2>&1 | tee "$attempt/run.txt"
statuses=("${PIPESTATUS[@]}")
status=${statuses[0]}
if [[ ${statuses[1]} -ne 0 ]]; then status=${statuses[1]}; fi
printf '%s\n' "$status" > "$attempt/exit_status.txt"
echo "Saved report: $report (exit $status)"
exit "$status"
