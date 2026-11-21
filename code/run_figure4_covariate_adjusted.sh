#!/usr/bin/env bash
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
figure_dir="$repo/results/manuscript/figure4_partner_displays"
if python3 -c 'import numpy, nibabel' 2>/dev/null; then
    py=(python3)
else
    py=(conda run -n srndna-ultimatum python3)
fi
stage=$(mktemp -d)
printf 'Working directory: %s\n' "$stage"
"${py[@]}" "$repo/code/extract_figure4_conditions.py" --repo "$repo" --adjusted \
    --output "$stage/all_estimates.tsv" | tee "$stage/adjusted_validation.txt"
"${py[@]}" -c '
import csv, math, sys
from pathlib import Path
stage, reference = map(Path, sys.argv[1:])
def cells(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    keys = {(r["age_group"], r["condition"]): r for r in rows}
    if len(rows) != 6 or len(keys) != 6:
        raise ValueError("Expected six unique cells")
    return keys
old, new = cells(reference), cells(stage/"all_estimates.tsv")
if old.keys() != new.keys():
    raise ValueError("Participant groups or partner labels changed")
fields = ("age_group", "condition", "n", "covariate_adjusted_mean", "covariate_adjusted_sem")
with (stage/"condition_estimates_adjusted.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for key, r in new.items():
        if r["n"] != old[key]["n"] or not all(math.isclose(float(r[k]), float(old[key][k]), rel_tol=1e-9, abs_tol=1e-9) for k in ("mean", "sem")):
            raise ValueError(f"Raw estimates differ from the verified Linux1 extraction for {key}")
        if not all(math.isfinite(float(r[k])) for k in fields[-2:]) or float(r["covariate_adjusted_sem"]) < 0:
            raise ValueError(f"Invalid adjusted estimates for {key}")
        writer.writerow(r)
' "$stage" "$figure_dir/condition_estimates.tsv"
cp "$stage/condition_estimates_adjusted.tsv" "$stage/adjusted_validation.txt" "$figure_dir/"
cat "$figure_dir/condition_estimates_adjusted.tsv"
