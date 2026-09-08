#!/usr/bin/env bash

# Read-only collection of production imaging provenance. This script never
# invokes feat, flameo, randomise, cluster inference, or any file-removal tool.

set -euo pipefail

usage() {
    echo "Usage: $0 --fmriprep-root DIR --group-dir DIR --output-dir DIR" >&2
}

fmriprep_root=""
group_dir=""
output_dir=""
while (($#)); do
    case "$1" in
        --fmriprep-root) fmriprep_root="$2"; shift 2 ;;
        --group-dir) group_dir="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -d "$fmriprep_root" && -d "$group_dir" && -n "$output_dir" ]] || { usage; exit 2; }
mkdir -p "$output_dir/design" "$output_dir/headers" "$output_dir/cluster-tables" \
    "$output_dir/fmriprep-provenance"

{
    date -u '+collected_utc=%Y-%m-%dT%H:%M:%SZ'
    echo "fmriprep_root=$fmriprep_root"
    echo "group_dir=$group_dir"
    command -v fslhd || true
    command -v feat || true
    if command -v fslversion >/dev/null 2>&1; then fslversion; fi
    if [[ -n "${FSLDIR:-}" && -f "${FSLDIR}/etc/fslversion" ]]; then
        echo "FSLDIR_version=$(<"${FSLDIR}/etc/fslversion")"
    fi
    if command -v python3 >/dev/null 2>&1; then python3 --version; fi
} > "$output_dir/environment.txt"

find "$fmriprep_root" -maxdepth 3 -type f \
    \( -name dataset_description.json -o -name '*desc-about.html' -o -name '*CITATION*' \) \
    -print | sort > "$output_dir/fmriprep-provenance-files.txt"

provenance_index=0
while IFS= read -r source; do
    [[ -f "$source" ]] || continue
    provenance_index=$((provenance_index + 1))
    target_name="$(printf '%03d_%s' "$provenance_index" "$(basename "$source")")"
    cp -p "$source" "$output_dir/fmriprep-provenance/$target_name"
done < "$output_dir/fmriprep-provenance-files.txt"

while IFS= read -r source; do
    [[ -f "$source" ]] || continue
    target="$output_dir/design/$(basename "$source")"
    cp -p "$source" "$target"
done < <(find "$group_dir" -maxdepth 2 -type f \
    \( -name design.fsf -o -name design.mat -o -name design.con -o -name design.grp -o -name design.fts \
       -o -name smoothness -o -name resels.dof -o -name dof \) \
    -print | sort)

while IFS= read -r source; do
    [[ -f "$source" ]] || continue
    cp -p "$source" "$output_dir/cluster-tables/$(basename "$source")"
done < <(find "$group_dir" -maxdepth 3 -type f \
    \( -name 'cluster_zstat*.txt' -o -name 'cluster_zstat*.html' -o -name 'report_log.html' \) \
    -print | sort)

while IFS= read -r image; do
    [[ -f "$image" ]] || continue
    name="$(basename "$image")"
    fslhd "$image" > "$output_dir/headers/${name}.fslhd.txt"
    fslstats "$image" -V > "$output_dir/headers/${name}.volume.txt"
done < <(find "$group_dir" -maxdepth 4 -type f \
    \( -name 'zstat*.nii.gz' -o -name 'cluster_mask_zstat*.nii.gz' -o -name 'thresh_zstat*.nii.gz' \
       -o -name 'cope*.nii.gz' -o -name 'mask.nii.gz' \) \
    -print | sort)

if command -v sha256sum >/dev/null 2>&1; then
    find "$output_dir" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$output_dir/SHA256SUMS"
else
    find "$output_dir" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > "$output_dir/SHA256SUMS"
fi

echo "PASS: read-only provenance bundle written to $output_dir"
