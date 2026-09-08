#!/usr/bin/env bash

# Run a command while preserving its exit status and a timestamped local log.

set -euo pipefail

if (($# < 2)); then
    echo "Usage: $0 LABEL COMMAND [ARG ...]" >&2
    exit 2
fi

label="$1"
shift
if [[ ! "$label" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: label may contain only letters, numbers, dot, underscore, and hyphen" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
project_root="$(dirname "$script_dir")"
log_dir="$project_root/logs/runs"
mkdir -p "$log_dir"
timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
log_path="$log_dir/${label}_${timestamp}_$$.log"

{
    echo "started_utc=$timestamp"
    echo "working_directory=$(pwd)"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
    "$@"
} 2>&1 | tee "$log_path"

echo "Log: $log_path"
