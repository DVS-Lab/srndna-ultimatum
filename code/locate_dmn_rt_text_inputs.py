#!/usr/bin/env python3
"""Locate retained EV/confound candidates without choosing or changing inputs.

Searches only named project input directories, never MRI files or FEAT outputs.
Matches participant plus filename (allowing run-1/run-01 spelling), fingerprints
small text candidates, and records absence/ambiguity. This does NOT establish
agreement with the original design or authorize substituting a candidate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

DEFAULT_ROOTS = tuple(
    Path(base)/'derivatives/fsl'/leaf
    for base in (
        '/ZPOOL/data/projects/srndna-ultimatum',
        '/ZPOOL/data/projects/srndna-ug',
        '/ZPOOL/data/projects/srndna-data',
        '/ZPOOL/data/datasets/ds003745-work',
    ) for leaf in ('EVfiles', 'confounds')
) + (Path('/ZPOOL/data/scratch/srndna-datapaper-lss/confounds'),)
ROLES = {'confoundev_files(1)', *(f'custom{i}' for i in range(1, 10))}
MAX_BYTES = 5_000_000


def normalized_name(name):
    return re.sub(r'run[-_]?0*(\d+)', lambda match: 'run-'+str(int(match[1])), name)


def wanted_inputs(inventory):
    return [r for r in inventory if r['status']=='missing' and r['role'] in ROLES]


def locate(inventory, roots):
    wanted = wanted_inputs(inventory)
    names = {normalized_name(Path(r['recorded']).name) for r in wanted}
    index, root_rows, errors = {}, [], []
    for root in roots:
        if not root.is_absolute() or root == Path('/'):
            raise ValueError('search roots must be explicit absolute input directories, not /')
        root_rows.append(dict(root=str(root), exists=root.is_dir()))
        if not root.is_dir():
            continue
        def failed(error):
            errors.append(str(error))
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=failed):
            dirs[:] = sorted(d for d in dirs if d != '.git' and not d.endswith(('.feat','.gfeat')))
            for name in sorted(files):
                key = normalized_name(name)
                if key not in names:
                    continue
                path = Path(directory)/name
                subjects = set(re.findall(r'(?<![A-Za-z0-9])sub-\d+(?!\d)', str(path)))
                for subject in subjects:
                    index.setdefault((subject,key), set()).add(path)
    rows, summary_rows, cache = [], [], {}
    for item in wanted:
        candidates = sorted(index.get((item['subject'], normalized_name(Path(item['recorded']).name)), []))
        valid_hashes, valid_count = set(), 0
        for path in candidates:
            try:
                real = path.resolve(strict=True)
                size = real.stat().st_size
                if real not in cache:
                    if size == 0 or size > MAX_BYTES:
                        raise ValueError(f'empty or exceeds {MAX_BYTES}-byte text-input limit')
                    cache[real] = hashlib.sha256(real.read_bytes()).hexdigest()
                fingerprint, error = cache[real], ''
                valid_hashes.add(fingerprint)
                valid_count += 1
            except (OSError, ValueError) as exc:
                real, size, fingerprint, error = '', '', '', str(exc)
            rows.append(dict(subject=item['subject'], run=item['run'], role=item['role'], recorded=item['recorded'],
                             candidate=str(path), resolved=str(real), bytes=size, sha256=fingerprint, error=error))
        state = ('not_found' if not candidates else 'unreadable' if not valid_count else
                 'different_copies' if len(valid_hashes)>1 else 'candidate_found')
        summary_rows.append(dict(subject=item['subject'],run=item['run'],role=item['role'],recorded=item['recorded'],
                                 status=state, candidates=len(candidates), readable_candidates=valid_count,
                                 distinct_sha256=len(valid_hashes)))
    summary = dict(required_missing_text_references=len(wanted),
                   references_with_readable_candidates=sum(r['readable_candidates']>0 for r in summary_rows),
                   references_without_readable_candidates=sum(r['readable_candidates']==0 for r in summary_rows),
                   references_with_different_copies=sum(r['status']=='different_copies' for r in summary_rows),
                   candidate_paths=len(rows), search_roots=root_rows, search_errors=errors,
                   note='Candidates only: no inputs replaced or regenerated, no FEAT jobs run. Matching copies do not prove agreement with an unavailable original; different copies need review.')
    return summary_rows, rows, summary


def write_table(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, default=repo/'results/reviewer/dmn_rt_input_audit/inputs.tsv')
    p.add_argument('--search-root', type=Path, action='append', default=[], help='add a known archived input directory to the bounded default roots')
    p.add_argument('--output-dir', type=Path, default=repo/'results/reviewer/dmn_rt_text_input_search')
    a = p.parse_args()
    with a.inventory.open() as stream:
        inventory = list(csv.DictReader(stream, delimiter='\t'))
    required, candidates, summary = locate(inventory, (*DEFAULT_ROOTS, *a.search_root))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(a.output_dir/'references.tsv', ['subject','run','role','recorded','status','candidates','readable_candidates','distinct_sha256'], required)
    write_table(a.output_dir/'candidates.tsv', ['subject','run','role','recorded','candidate','resolved','bytes','sha256','error'], candidates)
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(f'Search completed: {a.output_dir}; this is NOT model-launch approval.')
    return 0  # A successful search can establish absence; summary is explicit.


if __name__ == '__main__':
    raise SystemExit(main())
