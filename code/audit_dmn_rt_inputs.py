#!/usr/bin/env python3
"""Inventory RT-sensitivity inputs and surviving BOLD copies; never fit models.

Candidate locations are reported, never used as automatic replacements. Optional
hash comparison establishes equality between surviving copies only, not identity
to an unavailable original. Reports contain paths/provenance, not MRI contents.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

from audit_l1_designs import write_tsv
from prepare_dmn_rt_sensitivity import input_inventory, parse_input_maps

DEFAULT_CANDIDATES = (
    Path('/ZPOOL/data/projects/srndna-data/derivatives/fmriprep'),
    Path('/ZPOOL/data/projects/srndna-ug/derivatives/fmriprep'),
    Path('/ZPOOL/data/datasets/ds003745-work/derivatives/fmriprep'),
)


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def bold_candidates(inventory, roots, compare=False):
    rows, cache = [], {}
    for item in inventory:
        if item['role'] != 'feat_files(1)':
            continue
        _, marker, relative = item['recorded'].partition('/derivatives/fmriprep/')
        if not marker or '..' in Path(relative).parts or Path(relative).is_absolute():
            continue
        for root in roots:
            path = root / relative
            exists = path.is_file()
            digest = ''
            if exists and compare:
                real = path.resolve()
                if real not in cache:
                    print(f'HASH: {path}', flush=True)
                    cache[real] = sha256(real)
                digest = cache[real]
            rows.append(dict(subject=item['subject'], run=item['run'], root=str(root),
                             path=str(path), exists=int(exists), bytes=path.stat().st_size if exists else '',
                             sha256=digest))
    return rows


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, default=repo)
    p.add_argument('--production-fsl-root', type=Path, default=Path('/ZPOOL/data/projects/srndna-ultimatum/derivatives/fsl'))
    p.add_argument('--repaired-fsl-root', type=Path, default=Path('/ZPOOL/data/scratch/srndna-ultimatum-sub144-repair-v2/derivatives/fsl'))
    p.add_argument('--standard-image', type=Path,
                   default=Path(os.environ.get('FSLDIR', '/usr/local/fsl'))/'data/standard/MNI152_T1_2mm_brain.nii.gz')
    p.add_argument('--input-map', action='append', default=[], metavar='OLD=NEW')
    p.add_argument('--candidate-fmriprep-root', type=Path, action='append', help='repeat to replace the three default candidate locations')
    p.add_argument('--compare-bold-copies', action='store_true', help='stream SHA256 over surviving candidate BOLD files; may take time')
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    roots = a.candidate_fmriprep_root or DEFAULT_CANDIDATES
    inventory = input_inventory(a.repository.resolve(), a.production_fsl_root.resolve(),
                                a.repaired_fsl_root.resolve(), a.standard_image.resolve(), parse_input_maps(a.input_map))
    candidates = bold_candidates(inventory, roots, a.compare_bold_copies)
    output = a.output_dir or a.repository/'results/reviewer/dmn_rt_input_audit'
    output.mkdir(parents=True, exist_ok=True)
    write_tsv(output/'inputs.tsv', inventory)
    with (output/'bold_candidates.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['subject','run','root','path','exists','bytes','sha256'], delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(candidates)
    versions = {}
    for root in roots:
        description = root/'dataset_description.json'
        if description.is_file():
            data = json.loads(description.read_text())
            versions[str(root)] = {key: data[key] for key in ('GeneratedBy', 'PipelineDescription', 'Name', 'BIDSVersion') if key in data}
    missing = Counter(row['role'] for row in inventory if row['status'] == 'missing')
    summary = dict(required_references=len(inventory), missing_references=sum(missing.values()),
                   missing_by_role=dict(missing), candidate_bold_counts={str(root):sum(r['exists'] for r in candidates if r['root']==str(root)) for root in roots},
                   derivative_metadata=versions, hashes_computed=a.compare_bold_copies,
                   warning='Candidate availability/version agreement alone does not prove equality to the historical input. No candidate is automatically selected.')
    if a.compare_bold_copies:
        grouped = {}
        for row in candidates:
            if row['sha256']:
                grouped.setdefault((row['subject'], row['run']), []).append(row['sha256'])
        summary['multiple_copy_runs_checked'] = sum(len(v)>1 for v in grouped.values())
        summary['differing_copy_runs'] = [f'{s}/run-{r}' for (s,r),v in grouped.items() if len(set(v))>1]
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Inventory: {output}; no FEAT models launched or imaging files changed.')
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
