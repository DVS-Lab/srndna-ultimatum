#!/usr/bin/env python3
"""Collect all planned contrasts, including nulls, from DMN sensitivity models.

Missing poststatistics are errors, never interpreted as absence of clusters.
Only compact maps/designs/tables are copied; full FEAT trees stay on Linux1.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

from audit_l1_designs import write_tsv
from prepare_dmn_revision_checks import CONTRASTS, digest
from prepare_sub144_imaging_repair import fsf_value


def collect(manifest, destination, expected_jobs=2):
    with manifest.open() as stream:
        jobs = [r for r in csv.DictReader(stream, delimiter='\t') if r['stage']=='l3']
    if len(jobs) != expected_jobs:
        raise ValueError(f'expected {expected_jobs} L3 jobs in this batch')
    if len({r['run'] for r in jobs}) != len(jobs):
        raise ValueError('collection names must be unique')
    sources, summary = [], []
    for row in jobs:
        name, output = row['run'], Path(row['output'])
        if digest(Path(row['fsf'])) != row['fsf_sha256']:
            raise ValueError('prepared FSF changed')
        feat = output / 'cope1.feat'
        for file in ('design.fsf', 'design.mat', 'design.con', 'design.grp'):
            sources.append((output/file, Path(name)/'design'/file))
        for file in ('mask.nii.gz', 'stats/smoothness'):
            sources.append((feat/file, Path(name)/'poststats'/Path(file).name))
        contrasts = ([fsf_value(Path(row['fsf']).read_text(), f'conname_real.{i}')
                      for i in range(1,int(row['expected_zstats'])+1)]
                     if row.get('group_contract') == 'locked-source'
                     else [name for name, _ in CONTRASTS])
        for i, contrast in enumerate(contrasts, 1):
            table = feat / f'cluster_zstat{i}_std.txt'
            if not table.is_file():
                raise FileNotFoundError(f'missing cluster table; not a null result: {table}')
            with table.open() as stream:
                reader = csv.DictReader(stream, delimiter='\t')
                if not {'Cluster Index', 'Voxels', 'P'}.issubset(reader.fieldnames or []):
                    raise ValueError(f'invalid cluster table: {table}')
                clusters = list(reader)
            summary.append(dict(model=name, contrast=i, contrast_name=contrast,
                                clusters=len(clusters), voxels=sum(int(r['Voxels']) for r in clusters),
                                minimum_cluster_p=min(float(r['P']) for r in clusters) if clusters else '',
                                maximum_z=max(float(r['Z-MAX']) for r in clusters) if clusters else '',
                                status='complete'))
            for file in (f'stats/zstat{i}.nii.gz', f'thresh_zstat{i}.nii.gz',
                         f'cluster_mask_zstat{i}.nii.gz', f'cluster_zstat{i}_std.txt'):
                sources.append((feat/file, Path(name)/f'contrast{i}'/Path(file).name))
    sources.extend((p, Path(p.name)) for p in [manifest, manifest.parent/'analysis_plan.json', manifest.parent/'l3_preflight.tsv'])
    for optional in ('l1_preflight.tsv', 'l2_preflight.tsv', 'baseline_preflight.tsv', 'rt_construction.tsv', 'input_provenance.json', 'input_path_resolution.json', 'software.json', 'preparation_config.json'):
        if (manifest.parent/optional).is_file():
            sources.append((manifest.parent/optional, Path(optional)))
    for source, _ in sources:
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f'missing/empty result: {source}')
        if source.stat().st_size > 20_000_000:
            raise ValueError(f'not a compact output: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.dmn-collect-') as tmp:
        staging = Path(tmp)
        inventory = []
        for source, relative in sources:
            target = staging/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inventory.append(dict(source=str(source), artifact=str(relative), bytes=source.stat().st_size, sha256=digest(source)))
        write_tsv(staging/'inventory.tsv', inventory)
        write_tsv(staging/'summary.tsv', summary)
        if destination.exists():
            existing = {str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()}
            current = {str(p.relative_to(staging)):digest(p) for p in staging.rglob('*') if p.is_file()}
            if existing != current:
                raise FileExistsError(f'collection differs; inspect and use a new output root: {destination}')
        else:
            shutil.copytree(staging, destination)
    print(f'PASS: collected all {len(summary)} planned contrasts, including nulls: {destination}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    collect(a.manifest.resolve(), a.output_root.resolve())


if __name__ == '__main__':
    main()
