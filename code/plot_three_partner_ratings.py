#!/usr/bin/env python3
"""Plot retained last-complete Ultimatum ratings; no new behavioral tests."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARTNERS = ('computer', 'similar', 'dissimilar')
COLORS = ('#8a8a8a', '#287d8e', '#bc6545')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, delimiter='\t'):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write(path, rows):
    if not rows:
        raise ValueError(f'empty table: {path}')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def save_figure(fig, path):
    fig.savefig(path, dpi=220)
    if path.suffix=='.svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def selected_cells(rows):
    cells = {}
    for row in rows:
        if row['policy'] != 'last_complete_block' or row['age_group'] != 'all':
            continue
        key = row['timepoint'], row['rating_dimension'], row['partner']
        if key in cells:
            raise ValueError(f'duplicate ratings cell: {key}')
        n, mean, sd, sem = (float(row[k]) for k in ('n', 'mean', 'sd', 'sem'))
        if n != 42 or not 0 <= mean <= 10 or sd < 0 or not np.isfinite([mean,sd,sem]).all():
            raise ValueError(f'unexpected ratings sample/range: {key}')
        if not np.isclose(sem, sd/np.sqrt(n), rtol=1e-6):
            raise ValueError(f'ratings SEM mismatch: {key}')
        cells[key] = row
    expected = {(t,d,p) for t,d in [('pre','fairness'),('post','fairness'),
                ('pre','likeability'),('post','likeability'),('post','anger'),
                ('post','satisfaction')] for p in PARTNERS}
    if set(cells) != expected:
        raise ValueError('missing or unexpected primary ratings cells')
    return cells


def build(repo, output):
    source = repo/'results/reviewer/ultimatum_ratings'
    paths = [source/'ultimatum_ratings_descriptives.tsv',
             source/'ultimatum_ratings_within_subject_tests.tsv',
             source/'ultimatum_ratings_sample.tsv']
    contract = {'code': digest(__file__), 'inputs': {str(p.relative_to(repo)):digest(p) for p in paths}}
    output.mkdir(parents=True, exist_ok=True)
    cp = output/'contract.json'
    if (output/'complete.json').exists() and not cp.exists():
        raise ValueError('completed ratings outputs lack their input contract')
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError('ratings contract changed; use a new output directory')
    receipt=output/'complete.json'
    if receipt.exists():
        for name,checksum in json.loads(receipt.read_text())['outputs'].items():
            if digest(output/name)!=checksum: raise ValueError(f'completed ratings output changed: {name}')
        print(f'SKIP_COMPLETE: verified ratings figures: {output}',flush=True)
        return
    save_json(cp, contract)
    cells = selected_cells(read(paths[0]))
    write(output/'bar_source.tsv', list(cells.values()))
    tests = [r for r in read(paths[1]) if r['policy']=='last_complete_block']
    write(output/'retained_paired_tests.tsv', tests)
    for name, panels, shape in [
        ('three_partner_ratings', [('pre','fairness'),('pre','likeability'),
                                   ('post','fairness'),('post','likeability')], (2,2)),
        ('post_task_affect', [('post','anger'),('post','satisfaction')], (1,2))]:
        fig, axes = plt.subplots(*shape, figsize=(9,7 if shape[0]==2 else 4.5), squeeze=False)
        for ax, (timepoint, dimension) in zip(axes.flat, panels):
            records = [cells[timepoint,dimension,p] for p in PARTNERS]
            ax.bar(np.arange(3), [float(r['mean']) for r in records],
                   yerr=[float(r['sem']) for r in records], color=COLORS,
                   width=.65, capsize=4, edgecolor='white')
            ax.set_xticks(np.arange(3), ['Computer','Similar','Dissimilar'])
            ax.set_ylim(0,10); ax.set_yticks([0,2,4,6,8,10])
            ax.set_ylabel('Rating (0–10)')
            ax.set_title(f'{timepoint.capitalize()}-task {dimension}  ·  n = 42')
            ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(rect=(0,.12,1,1))
        fig.text(.08,.025,'Means ± participant SEM; last complete attempt.\n'
                 'Similar/dissimilar refer to partner age relative to the participant.\n'
                 'Pre/post samples overlap but are not identical (41 paired participants).',fontsize=9)
        for ext in ('png','svg'):
            save_figure(fig,output/f'{name}.{ext}')
        plt.close(fig)
    save_json(output/'status.json', {'complete':True,'new_tests':False,
        'policy':'last_complete_block','error_bars':'participant SD / sqrt(n)',
        'interpretation':'Partner rating levels, not demonstrated changes caused by the task.'})
    save_json(receipt,{'outputs':{p.name:digest(p) for p in output.iterdir()
        if p.is_file() and p.name!='complete.json'}})
    print(f'PASS: three-partner ratings figures: {output}', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/vmpfc_ratings/v1/ratings')
    args=parser.parse_args(); build(ROOT,args.output.resolve())
