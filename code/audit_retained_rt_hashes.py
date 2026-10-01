#!/usr/bin/env python3
"""Identify retained RT-file contents by exact SHA-256 match; never alter models.

Uses fingerprints exported from Linux1, not a claim that local source files
necessarily equal fitted inputs. A failure to match is UNKNOWN, never an omission.
The sub-144 inputs belong to the corrected baseline, not its historical fit.
"""
import argparse
import csv
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path


def read_tsv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def fmt(value, spec):
    return value if spec == 'raw' else format(float(value), spec)


def match_candidates(rows, ev, target):
    blocks = {float(r['onset']) for r in rows if r['trial_type'].startswith('block_')}
    responded = [r for r in rows if re.match(r'^event_(accept|reject)_', r['trial_type'])]
    first = {float(r['onset']) for r in responded if float(r['onset']) in blocks}
    candidates = {
        'companion_rows': [r for r in rows if r['trial_type'] == 'event_RT'],
        'all_responded': responded,
        'responded_excluding_block_first': [r for r in responded if float(r['onset']) not in first],
    }
    matches = []
    for label, selected in candidates.items():
        # Retained FEAT inputs use tab-delimited three-column text. Match several
        # historical numeric renderings, without changing numerical RT values.
        for onset_fmt, height_fmt in itertools.product(
                ('raw', '.6g', '.12g', '.15g'), ('raw', '.6g', '.12g', '.6f')):
            content = ''.join(
                fmt(r['onset'], onset_fmt) + '\t0\t' +
                fmt('1.0' if ev == 8 else r['response_time'], height_fmt) + '\n'
                for r in selected).encode()
            if sha(content) == target:
                matched_first = sorted(first & {float(r['onset']) for r in selected})
                matches.append(dict(candidate=label, onset_format=onset_fmt,
                                    height_format=height_fmt, n_rows=len(selected),
                                    block_first_onsets=matched_first, sha256=sha(content)))
    return responded, sorted(first), matches


def audit(repo):
    directory = repo / 'results/reviewer/dmn_rt_sensitivity'
    provenance = json.loads((directory / 'input_provenance.json').read_text())
    inventory = read_tsv(repo / 'results/reviewer/dmn_rt_input_audit_recovered/inputs.tsv')
    resolved = {(r['subject'], r['run'], r['role']): r['resolved'] for r in inventory}
    baseline = {(r['subject'], r['run']): r for r in read_tsv(directory / 'baseline_preflight.tsv')}
    construction = read_tsv(directory / 'rt_construction.tsv')
    runs = sorted({(r['subject'], r['run']) for r in construction})
    evidence = []
    trials = []
    for subject, run in runs:
        events = repo / f'source_data/bids/{subject}/func/{subject}_task-ultimatum_run-{run}_events.tsv'
        event_hash = sha(events.read_bytes())
        expected = {r['events_sha256'] for r in construction if (r['subject'], r['run']) == (subject, run)}
        if expected != {event_hash}:
            raise ValueError(f'Events changed since baseline audit: {subject} {run}')
        b = baseline[(subject, run)]
        if b['passed'] != '1' or b['source_design_sha256'] != b['compiled_sha256']:
            raise ValueError(f'No byte-identical baseline design replication: {subject} {run}')
        rows = read_tsv(events)
        hits = {}
        for ev in (8, 9):
            key = resolved[(subject, run, f'custom{ev}')]
            target = provenance[key]['sha256']
            responded, first, matches = match_candidates(rows, ev, target)
            patterns = {(m['n_rows'], tuple(m['block_first_onsets'])) for m in matches}
            if len(patterns) > 1:
                raise ValueError('Contradictory exact hash matches')
            match = matches[0] if matches else None
            hits[ev] = None if match is None else set(match['block_first_onsets'])
            evidence.append(dict(
                subject=subject, run=run, ev=ev,
                input_scope='corrected_sub144' if subject == 'sub-144' else 'retained_production',
                input_path=key, retained_sha256=target,
                matched_sha256='' if match is None else match['sha256'],
                match_status='EXACT_SHA256_MATCH' if match else 'UNKNOWN',
                candidates=';'.join(sorted({m['candidate'] for m in matches})),
                onset_format='' if match is None else match['onset_format'],
                height_format='' if match is None else match['height_format'],
                responded_trials=len(responded), matched_rt_rows='' if match is None else match['n_rows'],
                responded_block_first=len(first),
                block_first_included='' if match is None else len(match['block_first_onsets']),
                block_first_excluded='' if match is None else len(first)-len(match['block_first_onsets']),
                source_design_sha256=b['source_design_sha256'],
                replicated_design_sha256=b['compiled_sha256'], events_sha256=event_hash,
            ))
        for r in responded:
            onset = float(r['onset'])
            if onset in first:
                trials.append(dict(subject=subject, run=run, onset=r['onset'],
                                   response_time=r['response_time'], trial_type=r['trial_type'],
                                   ev8_included='UNKNOWN' if hits[8] is None else int(onset in hits[8]),
                                   ev9_included='UNKNOWN' if hits[9] is None else int(onset in hits[9])))
    known = [r for r in evidence if r['match_status'] == 'EXACT_SHA256_MATCH']
    ev9 = [r for r in known if r['ev'] == 9]
    summary = dict(
        method='Exact SHA-256 reconstruction of exported baseline RT input fingerprints',
        participants=len({s for s, _ in runs}), runs=len(runs), rt_files=len(evidence),
        exact_matches=len(known), unknown_files=len(evidence)-len(known),
        byte_identical_baseline_designs=len(runs),
        source_scope='92 retained production runs plus 2 repaired sub-144 runs; not original sub-144',
        responded_trials=sum(r['responded_trials'] for r in ev9),
        matched_rt_rows=sum(r['matched_rt_rows'] for r in ev9),
        responded_block_first=sum(r['responded_block_first'] for r in ev9),
        block_first_included=sum(r['block_first_included'] for r in ev9),
        block_first_excluded=sum(r['block_first_excluded'] for r in ev9),
        inclusion_by_subject=dict(Counter(r['subject'] for r in trials if r['ev9_included'] == 1)),
        evidence_sha256={str(path.relative_to(repo)): sha(path.read_bytes()) for path in
            [directory/'input_provenance.json', directory/'baseline_preflight.tsv',
             directory/'rt_construction.tsv', repo/'results/reviewer/dmn_rt_input_audit_recovered/inputs.tsv']},
    )
    return evidence, trials, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory; existing evidence is never overwritten')
    evidence, trials, summary = audit(args.repo)
    args.output.mkdir(parents=True)
    for name, rows in [('retained_rt_matches.tsv', evidence), ('block_first_trials.tsv', trials)]:
        with (args.output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['unknown_files'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())

