#!/usr/bin/env python3
"""Prepare two DMN-only RT sensitivities; never overwrite primary derivatives.

all-trial-rt: retain every non-RT EV; replace RT impulses using every responded
substantive decision row. response-duration: same RT coverage, but shorten the
six task/offer EVs from offer onset to response, retaining original amplitudes.
BIDS response_time is offer-onset referenced: do NOT add one second.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from audit_l1_designs import write_tsv
from make_ultimatum_3col import read_events, ev_rows, three_column, TRIAL_RE
from prepare_dmn_condition_bar_models import parse_inputs, SOURCE_FSF_RELATIVE
from prepare_dmn_revision_checks import digest, group_job, safe_empty
from prepare_sub144_imaging_repair import fsf_value, replace_fsf_value, render_l2

VARIANTS = ('all-trial-rt', 'response-duration')


def decisions(events):
    rows = [r for r in events if TRIAL_RE.match(r['trial_type'])]
    onsets = np.array([float(r['onset']) for r in rows])
    durations = np.array([float(r['duration']) for r in rows])
    rt = np.array([float(r['response_time']) for r in rows])
    if not len(rows) or len(set(onsets)) != len(rows) or not np.isfinite([onsets, durations, rt]).all():
        raise ValueError('missing, duplicated, or nonfinite substantive events')
    if np.any(rt <= 0) or np.any(rt > durations + .05):
        raise ValueError('response_time is outside its display epoch; inspect timing provenance')
    return rows


def replace_durations(original, rows, partner):
    original = np.asarray(original, float)
    if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all():
        raise ValueError('invalid original three-column EV')
    subset = [r for r in rows if TRIAL_RE.match(r['trial_type']).group('partner') == partner]
    if len(original) != len(subset):
        raise ValueError(f'{partner}: original EV count differs from corrected substantive events')
    used = set()
    result = original.copy()
    for i, (onset, duration, _) in enumerate(original):
        matches = [j for j, r in enumerate(subset) if abs(float(r['onset']) - onset) < .001]
        if len(matches) != 1 or matches[0] in used:
            raise ValueError('EV onset matching is ambiguous or incomplete')
        j = matches[0]
        used.add(j)
        if abs(duration - float(subset[j]['duration'])) > .001:
            raise ValueError('original duration differs from corrected event duration')
        result[i, 1] = float(subset[j]['response_time'])
    return result


def resolve_recorded(value, standard):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if value.startswith('/data/projects/'):
        remapped = Path('/ZPOOL/data/projects') / path.relative_to('/data/projects')
        if remapped.is_file():
            return remapped.resolve()
    if path.name == standard.name and '/data/standard/' in value:
        return standard
    raise FileNotFoundError(f'retained input not found (no replacement BOLD/network signal guessed): {value}')


def render_l1(text, output, evdir, events, variant, standard):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if (fsf_value(text, 'evs_orig'), fsf_value(text, 'evs_real')) != ('28', '28'):
        raise ValueError('expected original 28-EV DMN design without expanded derivatives')
    if fsf_value(text, 'evtitle8') != 'rt' or fsf_value(text, 'evtitle9') != 'rt_p':
        raise ValueError('unexpected RT EV ordering')
    rows = decisions(events)
    evdir.mkdir(parents=True, exist_ok=True)
    text = replace_fsf_value(text, 'outputdir', str(output), fmri=True)
    required = []
    for key, fmri in [('feat_files(1)', False), ('confoundev_files(1)', False), ('regstandard', True)]:
        path = resolve_recorded(fsf_value(text, key) or '', standard)
        text = replace_fsf_value(text, key, str(path), fmri=fmri)
        required.append(path)
    # Resolve only file-based EVs, not generated interactions or empty EVs.
    for ev in range(1, 29):
        if fsf_value(text, f'shape{ev}') not in ('2', '3'):
            continue
        if ev in (8, 9):
            continue
        source = resolve_recorded(fsf_value(text, f'custom{ev}') or '', standard)
        required.append(source)
        path = source
        if variant == 'response-duration' and ev <= 6:
            partner = ('computer', 'ingroup', 'outgroup')[(ev-1)//2]
            matrix = np.loadtxt(source, ndmin=2)
            changed = replace_durations(matrix, rows, partner)
            path = evdir / f'ev{ev}_response_duration.txt'
            path.write_text(three_column(changed.tolist()))
            required.append(path)
        text = replace_fsf_value(text, f'custom{ev}', str(path), fmri=True)
    rebuilt = ev_rows(events, 'substantive')
    for ev, name in ((8, 'event_RT'), (9, 'event_RT_pmod')):
        if fsf_value(text, f'shape{ev}') != '3':
            raise ValueError('RT EV is not a three-column input')
        path = evdir / f'{name}.txt'
        path.write_text(three_column(rebuilt[name]))
        text = replace_fsf_value(text, f'custom{ev}', str(path), fmri=True)
        required.append(path)
    return text, required, len(rows)


def prepare(repo, production, repaired, work, standard):
    source_group = repo / SOURCE_FSF_RELATIVE
    group_inputs = parse_inputs(source_group.read_text().splitlines())
    safe_empty(work, [repo, production, repaired])
    if not standard.is_file():
        raise FileNotFoundError(standard)
    jobs, counts, provenance = [], [], {}
    for variant in VARIANTS:
        group_replacements = {}
        for _, subject, l2_cope in group_inputs:
            source_root = repaired if subject == 'sub-144' else production
            l1_outputs = []
            for run in ('01', '02'):
                source = source_root / subject / f'L1_task-ultimatum_model-02_type-nppi-dmn_run-{run}_sm-6.feat/design.fsf'
                events = repo / 'source_data/bids' / subject / 'func' / f'{subject}_task-ultimatum_run-{run}_events.tsv'
                evdir = work / 'EVfiles' / variant / subject / run
                output = work / 'derivatives' / variant / subject / f'L1_run-{run}'
                fsf = work / 'fsf' / f'{variant}_{subject}_run-{run}.fsf'
                text, required, n = render_l1(source.read_text(), output, evdir, read_events(events), variant, standard)
                fsf.parent.mkdir(parents=True, exist_ok=True)
                fsf.write_text(text)
                required.extend([source, events])
                for path in required + [fsf]:
                    provenance[str(path)] = ({'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                                             if path.name.endswith('.nii.gz') else {'sha256': digest(path)})
                jobs.append(dict(stage='l1', model=variant, subject=subject, run=run,
                                 fsf=str(fsf), output=str(output)+'.feat', inputs='|'.join(map(str, required)),
                                 source_fsf=str(source), source_sha256=digest(source), fsf_sha256=digest(fsf),
                                 expected_copes=int(fsf_value(text, 'ncon_real')), expected_zstats='', n_evs=28, design_rank=''))
                l1_outputs.append(Path(str(output)+'.feat'))
                counts.append(dict(variant=variant, subject=subject, run=run, responded_trials=n,
                                   rt_rows=n, response_time_reference='offer_onset_no_added_second', events_sha256=digest(events)))
            l2_source = Path(l2_cope).parents[2] / 'design.fsf'
            l2_output = work / 'derivatives' / variant / subject / 'L2'
            l2_fsf = work / 'fsf' / f'{variant}_{subject}_L2.fsf'
            render_l2(l2_source, l2_fsf, l2_output, *l1_outputs)
            l2_text = replace_fsf_value(l2_fsf.read_text(), 'regstandard', str(standard), fmri=True)
            l2_fsf.write_text(l2_text)
            provenance[str(l2_source)] = {'sha256': digest(l2_source)}
            provenance[str(l2_fsf)] = {'sha256': digest(l2_fsf)}
            jobs.append(dict(stage='l2', model=variant, subject=subject, run='combined',
                             fsf=str(l2_fsf), output=str(l2_output)+'.gfeat', inputs='|'.join(map(str, l1_outputs)),
                             source_fsf=str(l2_source), source_sha256=digest(l2_source), fsf_sha256=digest(l2_fsf),
                             expected_copes=11, expected_zstats='', n_evs='', design_rank=''))
            group_replacements[subject] = Path(str(l2_output)+'.gfeat') / 'cope7.feat/stats/cope1.nii.gz'
        jobs.append(group_job(repo, work, variant, standard, replacements=group_replacements))
    fields = list(dict.fromkeys(k for row in jobs for k in row))
    write_tsv(work / 'revision_jobs.tsv', [{k: row.get(k, '') for k in fields} for row in jobs])
    write_tsv(work / 'rt_construction.tsv', counts)
    (work / 'input_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (work / 'analysis_plan.json').write_text(json.dumps({
        'variants': VARIANTS, 'primary_unchanged': True, 'subjects': 47,
        'expected_jobs': {'l1': 188, 'l2': 94, 'l3': 2},
        'response_time_reference': 'BIDS onset/response_time are offer-onset referenced; no +1 second',
        'duration_interpretation': 'Offer-onset to response; does not independently isolate postresponse activity.',
        'rt_policy': 'all substantive responded rows, zero-duration RT impulses; amplitudes 1 and raw response_time',
        'task_policy': 'retained task amplitudes, interaction structure, network signals, confounds, smoothing; only durations change in second variant'
    }, indent=2) + '\n')
    print(f'PASS: prepared 188 L1, 94 L2, 2 L3 sensitivity jobs: {work / "revision_jobs.tsv"}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--production-fsl-root', type=Path, required=True)
    p.add_argument('--repaired-fsl-root', type=Path, required=True)
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--standard-image', type=Path, required=True)
    a = p.parse_args()
    prepare(a.repository.resolve(), a.production_fsl_root.resolve(), a.repaired_fsl_root.resolve(), a.work_root.resolve(), a.standard_image.resolve())


if __name__ == '__main__':
    main()
