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


def parse_input_maps(values):
    """Explicit relocation only; no dataset search or preprocessing substitution."""
    result = []
    for value in values:
        old, separator, new = value.partition('=')
        if not separator or not old or not new:
            raise ValueError('input map must be an absolute OLD=NEW directory pair')
        old, new = Path(old), Path(new)
        if not old.is_absolute() or not new.is_absolute() or '..' in old.parts or '..' in new.parts:
            raise ValueError('input map directories must be absolute without ..')
        if old == Path('/') or new == Path('/') or any(old == x for x, _ in result):
            raise ValueError('root/duplicate input maps are not allowed')
        result.append((old, new))
    return tuple(sorted(result, key=lambda pair: len(pair[0].parts), reverse=True))


def resolve_recorded(value, standard, input_maps=()):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    # FEAT commonly records an image basename without the NIfTI extension.
    # Restrict normalization to the named standard image, not arbitrary inputs.
    def image_basename(name):
        for suffix in ('.nii.gz', '.nii'):
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name
    if ('/data/standard/' in value and standard.is_file()
            and image_basename(path.name) == image_basename(standard.name)):
        return standard.resolve()
    # An explicit mapping never overrides an existing recorded file. Match path
    # components, not string prefixes; the longest requested prefix wins.
    for old, new in input_maps:
        if path.is_relative_to(old):
            remapped = new / path.relative_to(old)
            if remapped.is_file():
                return remapped.resolve()
            raise FileNotFoundError(f'explicitly mapped input missing: {value} -> {remapped}')
    if value.startswith('/data/projects/'):
        remapped = Path('/ZPOOL/data/projects') / path.relative_to('/data/projects')
        if remapped.is_file():
            return remapped.resolve()
    raise FileNotFoundError(f'retained input not found (no replacement BOLD/network signal guessed): {value}')


def retained_inputs(text, include_rt=False):
    yield from ('feat_files(1)', 'confoundev_files(1)', 'regstandard')
    for ev in range(1, 29):
        if (include_rt or ev not in (8, 9)) and fsf_value(text, f'shape{ev}') in ('2', '3'):
            yield f'custom{ev}'


def resolve_input(value, standard, input_maps, source_fsf=None, role=None):
    """Optionally recover only the exact run's FEAT input copies, not other runs."""
    try:
        return resolve_recorded(value, standard, input_maps)
    except FileNotFoundError as error:
        if source_fsf is None:
            raise
        if role and re.fullmatch(r'custom\d+', role):
            candidate = source_fsf.parent/'custom_timing_files'/f'ev{role[6:]}.txt'
        elif role == 'confoundev_files(1)':
            text = source_fsf.read_text()
            # confoundevs.txt may combine external confounds and FEAT-generated
            # motion terms. Do not add the latter a second time in the rerun.
            if fsf_value(text, 'motionevs') not in (None, '0') or fsf_value(text, 'motionevsbeta') not in (None, ''):
                raise ValueError(f'cannot reuse combined FEAT confounds with automatic motion EVs: {source_fsf}')
            candidate = source_fsf.parent/'confoundevs.txt'
        else:
            raise
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
        raise FileNotFoundError(f'{error}; retained FEAT copy also missing/empty: {candidate}') from error


def input_inventory(repo, production, repaired, standard, input_maps=(), use_feat_input_copies=False):
    """Check all runs without creating a model directory or launching FSL."""
    rows = []
    for _, subject, l2_cope in parse_inputs((repo / SOURCE_FSF_RELATIVE).read_text().splitlines()):
        root = repaired if subject == 'sub-144' else production
        for run in ('01', '02'):
            source = root / subject / f'L1_task-ultimatum_model-02_type-nppi-dmn_run-{run}_sm-6.feat/design.fsf'
            paths = [('source_fsf', str(source)), ('events', str(repo/'source_data/bids'/subject/'func'/f'{subject}_task-ultimatum_run-{run}_events.tsv'))]
            if use_feat_input_copies:
                paths.append(('source_design', str(source.with_suffix('.mat'))))
            if source.is_file():
                text = source.read_text()
                paths += [(key, fsf_value(text, key) or '') for key in retained_inputs(text, include_rt=use_feat_input_copies)]
            for role, value in paths:
                try:
                    resolved = resolve_input(value, standard, input_maps, source if use_feat_input_copies else None, role)
                    error = ''
                except (FileNotFoundError, ValueError) as exc:
                    resolved, error = '', str(exc)
                rows.append(dict(subject=subject, run=run, role=role, recorded=value,
                                 resolved=str(resolved), status='missing' if error else 'found', error=error))
        source = Path(l2_cope).parents[2] / 'design.fsf'
        rows.append(dict(subject=subject, run='combined', role='source_l2_fsf', recorded=str(source),
                         resolved=str(source) if source.is_file() else '',
                         status='found' if source.is_file() else 'missing', error='' if source.is_file() else 'missing retained L2 FSF'))
    return rows


def render_l1(text, output, evdir, events, variant, standard, input_maps=(), resolutions=None, source_fsf=None):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if (fsf_value(text, 'evs_orig'), fsf_value(text, 'evs_real')) not in (('28', '28'), ('9', '9')):
        raise ValueError('expected retained 28-EV nPPI or 9-EV activation design without expanded derivatives')
    if fsf_value(text, 'evtitle8') != 'rt' or fsf_value(text, 'evtitle9') != 'rt_p':
        raise ValueError('unexpected RT EV ordering')
    rows = decisions(events)
    evdir.mkdir(parents=True, exist_ok=True)
    text = replace_fsf_value(text, 'outputdir', str(output), fmri=True)
    required = []
    for key, fmri in [('feat_files(1)', False), ('confoundev_files(1)', False), ('regstandard', True)]:
        recorded = fsf_value(text, key) or ''
        path = resolve_input(recorded, standard, input_maps, source_fsf, key)
        if resolutions is not None:
            resolutions[recorded] = str(path)
        text = replace_fsf_value(text, key, str(path), fmri=fmri)
        required.append(path)
    # Resolve only file-based EVs, not generated interactions or empty EVs.
    for ev in range(1, 29):
        if fsf_value(text, f'shape{ev}') not in ('2', '3'):
            continue
        if ev in (8, 9):
            continue
        recorded = fsf_value(text, f'custom{ev}') or ''
        source = resolve_input(recorded, standard, input_maps, source_fsf, f'custom{ev}')
        if resolutions is not None:
            resolutions[recorded] = str(source)
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


def baseline_copy(source, destination, standard, input_maps, resolutions):
    """Relocate original inputs only. Keep original RT EVs and every model setting."""
    text = source.read_text()
    required = [source, source.with_suffix('.mat')]
    for role in retained_inputs(text, include_rt=True):
        recorded = fsf_value(text, role) or ''
        resolved = resolve_input(recorded, standard, input_maps, source, role)
        text = replace_fsf_value(text, role, str(resolved), fmri=role not in ('feat_files(1)','confoundev_files(1)'))
        required.append(resolved)
        resolutions[recorded] = str(resolved)
    text = replace_fsf_value(text, 'outputdir', str(destination.with_suffix('')), fmri=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return required+[destination]


def prepare(repo, production, repaired, work, standard, input_maps=(), use_feat_input_copies=False):
    source_group = repo / SOURCE_FSF_RELATIVE
    group_inputs = parse_inputs(source_group.read_text().splitlines())
    if not standard.is_file():
        raise FileNotFoundError(standard)
    inventory = input_inventory(repo, production, repaired, standard, input_maps, use_feat_input_copies)
    missing = [r for r in inventory if r['status'] == 'missing']
    if missing:
        from collections import Counter
        by_role = dict(Counter(r['role'] for r in missing))
        raise FileNotFoundError(
            f'{len(missing)} required input references are missing; by role: {by_role}. '
            'No RT models were prepared or launched. For missing task EVs/confounds, '
            'run code/locate_dmn_rt_text_inputs.py; do not guess another directory mapping. '
            f'First: {missing[0]["recorded"]}')
    safe_empty(work, [repo, production, repaired, *[new for _, new in input_maps]])
    jobs, counts, provenance = [], [], {}
    resolutions = {}
    baselines = {}
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
                baseline_fields = {}
                if use_feat_input_copies:
                    if source not in baselines:
                        baseline_fsf = work/'baseline'/f'{subject}_run-{run}.fsf'
                        baseline_required = baseline_copy(source, baseline_fsf, standard, input_maps, resolutions)
                        baselines[source] = (baseline_fsf, baseline_required)
                    baseline_fsf, baseline_required = baselines[source]
                    baseline_fields = dict(baseline_fsf=str(baseline_fsf), baseline_sha256=digest(baseline_fsf),
                                           source_design=str(source.with_suffix('.mat')), source_design_sha256=digest(source.with_suffix('.mat')))
                text, required, n = render_l1(source.read_text(), output, evdir, read_events(events), variant, standard, input_maps, resolutions, source if use_feat_input_copies else None)
                if use_feat_input_copies:
                    required.extend(baseline_required)
                fsf.parent.mkdir(parents=True, exist_ok=True)
                fsf.write_text(text)
                required.extend([source, events])
                for path in required + [fsf]:
                    provenance[str(path)] = ({'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                                             if path.name.endswith('.nii.gz') else {'sha256': digest(path)})
                jobs.append(dict(stage='l1', model=variant, subject=subject, run=run,
                                 fsf=str(fsf), output=str(output)+'.feat', inputs='|'.join(map(str, required)),
                                 source_fsf=str(source), source_sha256=digest(source), fsf_sha256=digest(fsf),
                                 expected_copes=int(fsf_value(text, 'ncon_real')), expected_zstats='', n_evs=28, design_rank='', **baseline_fields))
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
    (work / 'input_path_resolution.json').write_text(json.dumps({
        'requested_maps': [[str(old), str(new)] for old, new in input_maps],
        'use_feat_input_copies': use_feat_input_copies,
        'recorded_to_resolved': resolutions,
        'limitation': 'Explicit relocation is not proof of identity to an unavailable original BOLD; compare surviving copies before choosing a root.'
    }, indent=2) + '\n')
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
    p.add_argument('--input-map', action='append', default=[], metavar='OLD=NEW')
    p.add_argument('--use-feat-input-copies', action='store_true')
    a = p.parse_args()
    prepare(a.repository.resolve(), a.production_fsl_root.resolve(), a.repaired_fsl_root.resolve(), a.work_root.resolve(), a.standard_image.resolve(), parse_input_maps(a.input_map), a.use_feat_input_copies)


if __name__ == '__main__':
    main()
