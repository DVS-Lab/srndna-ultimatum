#!/usr/bin/env python3
"""Partition model-02 display epochs into decision and post-response phases.

The transformation is structural, not text substitution: all EV-dependent
arrays are rebuilt and both contrast matrices are remapped. Everything outside
those arrays is retained. No source events, FSFs, or derivatives are edited.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from make_ultimatum_3col import PARTNERS, TRIAL_RE, three_column
from prepare_dmn_rt_sensitivity import decisions, replace_durations, resolve_input
from prepare_sub144_imaging_repair import fsf_value, replace_fsf_value

SINGLE = re.compile(r'(evtitle|shape|convolve|convolve_phase|tempfilt_yn|deriv_yn|custom)(\d+)$')
PAIR = re.compile(r'(ortho|interactions|interactionsd)(\d+)\.(\d+)$')
CONTRAST = re.compile(r'(con_orig|con_real)(\d+)\.(\d+)$')
SET = re.compile(r'^set fmri\(([^)]+)\)\s+(.+)$')
POST_MODELS = ('pooled', 'partner')


def post_ev_names(post_model):
    """One shared mean or three partner means; never offer/choice expansion."""
    if post_model not in POST_MODELS:
        raise ValueError(f'unknown post-response model: {post_model}')
    return ({8: 'post_response'} if post_model == 'pooled' else
            {i: f'post_response_{partner}' for i, partner in enumerate(PARTNERS, 8)})


def settings(text):
    result = {}
    for line in text.splitlines():
        match = SET.fullmatch(line.strip())
        if match:
            if match[1] in result:
                raise ValueError(f'duplicate FSF setting: {match[1]}')
            result[match[1]] = match[2]
        elif line.strip() and not line.lstrip().startswith(('#', 'set ')):
            raise ValueError('only declarative FEAT settings are supported')
    return result


def expected_contrasts(nppi):
    n, start = (28, 10) if nppi else (9, 0)
    values = np.zeros((11 if nppi else 10, n))
    for i in range(6):
        values[i, start + i] = 1
    values[6, [start+3, start+5]] = [1, -1]
    values[7, [start+1, start+3, start+5]] = [-2, 1, 1]
    values[8, [start+2, start+4]] = [1, -1]
    values[9, [start, start+2, start+4]] = [-2, 1, 1]
    if nppi:
        values[10, 9] = 1
    return values


def validate_source(text, *, template=False):
    """Fail closed on an unexpected legacy model, rather than guessing indices."""
    s = settings(text)
    n = int(s['evs_orig'])
    if n not in (9, 28) or s['evs_real'] != str(n) or s.get('evs_vox', '0') != '0':
        raise ValueError('expected a 9-EV activation or 28-EV nPPI model without expanded EVs')
    for key in s:
        if SINGLE.fullmatch(key) or PAIR.fullmatch(key) or CONTRAST.fullmatch(key):
            continue
        if re.search(r'\d', key) and not re.fullmatch(r'con(?:name|pic)_(?:orig|real)\.\d+|conmask\d+_\d+', key):
            raise ValueError(f'unhandled indexed FSF setting: {key}')
    if any(s.get(f'nftests_{mode}', '0') != '0' for mode in ('orig', 'real')):
        raise ValueError('F-tests require an explicit additional mapping')
    for i in range(1, n+1):
        if s.get(f'deriv_yn{i}') != '0':
            raise ValueError('temporal derivatives require a different real-EV mapping')
        shape = s.get(f'shape{i}')
        expected = {'3'} if i <= 9 else {'2'} if i == 10 or i >= 20 else {'4'}
        if i == 7:
            expected = {'3', '10', 'EV_SHAPE'} if template else {'3', '10'}
        if shape not in expected:
            raise ValueError(f'unexpected shape for original EV{i}: {shape}')
        if s.get(f'convolve{i}') != ('3' if i <= 9 else '0'):
            raise ValueError(f'unexpected convolution for original EV{i}')
    if (fsf_value(text, 'evtitle8'), fsf_value(text, 'evtitle9')) != ('rt', 'rt_p'):
        raise ValueError('unexpected dedicated RT ordering')
    active_ortho = {(int(m[2]), int(m[3])) for k,v in s.items()
                    if (m := PAIR.fullmatch(k)) and m[1] == 'ortho' and float(v) != 0}
    if active_ortho != {(i,j) for i in (2,4,6,9) for j in (0,i-1)}:
        raise ValueError('unexpected orthogonalization; explicit review required')
    expected = expected_contrasts(n == 28)
    for mode in ('orig', 'real'):
        if int(s[f'ncon_{mode}']) != len(expected):
            raise ValueError('unexpected contrast count')
        actual = np.array([[float(s[f'con_{mode}{c}.{i}']) for i in range(1,n+1)]
                           for c in range(1,len(expected)+1)])
        if not np.array_equal(actual, expected):
            raise ValueError(f'original {mode} contrasts do not match model-02')
    if n == 28:
        for i in range(11,20):
            active = {j for j in range(1,i) if s.get(f'interactions{i}.{j}', '0') == '1'}
            if active != {i-10,10}:
                raise ValueError(f'unexpected parents for interaction EV{i}')
            mode = 1 if i in (12,14,16,19) else 0
            if s.get(f'interactionsd{i}.{i-10}') != str(mode) or s.get(f'interactionsd{i}.10') != '2':
                raise ValueError(f'unexpected centering for interaction EV{i}')
    return n


def rebuild_fsf(text, ev_files, *, template=False, post_model='pooled'):
    """Pure transformation, suitable for retained FSFs and visible templates.

    ev_files maps NEW EV numbers 1..6 and 8 (pooled) or 8..10 (partner)
    to decision/post timing files. New PPI indices derive from this layout.
    The missed-trial file and all physiological signals retain their contents.
    """
    old_n = validate_source(text, template=template)
    nppi = old_n == 28
    post_names = post_ev_names(post_model)
    psychological = 7 + len(post_names)
    physiological = psychological + 1
    n = 2 * psychological + 10 if nppi else psychological
    s = settings(text)
    old_to_new = {i:i for i in range(1,8)}
    if nppi:
        old_to_new.update({10:physiological,
                           **{i:physiological+i-10 for i in range(11,18)},
                           **{i:2*psychological+2+i-20 for i in range(20,29)}})
    new_to_old = {new:old for old,new in old_to_new.items()}
    new_to_old.update({i:1 for i in post_names})
    if nppi:
        new_to_old.update({physiological+i:11 for i in post_names})
    if set(ev_files) != set(range(1,7)) | set(post_names):
        raise ValueError(f'exactly six decision and {len(post_names)} post-response timing files are required')
    if set(new_to_old) != set(range(1,n+1)):
        raise ValueError('incomplete or overlapping EV remapping')
    # Keep every non-EV setting and non-fmri input declaration verbatim. Old
    # comments are not copied because many describe obsolete EV numbers.
    lines = ['# Decision/post-response model; generated by decision_postresponse_model.py.',
             '# Preserve contrast numbers; nuisance post epochs have zero contrast weights.']
    for line in text.splitlines():
        stripped = line.strip()
        match = SET.fullmatch(stripped)
        if match:
            key = match[1]
            if SINGLE.fullmatch(key) or PAIR.fullmatch(key) or CONTRAST.fullmatch(key):
                continue
            lines.append(f'set fmri({key}) {n}' if key in ('evs_orig','evs_real') else line)
        elif stripped.startswith('set '):
            lines.append(line)
    def put(key, value):
        lines.append(f'set fmri({key}) {value}')
    for new in range(1,n+1):
        old = new_to_old[new]
        lines.append(f'\n# EV {new}')
        for key, value in s.items():
            m = SINGLE.fullmatch(key)
            if not m or int(m[2]) != old:
                continue
            prefix = m[1]
            if prefix == 'custom' and new in ev_files:
                value = quote(ev_files[new])
            if prefix == 'evtitle':
                if new <= 6:
                    value = quote('decision_' + PARTNERS[(new-1)//2] + ('_offer' if new % 2 == 0 else ''))
                elif new in post_names:
                    value = quote(post_names[new])
                elif nppi and physiological < new <= physiological+psychological:
                    value = quote(f'ppi_phase{new-physiological}')
            put(f'{prefix}{new}', value)
        for j in range(n+1):
            put(f'ortho{new}.{j}', int(new in (2,4,6) and j in (0,new-1)))
        if nppi and physiological < new <= physiological+psychological:
            psych = new-physiological
            for j in range(1,new):
                put(f'interactions{new}.{j}', int(j in (psych,physiological)))
                mode = 2 if j == physiological else 1 if j == psych and psych in (2,4,6) else 0
                put(f'interactionsd{new}.{j}', mode)
    lines.append('\n# Contrasts: unchanged numbering, remapped scientific EVs.')
    for mode in ('orig','real'):
        for c in range(1,int(s[f'ncon_{mode}'])+1):
            weights = {new: s[f'con_{mode}{c}.{old}'] for old,new in old_to_new.items()}
            for new in range(1,n+1):
                put(f'con_{mode}{c}.{new}', weights.get(new,'0'))
    return '\n'.join(lines)+'\n'


def quote(value):
    value = str(value)
    # FSFs are evaluated as Tcl. Reject paths that would perform substitution.
    if any(c in value for c in ('"','\n','\r','$','[',']','\\')):
        raise ValueError(f'unsafe Tcl path/title: {value!r}')
    return f'"{value}"'


def split_epochs(events):
    """Return unique responded rows and positive post epochs in recorded time."""
    rows = decisions(events)
    # Generic BIDS companion rows are not independent trials. Check only the
    # substantive responded trials and separately labeled missed trials.
    misses = [r for r in events if r['trial_type'] == 'missed_trial']
    intervals = sorted((float(r['onset']), float(r['duration'])) for r in rows + misses)
    if not np.isfinite(intervals).all() or any(d <= 0 for _, d in intervals):
        raise ValueError('nonfinite or nonpositive trial display interval')
    if any(onset < previous + duration - 1e-6
           for (previous,duration),(onset,_) in zip(intervals,intervals[1:])):
        raise ValueError('overlapping or duplicated responded/missed trial intervals')
    post = {p:[] for p in PARTNERS}
    for row in rows:
        onset, duration, rt = (float(row[k]) for k in ('onset','duration','response_time'))
        if rt >= duration:
            raise ValueError('response does not precede display offset; no clipping or invented post epoch')
        partner = TRIAL_RE.fullmatch(row['trial_type']).group('partner')
        post[partner].append((onset+rt, duration-rt, 1.0))
    if any(not values for values in post.values()):
        raise ValueError('a partner has no responded trials; this model requires all three partners')
    return rows, post


def render_l1(text, output, evdir, event_rows, standard, maps=(), resolutions=None, source_fsf=None,
              *, post_model='pooled'):
    """Resolve exact retained inputs and write new timing files in empty scratch."""
    n = validate_source(text)
    post_names = post_ev_names(post_model)
    rows, post = split_epochs(event_rows)
    missed = [r for r in event_rows if r['trial_type'] == 'missed_trial']
    if bool(missed) != (fsf_value(text,'shape7') == '3'):
        raise ValueError('retained missed-trial EV shape disagrees with source events')
    files, contents, required = {}, {}, []
    text = replace_fsf_value(text,'outputdir',str(output),fmri=True)
    def resolve(role):
        recorded = fsf_value(text,role) or ''
        path = resolve_input(recorded,standard,maps,source_fsf,role)
        if resolutions is not None:
            resolutions[recorded] = str(path)
        required.append(path)
        return path
    for role, fmri in (('feat_files(1)',False),('confoundev_files(1)',False),('regstandard',True)):
        text = replace_fsf_value(text,role,str(resolve(role)),fmri=fmri)
    for i in range(1,n+1):
        if i in (8,9) or fsf_value(text,f'shape{i}') not in ('2','3'):
            continue
        path = resolve(f'custom{i}')
        if i == 7:
            expected = np.array([(float(r['onset']),float(r['duration']),1.) for r in missed])
            actual = np.loadtxt(path,ndmin=2)
            if (actual.shape != expected.shape or not np.isfinite(actual).all() or
                    not np.allclose(actual[np.argsort(actual[:,0])],
                                    expected[np.argsort(expected[:,0])], atol=.001, rtol=0)):
                raise ValueError('retained missed-trial EV differs from source events')
        if i <= 6:
            original = np.loadtxt(path,ndmin=2)
            partner = PARTNERS[(i-1)//2]
            changed = replace_durations(original,rows,partner)
            # Keep retained amplitudes exactly: unit main effects and the
            # original raw-offer pmod, with its original FSL orthogonalization.
            files[i] = evdir/f'decision_{partner}{"_offer" if i%2==0 else ""}.txt'
            contents[files[i]] = three_column(changed.tolist())
        else:
            text = replace_fsf_value(text,f'custom{i}',str(path),fmri=True)
    for i, name in post_names.items():
        files[i] = evdir/f'{name}.txt'
        values = ([r for values in post.values() for r in values] if post_model == 'pooled'
                  else post[PARTNERS[i-8]])
        contents[files[i]] = three_column(sorted(values))
    changed = rebuild_fsf(text,files,post_model=post_model)
    # Complete validation before creating any files; never overwrite even a
    # partial earlier preparation. The orchestrator uses a separate work root.
    if evdir.exists():
        raise FileExistsError(f'new EV directory already exists: {evdir}')
    evdir.mkdir(parents=True)
    for path, content in contents.items():
        path.write_text(content)
    required.extend(contents)
    stats = dict(responded_trials=len(rows),**{f'post_rows_{p}':len(post[p]) for p in PARTNERS},
                 n_evs=7+len(post_names) if n==9 else 2*(7+len(post_names))+10,
                 minimum_post_duration=min(r[1] for values in post.values() for r in values),
                 decision_timing='offer_onset_to_response',post_timing='response_to_recorded_display_offset')
    return changed, required, stats


def public_template(source, *, post_model='pooled'):
    files = {i: f'PHASE_EVDIR/decision_{PARTNERS[(i-1)//2]}{"_offer" if i%2==0 else ""}.txt'
             for i in range(1,7)}
    files.update({i:f'PHASE_EVDIR/{name}.txt' for i,name in post_ev_names(post_model).items()})
    return '\n'.join(line.rstrip() for line in rebuild_fsf(source,files,template=True,
                                                          post_model=post_model).splitlines())+'\n'
