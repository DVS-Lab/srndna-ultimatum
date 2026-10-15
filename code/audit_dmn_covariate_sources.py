#!/usr/bin/env python3
"""Read-only provenance search; numerical matches are not proof of generation."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, re, subprocess
from pathlib import Path
import numpy as np
from prepare_sub144_imaging_repair import fsf_value
from prepare_ultimatum_l3_repair import parse_evs, parse_inputs

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('tsnr', 'fd_mean')
SKIP = {'.git', '.venv', 'node_modules', 'sourcedata', 'stimuli', 'psychopy',
        'single_trials', 'singletrial', 'fmriprep', 'EVfiles', 'logs', 'maps', 'fits'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def subject(value):
    value = str(value).strip()
    match = re.search(r'(?:sub-|SMITH-AgingDM-)(\d{3})', value)
    if match: return 'sub-' + match[1]
    if re.fullmatch(r'\d{3}(?:\.0+)?', value): return 'sub-' + value.split('.')[0]
    raise ValueError('no unambiguous participant ID')

def fsf_vectors(text):
    inputs = parse_inputs(text.splitlines())
    if not inputs: return {}
    subjects = [s for _, s, _ in inputs]
    if len(set(subjects)) != len(subjects): raise ValueError('duplicate FSF participants')
    evs = parse_evs(text.splitlines()); out = {}
    for i in range(1, int(fsf_value(text, 'evs_real') or 0)+1):
        title = (fsf_value(text, f'evtitle{i}') or '').lower()
        if title in METRICS:
            out[title] = {s: float(evs[row, i]) for row, s, _ in inputs}
    return out

def table_vectors(text, delimiter):
    records = list(csv.DictReader(io.StringIO(text.lstrip('\ufeff')), delimiter=delimiter))
    if not records: return []
    header = {k.lower(): k for k in records[0] if k}
    metrics = [m for m in METRICS if m in header]
    idcol = next((header[k] for k in ('subjid', 'participant_id', 'participant', 'subject', 'sub', 'bids_name') if k in header), None)
    if not metrics or not idcol: return []
    parsed = []
    for row in records:
        try:
            sid = subject(row[idcol])
            name = row[idcol]
            task = row.get(header.get('task', ''), '') or (re.search(r'task-([^_]+)', name)[1] if re.search(r'task-([^_]+)', name) else '')
            run = row.get(header.get('run', ''), '') or (re.search(r'run-(\d+)', name)[1] if re.search(r'run-(\d+)', name) else '')
            parsed.append((sid, task, run, row))
        except (ValueError, TypeError):
            raise ValueError('table contains an unparseable participant identifier')
    has_task = any(t for _, t, _, _ in parsed)
    if has_task:
        parsed = [r for r in parsed if r[1] == 'ultimatum']
        if not parsed: return []
    repeated = len({s for s, _, _, _ in parsed}) < len(parsed)
    if repeated and not all(run for _, _, run, _ in parsed):
        raise ValueError('duplicate participant rows without run identifiers')
    if repeated:
        keys = [(s, int(run)) for s, _, run, _ in parsed]
        if len(set(keys)) != len(keys): raise ValueError('duplicate participant/run records')
    selections = [('participant_values', parsed)] if not repeated else [
        ('ultimatum_mean_all_runs', parsed),
        ('ultimatum_run1', [r for r in parsed if int(r[2]) == 1]),
        ('ultimatum_run2', [r for r in parsed if int(r[2]) == 2])]
    flag = header.get('outlier_run_custom1')
    if repeated and flag:
        kept = [r for r in parsed if r[3][flag].strip().lower() in ('false', '0')]
        counts = {s: sum(t[0] == s for t in kept) for s, _, _, _ in kept}
        selections.append(('historical_kept_runs_min2', [r for r in kept if counts[r[0]] > 1]))
    output = []
    for policy, subset in selections:
        for metric in metrics:
            grouped = {}
            for sid, _, _, row in subset:
                value = float(row[header[metric]])
                if not np.isfinite(value): raise ValueError('nonfinite quality value')
                grouped.setdefault(sid, []).append(value)
            output.append((policy, metric, {s: float(np.mean(v)) for s, v in grouped.items()}))
    return output

def compare(target, candidate):
    common = sorted(set(target) & set(candidate))
    if len(common) < 3: return dict(n_overlap=len(common), status='insufficient_overlap')
    x = np.array([candidate[s] for s in common]); y = np.array([target[s] for s in common])
    if np.std(x) == 0: return dict(n_overlap=len(common), status='constant_candidate')
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    error = y - (slope*x + intercept)
    full = len(common) == len(target)
    residual = float(np.max(abs(error)))
    return dict(n_overlap=len(common), status='full_sample' if full else 'partial_sample',
                max_abs_direct=float(np.max(abs(y-x))), affine_slope=float(slope), affine_intercept=float(intercept),
                max_abs_affine=residual, correlation=float(np.corrcoef(x, y)[0, 1]),
                full_positive_affine_match=int(full and slope > 0 and residual < 2e-6))

def candidates(root):
    # Bounded metadata search: no scratch, images, raw events, logs, or subject data payloads.
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.') and not d.endswith(('.feat', '.gfeat'))
                         and not d.startswith('attempt-')
                         and (not d.startswith('sub-') or 'mriqc' in Path(base).parts))
        for name in sorted(names):
            p = Path(base)/name
            if p.is_symlink() or p.stat().st_size > 5_000_000: continue
            if p.suffix in ('.csv', '.tsv') and re.search(r'participant|cov|group_bold|outlier|quality|iqm', name, re.I): yield p
            elif p.suffix == '.fsf' and ('L3' in name or name == 'design.fsf'): yield p
            elif 'mriqc' in p.parts and re.fullmatch(r'sub-1\d\d_task-ultimatum_run-\d+_bold.json', name): yield p

def git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True)
    if result.returncode: raise ValueError(result.stderr.strip())
    return result.stdout

def audit(roots, output):
    targets = {}
    for family, directory in [('dmn', 'dmn-age'), ('ecn', 'ecn-sensitivity'), ('act', 'activation')]:
        p = ROOT/'results/reviewer/production_audits'/directory/'design/design.fsf'
        for metric, values in fsf_vectors(p.read_text()).items(): targets[(family, metric)] = values
    if set(targets) != {(f, m) for f in ('dmn', 'ecn', 'act') for m in METRICS}: raise ValueError('missing production targets')
    comparisons, errors, sources, history, detail = [], [], [], [], []
    seen = set(); qc_sets = {}
    def inspect(label, text, suffix, source_hash):
        vectors = [('fsf_column', m, v) for m, v in fsf_vectors(text).items()] if suffix == '.fsf' else table_vectors(text, ',' if suffix == '.csv' else '\t')
        for policy, metric, values in vectors:
            for (family, m), target in targets.items():
                if m != metric: continue
                result = compare(target, values)
                comparisons.append(dict(source=label, sha256=source_hash, policy=policy, target_family=family, metric=metric, **result))
                if result.get('full_positive_affine_match'):
                    for sid in sorted(target):
                        detail.append(dict(source=label, policy=policy, target_family=family, metric=metric, participant=sid,
                                           candidate=values[sid], retained=target[sid], residual=target[sid]-(result['affine_slope']*values[sid]+result['affine_intercept'])))
        return bool(vectors)
    for root in roots:
        if not root.is_dir(): errors.append(dict(source=str(root), issue='optional search root absent')); continue
        for p in candidates(root):
            resolved = str(p.resolve())
            if resolved in seen: continue
            seen.add(resolved)
            try:
                digest = sha(p)
                if p.suffix == '.json':
                    data=json.loads(p.read_text())
                    if not all(m in data for m in METRICS): raise ValueError('MRIQC JSON missing tsnr/fd_mean')
                    qroot=p.parents[2]
                    qc_sets.setdefault(str(qroot), []).append(dict(bids_name=p.stem, **{m:data[m] for m in METRICS}))
                    prov=data.get('provenance', {})
                    sources.append(dict(path=str(p),sha256=digest,
                        software=str(prov.get('software','')),
                        software_version=str(prov.get('version',prov.get('software_version','')))))
                    continue
                if inspect(str(p), p.read_text(encoding='utf-8-sig'), p.suffix, digest):
                    sources.append(dict(path=str(p), sha256=digest))
            except (ValueError, KeyError, UnicodeError, OSError) as e:
                errors.append(dict(source=str(p), issue=str(e)))
    for source, records in qc_sets.items():
        stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=['bids_name',*METRICS],delimiter='\t')
        writer.writeheader();writer.writerows(sorted(records,key=lambda r:r['bids_name']))
        content=stream.getvalue()
        try: inspect(source+' [direct MRIQC JSON aggregation]', content, '.tsv', hashlib.sha256(content.encode()).hexdigest())
        except (ValueError, KeyError) as e: errors.append(dict(source=source,issue=str(e)))
    # Trace commits only for the original covariate tables/templates/scripts, not generated result bundles.
    for root in roots:
        if not (root/'.git').exists(): continue
        try:
            paths = [p for p in git(root, 'ls-files').splitlines() if (
                not p.startswith('results/') and '/templates/' in '/'+p and p.endswith('.fsf') and ('L3' in p) and '/revision/' not in p)
                or (not p.startswith('results/') and re.search(r'participant|cov|Outlier', Path(p).name, re.I) and Path(p).suffix in ('.csv', '.tsv'))
                or Path(p).name == 'SRNDNA_OutlierID.py']
            for rel in paths:
                log = git(root, 'log', '--follow', '--format=%H%x09%aI%x09%cI%x09%s', '--', rel).splitlines()
                for line in log:
                    commit, author_date, commit_date, message = line.split('\t', 3)
                    label = f'{root}@{commit}:{rel}'
                    history.append(dict(repository=str(root), path=rel, commit=commit, author_date=author_date, commit_date=commit_date, subject=message))
                    if Path(rel).suffix not in ('.csv', '.tsv', '.fsf'): continue
                    try:
                        content = git(root, 'show', f'{commit}:{rel}')
                        inspect(label, content, Path(rel).suffix, hashlib.sha256(content.encode()).hexdigest())
                    except (ValueError, KeyError) as e: errors.append(dict(source=label, issue=str(e)))
        except ValueError as e: errors.append(dict(source=str(root), issue=str(e)))
    fields = ['source','sha256','policy','target_family','metric','n_overlap','status','max_abs_direct','affine_slope','affine_intercept','max_abs_affine','correlation','full_positive_affine_match']
    write_rows(output/'candidate_matches.tsv', comparisons, fields)
    write_rows(output/'matched_participant_values.tsv', detail, ['source','policy','target_family','metric','participant','candidate','retained','residual'])
    write_rows(output/'source_inventory.tsv', sources, ['path','sha256','software','software_version'])
    write_rows(output/'history.tsv', history, ['repository','path','commit','author_date','commit_date','subject'])
    write_rows(output/'search_issues.tsv', errors, ['source','issue'])
    matched = [r for r in comparisons if r.get('full_positive_affine_match') and r['target_family']=='dmn']
    summary = dict(search_complete=True, script_sha256=sha(__file__), roots=[str(p) for p in roots], candidate_files=len(sources), comparisons=len(comparisons),
                   dmn_matching_records=len(matched), covariates_changed=False,
                   interpretation='Full-sample affine matches identify numerically compatible sources, not proof of generation. Historical timestamps are commit dates, not acquisition dates. Missing roots and rejected tables are reported. No nuisance values replaced.')
    (output/'status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-root', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    audit(list(dict.fromkeys([ROOT]+[x.resolve() for x in a.search_root])), a.output)

if __name__=='__main__': main()
