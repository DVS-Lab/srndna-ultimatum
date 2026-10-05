#!/usr/bin/env python3
"""Read completed choice-extension chains; never compile, sample, or alter fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from export_gu_stan_diagnostics import (
    checked_file, digest, sha, read_chain, parameter_table, write_json,
    np, pd, plt, matplotlib,
)
from matplotlib.ticker import NullFormatter

SAMPLER = ('divergent__', 'treedepth__', 'energy__', 'accept_stat__', 'stepsize__', 'n_leapfrog__')
DEFAULT_SUBJECTS = ('sub-104', 'sub-107', 'sub-126', 'sub-128', 'sub-155', 'sub-156')
PAIRS = (('alpha', 'epsilon'), ('alpha', 'f0'), ('gamma', 'acceptance_bias'),
         ('f0', 'acceptance_bias'))


def discover(work, expected_jobs):
    """Verify completion provenance before producing any results."""
    work = Path(work).resolve()
    if expected_jobs < 1:
        raise ValueError('expected jobs must be positive')
    jobs, hashes = [], {}
    for cp in sorted((work / 'fits').glob('*/configuration.json')):
        config = json.loads(cp.read_text())
        if config['phase'] != 'fit':
            continue
        if config['model'] not in ('rw_positive', 'rw_positive_bias'):
            raise ValueError(f'not a choice-extension fit: {cp}')
        job = cp.parent
        completion = job / 'completed.json'
        record = json.loads(completion.read_text())
        if record['job'] != job.name or record['config_hash'] != digest(config):
            raise ValueError(f'completion/configuration mismatch: {job}')
        attempt = record['attempt'] + '/'
        paths = {relative: checked_file(job, record, relative)
                 for relative in record['files'] if relative.startswith(attempt)}
        data_path = paths[attempt + 'data.json']
        data = json.loads(data_path.read_text())
        if digest(data) != config['data_hash']:
            raise ValueError(f'data fingerprint mismatch: {job}')
        if json.loads(paths[attempt + 'diagnostics.json'].read_text()) != record['diagnostics']:
            raise ValueError(f'diagnostics/completion mismatch: {job}')
        table = parameter_table(pd.read_csv(paths[attempt + 'stan_summary.tsv'], sep='\t', index_col=0), config, data)
        # Bias theta copies across partners are the same parameter, not replicates.
        table = table.loc[[v for v in table.index if not (
            table.loc[v, 'family'] == 'theta' and table.loc[v, 'parameter'] == 'acceptance_bias'
            and data['partner'][int(v.split('[')[1].split(',')[0])-1] != 1)]]
        csvs = sorted(p for relative, p in paths.items() if relative.endswith('.csv'))
        if len(csvs) != config['chains']:
            raise ValueError(f'wrong chain count: {job}')
        for p in (cp, completion, *paths.values()):
            hashes[str(p.relative_to(work))] = sha(p)
        jobs.append((job.name, config, record, data, table, csvs, paths))
        print(f'VERIFIED: {job.name}', flush=True)
    if len(jobs) != expected_jobs:
        raise ValueError(f'expected {expected_jobs} completed fits, found {len(jobs)}')
    return jobs, hashes


def compare_draws(values, divergent):
    """Descriptive contrasts only; no independent-draw p-values or causal claims."""
    divergent = np.asarray(divergent, bool)
    baseline = values[~divergent]
    flagged = values[divergent]
    row = dict(n_divergent=len(flagged), n_nondivergent=len(baseline),
               divergent_median=np.nan, nondivergent_median=np.nan,
               standardized_mean_difference=np.nan,
               divergent_mean_percentile=np.nan, divergent_tail_fraction=np.nan)
    if len(baseline):
        row['nondivergent_median'] = float(np.median(baseline))
    if len(flagged):
        row['divergent_median'] = float(np.median(flagged))
    if len(baseline) and len(flagged):
        ordered = np.sort(baseline)
        pct = (np.searchsorted(ordered, flagged, side='left') +
               np.searchsorted(ordered, flagged, side='right')) / (2 * len(baseline))
        sd = float(np.std(baseline, ddof=1)) if len(baseline)>1 else 0.
        row.update(divergent_mean_percentile=float(pct.mean()),
                   divergent_tail_fraction=float(((pct < .05) | (pct > .95)).mean()))
        if sd > 0:
            row['standardized_mean_difference'] = float((flagged.mean()-baseline.mean())/sd)
    return row


def summarize_chains(name, config, record, table, chains):
    indices = {v: i for i, v in enumerate((*SAMPLER, *table.index))}
    chain_rows, comparisons = [], []
    for chain_id, values in chains:
        d = values[:, indices['divergent__']]
        depth = values[:, indices['treedepth__']]
        accept = values[:, indices['accept_stat__']]
        if not np.isin(d, (0, 1)).all() or (depth < 0).any() or not np.equal(depth, np.floor(depth)).all():
            raise ValueError('invalid sampler flags/depth')
        if (accept < 0).any() or (accept > 1).any():
            raise ValueError('invalid acceptance statistic')
        energy = values[:, indices['energy__']]
        variance = np.var(energy)
        chain_rows.append(dict(job=name, model=config['model'], stage=config['stage'], chain=chain_id,
            draws=len(values), divergences=int(d.sum()),
            max_depth_hits=int((depth >= config['max_treedepth']).sum()),
            max_treedepth=config['max_treedepth'],
            ebfmi=float(np.mean(np.diff(energy)**2)/variance) if variance else np.nan,
            median_stepsize=float(np.median(values[:, indices['stepsize__']])),
            median_leapfrogs=float(np.median(values[:, indices['n_leapfrog__']])),
            mean_accept_stat=float(accept.mean()), inference_eligible=record['inference_eligible']))
    for key in ('divergences', 'max_depth_hits'):
        if sum(r[key] for r in chain_rows) != record['diagnostics'][key]:
            raise ValueError(f'{name}: raw-chain {key} differs from saved diagnostics')
    # Chain-stratified comparisons prevent a pooled chain difference being hidden.
    for chain_id, values in [*chains, ('pooled', np.concatenate([v for _, v in chains]))]:
        d = values[:, indices['divergent__']].astype(bool)
        for variable in table.index:
            comparisons.append(dict(job=name, model=config['model'], stage=config['stage'],
                chain=chain_id, **table.loc[variable, ['family','parameter','subject','partner','contrast','scale']].to_dict(),
                variable=variable, **compare_draws(values[:, indices[variable]], d)))
    return chain_rows, comparisons


def plot_subject(target, name, subject, config, data, table, chains):
    parameters = config['parameters']
    units = [u for u, s in enumerate(data['subject']) if config['subjects'][s-1] == subject]
    if len(units) != 3:
        raise ValueError(f'expected three partner units: {subject}')
    pairs = PAIRS if 'acceptance_bias' in parameters else (('alpha','epsilon'),('alpha','f0'),('gamma','f0'),('f0','epsilon'))
    columns = list(SAMPLER) + list(table.index)
    pooled = np.concatenate([v for _, v in chains])
    d = pooled[:, columns.index('divergent__')].astype(bool)
    depth = pooled[:, columns.index('treedepth__')] >= config['max_treedepth']
    baseline = np.flatnonzero(~d & ~depth)
    baseline = baseline[np.linspace(0,len(baseline)-1,min(1200,len(baseline)),dtype=int)] if len(baseline) else baseline
    hit = np.flatnonzero(depth & ~d)
    hit = hit[np.linspace(0,len(hit)-1,min(1200,len(hit)),dtype=int)] if len(hit) else hit
    fig, axes = plt.subplots(3,4,figsize=(14,9), squeeze=False)
    for row, u in enumerate(sorted(units,key=lambda i:data['partner'][i])):
        for col, (a,b) in enumerate(pairs):
            def variable(parameter):
                unit = u
                if parameter == 'acceptance_bias':
                    unit = next(v for v in units if data['partner'][v] == 1)
                return f'theta[{unit+1},{parameters.index(parameter)+1}]'
            x, y = (pooled[:, columns.index(variable(p))] for p in (a,b))
            ax = axes[row,col]
            ax.scatter(x[baseline],y[baseline],s=3,c='.6',alpha=.3,label='Other transitions (display subsample)')
            ax.scatter(x[hit],y[hit],s=5,c='#c78c25',alpha=.4,label='Depth limit (display subsample)')
            ax.scatter(x[d],y[d],s=18,c='#bd2929',marker='x',label='Divergent (all)')
            if a in ('alpha','gamma') and np.all(x>0):
                ax.set_xscale('log')
                ax.xaxis.set_minor_formatter(NullFormatter())
            if b in ('alpha','gamma') and np.all(y>0):
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(a + (' (log axis)' if a in ('alpha','gamma') else ''))
            ax.set_ylabel(b + (' (log axis)' if b in ('alpha','gamma') else ''))
            ax.set_title(('computer','similar','dissimilar')[data['partner'][u]-1], fontsize=9)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle(f'{name} / {subject}\nTransition endpoints; association with flags is not participant causation',fontsize=11)
    fig.tight_layout(rect=(0,.045,1,.935))
    fig.savefig(target/f'{name}_{subject}_pairs.png',dpi=125)
    plt.close(fig)


def export(work, output, expected_jobs=4, subjects=DEFAULT_SUBJECTS):
    work, output = Path(work).resolve(), Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    if output == work or work in output.parents or output in work.parents:
        raise ValueError('output must be outside the chain work root')
    if output == root or output in root.parents or (root in output.parents and root/'results/norm_learning' not in output.parents):
        raise ValueError('repository outputs belong under results/norm_learning')
    jobs, hashes = discover(work, expected_jobs)
    for _,config,*_ in jobs:
        if not set(subjects) <= set(config['subjects']):
            raise ValueError('requested plot subject missing from fit')
    options = dict(work_root=str(work), expected_jobs=expected_jobs, subjects=list(subjects),
        sources=hashes, exporter_sha256=sha(__file__),
        helper_sha256=sha(Path(__file__).with_name('export_gu_stan_diagnostics.py')),
        numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__)
    fingerprint = digest(options)
    if output.exists():
        manifest = json.loads((output/'export_manifest.json').read_text())
        if manifest['fingerprint'] != fingerprint:
            raise FileExistsError('different audit exists; choose a new output directory')
        for rel, expected in manifest['outputs'].items():
            p = output/rel
            if not p.is_file() or sha(p) != expected:
                raise ValueError(f'changed/missing audit output: {p}')
        print(f'ALREADY_CURRENT: {output}')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gu-divergence-',dir=output.parent) as tmp:
        target = Path(tmp)/'export'
        target.mkdir()
        counts, comparisons, console = [], [], []
        for name, config, record, data, table, csvs, paths in jobs:
            variables = list(SAMPLER) + list(table.index)
            chains = sorted((read_chain(p, variables, config['samples']) for p in csvs), key=lambda x: x[0])
            if len({c for c,_ in chains}) != config['chains']:
                raise ValueError(f'duplicate chain IDs: {name}')
            rows, contrast = summarize_chains(name,config,record,table,chains)
            counts.extend(rows)
            comparisons.extend(contrast)
            for subject in subjects:
                plot_subject(target,name,subject,config,data,table,chains)
            for rel,path in paths.items():
                if path.name.endswith('stdout.txt'):
                    lines=path.read_text(errors='replace').splitlines()
                    console.append(dict(job=name,file=rel,
                        nan_logit_lines=sum('Logit transformed probability parameter' in s and 'nan' in s.lower() for s in lines),
                        exception_lines=sum('Exception:' in s for s in lines)))
            print(f'AUDITED: {name}; divergences={sum(r["divergences"] for r in rows)}; '
                  f'depth_hits={sum(r["max_depth_hits"] for r in rows)}',flush=True)
        pd.DataFrame(counts).to_csv(target/'sampler_by_chain.tsv',sep='\t',index=False)
        pd.DataFrame(comparisons).to_csv(target/'parameter_divergence_associations.tsv',sep='\t',index=False,na_rep='n/a')
        pd.DataFrame(console,columns=['job','file','nan_logit_lines','exception_lines']).to_csv(target/'console_warning_counts.tsv',sep='\t',index=False)
        (target/'README.txt').write_text(
            'Diagnostic-only audit of fingerprint-verified completed choice-extension fits.\n'
            'No sampling, exclusions, parameter changes, or inference-eligibility upgrades.\n'
            'Draw tables summarize retained post-warmup transitions only.\n'
            'Console counts include the entire console, potentially warmup: not retained divergence counts.\n'
            'All parameters are summarized within each chain and pooled; duplicate bias theta copies are removed.\n'
            'Percentiles use the same-chain nondivergent empirical CDF (midrank ties); pooled rows use pooled draws.\n'
            'Tail fraction is below 5th or above 95th nondivergent percentiles.\n'
            'Standardized differences divide by nondivergent sample SD; n/a means undefined, not zero.\n'
            'These descriptive associations are not p-values, causal attribution, or parameter-recovery evidence.\n'
            'Plots show transition endpoints, not the entire divergent trajectory. All divergent endpoints appear.\n'
            'Other/depth-limited points are display-subsampled; numerical summaries use every retained draw.\n'
            'Requested participants illustrate hypotheses, not an exhaustive search or grounds for exclusion.\n'
            'Raw chains remain in scratch.\n')
        write_json(target/'export_manifest.json',dict(fingerprint=fingerprint,configuration=options,
            outputs={p.name:sha(p) for p in target.iterdir()}))
        target.replace(output)
    print(f'EXPORTED: {len(jobs)} fits -> {output}; no models rerun',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-jobs',type=int,default=4)
    parser.add_argument('--subjects',nargs='+',default=list(DEFAULT_SUBJECTS))
    args=parser.parse_args()
    export(args.work_root,args.output_dir,args.expected_jobs,args.subjects)


if __name__=='__main__':
    main()
