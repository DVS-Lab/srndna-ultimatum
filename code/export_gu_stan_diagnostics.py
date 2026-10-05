#!/usr/bin/env python3
"""Export compact, labeled Stan diagnostics without modifying or refitting models.

Only completed, fingerprint-verified fits are read. Raw posterior CSVs remain
in scratch. Trace plots are diagnostic displays, not scientific effect figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-stan-diagnostic-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PARTNERS = ('computer', 'similar', 'dissimilar')
CONTRASTS = ('human_minus_computer', 'similar_minus_dissimilar')
CORE = re.compile(r'^(mu|effect|age_beta|sigma_subject|sigma_contrast|z_subject|z_contrast|theta|bias_mu|bias_sigma|bias_z)\[([0-9,]+)\]$')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checked_file(job, record, relative):
    path = (job/relative).resolve()
    if job.resolve() not in path.parents:
        raise ValueError(f'path leaves job directory: {relative}')
    expected = record['files'].get(relative)
    if expected is None or not path.is_file() or sha(path) != expected:
        raise ValueError(f'missing or modified completed input: {path}')
    return path


def variable_label(variable, config, data):
    match = CORE.fullmatch(variable)
    if not match:
        raise ValueError(f'unknown core parameter: {variable}')
    family, indices = match.groups()
    index = [int(i)-1 for i in indices.split(',')]
    if family.startswith('bias_'):
        return dict(variable=variable, family=family, parameter='acceptance_bias',
                    subject=config['subjects'][index[0]] if family == 'bias_z' else '',
                    partner='all', contrast='',
                    scale={'bias_mu': 'log_odds', 'bias_sigma': 'log_odds_standard_deviation',
                           'bias_z': 'standard_normal_deviation'}[family])
    parameter = config['parameters'][index[-1]]
    row = dict(variable=variable, family=family, parameter=parameter,
               subject='', partner='', contrast='', scale='latent')
    if family == 'theta':
        unit = index[0]
        row.update(subject=config['subjects'][data['subject'][unit]-1],
                   partner=PARTNERS[data['partner'][unit]-1], scale='natural')
    elif family == 'z_subject':
        row.update(subject=config['subjects'][index[0]], scale='standard_normal_deviation')
    elif family == 'z_contrast':
        row.update(subject=config['subjects'][index[1]], contrast=CONTRASTS[index[0]],
                   scale='standard_normal_deviation')
    elif family in ('effect', 'sigma_contrast'):
        row['contrast'] = CONTRASTS[index[0]]
    elif family == 'age_beta':
        row['contrast'] = 'older_minus_younger'
    if family.startswith('sigma'):
        row['scale'] = 'latent_standard_deviation'
    if family == 'theta' and parameter == 'acceptance_bias':
        row.update(partner='all', scale='log_odds')
    return row


def parameter_table(summary, config, data):
    if not summary.index.is_unique:
        raise ValueError('duplicate variables in Stan summary')
    core = summary.loc[[i for i in summary.index if CORE.fullmatch(i)]].copy()
    if core.empty:
        raise ValueError('no monitored parameters in Stan summary')
    labels = pd.DataFrame([variable_label(i, config, data) for i in core.index]).set_index('variable')
    core = labels.join(core)
    for col in ('R_hat', 'ESS_bulk', 'ESS_tail'):
        core[col] = pd.to_numeric(core[col], errors='raise')
    core['nonfinite_diagnostic'] = ~np.isfinite(core[['R_hat','ESS_bulk','ESS_tail']]).all(axis=1)
    core['rhat_flag'] = core.R_hat > 1.01
    core['bulk_ess_flag'] = core.ESS_bulk < 400
    core['tail_ess_flag'] = core.ESS_tail < 400
    core['flagged'] = core[['nonfinite_diagnostic','rhat_flag','bulk_ess_flag','tail_ess_flag']].any(axis=1)
    return core


def worst_variables(table, count):
    # Include the worst of each diagnostic before filling by R-hat rank.
    ranks = [list(table.sort_values(c, ascending=ascending, na_position='first').index)
             for c, ascending in [('R_hat', False), ('ESS_bulk', True), ('ESS_tail', True)]]
    selected = []
    for rank in ranks:
        if rank[0] not in selected:
            selected.append(rank[0])
    for variable in ranks[0]:
        if variable not in selected:
            selected.append(variable)
    return selected[:count]


def stan_column(variable):
    return variable.replace('[', '.').replace(',', '.').replace(']', '')


def read_chain(path, variables, samples):
    chain_id, saved_warmup, header = None, None, None
    with path.open() as stream:
        for line in stream:
            if line.startswith('#'):
                found = re.match(r'#\s*id\s*=\s*(\d+)', line)
                if found:
                    chain_id = int(found.group(1))
                found = re.match(r'#\s*save_warmup\s*=\s*(false|true|0|1)\b', line)
                if found:
                    saved_warmup = found.group(1) in ('true', '1')
            elif line.strip():
                header = line.strip().split(',')
                break
    if chain_id is None or saved_warmup != 0 or header is None:
        raise ValueError(f'expected chain ID and posterior-only CSV (save_warmup=0): {path}')
    columns = [stan_column(v) for v in variables]
    if not set(columns) <= set(header):
        raise ValueError(f'selected parameter columns missing in {path}')
    frame = pd.read_csv(path, comment='#', usecols=columns)[columns]
    if len(frame) != samples or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f'incomplete or nonfinite chain: {path}; expected {samples} draws')
    return chain_id, frame.to_numpy()


def plot_traces(directory, name, table, variables, chains):
    values = np.stack([a for _, a in chains])  # chain x draw x variable
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(len(variables), 2, figsize=(13, 2.8*len(variables)), squeeze=False)
    stats, pairs = [], []
    n = values.shape[1]
    stride = max(1, int(np.ceil(n/750)))
    for j, variable in enumerate(variables):
        label = table.loc[variable]
        title = ' / '.join(str(label[k]) for k in ('parameter','subject','partner','contrast') if label[k])
        ranks = rankdata(values[:,:,j].ravel()).reshape(len(chains), n)
        bins = np.linspace(.5, ranks.size+.5, 21)
        for c, (chain_id, _) in enumerate(chains):
            x = values[c,:,j]
            color = colors[c % len(colors)]
            axes[j,0].plot(np.arange(1, n+1)[::stride], x[::stride], color=color,
                           lw=.65, alpha=.8, label=f'Chain {chain_id}')
            counts, _ = np.histogram(ranks[c], bins=bins)
            axes[j,1].step((bins[:-1]+bins[1:])/2, counts, where='mid', color=color,
                           label=f'Chain {chain_id}')
            stats.append(dict(variable=variable, chain=chain_id, draws=n,
                mean=float(x.mean()), sd=float(x.std(ddof=1)), q025=float(np.quantile(x,.025)),
                q975=float(np.quantile(x,.975)), first_half_mean=float(x[:n//2].mean()),
                second_half_mean=float(x[n//2:].mean())))
        axes[j,0].set_title(f'{variable}: {title}\n{label["scale"]}', fontsize=9, loc='left')
        axes[j,1].set_title(f'R-hat {label.R_hat:.4f}; ESS bulk {label.ESS_bulk:.0f}, tail {label.ESS_tail:.0f}', fontsize=9)
        axes[j,1].axhline(n/20, color='.5', ls='--', lw=.8)
        axes[j,0].set_xlabel('Retained post-warmup draw')
        axes[j,1].set_xlabel('Pooled rank (all draws; 20 bins)')
        axes[j,1].set_ylabel('Draws per chain')
        for ax in axes[j]:
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(ncol=min(4,len(chains)), fontsize=8, loc='best')
    fig.suptitle(f'{name}\nSampling diagnostics; trace display stride {stride}; not an effect figure', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(directory/f'{name}_traces.png', dpi=140)
    plt.close(fig)
    for c, (chain_id, _) in enumerate(chains):
        for j in range(len(variables)):
            for k in range(j+1, len(variables)):
                a, b = values[c,:,j], values[c,:,k]
                correlation = float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else np.nan
                pairs.append(dict(chain=chain_id, variable_a=variables[j], variable_b=variables[k],
                                  correlation=correlation))
    return stats, pairs


def plot_overview(rows, path):
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(table)*.48)))
    labels = table['job'].str.replace('fit-', '', regex=False).str.replace('-prior1', '', regex=False)
    y = np.arange(len(table))
    for ax in axes:
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.spines[['top','right']].set_visible(False)
    axes[0].scatter(table.max_rhat, y, c=['#9d302b' if x else '#286b78' for x in table.flagged_parameters.gt(0)])
    axes[0].axvline(1.01, ls='--', color='.4')
    axes[0].set_xlabel('Maximum parameter R-hat (target ≤ 1.01)')
    axes[1].scatter(table.min_bulk_ess, y-.1, label='Bulk', marker='o')
    axes[1].scatter(table.min_tail_ess, y+.1, label='Tail', marker='s')
    axes[1].axvline(400, ls='--', color='.4')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Minimum parameter ESS (target ≥ 400; log scale)')
    axes[1].legend()
    fig.suptitle('Stan convergence audit — all completed fits, including flagged fits')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def export(work, output, phase='fit', expected_jobs=8, trace_count=4):
    work, output = Path(work).resolve(), Path(output).resolve()
    if output == work or work in output.parents or output in work.parents:
        raise ValueError('export must be outside the Stan work root')
    root = Path(__file__).resolve().parents[1]
    if output == root or output in root.parents or (root in output.parents and root/'results/norm_learning' not in output.parents):
        raise ValueError('repository exports belong under results/norm_learning only')
    if not 3 <= trace_count <= 6 or expected_jobs < 1:
        raise ValueError('trace count must be 3..6 and expected jobs positive')
    jobs, source_hashes = [], {}
    for cp in sorted((work/'fits').glob('*/configuration.json')):
        config = json.loads(cp.read_text())
        if config['phase'] != phase:
            continue
        job = cp.parent
        completion = job/'completed.json'
        if not completion.is_file():
            raise ValueError(f'incomplete fit; no export presented as complete: {job.name}')
        record = json.loads(completion.read_text())
        if record['config_hash'] != digest(config) or record['job'] != job.name:
            raise ValueError(f'configuration fingerprint mismatch: {job}')
        relative = record['attempt']+'/'
        data_path = checked_file(job, record, relative+'data.json')
        data = json.loads(data_path.read_text())
        if digest(data) != config['data_hash']:
            raise ValueError(f'data fingerprint mismatch: {job}')
        summary_path = checked_file(job, record, relative+'stan_summary.tsv')
        diagnostics_path = checked_file(job, record, relative+'diagnostics.json')
        if json.loads(diagnostics_path.read_text()) != record['diagnostics']:
            raise ValueError(f'diagnostic record mismatch: {job}')
        table = parameter_table(pd.read_csv(summary_path, sep='\t', index_col=0), config, data)
        variables = worst_variables(table, trace_count)
        chains = []
        used = [cp, completion, data_path, summary_path, diagnostics_path]
        for relative_csv in sorted(record['files']):
            if relative_csv.startswith(relative) and relative_csv.endswith('.csv'):
                chain = checked_file(job, record, relative_csv)
                chains.append(read_chain(chain, variables, config['samples']))
                used.append(chain)
        ids = [c for c, _ in chains]
        if len(ids) != config['chains'] or len(set(ids)) != len(ids):
            raise ValueError(f'wrong number of unique completed chains: {job}')
        for path in used:
            source_hashes[str(path.relative_to(work))] = sha(path)
        jobs.append((job.name, config, record, table, variables, sorted(chains)))
        print(f'CHECKED: {job.name}; {len(table)} parameters; {int(table.flagged.sum())} flagged', flush=True)
    if len(jobs) != expected_jobs:
        raise ValueError(f'expected {expected_jobs} completed {phase} fits; found {len(jobs)}')
    config = dict(work_root=str(work), phase=phase, expected_jobs=expected_jobs,
        trace_count=trace_count, source_hashes=source_hashes, exporter_sha256=sha(__file__),
        numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__)
    fingerprint = digest(config)
    if output.exists() and any(output.iterdir()):
        manifest_path = output/'export_manifest.json'
        if not manifest_path.is_file():
            raise FileExistsError(f'nonempty unmanaged output directory: {output}')
        prior = json.loads(manifest_path.read_text())
        if prior['fingerprint'] != fingerprint:
            raise FileExistsError('export inputs/options changed; use a NEW output directory')
        for relative, expected in prior['outputs'].items():
            candidate = output/relative
            if not candidate.is_file() or sha(candidate) != expected:
                raise ValueError(f'export file changed/missing: {candidate}; use a new output directory')
        print(f'ALREADY_CURRENT: verified {output}', flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gu-diagnostic-export-', dir=output.parent) as tmp:
        target = Path(tmp)/'export'
        target.mkdir()
        all_tables, worst_tables, summaries, chain_stats, pairs = [], [], [], [], []
        for name, config_job, record, table, variables, chains in jobs:
            table = table.copy()
            table.insert(0, 'job', name)
            table.insert(1, 'model', config_job['model'])
            table.insert(2, 'stage', config_job['stage'])
            all_tables.append(table.reset_index())
            worst_tables.append(table.loc[variables].reset_index())
            summaries.append(dict(job=name, phase=config_job['phase'], model=config_job['model'],
                stage=config_job['stage'], parameters=len(table), flagged_parameters=int(table.flagged.sum()),
                **record['diagnostics'], inference_eligible=record['inference_eligible']))
            stats, corr = plot_traces(target, name, table, variables, chains)
            chain_stats.extend(dict(job=name, **row) for row in stats)
            pairs.extend(dict(job=name, **row) for row in corr)
        all_parameters = pd.concat(all_tables, ignore_index=True)
        all_parameters.to_csv(target/'parameter_diagnostics.tsv', sep='\t', index=False)
        all_parameters[all_parameters.flagged].to_csv(target/'flagged_parameters.tsv', sep='\t', index=False)
        pd.concat(worst_tables).to_csv(target/'plotted_parameters.tsv', sep='\t', index=False)
        pd.DataFrame(summaries).to_csv(target/'fit_diagnostics.tsv', sep='\t', index=False)
        pd.DataFrame(chain_stats).to_csv(target/'chain_summaries.tsv', sep='\t', index=False)
        pd.DataFrame(pairs).to_csv(target/'selected_pair_correlations.tsv', sep='\t', index=False)
        plot_overview(summaries, target/'convergence_overview.png')
        (target/'README.txt').write_text(
            'Posterior sampling diagnostics, not a new analysis or model refit.\n'
            'All completed fits are included, including those passing diagnostics.\n'
            'Parameter labels use the saved fit configuration and unit/subject mapping.\n'
            'Flags: R-hat >1.01, bulk/tail ESS <400, or nonfinite diagnostics.\n'
            'Trace variables include the worst R-hat/bulk ESS/tail ESS, then further R-hat ranks.\n'
            'Trace display uses at most 750 points per chain; rank histograms and summaries use all draws.\n'
            'Dashed rank-histogram line is the uniform expected count, not a significance threshold.\n'
            'Between/within-chain behavior and selected correlations require interpretation;\n'
            'these plots do not establish parameter identifiability or recovery.\n'
            'Original inference-eligibility labels are retained, never upgraded by this exporter.\n'
            'No raw draws, model outputs, or sampling settings were modified.\n')
        manifest = dict(fingerprint=fingerprint, configuration=config,
            outputs={p.name: sha(p) for p in target.iterdir() if p.is_file()})
        write_json(target/'export_manifest.json', manifest)
        target.replace(output)
    print(f'EXPORTED: {len(jobs)} fits, {len(jobs)+1} figures, six diagnostic tables -> {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['fit','smoke','recovery'], default='fit')
    parser.add_argument('--expected-jobs', type=int, default=8)
    parser.add_argument('--trace-count', type=int, default=4)
    args = parser.parse_args()
    export(args.work_root, args.output_dir, args.phase, args.expected_jobs, args.trace_count)


if __name__ == '__main__':
    main()
