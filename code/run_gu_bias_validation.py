#!/usr/bin/env python3
"""Read-only empirical warning audit plus eight conditional recovery fits.

No observed-data refits or scientific-model changes. This is a bounded recovery
screen conditional on a diagnostically flagged posterior, NOT SBC or a release
of individual imaging covariates. Completed synthetic fits are verified/skipped.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import re
import sys
import traceback

import run_gu_choice_extensions as ext
import run_gu_bias_pilot as pilot
import validate_gu_stable as validation
from audit_gu_divergences import discover
from export_gu_stan_diagnostics import read_chain
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

ROOT = ext.ROOT
MODEL = 'rw_positive_bias'
SEED = 20261007
SOURCES = ['code/run_gu_bias_validation.py', 'code/run_gu_bias_validation.sh',
    'code/run_gu_choice_extensions.py', 'code/run_gu_bias_pilot.py',
    'code/validate_gu_stable.py', 'code/stan/gu_choice_stable.stan',
    'code/stan/gu_choice_extensions.stan',
    'code/stan/gu_stable_functions.stan', 'code/stan/gu_stable_kernel_check.stan',
    'code/audit_gu_divergences.py', 'code/export_gu_stan_diagnostics.py',
    'code/run_gu_norm_learning.py', 'code/gu_norm_model.py', 'code/build_event_corrected_trials.py']


def warning_rows(text):
    """Bracket exceptions by progress lines; never pretend to know exact iteration.

    Stream buffering can limit temporal precision. A warmup/sampling boundary
    is explicitly ambiguous rather than automatically classified as warmup.
    """
    lines = text.splitlines()
    progress = []
    for i, line in enumerate(lines):
        m = re.search(r'Iteration:\s*(\d+)\s*/.*\((Warmup|Sampling)\)', line)
        if m:
            progress.append((i, int(m[1]), m[2].lower()))
    rows = []
    for i, line in enumerate(lines):
        if 'Exception:' not in line and 'Rejecting initial value' not in line:
            continue
        before = [p for p in progress if p[0] < i]
        after = [p for p in progress if p[0] > i]
        left, right = before[-1] if before else None, after[0] if after else None
        if left and right and left[2] == right[2]:
            phase = left[2]
        elif left and right:
            phase = 'warmup_sampling_boundary_unknown'
        elif not left and right and right[2] == 'warmup':
            phase = 'initialization_or_warmup'
        else:
            phase = 'unknown'
        kind = ('nan_logit' if 'Logit transformed probability' in line and 'nan' in line
            else 'unrepresentable_cancellation' if 'unrepresentable derivatives' in line
            else 'initialization' if 'Rejecting initial value' in line else 'other_exception')
        rows.append(dict(line=i+1, kind=kind, phase=phase,
            preceding_iteration=left[1] if left else None,
            following_iteration=right[1] if right else None, message=line.strip()))
    return rows


def audit_sources(items, output):
    rows, counts = [], []
    for name, config, record, data, _, _, paths in items:
        for relative, path in sorted(paths.items()):
            if not relative.endswith('-stdout.txt'):
                continue
            text = path.read_text()
            match = re.search(r'^id\s*=\s*(\d+)', text, re.M)
            chain = int(match[1]) if match else None
            selected = warning_rows(text)
            rows.extend(dict(job=name, stage=config['stage'], chain=chain,
                             source_file=relative, **r) for r in selected)
            counts.append(dict(job=name, stage=config['stage'], chain=chain,
                source_file=relative, exception_lines=len(selected),
                warmup=sum(r['phase']=='warmup' for r in selected),
                sampling=sum(r['phase']=='sampling' for r in selected),
                initialization_or_warmup=sum(r['phase']=='initialization_or_warmup' for r in selected),
                boundary_or_unknown=sum('unknown' in r['phase'] for r in selected)))
    if rows:
        ext.write_table(output/'empirical_warning_context.tsv', rows)
    else:
        (output/'empirical_warning_context.tsv').write_text('job\tstage\tchain\tline\tkind\tphase\tmessage\n')
    ext.write_table(output/'empirical_warning_counts.tsv', counts)
    ext.atomic_json(output/'empirical_warning_audit.json', dict(exception_lines=len(rows),
        phase_counts={p:sum(r['phase']==p for r in rows) for p in
            ('initialization_or_warmup','warmup','sampling','warmup_sampling_boundary_unknown','unknown')},
        interpretation='Progress-line bracketing, not exact iterations. Buffered output can limit precision. '
                       'Exception counts are not retained divergence counts.'))


def select_truths(item):
    """One whole joint draw per half of each chain; no diagnostic/value filtering."""
    _, config, record, data, _, csvs, _ = item
    if config['chains'] != 4:
        raise ValueError('Expected four source chains')
    variables = [f'theta[{u+1},{k+1}]' for u in range(data['U']) for k in range(5)]
    variables += ['divergent__', 'treedepth__']
    seen, selected = set(), []
    for path in sorted(csvs):
        chain, values = read_chain(path, variables, config['samples'])
        if chain in seen:
            raise ValueError('Duplicate source chain ID')
        seen.add(chain)
        for half, indices in enumerate(np.array_split(np.arange(len(values)), 2), 1):
            rng = np.random.default_rng(SEED + chain*101 + half)
            index = int(rng.choice(indices))
            truth = values[index,:data['U']*5].reshape(data['U'],5).copy()
            selected.append((dict(chain=chain, half=half, draw_zero_based=index,
                source_file=path.name, source_divergent=int(values[index,-2]),
                source_treedepth=int(values[index,-1]),
                source_diagnostics_passed=record['diagnostics']['passed']), truth))
    if seen != {1,2,3,4}:
        raise ValueError('Expected source chain IDs 1..4')
    return sorted(selected, key=lambda t:(t[0]['chain'],t[0]['half']))


def simulate(data, truth, seed):
    truth = np.asarray(truth)
    if truth.shape != (data['U'],5) or not np.isfinite(truth).all():
        raise ValueError('Nonfinite or mismatched generating parameters')
    for subject in set(data['subject']):
        bias = truth[np.array(data['subject']) == subject,4]
        if not np.all(bias == bias[0]):
            raise ValueError('Generating bias must be shared across partners')
    z = ext.logits(truth, data['offers'], MODEL)
    if not np.isfinite(z).all():
        raise ValueError('Generating logits are not finite; do not silently clip or skip this truth')
    new = copy.deepcopy(data)
    y = np.random.default_rng(seed).binomial(1,expit(z))
    new['choice'] = np.where(np.array(data['observed'],bool),y,0).tolist()
    return new


def recovery_metrics(table, kind, job, passed):
    """Subject is the independent recovery unit; shared bias is counted once."""
    table = table[table.subject != 'sample_mean']
    groups = ['parameter','partner'] if kind == 'parameter' else ['parameter','contrast']
    rows = []
    for keys, frame in table.groupby(groups, sort=True):
        if frame.subject.duplicated().any() or len(frame) < 3:
            raise ValueError('Recovery groups require one record per participant')
        x, y = frame.truth.to_numpy(), frame['mean'].to_numpy()
        rho = float(spearmanr(x,y).statistic) if np.ptp(x) and np.ptp(y) else None
        error = y-x
        rows.append(dict(job=job, diagnostic_passed=passed, target=kind,
            parameter=keys[0], partner_or_contrast=keys[1], n_subjects=len(frame),
            spearman_r=rho, rmse=float(np.sqrt(np.mean(error**2))),
            mean_error=float(error.mean()), coverage_95=float(((frame.q025<=x)&(x<=frame.q975)).mean()),
            median_interval_width=float((frame.q975-frame.q025).median())))
    return rows


def summarize_recovery(records, work, output):
    rows, group_rows = [], []
    for record in records:
        job = work/'fits'/record['job']; attempt=job/record['attempt']
        passed=record['diagnostics']['passed']
        truth=np.array(json.loads((job/'simulation_truth.json').read_text()))
        data=json.loads((attempt/'data.json').read_text())
        for kind, name in [('parameter','parameters'),('partner_difference','partner_contrasts')]:
            t=pd.read_csv(attempt/f'{name}.tsv',sep='\t')
            rows.extend(recovery_metrics(t,kind,record['job'],passed))
            if kind=='partner_difference':
                true=ext.difference_draws(truth,data)
                for r in t[t.subject=='sample_mean'].to_dict('records'):
                    value=float(true[r['contrast']][:,ext.MODELS[MODEL][1].index(r['parameter'])].mean())
                    group_rows.append(dict(job=record['job'],diagnostic_passed=passed,
                        contrast=r['contrast'],parameter=r['parameter'],truth=value,
                        mean=r['mean'],q025=r['q025'],q975=r['q975'],
                        covered_95=r['q025']<=value<=r['q975']))
    ext.write_table(output/'recovery_by_subject_metric.tsv',rows)
    ext.write_table(output/'sample_mean_contrast_recovery.tsv',group_rows)
    table=pd.DataFrame(rows)
    summaries=[]
    for label, frame in [('all_completed_descriptive',table),
                          ('diagnostic_pass_only',table[table.diagnostic_passed])]:
        for key, sub in frame.groupby(['target','parameter','partner_or_contrast']):
            summaries.append(dict(selection=label,target=key[0],parameter=key[1],partner_or_contrast=key[2],
                completed_replicates=len(sub),median_spearman_r=None if sub.spearman_r.isna().all() else float(sub.spearman_r.median()),
                median_rmse=float(sub.rmse.median()),mean_coverage_95=float(sub.coverage_95.mean())))
    ext.write_table(output/'recovery_summary_by_target.tsv',summaries)
    import matplotlib.pyplot as plt
    table['label']=table.parameter+' / '+table.partner_or_contrast
    grid=table.pivot(index='job',columns='label',values='spearman_r')
    fig,ax=plt.subplots(figsize=(16,5))
    im=ax.imshow(grid.to_numpy(float),vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
    ax.set_xticks(range(len(grid.columns)),grid.columns,rotation=65,ha='right',fontsize=8)
    pass_map={r['job']:r['diagnostics']['passed'] for r in records}
    ax.set_yticks(range(len(grid.index)),[f'{s.rsplit("rep",1)[-1]} ({"pass" if pass_map[s] else "flagged"})' for s in grid.index])
    ax.set_title('Conditional recovery screen: participant rank correlation\nFlagged fits retained for diagnosis; not validated imaging measures')
    fig.colorbar(im,ax=ax,label='Spearman r');fig.tight_layout()
    fig.savefig(output/'recovery_overview.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,default=Path('/ZPOOL/data/scratch/srndna-gu-bias-stable-v1'))
    parser.add_argument('--work-root',type=Path,default=Path('/ZPOOL/data/scratch/srndna-gu-bias-validation-v1'))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/norm_learning/stan-bias-validation-v1')
    parser.add_argument('--cmdstan',type=Path,default=Path('/ZPOOL/data/scratch/srndna-stan-toolchain/cmdstan-2.40.0'))
    parser.add_argument('--execute',action='store_true')
    a=parser.parse_args()
    work,baseline,output=a.work_root.resolve(),a.baseline_root.resolve(),a.output_dir.resolve()
    pilot.validate_paths(work,baseline,output)
    work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    with (work/'.validation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            units,_,_=ext.load_trials(ROOT)
            items,hashes=discover(baseline,2)
            for _,config,_,data,_,_,_ in items:
                if config['model']!=MODEL or config.get('likelihood_implementation')!='log_scale_stable_v1':
                    raise ValueError('Expected completed stable bias fits')
                if config['data_hash']!=ext.digest(ext.build_data(units,MODEL,config['stage'])):
                    raise ValueError('Current data/priors differ from empirical fits')
                for name in ['code/stan/gu_choice_stable.stan','code/stan/gu_stable_functions.stan']:
                    if config['sources'][name]!=ext.sha(ROOT/name):
                        raise ValueError('Empirical Stan kernel differs from current code')
            full=[item for item in items if item[1]['stage']=='full']
            if len(full)!=1 or {i[1]['stage'] for i in items}!={'full','run1'}:
                raise ValueError('Need exactly one full and one run-1 source fit')
            import cmdstanpy
            cmdstanpy.set_cmdstan_path(str(a.cmdstan.resolve()))
            version=next(line.split(':=',1)[1].strip() for line in (a.cmdstan/'makefile').read_text().splitlines()
                         if line.startswith('CMDSTAN_VERSION :='))
            if version!='2.40.0' or cmdstanpy.__version__!='1.3.0':
                raise ValueError('Requires CmdStan 2.40.0 and CmdStanPy 1.3.0')
            guard=dict(baseline=str(baseline),output=str(output),baseline_hashes=hashes,
                sources={s:ext.sha(ROOT/s) for s in SOURCES},seed=SEED,replicates=8,
                chains=4,threads_per_chain=2,warmup=4000,samples=4000,adapt_delta=.995,max_treedepth=12,
                recovery_kind='Posterior-conditioned joint draws, not SBC or independent validation')
            ext.config_guard(work/'workflow',guard)
            ext.atomic_json(output/'workflow_configuration.json',guard)
            audit_sources(items,output)
            print('AUDITED: empirical console warnings; observed-data fits are not changed.',flush=True)
            print('PLAN: 8 conditional recovery fits, all participants/partners and missed-trial masks retained.',flush=True)
            print('CPU: up to 5 concurrent fits x 4 chains x 2 threads = 40.',flush=True)
            if not a.execute:
                print('PASS: audit and preparation checks; add --execute to run recovery.',flush=True)
                return 0
            source=full[0];jobs=[];selections=[]
            for rep,(selection,truth) in enumerate(select_truths(source),1):
                seed=SEED+rep*1009
                data=simulate(source[3],truth,seed)
                config=copy.deepcopy(source[1])
                config.update(phase='recovery',replicate=rep,seed=seed,samples=4000,warmup=4000,
                    chains=4,threads_per_chain=2,adapt_delta=.995,max_treedepth=12,
                    data_hash=ext.digest(data),truth_hash=ext.digest(truth.tolist()),truth_source=selection,
                    source_completion_hash=ext.sha(baseline/'fits'/source[0]/'completed.json'),
                    recovery_kind=guard['recovery_kind'],software=dict(cmdstan=version,cmdstanpy=cmdstanpy.__version__))
                config['sources'].update(guard['sources'])
                directory=work/'fits'/f'recovery-{MODEL}-full-conditional-rep{rep:03d}'
                ext.config_guard(directory,config)
                truth_path=directory/'simulation_truth.json'
                if truth_path.exists() and ext.digest(json.loads(truth_path.read_text()))!=config['truth_hash']:
                    raise ValueError(f'Changed generating truth: {truth_path}')
                if not truth_path.exists():ext.atomic_json(truth_path,truth.tolist())
                jobs.append((directory,data,config,truth));selections.append(dict(job=directory.name,**selection))
            ext.write_table(output/'generating_draws.tsv',selections)
            compiled=validation.compile_model('gu_choice_stable',work)
            audit=validation.validate(compiled,validation.compile_model('gu_choice_extensions',work),
                validation.compile_model('gu_stable_kernel_check',work),units,work)
            ext.atomic_json(output/'compiled_equivalence.json',audit)
            options=argparse.Namespace(chains=4,threads_per_chain=2,warmup=4000,samples=4000,adapt_delta=.995,max_treedepth=12)
            records,errors=[],[]
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures={pool.submit(ext.fit_job,j,compiled.exe_file,options,units):j[0].name for j in jobs}
                for future in as_completed(futures):
                    try:records.append(future.result())
                    except Exception as exc:
                        errors.append(dict(job=futures[future],error=str(exc)))
                        print(f'FAILED: {futures[future]}: {exc}',flush=True)
                    # Small checkpoint even if a later fit fails or the session ends.
                    ext.atomic_json(output/'progress.json',dict(completed=[r['job'] for r in records],failed=errors,planned=8))
            records.sort(key=lambda r:r['job'])
            if records:
                ext.collect(records,work,output)
                provenance=json.loads((output/'provenance.json').read_text())
                provenance['recovery']=guard['recovery_kind']+'; source empirical fit remains diagnostically flagged'
                ext.atomic_json(output/'provenance.json',provenance)
                summarize_recovery(records,work,output)
            if len(records)==8 and not errors:
                pilot.run_logged([sys.executable,'-u',str(ROOT/'code/export_gu_stan_diagnostics.py'),
                    '--work-root',str(work),'--phase','recovery','--expected-jobs',str(len(records)),
                    '--output-dir',str(output/f'diagnostics-completed-{len(records)}')],output/'export.txt')
            flagged=[r['job'] for r in records if not r['diagnostics']['passed']]
            ext.atomic_json(output/'workflow_status.json',dict(workflow_complete=len(records)==8 and not errors,
                completed_fits=len(records),planned_fits=8,failed=errors,diagnostic_review_required=flagged,
                empirical_fits_refitted=False,parameter_recovery_validated=False,imaging_covariates_released=False,
                interpretation='Bounded conditional screen; review recovery by parameter, partner and contrast before scientific use.'))
            print(f'EXPORTED: {len(records)}/8 recovery fits, {len(flagged)} flagged; no observed-data rerun or imaging release.',flush=True)
            return 1 if errors else 0
        except Exception as exc:
            ext.atomic_json(output/'workflow_status.json',dict(workflow_complete=False,error=str(exc)))
            traceback.print_exc();return 1


if __name__=='__main__':
    raise SystemExit(main())
