"""Opt-in real Stan plumbing test; deliberately short, never inferential."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import run_gu_bias_validation as v


@unittest.skipUnless(os.environ.get('SRNDNA_STABLE_INTEGRATION')=='1','optional compiled Stan test')
class IntegrationTest(unittest.TestCase):
    def test_joint_draw_simulation_fitting_resume_metrics_and_exports(self):
        import cmdstanpy
        from export_gu_stan_diagnostics import export
        cmdstanpy.set_cmdstan_path(os.environ['CMDSTAN'])
        with tempfile.TemporaryDirectory(prefix='gu-recovery-integration-') as tmp:
            base=Path(tmp);work=base/'recovery';out=base/'exports';work.mkdir();out.mkdir()
            units,_,_=v.ext.load_trials(v.ROOT);units=units[:9]
            data=v.ext.build_data(units,v.MODEL)
            # Reuse the content-addressed test build if supplied; otherwise compile in temporary scratch.
            compiled=v.validation.compile_model('gu_choice_stable',Path(os.environ.get('SRNDNA_TEST_BUILD',str(base))))
            config=dict(model=v.MODEL,stage='full',phase='fit',prior_scale=1.,seed=17,
                data_hash=v.ext.digest(data),parameters=list(v.ext.MODELS[v.MODEL][1]),
                subjects=list(dict.fromkeys(u['subject'] for u in units)),
                chains=4,samples=100,warmup=100,max_treedepth=10,age_mode='none',sources={})
            options=argparse.Namespace(chains=4,threads_per_chain=1,samples=100,warmup=100,
                                       adapt_delta=.9,max_treedepth=10)
            source=base/'source';v.ext.config_guard(source,config)
            record=v.ext.fit_job((source,data,config,None),compiled.exe_file,options,units)
            csvs=sorted((source/record['attempt']).glob('*.csv'))
            item=('source',config,record,data,None,csvs,{})
            selected=v.select_truths(item);self.assertEqual(len(selected),8)
            records=[]
            for rep,(selection,truth) in enumerate(selected,1):
                simulated=v.simulate(data,truth,rep+31)
                c=dict(config,phase='recovery',seed=rep+31,data_hash=v.ext.digest(simulated),
                       truth_hash=v.ext.digest(truth.tolist()))
                directory=work/'fits'/f'recovery-test-rep{rep:03d}'
                v.ext.config_guard(directory,c)
                v.ext.atomic_json(directory/'simulation_truth.json',truth.tolist())
                job=(directory,simulated,c,truth)
                completed=v.ext.fit_job(job,compiled.exe_file,options,units)
                self.assertEqual(v.ext.fit_job(job,compiled.exe_file,options,units),completed)
                records.append(completed)
            v.ext.collect(records,work,out)
            v.summarize_recovery(records,work,out)
            export(work,out/'diagnostics',phase='recovery',expected_jobs=8)
            metrics=v.pd.read_csv(out/'recovery_by_subject_metric.tsv',sep='\t')
            self.assertEqual(set(metrics.n_subjects),{3})
            self.assertEqual(len(metrics),8*(4*3+1+4*2))
            groups=v.pd.read_csv(out/'sample_mean_contrast_recovery.tsv',sep='\t')
            self.assertEqual(len(groups),8*8)
            self.assertTrue((out/'recovery_overview.png').is_file())
            print('PASS: eight real short synthetic fits, recovery metrics, diagnostic plots and resume; not scientific validation')


if __name__=='__main__':unittest.main()
