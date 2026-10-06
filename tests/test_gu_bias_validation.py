import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import run_gu_bias_validation as v


class ValidationTests(unittest.TestCase):
    def test_warning_phase_brackets_not_guessed(self):
        text='''Exception: initialization
Iteration: 1 / 8000 (Warmup)
Exception: warm
Iteration: 4000 / 8000 (Warmup)
Exception: boundary
Iteration: 4001 / 8000 (Sampling)
Exception: retained
Iteration: 4100 / 8000 (Sampling)
Exception: trailing
'''
        rows=v.warning_rows(text)
        self.assertEqual([r['phase'] for r in rows],['initialization_or_warmup','warmup',
            'warmup_sampling_boundary_unknown','sampling','unknown'])
        self.assertEqual(rows[3]['preceding_iteration'],4001)
        self.assertEqual(rows[3]['following_iteration'],4100)
        self.assertEqual(v.warning_rows('no warnings'),[])

    def test_joint_truth_selection_keeps_all_chains_halves_and_flags(self):
        values=np.arange(10*17,dtype=float).reshape(10,17)
        values[:,-2]=1;values[:,-1]=12
        def read(path,variables,samples):return int(path.name),values
        item=('fit',dict(chains=4,samples=10),dict(diagnostics=dict(passed=False)),
              dict(U=3),None,[Path(str(c)) for c in (4,2,1,3)],None)
        with patch.object(v,'read_chain',side_effect=read):
            a=v.select_truths(item);b=v.select_truths(item)
        self.assertEqual(len(a),8)
        self.assertEqual([x[0] for x in a],[x[0] for x in b])
        for info,truth in a:
            self.assertEqual(info['source_divergent'],1)
            self.assertEqual(info['source_diagnostics_passed'],False)
            self.assertEqual(info['half'],1 if info['draw_zero_based']<5 else 2)
            np.testing.assert_array_equal(truth,values[info['draw_zero_based'],:15].reshape(3,5))

    def test_simulation_preserves_masks_histories_and_bidirectional_bias(self):
        data=dict(U=3,subject=[1,1,1],offers=[[1.,2.,4.]]*3,observed=[[1,0,1]]*3,
                  use_choice=[[1,0,1]]*3,choice=[[0,0,0]]*3,run=[[1,1,2]]*3)
        before=copy.deepcopy(data)
        for bias,expected in [(100,1),(-100,0)]:
            truth=np.array([[1.,1.,3.,.1,bias]]*3)
            simulated=v.simulate(data,truth,42)
            self.assertEqual(simulated['choice'],[[expected,0,expected]]*3)
            self.assertEqual(data,before)
            for k in data:
                if k!='choice':self.assertEqual(simulated[k],data[k])
            truth[1,4]+=1
            with self.assertRaises(ValueError):v.simulate(data,truth,42)

    def test_metrics_count_participants_once_and_zero_rank_variance(self):
        rows=[dict(subject=f'sub-{i}',parameter='acceptance_bias',partner='all',truth=i,
                   mean=i,q025=i-1,q975=i+1) for i in range(4)]
        frame=pd.DataFrame(rows)
        r=v.recovery_metrics(frame,'parameter','job',False)[0]
        self.assertEqual(r['n_subjects'],4);self.assertEqual(r['spearman_r'],1)
        self.assertEqual(r['rmse'],0);self.assertEqual(r['coverage_95'],1)
        with self.assertRaises(ValueError):v.recovery_metrics(pd.concat([frame,frame]),'parameter','job',False)
        frame['truth']=1
        self.assertIsNone(v.recovery_metrics(frame,'parameter','job',False)[0]['spearman_r'])

    def test_orchestration_dry_run_flags_and_partial_failure(self):
        for execute,fail in [(False,False),(True,False),(True,True)]:
            with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                root=Path(tmp);base=root/'baseline';out=root/'out';work=root/'work'
                tool=root/'cmdstan';tool.mkdir();(tool/'makefile').write_text('CMDSTAN_VERSION := 2.40.0\n')
                items=[]
                for stage in ('full','run1'):
                    config=dict(model=v.MODEL,stage=stage,chains=4,phase='fit',sources={s:v.ext.sha(v.ROOT/s)
                        for s in ('code/stan/gu_choice_stable.stan','code/stan/gu_stable_functions.stan')},
                        data_hash=v.ext.digest({}),likelihood_implementation='log_scale_stable_v1')
                    job=base/'fits'/stage;job.mkdir(parents=True);(job/'completed.json').write_text('{}')
                    items.append((stage,config,dict(diagnostics=dict(passed=False)),{},None,[],{}))
                stack.enter_context(patch.object(v,'SOURCES',[]))
                stack.enter_context(patch.object(v.pilot,'validate_paths'))
                stack.enter_context(patch.object(v.ext,'load_trials',return_value=([],{},{})))
                stack.enter_context(patch.object(v.ext,'build_data',return_value={}))
                stack.enter_context(patch.object(v,'discover',return_value=(items,{})))
                stack.enter_context(patch.object(v,'audit_sources'))
                stack.enter_context(patch('cmdstanpy.set_cmdstan_path'))
                stack.enter_context(patch.object(v,'select_truths',return_value=[(dict(chain=c,half=h),np.ones((3,5)))
                    for c in range(1,5) for h in (1,2)]))
                stack.enter_context(patch.object(v,'simulate',return_value={}))
                stack.enter_context(patch.object(v.validation,'compile_model',return_value=SimpleNamespace(exe_file='fake')))
                stack.enter_context(patch.object(v.validation,'validate',return_value={'passed':True}))
                def fit(job,*args):
                    if fail and job[0].name.endswith('001'):raise RuntimeError('test failure')
                    return dict(job=job[0].name,diagnostics=dict(passed=False))
                fits=stack.enter_context(patch.object(v.ext,'fit_job',side_effect=fit))
                def collect(records,work,out):v.ext.atomic_json(out/'provenance.json',{})
                stack.enter_context(patch.object(v.ext,'collect',side_effect=collect))
                summary=stack.enter_context(patch.object(v,'summarize_recovery'))
                exporter=stack.enter_context(patch.object(v.pilot,'run_logged'))
                args=['runner','--baseline-root',str(base),'--work-root',str(work),'--output-dir',str(out),'--cmdstan',str(tool)]
                stack.enter_context(patch('sys.argv',args+(['--execute'] if execute else [])))
                self.assertEqual(v.main(),1 if fail else 0)
                self.assertEqual(fits.call_count,8 if execute else 0)
                self.assertEqual(exporter.call_count,int(execute and not fail))
                if execute:
                    status=json.loads((out/'workflow_status.json').read_text())
                    self.assertEqual(status['completed_fits'],7 if fail else 8)
                    self.assertEqual(status['workflow_complete'],not fail)
                    self.assertFalse(status['parameter_recovery_validated'])
                    self.assertEqual(summary.call_count,1)


if __name__=='__main__':unittest.main()
