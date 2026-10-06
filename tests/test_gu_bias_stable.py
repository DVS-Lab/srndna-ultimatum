import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import run_gu_choice_extensions as ext
import run_gu_bias_stable as stable
import run_gu_bias_pilot as pilot
import validate_gu_stable as validation


class StableTests(unittest.TestCase):
    def test_config_preserves_model_data_priors_seed(self):
        baseline = dict(model=pilot.MODEL, stage='full', phase='fit', data_hash='data',
            seed=123, sources={'old':'hash'}, prior_scale=1, bias_prior={'mu_sd':1.5},
            subjects=['sub-143','sub-144'], parameters=['alpha','gamma','f0','epsilon','acceptance_bias'])
        before = copy.deepcopy(baseline)
        config = stable.job_config(baseline, {'new':'source'})
        self.assertEqual(before, baseline)
        for key in ('model','stage','phase','data_hash','seed','prior_scale','bias_prior','subjects','parameters'):
            self.assertEqual(config[key], baseline[key])
        self.assertEqual(config['samples'], 6000)
        self.assertEqual(config['sources'], {'old':'hash','new':'source'})
        self.assertEqual(config['reference_config_hash'], ext.digest(baseline))

    def test_stan_prior_and_parameter_blocks_unchanged(self):
        old = ext.STAN.read_text()
        new = (validation.STAN_DIR/'gu_choice_stable.stan').read_text()
        for start, end in [('data {','transformed parameters {'), ('model {','  target += reduce_sum')]:
            self.assertEqual(old.split(start,1)[1].split(end,1)[0],
                             new.split(start,1)[1].split(end,1)[0])
        self.assertIn('matrix[U,K+B] theta;', new.split('generated quantities {')[1])
        self.assertNotIn('exp(',new.split('transformed parameters {')[1].split('model {')[0])

    def test_reference_both_bias_signs_and_overflow(self):
        for b in (-3.,0.,3.):
            for y in (0,1):
                p=dict(log_alpha=0.,log_gamma=705.,norm=2.,bias=b)
                r=validation.reference_kernel(p,1.,y)
                self.assertTrue(np.isfinite(r).all())
                self.assertAlmostEqual(r[0], -np.logaddexp(0,-b if y else b))
        p=dict(log_alpha=1000.,log_gamma=-1000.,norm=2.,bias=0.)
        self.assertAlmostEqual(validation.reference_kernel(p,1,1)[0], -np.logaddexp(0,1))

    def test_completed_fit_skipped_and_tampering_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); p=d/'evidence.txt'; p.write_text('retained')
            config={'model':'same'}
            record=dict(config_hash=ext.digest(config),files={'evidence.txt':ext.sha(p)})
            ext.atomic_json(d/'completed.json',record)
            result=ext.fit_job((d,{},config,None),'not-used',None,[])
            self.assertEqual(result,record)
            p.write_text('altered')
            with self.assertRaises(ValueError):
                ext.completed(d,ext.digest(config))

    def test_path_guards_preserve_original_fits(self):
        root=Path('/scratch/reference')
        output=ext.ROOT/'results/norm_learning/stable-test'
        with self.assertRaises(ValueError): pilot.validate_paths(root,root,output)
        with self.assertRaises(ValueError): pilot.validate_paths(ext.ROOT/'scratch',root,output)

    def test_workflow_exports_flags_and_records_failure(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root=Path(tmp); output=root/'exports'; work=root/'work'
                cmdstan=root/'cmdstan'; cmdstan.mkdir()
                (cmdstan/'makefile').write_text('CMDSTAN_VERSION := 2.40.0\n')
                reference=[(dict(stage=s,sources={},model=pilot.MODEL,seed=1),dict(job='fit-'+s))
                           for s in ('full','run1')]
                stack.enter_context(patch.object(stable,'SOURCES',[]))
                stack.enter_context(patch.object(pilot,'validate_paths'))
                stack.enter_context(patch.object(ext,'load_trials',return_value=([],{},{})))
                stack.enter_context(patch.object(pilot,'baseline_records',return_value=reference))
                stack.enter_context(patch.object(ext,'build_data',return_value={}))
                stack.enter_context(patch.object(validation,'compile_model',return_value=SimpleNamespace(exe_file='fake')))
                stack.enter_context(patch.object(validation,'validate',return_value={'passed':True}))
                fit=stack.enter_context(patch.object(ext,'fit_job',
                    side_effect=RuntimeError('simulated fit failure') if fail else None,
                    return_value={'job':'fit-test'}))
                collect=stack.enter_context(patch.object(ext,'collect'))
                export=stack.enter_context(patch.object(pilot,'run_logged'))
                stack.enter_context(patch.object(pilot,'comparison',return_value=[
                    dict(batch='pilot',stage='full',inference_eligible=False)]))
                stack.enter_context(patch('cmdstanpy.set_cmdstan_path'))
                stack.enter_context(patch('sys.argv',['runner','--execute','--cmdstan',str(cmdstan),
                    '--work-root',str(work),'--output-dir',str(output)]))
                self.assertEqual(stable.main(),1 if fail else 0)
                status=json.loads((output/'workflow_status.json').read_text())
                self.assertEqual(status['workflow_complete'],not fail)
                self.assertEqual(fit.call_count,2)
                self.assertEqual(export.call_count,0 if fail else 2)
                self.assertEqual(collect.call_count,0 if fail else 1)
                if not fail:
                    self.assertEqual(status['diagnostic_review_required'],['full'])
                    self.assertFalse(status['imaging_covariates_released'])


if __name__ == '__main__':
    unittest.main()
