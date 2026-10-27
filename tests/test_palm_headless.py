import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import resume_palm_headless as headless


class HeadlessTests(unittest.TestCase):
    def maps(self, root, mask, count=1, value=.96):
        root.mkdir(exist_ok=True)
        for unit in ('tfce', 'clustere'):
            for c in range(1,count+1):
                for suffix in ('', '_fwep'):
                    nib.save(nib.Nifti1Image(np.full(mask.shape,value),mask.affine),
                             root/f'palm_{unit}_zvstat{suffix}_c{c}.nii')

    def test_child_environment_is_isolated_and_preserves_licensing(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=dict(HOME='/home/researcher',MLM_LICENSE_FILE='license-location',
                CONDA_PREFIX='/tools/conda/envs/project',PATH='/tools/conda/envs/project/bin:/usr/bin:/usr/local/fsl/bin',
                LD_LIBRARY_PATH='/tools/conda/lib',LD_PRELOAD='injected.so',
                DISPLAY='localhost:10.0',MATLABPATH='/old/startup',OMP_NUM_THREADS='40')
            before=original.copy()
            env=headless.child_environment(Path(tmp),original)
            self.assertEqual(original,before)
            self.assertEqual(env['HOME'],original['HOME'])
            self.assertEqual(env['MLM_LICENSE_FILE'],original['MLM_LICENSE_FILE'])
            self.assertEqual(env['PATH'],'/usr/bin:/usr/local/fsl/bin')
            self.assertEqual(env['OMP_NUM_THREADS'],'1')
            for key in ('DISPLAY','LD_LIBRARY_PATH','LD_PRELOAD','MATLABPATH'):
                self.assertNotIn(key,env)
            self.assertTrue(Path(env['MATLAB_PREFDIR']).is_dir())
            self.assertTrue(Path(env['TMPDIR']).is_dir())

    def test_command_preserves_statistical_arguments_and_quotes(self):
        args=headless.palm.arguments(Path('/input'), 'tfce', Path('/out/palm'),10000,20261018)
        cmd=headless.command('/original/matlab',Path("/tools/PALM's"),args)
        self.assertEqual(cmd[1:4],headless.FLAGS)
        self.assertIn("addpath('/tools/PALM''s')",cmd[-1])
        for value in args:
            self.assertIn(headless.palm.quote(value),cmd[-1])
        self.assertIn('restoredefaultpath;',cmd[-1])
        self.assertNotIn('-nojit',cmd)
        with self.assertRaisesRegex(ValueError,'no engine substitution'):
            headless.command('/tools/octave',Path('/palm'),args)

    def test_smoke_requires_equal_raw_and_corrected_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference=root/'reference';candidate=root/'candidate'
            mask=nib.Nifti1Image(np.ones((2,2,2)),np.eye(4))
            self.maps(reference,mask);self.maps(candidate,mask)
            nib.save(mask,reference/'mask.nii.gz')
            self.assertEqual(len(headless.compare_smoke(reference,candidate,1)),4)
            nib.save(nib.Nifti1Image(np.zeros((2,2,2)),np.eye(4)),candidate/'palm_tfce_zvstat_c1.nii')
            with self.assertRaisesRegex(ValueError,'differs from original'):
                headless.compare_smoke(reference,candidate,1)

    def test_launch_records_failure_and_uses_child_only_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp)
            with patch.object(headless.subprocess,'run',return_value=subprocess.CompletedProcess([], -11)) as run:
                with self.assertRaisesRegex(RuntimeError,r'failed \(-11\)'):
                    headless.launch('/original/matlab',Path('/palm'),['-T'],attempt)
            self.assertEqual(run.call_args.kwargs['cwd'],attempt)
            self.assertNotIn('DISPLAY',run.call_args.kwargs['env'])
            self.assertEqual(json.loads((attempt/'exit_status.json').read_text())['returncode'],-11)
            self.assertTrue((attempt/'command.json').is_file())

    def test_completed_job_never_launches_and_receipt_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs';inputs.mkdir()
            mask=nib.Nifti1Image(np.ones((2,2,2)),np.eye(4));nib.save(mask,inputs/'mask.nii.gz')
            model=dict(name='test',c=[[1]],labels=['positive'])
            attempt=root/'jobs/test/tfce/attempt-original';attempt.mkdir(parents=True)
            self.maps(attempt,mask)
            receipt=attempt.parent/'complete.json'
            headless.palm.pa.write_json(receipt,dict(attempt=str(attempt),outputs={str(p):headless.palm.pa.digest(p) for p in attempt.iterdir()}))
            original=receipt.read_bytes()
            runtime=headless.HeadlessRecovery(root,root/'report',root/'PALM')
            with patch.object(headless,'launch',side_effect=AssertionError('must reuse')):
                rows=runtime.job(model,'tfce',inputs,root,root/'report','matlab',root/'PALM',10000)
            self.assertEqual(len(rows),1)
            self.assertEqual(receipt.read_bytes(),original)

    def test_successful_new_job_keeps_old_failed_attempt_and_marks_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs';inputs.mkdir()
            mask=nib.Nifti1Image(np.ones((2,2,2)),np.eye(4));nib.save(mask,inputs/'mask.nii.gz')
            failed=root/'jobs/test/tfce/attempt-failed/run.txt';failed.parent.mkdir(parents=True);failed.write_text('old failure')
            model=dict(name='test',c=[[1]],labels=['positive'])
            runtime=headless.HeadlessRecovery(root,root/'report',root/'PALM')
            runtime.profile=root/'profile';runtime.profile.mkdir();(runtime.profile/'contract.json').write_text('{}')
            def launch(engine,palm,args,attempt):
                self.assertEqual(args[args.index('-n')+1],'10000')
                self.assertEqual(args[args.index('-seed')+1],'20261018')
                self.maps(attempt,mask)
                (attempt/'run.txt').write_text('completed')
            with patch.object(headless,'launch',side_effect=launch):
                rows=runtime.job(model,'tfce',inputs,root,root/'report','matlab',root/'PALM',10000)
            self.assertEqual(rows[0]['significant_voxels'],8)
            self.assertEqual(failed.read_text(),'old failure')
            saved=json.loads((failed.parent.parent/'complete.json').read_text())
            self.assertIn('runtime_contract_sha256',saved)
            self.assertIn('attempt-headless-',saved['attempt'])

    def test_failed_new_job_has_no_completion_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runtime=headless.HeadlessRecovery(root,root/'report',root/'PALM')
            runtime.profile=root/'profile';runtime.profile.mkdir()
            def launch(engine,palm,args,attempt):
                (attempt/'run.txt').write_text('fatal error')
                raise RuntimeError('engine crash')
            with patch.object(headless,'launch',side_effect=launch):
                with self.assertRaisesRegex(RuntimeError,'engine crash'):
                    runtime.job(dict(name='test',c=[[1]]),'tfce',root,root,root/'report','matlab',root/'PALM',10000)
            self.assertFalse((root/'jobs/test/tfce/complete.json').exists())
            self.assertEqual(len(list((root/'report').rglob('run.txt'))),1)

    def test_gate_failure_blocks_remaining_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';report=root/'report';work.mkdir();report.mkdir()
            (work/'smoke').mkdir();smoke=work/'smoke/run.txt';smoke.write_text('original')
            headless.palm.pa.write_json(work/'smoke/complete.json',dict(outputs={str(smoke):headless.palm.pa.digest(smoke)}))
            headless.palm.pa.write_json(work/'contract.json',dict(models=['test'],engine='matlab',shuffles=10000))
            def gate(pending,frozen):
                self.assertEqual(len(pending),3)
                raise RuntimeError('smoke mismatch')
            with patch.object(headless.recovery.subprocess,'run'), \
                 patch.object(headless.palm.pa,'standardized_dmn_catalog',return_value=([dict(name='test')],{},None)), \
                 patch.object(headless.recovery,'validate_participants'), \
                 patch.object(headless.palm,'job',side_effect=AssertionError('no work allowed')) as original, \
                 patch.object(headless,'launch',side_effect=AssertionError('no MATLAB allowed')):
                with self.assertRaisesRegex(RuntimeError,'smoke mismatch'):
                    headless.recovery.recover(work,report,root,root,1,True,before_pending=gate)
                self.assertEqual(original.call_count,0)
            state=json.loads((report/'status.json').read_text())
            self.assertEqual(state['stage'],'runtime-validation-failed')

    def test_validation_starts_smoke_and_canary_before_other_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';report=root/'report';work.mkdir();report.mkdir()
            (work/'contract.json').write_text('{}')
            original=work/'smoke';original.mkdir()
            for name in ('copes.nii.gz','mask.nii.gz','design.mat','design.con','variance_groups.csv'):
                (original/name).write_text('original')
            engine=root/'matlab';engine.write_text('original engine')
            frozen=dict(engine=str(engine),engine_sha256=headless.palm.pa.digest(engine),models=['model'],shuffles=10000)
            runtime=headless.HeadlessRecovery(work,report,root/'PALM')
            model=dict(name='model',c=[[1]])
            runtime.models=[model];calls=[]
            def smoke(engine,palm,args,attempt):
                calls.append('smoke')
                self.assertEqual(args[args.index('-n')+1],'20')
                (attempt/'run.txt').write_text('smoke complete')
            def job(m,method,*args):
                calls.append('canary')
                self.assertEqual(m,model)
                receipt=work/'jobs/model/tfce/complete.json';receipt.parent.mkdir(parents=True)
                receipt.write_text('{}')
            with patch.object(headless,'launch',side_effect=smoke), \
                 patch.object(headless,'compare_smoke',return_value=[dict(map='test',max_abs_difference=0)]), \
                 patch.object(runtime,'job',side_effect=job):
                runtime.validate([(model,'tfce')],frozen)
            self.assertEqual(calls,['smoke','canary'])
            self.assertTrue((runtime.profile/'canary_complete.json').is_file())


if __name__=='__main__':
    unittest.main()
