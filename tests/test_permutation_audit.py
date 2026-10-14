from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import run_permutation_audit as audit


class PermutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models,cls.sample,_=audit.catalog(ROOT)
        cls.tests,cls.mapping=audit.plan_tests(cls.models,20261014)

    def test_catalog_covers_every_retained_model_and_contrast(self):
        self.assertEqual(len(self.models),36)
        self.assertEqual(len(self.mapping),252)
        self.assertEqual(len(self.tests),174)
        self.assertEqual(sum(m['robust'] for m in self.models),3)
        self.assertTrue(all(len(m['subjects'])==47 for m in self.models))

    def test_dedup_keeps_both_directions_and_robust_aliases(self):
        by={(r['model'],r['contrast']):r['test'] for r in self.mapping}
        for timing in ('full-rt','pooled-post','partner-post'):
            for k in range(1,9):
                self.assertEqual(by[timing+'__activation-norm',k],by['crossed__'+timing+'_act_acceptance-intercept',k])
            for k in range(1,7):
                self.assertEqual(by[timing+'__dmn-simple-effects',k],by[timing+'__dmn-outlier-deweighted',k])
            self.assertNotEqual(by[timing+'__activation-norm',1],by[timing+'__activation-norm',2])

    def test_column_permutation_preserves_estimand(self):
        rng=np.random.default_rng(8); x=rng.normal(size=(12,4)); c=np.array([[0,1,-1,0]])
        a,b=audit.canonical_design(x,c)
        order=[3,2,0,1]; other,contrast=audit.canonical_design(x[:,order],c[:,order])
        np.testing.assert_array_equal(a,other); np.testing.assert_array_equal(b,contrast)
        np.testing.assert_allclose(x@np.linalg.pinv(x),a@np.linalg.pinv(a))

    def test_commands_keep_means_and_do_not_misuse_variance_groups(self):
        for method in audit.METHODS:
            cmd=audit.command(self.tests[0],method,Path('/tmp/work'),Path('/tmp/out'),10000,Path('/fsl/randomise'))
            self.assertNotIn('-D',cmd); self.assertNotIn('-e',cmd); self.assertNotIn('-1',cmd)
            self.assertTrue(any(s.startswith('--seed=') for s in cmd)); self.assertIn('-N',cmd)
            self.assertEqual(cmd[cmd.index('-n')+1],'10000')
            self.assertEqual(cmd[-len(audit.METHODS[method]):],audit.METHODS[method])

    def test_nuisance_audit_does_not_modify_models(self):
        before=json.dumps(self.models,sort_keys=True)
        with tempfile.TemporaryDirectory() as tmp:
            audit.nuisance_audit(self.models,self.sample,Path(tmp))
            rows=audit.read_rows(Path(tmp)/'nuisance_source_comparison.tsv')
            self.assertEqual(len(rows),72)
            self.assertTrue(any(float(r['max_abs_vs_participant_table'])>.01 for r in rows))
        self.assertEqual(before,json.dumps(self.models,sort_keys=True))

    def test_residual_variance_audit_is_descriptive(self):
        rng=np.random.default_rng(9); older=np.arange(47)>=25
        x=np.column_stack([~older,older,rng.normal(size=47)])
        y=rng.normal(size=(47,1000)); y[older]*=3
        d,s=audit.residual_audit(x,y,older)
        self.assertGreater(d['older_over_younger_variance_median'],5)
        self.assertEqual(d['n_young'],25); self.assertEqual(d['n_old'],22)
        self.assertEqual(s.shape,(47,))

    def test_input_hash_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input'; p.write_text('original')
            record={str(p):audit.digest(p)}
            audit.verify_files(record); p.write_text('changed')
            with self.assertRaises(ValueError): audit.verify_files(record)

    def fixture_block(self,base):
        source=base/'source'; source.mkdir()
        mask=source/'mask.nii.gz'
        nib.save(nib.Nifti1Image(np.ones((3,3,3),dtype=np.uint8),np.eye(4)),mask)
        paths=[]
        for i in range(4):
            directory=source/f'sub-{i}'/'cope7.feat'/'stats'; directory.mkdir(parents=True)
            cope=directory/'cope1.nii.gz'
            data=np.full((3,3,3),float(i),dtype=np.float32); data[0,0,0]=0
            nib.save(nib.Nifti1Image(data,np.eye(4)),cope)
            nib.save(nib.Nifti1Image(np.ones((3,3,3),dtype=np.float32),np.eye(4)),directory/'varcope1.nii.gz')
            paths.append(str(cope))
        return dict(block='fixture',mask=str(mask),subjects=[f'sub-{i}' for i in range(4)],images=paths)

    def test_merge_order_mask_and_source_hash_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); m=self.fixture_block(base)
            block=audit.prepare_block(m,base/'work')
            image=nib.load(block/'copes.nii.gz').get_fdata()
            np.testing.assert_equal(image[1,1,1,:],[0,1,2,3])
            coverage=json.loads((block/'coverage.json').read_text())
            self.assertEqual(coverage['constant_voxels_removed'],1)
            self.assertEqual(coverage['tested_voxels'],26)
            self.assertEqual(block,audit.prepare_block(m,base/'work'))
            nib.save(nib.Nifti1Image(np.ones((3,3,3)),np.eye(4)),m['images'][0])
            with self.assertRaises(ValueError): audit.prepare_block(m,base/'work')

    def test_geometry_error_never_marks_block_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); m=self.fixture_block(base)
            affine=np.eye(4); affine[0,3]=3
            nib.save(nib.Nifti1Image(np.ones((3,3,3)),affine),m['images'][1])
            with self.assertRaisesRegex(ValueError,'geometry mismatch'):
                audit.prepare_block(m,base/'work')
            self.assertFalse((base/'work/blocks/fixture/complete.json').exists())

    @unittest.skipUnless(os.environ.get('SRNDNA_TEST_RANDOMISE'),'real FSL integration test')
    def test_end_to_end_preparation_execution_collection_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); m=self.fixture_block(base)
            sample={s:dict(older=int(i>=2),tsnr=i,fd_mean=-i) for i,s in enumerate(m['subjects'])}
            table=base/'people.csv'; table.write_text('fixture source table\n')
            m.update(name='fixture',scope='fixture',group='fixture-group',x=[[1]]*4,c=[[1]],labels=['mean'],
                     nuisance=dict(tsnr=[0,1,2,3],fd_mean=[0,-1,-2,-3]),robust=False,
                     source_hashes={m['mask']:audit.digest(m['mask'])})
            args=['run_permutation_audit.py','--work-root',str(base/'work'),
                  '--report-root',str(base/'report'),'--jobs','2','--permutations','100','--execute']
            binary=os.environ['SRNDNA_TEST_RANDOMISE']
            with patch.object(audit,'catalog',return_value=([m],sample,table)), \
                 patch.object(sys,'argv',args), patch.object(audit.shutil,'which',return_value=binary):
                self.assertEqual(audit.main(),0)
                self.assertEqual(audit.main(),0)
            status=json.loads((base/'report/status.json').read_text())
            self.assertTrue(status['complete'])
            self.assertEqual(status['unique_method_tests'],3)
            self.assertTrue((base/'report/input_provenance.json').exists())

    @unittest.skipUnless(os.environ.get('SRNDNA_TEST_RANDOMISE'),'real FSL integration test')
    def test_real_randomise_three_methods_resume_and_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'work'; report=Path(tmp)/'report'
            block=work/'blocks'/'b'; group=work/'groups'/'g'; testdir=work/'tests'/'t'
            for p in (block,group,testdir): p.mkdir(parents=True)
            rng=np.random.default_rng(12)
            data=rng.normal(size=(5,5,5,12)).astype(np.float32)
            nib.save(nib.Nifti1Image(data,np.eye(4)),block/'copes.nii.gz')
            nib.save(nib.Nifti1Image(np.ones((5,5,5),dtype=np.uint8),np.eye(4)),block/'mask.nii.gz')
            x=np.column_stack([np.ones(12),np.arange(12)-5.5])
            audit.vest(group/'design.mat',x)
            audit.vest(testdir/'design.con',[[0,1]],contrast=True)
            test=dict(test='t',group='g',block='b',contrast=[0,1],seed=42)
            binary=Path(os.environ['SRNDNA_TEST_RANDOMISE'])
            # A failed attempt is retained; a successful retry uses a new path.
            with patch.object(audit.subprocess,'run',return_value=type('Failure',(),{'returncode':17})()):
                with self.assertRaisesRegex(RuntimeError,'exit 17'):
                    audit.run_test(test,'tfce',work,100,binary)
            partial,=(work/'jobs/t/tfce').glob('attempt-*')
            for method in audit.METHODS:
                result=audit.run_test(test,method,work,100,binary)
                self.assertEqual(result,audit.run_test(test,method,work,100,binary))
                self.assertEqual(result['permutations_recorded'],100)
            self.assertTrue(partial.is_dir())
            mapping=[dict(model='fixture',test='t',contrast=1)]
            self.assertEqual(audit.collect([test],mapping,work,report),3)
            rows=audit.read_rows(report/'unique_test_summary.tsv')
            self.assertEqual(len(rows),3)
            self.assertTrue(all(0<=float(r['minimum_spatial_fwe_p'])<=1 for r in rows))
            receipt=json.loads((work/'jobs/t/tfce/complete.json').read_text())
            Path(receipt['attempt'],'randomise_tstat1.nii.gz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): audit.run_test(test,'tfce',work,100,binary)


if __name__=='__main__': unittest.main()
