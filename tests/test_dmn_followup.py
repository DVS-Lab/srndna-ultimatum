import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import finish_dmn_diagnostics as dd
import run_dmn_heteroscedasticity as hv


class FollowupTests(unittest.TestCase):
    def test_hc3_matches_sandwich_and_column_reordering(self):
        rng=np.random.default_rng(123)
        x=np.c_[np.ones(47),rng.normal(size=(47,5))]
        y=x@np.arange(6)+rng.normal(size=47)*np.linspace(.5,3,47)
        c=np.array([1,-1,0,0,0,0.])
        inv=np.linalg.inv(x.T@x);b=inv@x.T@y;e=y-x@b;h=np.sum((x@inv)*x,axis=1)
        cov=inv@x.T@np.diag(e**2/(1-h)**2)@x@inv
        self.assertAlmostEqual(dd.hc3_se(x,y,c),np.sqrt(c@cov@c))
        perm=[5,3,1,4,0,2]
        self.assertAlmostEqual(dd.hc3_se(x[:,perm],y,c[perm]),dd.hc3_se(x,y,c))

    def test_groups_follow_ids_not_positions(self):
        sample={f'sub-{i}':{'older':str(int(i>=25))} for i in range(47)}
        model={'subjects':list(reversed(sample))}
        groups=hv.variance_groups(model,sample)
        self.assertEqual(groups[:22].tolist(),[2]*22)
        self.assertEqual(groups[22:].tolist(),[1]*25)

    def test_signflip_contract(self):
        for method in hv.METHODS:
            args=hv.arguments(Path('/inputs'),method,Path('/outputs/palm'),10000,13)
            self.assertIn('-vg',args);self.assertIn('-ise',args)
            for bad in ('-ee','-eb','-demean','-twotail','-accel'):self.assertNotIn(bad,args)
            self.assertEqual(args[args.index('-n')+1],'10000')
            self.assertIn('-save1-p',args)

    def test_engine_escaping(self):
        command=hv.engine_command('/usr/bin/octave',Path("/a'b"),['-i',"a'b.nii"])
        self.assertIn("addpath('/a''b')",command[-1]);self.assertIn("'a''b.nii'",command[-1])
        self.assertIn('pkg load image',command[-1])

    def test_probability_semantics_and_missing_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mask=nib.Nifti1Image(np.ones((2,2,2)),np.eye(4))
            data=np.array([.951,.95,.9,1,0,0,0,0]).reshape((2,2,2))
            nib.save(nib.Nifti1Image(data,np.eye(4)),root/'palm_tfce_zvstat_fwep_c1.nii.gz')
            result=hv.probability_maps(root,'tfce',1,mask)
            self.assertEqual(result[0][1]['significant_voxels'],2)
            self.assertEqual(result[0][1]['minimum_spatial_fwe_p'],0)
            with self.assertRaises(ValueError):hv.probability_maps(root,'tfce',2,mask)

    def test_resume_requires_verified_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=root/'input';inp.mkdir();report=root/'report'
            mask=nib.Nifti1Image(np.ones((2,2,2)),np.eye(4));nib.save(mask,inp/'mask.nii.gz')
            model={'name':'test','c':[[1],[ -1]],'labels':['positive','negative']}
            folder=root/'jobs/test/tfce';attempt=folder/'attempt-old';attempt.mkdir(parents=True)
            for c in (1,2):nib.save(mask,attempt/f'palm_tfce_zvstat_fwep_c{c}.nii.gz')
            hv.pa.write_json(folder/'complete.json',dict(attempt=str(attempt),outputs={str(p):hv.pa.digest(p) for p in attempt.iterdir()}))
            with patch.object(hv.subprocess,'run',side_effect=AssertionError('must not rerun')):
                rows=hv.job(model,'tfce',inp,root,report,'octave',root,10000)
            self.assertEqual(len(rows),2)
            (attempt/'palm_tfce_zvstat_fwep_c1.nii.gz').write_bytes(b'changed')
            with self.assertRaises(ValueError):hv.job(model,'tfce',inp,root,report,'octave',root,10000)

    def test_all_retained_models_have_288_summary_cells(self):
        root=Path(__file__).resolve().parents[1]
        config=json.loads((root/'results/dmn_covariate_standardized/v1/randomise/config.json').read_text())
        self.assertEqual(len(config['models']),15)
        self.assertEqual(sum(len(m['c']) for m in config['models'])*3,288)
        for m in config['models']:
            x=np.array(m['x']); self.assertEqual(np.linalg.matrix_rank(x),x.shape[1])


if __name__=='__main__':unittest.main()
