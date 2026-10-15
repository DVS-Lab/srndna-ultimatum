import csv, io, json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import audit_dmn_covariate_sources as audit
import plot_permutation_followup as plot
from run_permutation_audit import vest

class CovariateTests(unittest.TestCase):
    def test_subjects(self):
        self.assertEqual(audit.subject('sub-143_task-ultimatum_run-01_bold'),'sub-143')
        self.assertEqual(audit.subject('143.0'),'sub-143')
        with self.assertRaises(ValueError): audit.subject('unknown')

    def test_affine_requires_full_positive_match(self):
        a={f'sub-{i}':float(i) for i in range(104,110)}
        b={s:2*v+7 for s,v in a.items()}
        self.assertEqual(audit.compare(a,b)['full_positive_affine_match'],1)
        self.assertEqual(audit.compare(a,{s:-v for s,v in b.items()})['full_positive_affine_match'],0)
        del b['sub-104']
        self.assertEqual(audit.compare(a,b)['full_positive_affine_match'],0)

    def test_alignment_is_by_id(self):
        text='participant_id,tsnr,fd_mean\nsub-105,3,4\nsub-104,1,2\n'
        vectors=audit.table_vectors(text,',')
        self.assertEqual(vectors[0][2],{'sub-105':3.,'sub-104':1.})

    def test_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            audit.table_vectors('Sub,tsnr\nsub-104,1\nsub-104,2\n',',')
        with self.assertRaises(ValueError):
            audit.table_vectors('Sub,task,run,tsnr\nsub-104,ultimatum,1,1\nsub-104,ultimatum,01,2\n',',')

    def test_mriqc_task_filter_and_run_means(self):
        text='bids_name\ttsnr\tfd_mean\nsub-104_task-trust_run-1_bold\t99\t99\nsub-104_task-ultimatum_run-01_bold\t2\t4\nsub-104_task-ultimatum_run-02_bold\t4\t6\n'
        values={(p,m):v for p,m,v in audit.table_vectors(text,'\t')}
        self.assertEqual(values['ultimatum_mean_all_runs','tsnr'],{'sub-104':3.})
        self.assertEqual(values['ultimatum_run2','fd_mean'],{'sub-104':6.})

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): audit.table_vectors('Sub,tsnr\nsub-104,nan\n',',')

class PlotTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(14)
        self.age=np.r_[np.zeros(25),np.ones(22)]
        b=self.rng.normal(size=47);b-=b.mean()
        self.x=np.column_stack([self.rng.normal(size=(47,4)),1-self.age,self.age,b*(1-self.age),b*self.age])
        self.c=np.array([0,0,0,0,0,0,-1,1])
        self.y=self.x@np.arange(8)+self.rng.normal(size=47)

    def test_adjustment_and_leave_one_out(self):
        d=plot.diagnostic(self.x,self.y,self.c)
        beta=np.linalg.lstsq(self.x,self.y,rcond=None)[0]
        np.testing.assert_allclose(d['beta'],beta)
        np.testing.assert_allclose(d['adjusted'],self.y-self.x[:,:4]@beta[:4])
        self.assertAlmostEqual(d['loo'][4],self.c@np.linalg.lstsq(np.delete(self.x,4,0),np.delete(self.y,4),rcond=None)[0])
        self.assertEqual(d['loo'].shape,(47,))

    def test_rank_and_grid_guards(self):
        x=self.x.copy();x[:,1]=x[:,0]
        with self.assertRaises(ValueError):plot.diagnostic(x,self.y,self.c)
        a=nib.Nifti1Image(np.zeros((3,3,3)),np.eye(4)); b=nib.Nifti1Image(np.zeros((4,3,3)),np.eye(4))
        with self.assertRaises(ValueError):plot.same_grid(a,b)

    def test_extraction_hash_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cope.nii.gz'; im=nib.Nifti1Image(np.ones((3,3,3)),np.eye(4));nib.save(im,p)
            model={'subjects':['sub-104'],'images':[str(p)]}
            with self.assertRaises(ValueError):plot.extract(model,np.ones(im.shape,bool),im,{}, {str(p):'wrong'})
            np.testing.assert_allclose(plot.extract(model,np.ones(im.shape,bool),im,{}, {str(p):plot.digest(p)}),[1])

    def test_end_to_end_synthetic_extract_export_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);report=root/'permutations';report.mkdir();out=root/'output'
            aff=np.diag([4.,4.,4.,1.]);shape=(7,8,9)
            standard=root/'standard.nii.gz';nib.save(nib.Nifti1Image(np.ones(shape),aff),standard)
            participants=[f'sub-{i}' for i in range(104,151)]
            inventory=[];mapping=[];models=[];input_hashes={};flame_inventory=[]
            target=plot.TARGETS['activation-age-interaction']
            for timing in plot.TIMINGS:
                name=f'crossed__{timing}_act_acceptance-intercept';group=timing
                directory=root/'results/crossed_behavior_l3/v1/imaging'/name.removeprefix('crossed__')/'design'
                directory.mkdir(parents=True)
                # Reverse FSF order to test ID alignment against sorted image data.
                order=list(reversed(range(47))); images=[]
                for k,s in enumerate(participants):
                    p=root/timing/f'{s}.nii.gz';p.parent.mkdir(exist_ok=True)
                    nib.save(nib.Nifti1Image(np.full(shape,self.y[k]),aff),p);images.append(str(p));input_hashes[str(p)]=plot.digest(p)
                fsf='\n'.join(f'set feat_files({j+1}) "{images[k]}"' for j,k in enumerate(order))
                (directory/'design.fsf').write_text(fsf)
                vest(directory/'design.mat',self.x[order]);vest(directory/'design.con',self.c,contrast=True)
                models.append(dict(name=name,subjects=participants,images=images,source_hashes={str(p):plot.digest(p) for p in directory.iterdir()}))
                maskpath=report/'designs'/group/'mask.nii.gz';maskpath.parent.mkdir(parents=True)
                nib.save(nib.Nifti1Image(np.ones(shape),aff),maskpath)
                mask=np.zeros(shape);mask[2:5,3:6,4:7]=1
                old=directory.parent/'contrast1/cluster_mask_zstat1.nii.gz';old.parent.mkdir()
                nib.save(nib.Nifti1Image(mask,aff),old)
                flame_inventory.append(dict(artifact=str(old.relative_to(root/'results/crossed_behavior_l3/v1/imaging')),sha256=plot.digest(old)))
                for method in plot.METHODS:
                    folder=report/'maps'/timing/method;folder.mkdir(parents=True)
                    paths=[folder/'randomise_tstat1.nii.gz',folder/('randomise_'+('tfce' if method=='tfce' else 'clustere')+'_corrp_tstat1.nii.gz')]
                    for p,data in zip(paths,[mask*4,mask*.99]):
                        nib.save(nib.Nifti1Image(data,aff),p);inventory.append(dict(artifact=str(p.relative_to(report)),sha256=plot.digest(p)))
                    mapping.append(dict(model=name,contrast=1,method=method,test=timing,group=group,significant_voxels=27,minimum_spatial_fwe_p=.01))
            plot.table(report/'map_inventory.tsv',inventory);plot.table(report/'model_contrast_summary.tsv',mapping)
            plot.table(root/'results/crossed_behavior_l3/v1/imaging/inventory.tsv',flame_inventory)
            (report/'config.json').write_text(json.dumps(dict(models=models)))
            (report/'status.json').write_text(json.dumps(dict(complete=True)))
            (report/'input_provenance.json').write_text(json.dumps(input_hashes))
            with patch.object(plot,'ROOT',root),patch.object(plot,'TARGETS',{'activation-age-interaction':target}):
                plot.build(report,out,standard)
                self.assertTrue(json.loads((out/'status.json').read_text())['complete'])
                values=plot.rows(out/'activation-age-interaction/participant_extractions.tsv')
                self.assertEqual(len(values),141)
                np.testing.assert_allclose([float(r['raw_cope7']) for r in values[:47]],self.y)
                self.assertEqual(len(list((out/'activation-age-interaction').glob('*.png'))),5)
                # Test resume contract without rendering another identical set.
                with patch.object(plot,'overlays'),patch.object(plot,'plots_and_tables'):
                    plot.build(report,out,standard)
                (report/'model_contrast_summary.tsv').write_text('changed')
                with self.assertRaises(ValueError):plot.build(report,out,standard)

if __name__=='__main__':unittest.main()
