import json
from pathlib import Path
import sys
import tempfile
import unittest
import csv
from unittest.mock import patch
import nibabel as nib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import analyze_corrected_vmpfc as vm
import plot_three_partner_ratings as rat


class VmpfcRatingsTests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(481)
        subjects=[f'sub-{i+100}' for i in range(47)]
        sample={s:dict(older=int(i>=25),ismale=rng.integers(0,2),
            tsnr=rng.normal(),fd_mean=rng.normal()) for i,s in enumerate(subjects)}
        rt={s:dict(corrected_mean_rt_z=rng.normal()) for s in subjects}
        return subjects,sample,rt

    def test_design_id_alignment_and_centering(self):
        subjects,sample,rt=self.sample()
        x,names,older=vm.make_design(subjects,sample,rt,True)
        other,_,_=vm.make_design(subjects[::-1],sample,rt,True)
        np.testing.assert_allclose(other[::-1],x,atol=1e-14)
        np.testing.assert_allclose(x[:,1:].mean(0),0,atol=1e-14)
        self.assertEqual(names,['intercept','older_centered','ismale','tsnr','fd_mean','corrected_mean_rt_z'])
        self.assertEqual(np.sum(older),22)

    def test_hc3_against_direct_sandwich(self):
        subjects,sample,rt=self.sample();x,_,_=vm.make_design(subjects,sample,rt,True)
        rng=np.random.default_rng(23); y=x@np.arange(6)+rng.normal(size=47)*np.linspace(.4,2,47)
        c=np.array([0,1,0,0,0,0.]);result,_,_=vm.estimate(x,y,c)
        inv=np.linalg.inv(x.T@x);b=inv@x.T@y;e=y-x@b;h=np.sum(x*(x@inv),axis=1)
        cov=inv@x.T@np.diag((e/(1-h))**2)@x@inv
        self.assertAlmostEqual(result['estimate'],b[1])
        self.assertAlmostEqual(result['hc3_se'],np.sqrt(cov[1,1]))
        self.assertEqual(result['df'],41)

    def test_pairing_joint_test_invariance(self):
        subjects,sample,rt=self.sample();x,_,_=vm.make_design(subjects,sample,rt,True)
        rng=np.random.default_rng(12);y=rng.normal(size=(47,3))+rng.normal(size=(47,1))*3
        test=vm.joint_age_test(x,y)
        np.testing.assert_allclose(vm.joint_age_test(x[::-1],y[::-1])['wald_chi2'],test['wald_chi2'])
        np.testing.assert_allclose(vm.joint_age_test(x,y[:,[2,0,1]])['wald_chi2'],test['wald_chi2'])

    def test_holm(self):
        np.testing.assert_allclose(vm.holm([.04,.001,.02]),[.04,.003,.04])

    def test_roi_zero_values_included_and_partial_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); shape=(3,3,3); m=np.zeros(shape);m[1,1,1:]=1
            mask=nib.Nifti1Image(m,np.eye(4));y=np.zeros(shape);y[1,1,2]=4
            nib.save(nib.Nifti1Image(y,np.eye(4)),root/'cope.nii.gz')
            nib.save(nib.Nifti1Image(np.ones(shape),np.eye(4)),root/'mask.nii.gz')
            result=vm.extract_mean(root/'cope.nii.gz',root/'mask.nii.gz',mask)
            self.assertEqual(result,(2.,2,False))
            support=np.ones(shape);support[1,1,1]=0
            nib.save(nib.Nifti1Image(support,np.eye(4)),root/'mask.nii.gz')
            with self.assertRaises(ValueError):vm.extract_mean(root/'cope.nii.gz',root/'mask.nii.gz',mask)

    def test_rating_policy_and_retained_sem(self):
        path=rat.ROOT/'results/reviewer/ultimatum_ratings/ultimatum_ratings_descriptives.tsv'
        rows=rat.read(path);cells=rat.selected_cells(rows)
        self.assertEqual(len(cells),18)
        self.assertAlmostEqual(float(cells['pre','likeability','similar']['n']),42)
        with self.assertRaises(ValueError): rat.selected_cells(rows+[next(iter(cells.values()))])
        damaged=[dict(row) for row in rows]; target=next(r for r in damaged if r['policy']=='last_complete_block' and r['age_group']=='all')
        target['sem']='99'
        with self.assertRaises(ValueError):rat.selected_cells(damaged)

    def test_mock_full_analysis_outputs_no_palM(self):
        subjects,sample,rt=self.sample();rng=np.random.default_rng(99)
        records=[dict(timing=t,subject=s,partner=p,mean_cope=float(rng.normal()))
            for t in vm.REPORTS for s in subjects for p,_,_ in vm.CONDITIONS]
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            with patch.object(vm,'save_figure'):
                vm.analyze(subjects,sample,rt,records,output)
            tests=rat.read(output/'roi_tests.tsv');bars=rat.read(output/'bar_source.tsv')
            self.assertEqual(len(tests),72);self.assertEqual(len(bars),36)
            self.assertEqual(len(rat.read(output/'age_partner_omnibus.tsv')),6)
            self.assertTrue(all(0<=float(r['p_holm_family_across_timings_and_adjustments'])<=1 for r in tests))
            # Adjustment preserves within-group means at the fitted reference,
            # with SEM calculated over participants, not across conditions.
            self.assertEqual({r['n'] for r in bars},{'25','22'})

    def test_complete_receipt_rejects_tampered_ratings(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            with patch.object(rat,'save_figure'):
                rat.build(rat.ROOT,output)
                rat.build(rat.ROOT,output)
            (output/'bar_source.tsv').write_text('damaged')
            with self.assertRaises(ValueError):rat.build(rat.ROOT,output)

    def test_l1_contrast_mapping_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fsf=root/'prepared.fsf'
            text=['set fmri(evs_orig) 6','set fmri(evs_real) 6']
            for _,cope,name in vm.CONDITIONS:
                text.extend([f'set fmri(conname_real.{cope}) "{name}"',
                             f'set fmri(conname_orig.{cope}) "{name}"'])
                for j in range(1,7):
                    for kind in ('orig','real'):
                        text.append(f'set fmri(con_{kind}{cope}.{j}) {int(j==cope)}')
            fsf.write_text('\n'.join(text)+'\n')
            con=root/'design.con'
            con.write_text('/NumWaves 6\n/NumContrasts 6\n/Matrix\n'+
                '\n'.join(' '.join(str(v) for v in row) for row in np.eye(6)))
            row=dict(fsf=str(fsf),fsf_sha256=vm.digest(fsf),output=str(root))
            vm.validate_l1(row,{})
            con.write_text(con.read_text().replace('0.0 1.0','0.0 0.0',1))
            with self.assertRaises(ValueError):vm.validate_l1(row,{})
            fsf.write_text(fsf.read_text()+'# changed\n')
            with self.assertRaises(ValueError):vm.validate_l1(row,{})

    def test_retained_feat_nuisance_contrast_columns(self):
        source=rat.ROOT/'results/decision_postresponse_partner/design_examples/act_sub-105_run-01'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fsf=root/'prepared.fsf';con=root/'design.con'
            fsf.write_text((source/'design.fsf').read_text())
            con.write_text((source/'design.con').read_text())
            row=dict(fsf=str(fsf),fsf_sha256=vm.digest(fsf),output=str(root))
            self.assertEqual(int(vm.fsf_value(fsf.read_text(),'evs_real')),10)
            matrix=vm.read_vest_matrix(con)
            self.assertEqual(matrix.shape[1],28)
            vm.validate_l1(row,{})
            # A nuisance weight must not silently enter an offer contrast.
            matrix[1,-1]=1
            con.write_text('/NumWaves 28\n/NumContrasts 10\n/Matrix\n'+
                '\n'.join(' '.join(str(v) for v in r) for r in matrix))
            with self.assertRaisesRegex(ValueError,'compiled L1 contrast mismatch'):
                vm.validate_l1(row,{})
            con.write_text((source/'design.con').read_text())
            # Missing weights inside evs_real must never be zero-filled.
            text='\n'.join(line for line in fsf.read_text().splitlines()
                if not line.startswith('set fmri(con_real2.10)'))+'\n'
            fsf.write_text(text);row['fsf_sha256']=vm.digest(fsf)
            with self.assertRaisesRegex(ValueError,'missing modeled contrast weight'):
                vm.validate_l1(row,{})

    def test_manifest_linked_extraction_fixture(self):
        subjects,sample,rt=self.sample()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            report=root/'reports/full-rt';report.mkdir(parents=True)
            masks=root/'masks';masks.mkdir()
            mask=nib.Nifti1Image(np.ones((2,2,2)),np.eye(4))
            nib.save(mask,masks/'roi-VMPFC.nii.gz')
            jobs=[];images=[]
            prepared=source/'l2.fsf';prepared.write_text('fixture fixed effects')
            for subject in subjects:
                l2=source/subject/'L2.gfeat';l2.mkdir(parents=True)
                jobs.append(dict(stage='l2',model='act',subject=subject,run='combined',
                    output=str(l2),fsf=str(prepared),fsf_sha256=vm.digest(prepared)))
                for run in ('01','02'):
                    jobs.append(dict(stage='l1',model='act',subject=subject,run=run,
                        output='validated separately',fsf='',fsf_sha256=''))
                images.append(str(l2/'cope7.feat/stats/cope1.nii.gz'))
                for _,cope,_ in vm.CONDITIONS:
                    folder=l2/f'cope{cope}.feat';(folder/'stats').mkdir(parents=True)
                    nib.save(nib.Nifti1Image(np.full((2,2,2),float(cope)),np.eye(4)),folder/'stats/cope1.nii.gz')
                    nib.save(mask,folder/'mask.nii.gz')
            rat.write(report/'revision_jobs.tsv',jobs)
            cfgroot=root/'results/permutation_audit/v1';cfgroot.mkdir(parents=True)
            rat.save_json(cfgroot/'config.json',{'models':[dict(name='full-rt__activation-norm',
                subjects=subjects,images=images,nuisance={
                    k:[float(rt[s]['corrected_mean_rt_z'] if k=='RT' else sample[s][k]) for s in subjects]
                    for k in ('ismale','tsnr','fd_mean','RT')})]})
            with patch.object(vm,'REPORTS',{'full-rt':'reports/full-rt'}), \
                 patch.object(vm,'ROI_SHA256',vm.digest(masks/'roi-VMPFC.nii.gz')), \
                 patch.object(vm,'load_sample',return_value=(sample,rt)), \
                 patch.object(vm,'validate_l1') as validate:
                _,_,_,records=vm.collect(root,{})
                self.assertEqual(validate.call_count,94)
                self.assertEqual(len(records),141)
                self.assertEqual({r['mean_cope'] for r in records},{2.,4.,6.})


if __name__=='__main__':unittest.main()
