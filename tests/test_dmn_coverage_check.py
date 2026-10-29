import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import run_dmn_coverage_check as check
import test_palm_partner_motion as fixtures


class DmnCoverageCheckTests(unittest.TestCase):
    def setUp(self):
        # Fixture counts and mocked fits are not real study results. Keep them
        # out of Linux batch logs while retaining unittest's stderr diagnostics.
        capture=redirect_stdout(io.StringIO());capture.__enter__()
        self.addCleanup(capture.__exit__,None,None,None)

    def test_pseudonymous_order_table_is_tracked_but_private_tables_stay_ignored(self):
        for path, expected in [('results/dmn_followup/v1/coverage_check/v1/preparation/participants.tsv',1),
                               ('source_data/participants.tsv',0)]:
            result=check.subprocess.run(['git','check-ignore',path],cwd=check.ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,expected)

    def test_film_threshold_parser_modern_legacy_html_and_no_defaults(self):
        self.assertEqual(check.film_thresholds('/fsl/film_gls --thr=1000.0000 --in=data\n'), [1000])
        self.assertEqual(check.film_thresholds('film_gls --thr 1000.0 --pd=design.mat'), [1000])
        self.assertEqual(check.film_thresholds('<p>film_gls -rn stats -sa -ms 5 filtered_func_data design.mat 1000.0</p>'), [1000])
        self.assertEqual(check.film_thresholds('film_gls --in=data --ms=5'), [])
        self.assertEqual(check.film_thresholds('fslmaths image -thr 1000'), [])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'logs').mkdir()
            (p/'report_log.html').write_text('film_gls --thr=1000 --con=design.con\n')
            (p/'logs/stats.o1').write_text('film_gls --thr=1000\n')
            value,evidence=check.film_evidence(p)
            self.assertEqual(value,1000);self.assertEqual(len(evidence),2)
            (p/'logs/stats.o2').write_text('film_gls --thr=0\n')
            with self.assertRaisesRegex(ValueError,'unique explicit'):check.film_evidence(p)

    def test_modern_pd_before_threshold_does_not_parse_next_option_as_number(self):
        command='/usr/local/fsl/bin/film_gls --in=filtered_func_data --rn=stats --pd=design.mat --thr=1000.000000 --sa --ms=5 --con=design.con'
        self.assertEqual(check.film_thresholds(command),[1000])
        self.assertEqual(check.film_thresholds('film_gls --pd=design.mat --con=design.con --thr 1e+3'),[1000])
        self.assertEqual(check.film_thresholds('film_gls --pd=design.mat --sa'),[])
        self.assertEqual(check.film_thresholds('film_gls --thr=-- --pd=design.mat'),[])
        self.assertEqual(check.film_thresholds('film_gls --thr=1000garbage'),[])
        self.assertEqual(check.film_thresholds('film_gls input design.mat -1.25e-3'),[-.00125])

    def test_ambiguous_real_command_evidence_survives_preflight_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'logs').mkdir()
            (p/'report_log.html').write_text('film_gls --pd=design.mat --thr=1000\n')
            (p/'logs/stats.o1').write_text('film_gls --pd=design.mat --thr=500\n')
            snapshot=p/'saved-command-evidence.json'
            with self.assertRaisesRegex(ValueError,'unique explicit'):
                check.film_evidence(p,snapshot)
            saved=json.loads(snapshot.read_text());self.assertEqual(len(saved),2)
            self.assertTrue(any('--thr=1000' in line for row in saved for line in row['commands']))

    def test_validity_keeps_true_zero_cope_and_rejects_missing_variance_or_mask(self):
        valid=check.valid_estimate(np.array([0,1,np.nan,1,1,1]),
            np.array([1,0,1,-1,np.inf,1]),np.array([1,1,1,1,1,0]))
        np.testing.assert_array_equal(valid,[1,0,0,0,0,0])

    def fixture(self, root):
        model, baseline=fixtures.PartnerMotionTests().fixture(root)
        def save(path, data):
            path.parent.mkdir(parents=True,exist_ok=True)
            nib.save(nib.Nifti1Image(np.asarray(data,float),np.eye(4)),path)
        shape=(4,4,4);mask=root/'wholebrain.nii.gz';save(mask,np.ones(shape))
        model['mask']=str(mask)
        config=root/'results/dmn_covariate_standardized/v1/randomise/config.json'
        config.write_text(json.dumps(dict(models=[model])))
        flagged=model['subjects'][:3]
        for index, image_name in enumerate(model['images']):
            image=Path(image_name);save(image.with_name('varcope1.nii.gz'),np.ones(shape))
            save(image.parent.parent/'mask.nii.gz',np.ones(shape))
            for run in ('01','02'):
                folder=image.parents[3]/f'L1_run-{run}.feat';save(folder/'mask.nii.gz',np.ones(shape))
                if index<3:
                    mean=np.full(shape,2000.)
                    if run=='01':
                        var=nib.load(folder/'stats/varcope7.nii.gz').get_fdata();var.flat[index]=0
                        save(folder/'stats/varcope7.nii.gz',var);mean.flat[index]=800
                    save(folder/'mean_func.nii.gz',mean)
                    (folder/'report_log.html').write_text('film_gls --in=filtered_func_data --thr=1000 --pd=design.mat\n')
        seeds=[]
        for i in range(1,5):
            for method in check.pa.METHODS:
                seeds.append(dict(model=model['name'],contrast=i,method=method,seed=100+i,permutations_recorded=10000))
        check.pa.write_rows(config.with_name('model_contrast_summary.tsv'),seeds)
        hashes=config.with_name('input_provenance.json')
        audit=root/'audit';audit.mkdir()
        save(audit/'selected_roi.nii.gz',nib.load(baseline/'selected_age_roi.nii.gz').get_fdata())
        check.pa.write_json(audit/'status.json',dict(complete=True,detailed_participants=flagged))
        check.pa.write_json(audit/'provenance.json',dict(sources={str(p):check.pa.digest(p) for p in (config,hashes)}))
        inventory=[dict(path=str(Path(p)),sha256=check.pa.digest(Path(p)),exists=1) for p in model['images']]
        check.pa.write_rows(audit/'input_inventory.tsv',inventory)
        check.pa.write_rows(audit/'inventory.tsv',[dict(path=p.name,sha256=check.pa.digest(p)) for p in audit.iterdir()])
        work=root/'new-work';work.mkdir()
        return model,audit,work

    def test_preparation_full_sample_whole_brain_rule_and_resume_preserve_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,audit,work=self.fixture(root)
            before={p:check.pa.digest(p) for p in root.rglob('*') if p.is_file()}
            result=check.prepare(root,work,audit)
            self.assertEqual(result['coverage']['tested_voxels'],61)
            self.assertEqual(result['coverage']['previous_roi_valid_voxels'],48)
            self.assertEqual(result['coverage']['participants'],47)
            self.assertEqual(result['randomise_seeds'],{'1':101,'2':102,'3':103,'4':104})
            self.assertEqual(len(check.pa.read_rows(Path(result['root'])/'film_thresholds.tsv')),6)
            data=nib.load(Path(result['root'])/'inputs/copes.nii.gz').get_fdata()
            self.assertEqual(data.shape,(4,4,4,47))
            for i,path in enumerate(model['images']):np.testing.assert_allclose(data[...,i],nib.load(path).get_fdata())
            with patch.object(check,'film_evidence',side_effect=AssertionError('must reuse completed preparation')):
                self.assertEqual(check.prepare(root,work,audit),result)
            for p,digest in before.items():self.assertEqual(check.pa.digest(p),digest)
            changed=Path(model['images'][0]);changed.write_bytes(b'tampered')
            with self.assertRaises(ValueError):check.prepare(root,work,audit)

    def test_failed_film_check_does_not_prepare_models_or_launch_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,audit,work=self.fixture(root)
            folder=Path(model['images'][0]).parents[3]/'L1_run-01.feat'
            (folder/'report_log.html').write_text('No retained film command\n')
            with patch.object(check.headless,'launch',side_effect=AssertionError('must not launch')):
                with self.assertRaisesRegex(ValueError,'unique explicit'):check.prepare(root,work,audit)
            self.assertFalse((work/'prepared.json').exists())

    def test_completed_job_receipt_is_required_for_resume_and_tampering_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); artifact=p/'map';artifact.write_bytes(b'result')
            receipt=p/'complete.json';check.pa.write_json(receipt,dict(contract='locked',outputs={str(artifact):check.pa.digest(artifact)}))
            self.assertEqual(check.verify_receipt(receipt,'locked')['contract'],'locked')
            with self.assertRaises(ValueError):check.verify_receipt(receipt,'changed')
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):check.verify_receipt(receipt,'locked')

    def test_all_job_outputs_summary_and_resume_with_mocked_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model,audit,work=self.fixture(root)
            prepared=check.prepare(root,work,audit);report=root/'report';report.mkdir()
            def save(path,value):nib.save(nib.Nifti1Image(np.full((4,4,4),value),np.eye(4)),path)
            def fake_palm(engine,palm_root,args,attempt,mode):
                unit='tfce' if '-T' in args else 'clustere'
                for i in range(1,5):save(attempt/f'palm_{unit}_tstat_fwep_c{i}.nii.gz',.96)
                (attempt/'run.txt').write_text('MOCK test only\n')
            def fake_randomise(cmd,**kwargs):
                prefix=Path(cmd[cmd.index('-o')+1]);unit='tfce' if '-T' in cmd else 'clustere'
                save(prefix.parent/'randomise_tstat1.nii.gz',3.)
                save(prefix.parent/f'randomise_{unit}_corrp_tstat1.nii.gz',.97)
                np.savetxt(prefix.parent/f'randomise_{unit}_corrp_tstat1.txt',np.ones(check.N))
                self.assertEqual(cmd[cmd.index('-n')+1],'10000')
                self.assertIn('-R',cmd);self.assertIn('-N',cmd)
                return SimpleNamespace(returncode=0)
            rt=dict(engine='matlab',palm_root=str(root/'palm'),mode='clean')
            with patch.object(check.headless,'launch',side_effect=fake_palm),patch.object(check.subprocess,'run',side_effect=fake_randomise):
                for method in check.palm.METHODS:check.run_job('palm',method,0,prepared,work,report,'locked',rt,'randomise')
                for i in range(1,5):
                    for method in check.pa.METHODS:check.run_job('randomise',method,i,prepared,work,report,'locked',rt,'randomise')
            self.assertEqual(check.summarize(prepared,work,report,'locked'),15)
            rows=check.pa.read_rows(report/'summary.tsv');self.assertEqual(len(rows),24)
            self.assertEqual({r['contrast'] for r in rows},{'1','2','3','4'})
            with patch.object(check.headless,'launch',side_effect=AssertionError('do not repeat completed jobs')):
                check.run_job('palm','tfce',0,prepared,work,report,'locked',rt,'randomise')


if __name__=='__main__':unittest.main()
