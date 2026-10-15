"""Source-locked DMN-only rerun tests; no research model fitting."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import run_dmn_covariate_standardization as dmn
import run_permutation_audit as pa
from prepare_sub144_imaging_repair import fsf_value
from prepare_ultimatum_l3_repair import parse_inputs


class StandardizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path('/tmp/dmn-standardization-fixture')
        cls.plans,cls.sources,cls.changes=dmn.build_plan(ROOT,cls.work,Path('/fixture/standard.nii.gz'))
        cls.sample={r['subjID']:r for r in pa.read_rows(ROOT/'behavioral_analyses/data/participant_L3_47.csv',',')}

    def test_all_fifteen_models_and_ninety_six_contrasts(self):
        self.assertEqual(len(self.plans),15)
        self.assertEqual(sum(j['expected_zstats'] for j,_ in self.plans),96)
        self.assertEqual(sum(fsf_value(t,'robust_yn')=='1' for _,t in self.plans),3)
        self.assertEqual(len(self.changes),15*47*2)

    def test_only_quality_columns_and_output_paths_change(self):
        for job,text in self.plans:
            source=Path(job['source_fsf']).read_text(); before=dmn.matrix(source); after=dmn.matrix(text)
            titles=[fsf_value(text,f'evtitle{i}') for i in range(1,after.shape[1]+1)]
            allowed=[titles.index(name) for name in ('tsnr','fd_mean')]
            unchanged=[i for i in range(after.shape[1]) if i not in allowed]
            np.testing.assert_array_equal(before[:,unchanged],after[:,unchanged])
            self.assertEqual(parse_inputs(source.splitlines()),parse_inputs(text.splitlines()))
            self.assertEqual(np.linalg.matrix_rank(after),after.shape[1])
            for row,subject,_ in parse_inputs(text.splitlines()):
                for metric in ('tsnr','fd_mean'):
                    self.assertAlmostEqual(after[row-1,titles.index(metric)],float(self.sample[subject][metric]),12)
            self.assertTrue(any(abs(before[:,i]-after[:,i]).max()>.01 for i in allowed))

    def test_id_alignment_and_missing_nonfinite_rejection(self):
        job,_=self.plans[0]; source=Path(job['source_fsf']).read_text()
        forward=dmn.render(source,self.sample,Path('/out'),Path('/std'))[0]
        reverse=dmn.render(source,dict(reversed(list(self.sample.items()))),Path('/out'),Path('/std'))[0]
        self.assertEqual(forward,reverse)
        bad=copy.deepcopy(self.sample); bad.pop(next(iter(bad)))
        with self.assertRaises(ValueError): dmn.render(source,bad,Path('/out'),Path('/std'))
        bad=copy.deepcopy(self.sample); bad[next(iter(bad))]['tsnr']='nan'
        with self.assertRaises(ValueError): dmn.render(source,bad,Path('/out'),Path('/std'))

    def test_reject_unrelated_fsf_mutation(self):
        job,_=self.plans[0]; source=Path(job['source_fsf']).read_text()
        text,_,allowed=dmn.render(source,self.sample,Path('/out'),Path('/std'))
        dmn.unchanged_settings(source,text,allowed)
        with self.assertRaises(ValueError):
            dmn.unchanged_settings(source,text+'\nset fmri(z_thresh) 2.6\n',allowed)

    def test_standardized_catalog_and_seed_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp); inventory=[]; jobs=[]
            for job,text in self.plans:
                jobs.append(job)
                root=report/job['run']; design=root/'design'; design.mkdir(parents=True)
                (design/'design.fsf').write_text(text)
                pa.vest(design/'design.mat',dmn.matrix(text))
                n=job['n_evs']; k=job['expected_zstats']
                c=np.array([[float(fsf_value(text,f'con_real{i}.{j}')) for j in range(1,n+1)] for i in range(1,k+1)])
                pa.vest(design/'design.con',c,True)
                shutil.copy2(Path(job['source_fsf']).with_suffix('.grp'),design/'design.grp')
                post=root/'poststats'; post.mkdir()
                shutil.copy2(job['historical_mask'],post/'mask.nii.gz')
                for f in root.rglob('*'):
                    if f.is_file(): inventory.append(dict(artifact=str(f.relative_to(report)),sha256=pa.digest(f)))
            pa.write_rows(report/'revision_jobs.tsv',jobs)
            pa.write_rows(report/'inventory.tsv',inventory)
            pa.write_json(report/'analysis_plan.json',dict(covariate_policy=dmn.POLICY))
            models,_,_=pa.standardized_dmn_catalog(ROOT,report)
            tests,mapping=pa.plan_tests(models,20261014)
            pa.retain_reference_seeds(tests,mapping,ROOT/'results/permutation_audit/v1')
            self.assertEqual(len(mapping),96)
            self.assertEqual(len(tests),66)
            old={(r['model'],int(r['contrast'])):int(r['seed']) for r in pa.read_rows(ROOT/'results/permutation_audit/v1/model_contrast_summary.tsv')}
            new={t['test']:t['seed'] for t in tests}
            self.assertTrue(all(old[r['model'],r['contrast']]==new[r['test']] for r in mapping))
            self.assertTrue(all(np.allclose(m['nuisance']['tsnr'],[float(self.sample[s]['tsnr']) for s in m['subjects']]) for m in models))
            # Source tampering must be caught before permutation preparation.
            (report/jobs[0]['run']/'design/design.fsf').write_text('changed')
            with self.assertRaises((ValueError,TypeError,KeyError)): pa.standardized_dmn_catalog(ROOT,report)

    def test_prepare_resume_and_changed_fsf_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); fsl=base/'fsl'; (fsl/'etc').mkdir(parents=True)
            (fsl/'etc/fslversion').write_text('test-version')
            binary=base/'binary'; binary.write_text('test-binary')
            work=base/'work'; standard=base/'standard.nii.gz'
            with patch.dict(os.environ,FSLDIR=str(fsl)), patch.object(dmn.shutil,'which',return_value=str(binary)), patch.object(dmn,'check_images',return_value={}):
                first=dmn.prepare(ROOT,work,standard)
                before={str(p):pa.digest(p) for p in work.rglob('*') if p.is_file()}
                self.assertEqual(dmn.prepare(ROOT,work,standard),first)
                self.assertEqual(before,{str(p):pa.digest(p) for p in work.rglob('*') if p.is_file()})
                path=Path(pa.read_rows(first)[0]['fsf']); path.write_text(path.read_text()+'\n# changed\n')
                with self.assertRaisesRegex(ValueError,'prepared design changed'): dmn.prepare(ROOT,work,standard)

    def test_comparison_covers_all_results_including_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp); summary=[]; names={j['run'] for j,_ in self.plans}
            for job,_ in self.plans:
                scope,name=job['run'].split('__',1)
                old=pa.read_rows(ROOT/pa.REPORTS[scope]/'summary.tsv')
                summary.extend(dict(r,model=job['run']) for r in old if r['model']==name)
            pa.write_rows(report/'imaging/summary.tsv',summary)
            permutation=[r for r in pa.read_rows(ROOT/'results/permutation_audit/v1/model_contrast_summary.tsv') if r['model'] in names]
            pa.write_rows(report/'randomise/model_contrast_summary.tsv',permutation)
            dmn.compare(ROOT,report)
            flame=pa.read_rows(report/'flame_historical_comparison.tsv')
            randomise=pa.read_rows(report/'randomise_historical_comparison.tsv')
            self.assertEqual(len(flame),96); self.assertEqual(len(randomise),288)
            self.assertTrue(any(r['historical_clusters']=='0' for r in flame))
            self.assertTrue(all(r['historical_clusters']==r['standardized_clusters'] for r in flame))
            self.assertTrue(all(r['historical_p']==r['standardized_p'] for r in randomise))

    @unittest.skipUnless(os.environ.get('SRNDNA_TEST_FEAT_MODEL'),'optional installed FSL compiler check')
    def test_installed_fsl_compiles_all_fifteen_designs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for job,text in self.plans:
                root=Path(tmp)/job['run']; root.with_suffix('.fsf').write_text(text)
                result=subprocess.run([os.environ['SRNDNA_TEST_FEAT_MODEL'],str(root)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                x=pa.read_vest_matrix(root.with_suffix('.mat'))
                c=pa.read_vest_matrix(root.with_suffix('.con'))
                np.testing.assert_allclose(x,dmn.matrix(text),atol=1e-5,rtol=0)
                expected=[[float(fsf_value(text,f'con_real{i}.{j}')) for j in range(1,x.shape[1]+1)] for i in range(1,job['expected_zstats']+1)]
                np.testing.assert_array_equal(c,expected)
                self.assertEqual(np.linalg.matrix_rank(x),x.shape[1])


class BatchTests(unittest.TestCase):
    def fixture(self,fail):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); binary=base/'bin'; binary.mkdir()
            for name in ('feat_model','randomise'):
                tool=binary/name; tool.write_text('#!/bin/sh\nexit 0\n'); tool.chmod(0o755)
            python=binary/'python3'
            python.write_text('''#!/bin/bash
[[ "$1" == --version ]] && exit 0
[[ "$1" == -m ]] && exit 0
phase=prepare
for arg in "$@"; do [[ "$arg" == --execute ]] && phase=execute; done
echo "$phase" >> "$BATCH_CALLS"
[[ "$phase" == "$BATCH_FAIL" ]] && exit 17
exit 0
'''); python.chmod(0o755)
            report=base/'report'; calls=base/'calls'
            env=dict(os.environ,PATH=str(binary)+':'+os.environ['PATH'],BATCH_CALLS=str(calls),BATCH_FAIL=fail)
            result=subprocess.run(['bash',str(ROOT/'code/run_dmn_covariate_standardization_batch.sh'),
                                   str(base/'work'),str(report)],env=env,capture_output=True,text=True)
            attempt,=report.glob('attempt-*')
            self.assertEqual(int((attempt/'exit_status.txt').read_text()),result.returncode)
            self.assertTrue((attempt/'run.txt').is_file())
            return result.returncode,calls.read_text().splitlines()

    def test_wrapper_success_and_failure_gating(self):
        self.assertEqual(self.fixture('none'),(0,['prepare','execute']))
        self.assertEqual(self.fixture('prepare'),(17,['prepare']))
        self.assertEqual(self.fixture('execute'),(17,['prepare','execute']))


if __name__=='__main__': unittest.main()
