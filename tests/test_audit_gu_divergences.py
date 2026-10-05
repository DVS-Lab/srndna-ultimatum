import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from audit_gu_divergences import compare_draws, discover, export, digest, sha, SAMPLER


class DivergenceAuditTests(unittest.TestCase):
    def fixture(self, base, bias=True):
        work=base/'work'
        job=work/'fits/fit-test'
        attempt=job/'attempt-test'
        attempt.mkdir(parents=True)
        data=dict(subject=[1,1,1],partner=[3,1,2])
        parameters=['alpha','gamma','f0','epsilon'] + (['acceptance_bias'] if bias else [])
        config=dict(phase='fit',model='rw_positive_bias' if bias else 'rw_positive',stage='full',
                    subjects=['sub-107'],parameters=parameters,
                    data_hash=digest(data),samples=6,chains=2,max_treedepth=12)
        (job/'configuration.json').write_text(json.dumps(config))
        (attempt/'data.json').write_text(json.dumps(data))
        variables=[f'theta[{u},{k}]' for u in range(1,4) for k in range(1,len(parameters)+1)]
        pd.DataFrame(dict(R_hat=[1.001]*len(variables),ESS_bulk=[800]*len(variables),ESS_tail=[900]*len(variables)),
                     index=variables).to_csv(attempt/'stan_summary.tsv',sep='\t')
        diag=dict(divergences=2,max_depth_hits=2,passed=False)
        (attempt/'diagnostics.json').write_text(json.dumps(diag))
        for c in (1,2):
            # Distinct chain means expose pooled-versus-within-chain differences.
            frame=pd.DataFrame({v: np.arange(6,dtype=float)+c for v in variables})
            frame.columns=[v.replace('[','.').replace(',','.').replace(']','') for v in variables]
            for u in range(1,4):
                frame[f'theta.{u}.4']=np.linspace(.01,.6,6)
                if bias:
                    frame[f'theta.{u}.5']=np.linspace(-2,3,6)
            for key,values in dict(divergent__=[0,0,0,0,0,1],treedepth__=[5,5,5,5,12,5],
                energy__=[1,2,1,2,1,2],accept_stat__=[.99]*6,stepsize__=[.01]*6,n_leapfrog__=[31]*6).items():
                frame[key]=values
            path=attempt/f'chain{c}.csv'
            path.write_text(f'# id = {c}\n# save_warmup = 0\n'+frame.to_csv(index=False))
        (attempt/'chain1_0-stdout.txt').write_text('Exception: Logit transformed probability parameter is nan\n')
        record=dict(job=job.name,attempt=attempt.name,config_hash=digest(config),
                    diagnostics=diag,inference_eligible=False,
                    files={str(p.relative_to(job)):sha(p) for p in attempt.iterdir()})
        (job/'completed.json').write_text(json.dumps(record))
        return work,job

    def test_percentiles_and_degenerate_groups(self):
        out=compare_draws(np.array([1.,2.,3.,20.]),np.array([0,0,0,1]))
        self.assertEqual(out['divergent_mean_percentile'],1.)
        self.assertEqual(out['divergent_tail_fraction'],1.)
        self.assertEqual(out['standardized_mean_difference'],18.)
        self.assertTrue(np.isnan(compare_draws(np.ones(4),[0,0,0,1])['standardized_mean_difference']))
        self.assertEqual(compare_draws(np.ones(4),[0,0,0,1])['divergent_mean_percentile'],.5)
        self.assertTrue(np.isnan(compare_draws(np.ones(4),[0,0,0,0])['divergent_median']))
        self.assertTrue(np.isnan(compare_draws(np.ones(4),[1,1,1,1])['nondivergent_median']))

    def test_export_integrity_chain_strata_and_bias_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            work,job=self.fixture(base)
            before={str(p):sha(p) for p in work.rglob('*') if p.is_file()}
            jobs,_=discover(work,1)
            table=jobs[0][4]
            self.assertEqual(len(table),13)
            self.assertIn('theta[2,5]',table.index)
            self.assertNotIn('theta[1,5]',table.index)
            out=base/'out'
            export(work,out,1,['sub-107'])
            summary=pd.read_csv(out/'sampler_by_chain.tsv',sep='\t')
            self.assertEqual(summary.divergences.sum(),2)
            self.assertEqual(summary.max_depth_hits.sum(),2)
            self.assertFalse(summary.inference_eligible.any())
            comparisons=pd.read_csv(out/'parameter_divergence_associations.tsv',sep='\t')
            self.assertEqual(len(comparisons),13*3)
            self.assertEqual(set(comparisons.chain),{'1','2','pooled'})
            self.assertTrue((out/'fit-test_sub-107_pairs.png').is_file())
            self.assertEqual(pd.read_csv(out/'console_warning_counts.tsv',sep='\t').nan_logit_lines.sum(),1)
            export(work,out,1,['sub-107'])
            self.assertEqual(before,{str(p):sha(p) for p in work.rglob('*') if p.is_file()})
            (out/'sampler_by_chain.tsv').write_text('modified')
            with self.assertRaisesRegex(ValueError,'changed/missing'):
                export(work,out,1,['sub-107'])

    def test_refuses_changed_chain_incomplete_job_and_unsafe_output(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            work,job=self.fixture(base)
            with self.assertRaises(ValueError): export(work,work/'bad',1,['sub-107'])
            with self.assertRaises(ValueError): export(work,base/'bad',2,['sub-107'])
            with self.assertRaises(ValueError): export(work,base/'bad',1,['sub-999'])
            chain=job/'attempt-test/chain1.csv'
            chain.write_text(chain.read_text()+'1,2\n')
            with self.assertRaisesRegex(ValueError,'modified completed input'):
                discover(work,1)
            (job/'completed.json').unlink()
            with self.assertRaises(FileNotFoundError): discover(work,1)

    def test_refuses_raw_sampler_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            work,job=self.fixture(base)
            path=job/'completed.json'
            record=json.loads(path.read_text())
            record['diagnostics']['divergences']=99
            diag=job/'attempt-test/diagnostics.json'
            diag.write_text(json.dumps(record['diagnostics']))
            record['files']['attempt-test/diagnostics.json']=sha(diag)
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'raw-chain divergences differs'):
                export(work,base/'out',1,['sub-107'])
            self.assertFalse((base/'out').exists())

    def test_no_bias_model_plot_and_all_natural_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            work,_=self.fixture(base,bias=False)
            export(work,base/'out',1,['sub-107'])
            frame=pd.read_csv(base/'out/parameter_divergence_associations.tsv',sep='\t')
            self.assertEqual(len(frame),12*3)
            self.assertNotIn('acceptance_bias',set(frame.parameter))


if __name__=='__main__':
    unittest.main()
