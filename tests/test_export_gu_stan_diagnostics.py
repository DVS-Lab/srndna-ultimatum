import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from export_gu_stan_diagnostics import (variable_label, parameter_table, worst_variables,
    read_chain, checked_file, digest, sha, export)


class ExportDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(parameters=['alpha','gamma','f0','epsilon'],
                           subjects=['sub-143','sub-144'])
        self.data = dict(subject=[2,1,2], partner=[3,1,2])

    def test_labels_use_recorded_mapping(self):
        label = variable_label('theta[1,4]', self.config, self.data)
        self.assertEqual((label['subject'],label['partner'],label['parameter']),
                         ('sub-144','dissimilar','epsilon'))
        label = variable_label('z_subject[2,3]', self.config, self.data)
        self.assertEqual((label['subject'],label['parameter']), ('sub-144','f0'))
        self.assertEqual(variable_label('z_contrast[2,1,4]',self.config,self.data)['contrast'],
                         'similar_minus_dissimilar')
        self.assertEqual(variable_label('effect[1,1]',self.config,self.data)['contrast'],
                         'human_minus_computer')
        label = variable_label('age_beta[1,4]',self.config,self.data)
        self.assertEqual((label['parameter'],label['contrast'],label['scale']),
                         ('epsilon','older_minus_younger','latent'))

    def test_age_effects_enter_convergence_flags(self):
        frame = pd.DataFrame(dict(R_hat=[1.02], ESS_bulk=[350], ESS_tail=[800]),
                             index=['age_beta[1,1]'])
        table = parameter_table(frame,self.config,self.data)
        self.assertTrue(table.loc['age_beta[1,1]','flagged'])

    def test_flags_and_selection_include_nan_and_all_diagnostics(self):
        frame = pd.DataFrame(dict(R_hat=[1.,1.1,1.002,np.nan,1.],
            ESS_bulk=[1000,1000,3,1000,1000], ESS_tail=[1000,1000,800,2,1000]),
            index=['mu[1]','mu[2]','mu[3]','mu[4]','train_log_lik[1]'])
        table = parameter_table(frame,self.config,self.data)
        self.assertEqual(len(table),4)
        self.assertEqual(int(table.flagged.sum()),3)
        self.assertFalse(table.loc['mu[1]','flagged'])
        self.assertIn('mu[4]', worst_variables(table,3))
        self.assertIn('mu[3]', worst_variables(table,3))

    def test_csv_order_and_saved_warmup_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'chain.csv'
            path.write_text('# id = 3\n# save_warmup = 0 (Default)\nlp__,mu.2,theta.1.4\n0,2,0.5\n0,4,0.7\n# Elapsed time\n')
            chain_id, values=read_chain(path,['theta[1,4]','mu[2]'],2)
            self.assertEqual(chain_id,3)
            np.testing.assert_equal(values,[[.5,2],[.7,4]])
            path.write_text(path.read_text().replace('save_warmup = 0','save_warmup = false'))
            self.assertEqual(read_chain(path,['mu[2]'],2)[0],3)
            with self.assertRaises(ValueError):
                read_chain(path,['mu[2]'],3)
            path.write_text(path.read_text().replace('save_warmup = false','save_warmup = true'))
            with self.assertRaises(ValueError):
                read_chain(path,['mu[2]'],2)

    def test_integrity_and_path_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)/'job'
            job.mkdir()
            path=job/'summary.tsv'
            path.write_text('original')
            record={'files':{'summary.tsv':sha(path)}}
            self.assertEqual(checked_file(job,record,'summary.tsv'),path.resolve())
            path.write_text('modified')
            with self.assertRaises(ValueError):
                checked_file(job,record,'summary.tsv')
            with self.assertRaises(ValueError):
                checked_file(job,record,'../outside')

    def test_full_export_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'work'
            job=work/'fits/fit-demo'
            attempt=job/'attempt-test'
            attempt.mkdir(parents=True)
            config=dict(self.config, phase='fit', model='rw_free', stage='full', samples=4, chains=2,
                        data_hash=digest(self.data))
            (job/'configuration.json').write_text(json.dumps(config))
            (attempt/'data.json').write_text(json.dumps(self.data))
            summary=pd.DataFrame(dict(Mean=[.1,.2,.3],R_hat=[1.005,1.02,1.003],
                ESS_bulk=[700,300,600],ESS_tail=[800,500,300]),index=['mu[1]','mu[2]','mu[3]'])
            summary.to_csv(attempt/'stan_summary.tsv',sep='\t',index_label='variable')
            diag=dict(max_rhat=1.02,min_bulk_ess=300,min_tail_ess=300,passed=False,
                      divergences=0,max_depth_hits=0,min_ebfmi=.8,finite_diagnostics=True)
            (attempt/'diagnostics.json').write_text(json.dumps(diag))
            for c in (1,2):
                (attempt/f'chain{c}.csv').write_text(f'# id = {c}\n# save_warmup = 0\nmu.1,mu.2,mu.3\n.1,.2,.3\n.2,.3,.4\n.3,.4,.5\n.4,.5,.6\n')
            record=dict(job=job.name,config_hash=digest(config),attempt=attempt.name,
                diagnostics=diag,inference_eligible=False,
                files={str(p.relative_to(job)):sha(p) for p in attempt.iterdir()})
            (job/'completed.json').write_text(json.dumps(record))
            out=Path(tmp)/'export'
            export(work,out,expected_jobs=1,trace_count=3)
            self.assertTrue((out/'convergence_overview.png').is_file())
            self.assertTrue((out/'fit-demo_traces.png').is_file())
            table=pd.read_csv(out/'parameter_diagnostics.tsv',sep='\t')
            self.assertEqual(len(table),3)
            self.assertEqual(table.flagged.sum(),2)
            first=sha(out/'export_manifest.json')
            export(work,out,expected_jobs=1,trace_count=3)
            self.assertEqual(sha(out/'export_manifest.json'),first)
            with self.assertRaises(ValueError):
                export(work,Path(tmp)/'bad',expected_jobs=8)


if __name__ == '__main__':
    unittest.main()
