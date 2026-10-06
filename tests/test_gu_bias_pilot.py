import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import run_gu_bias_pilot as pilot
import numpy as np


class BiasPilotTests(unittest.TestCase):
    def test_arithmetic_and_extremes(self):
        audit = pilot.arithmetic_audit()
        self.assertTrue(audit['equivalence_passed'])
        self.assertEqual(audit['random_histories_checked'], 1000)
        self.assertEqual(audit['boundary_stress'][0]['infinite_logits'], 0)
        self.assertGreater(audit['boundary_stress'][1]['infinite_logits'], 0)
        self.assertGreater(audit['boundary_stress'][2]['nan_logits'], 0)
        for bias in [-30, 30]:
            z = pilot.reference_logits([2., 3., 0., .2, bias], [1.,2.])
            np.testing.assert_allclose(z, np.array([3.,6.])+bias)

    def test_command_cpu_budget_and_no_model_changes(self):
        args = pilot.sampling_command(Path('/tmp/pilot'), Path('/tmp/output'), Path('/tmp/stan'), True)
        self.assertEqual(args.count('--execute'), 1)
        self.assertEqual(args[args.index('--models')+1], 'rw_positive_bias')
        self.assertNotIn('--prior-scales', args)
        self.assertNotIn('--age-mode', args)
        self.assertEqual(args[args.index('--samples')+1], '2000')
        self.assertEqual(args[args.index('--adapt-delta')+1], '.995')
        self.assertEqual(2*int(args[args.index('--chains')+1])*int(args[args.index('--threads-per-chain')+1]),40)
        self.assertNotIn('--execute', pilot.sampling_command(Path('/tmp/a'),Path('/tmp/b'),Path('/tmp/c'),False))

    def test_logging_flags_versus_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'run.txt'
            command = [sys.executable, '-c', 'print("flagged fits");raise SystemExit(2)']
            self.assertEqual(pilot.run_logged(command,log,(0,2)), 2)
            self.assertIn('EXIT_STATUS: 2', log.read_text())
            with self.assertRaises(RuntimeError):
                pilot.run_logged(command,log)

    def test_paths_protect_baseline_and_repository(self):
        output = pilot.ROOT/'results/norm_learning/test-pilot'
        pilot.validate_paths(Path('/tmp/pilot'),Path('/tmp/baseline'),output)
        for work in [pilot.ROOT, pilot.ROOT/'temp',Path('/tmp/baseline'),Path('/tmp')]:
            with self.assertRaises(ValueError):
                pilot.validate_paths(work,Path('/tmp/baseline'),output)
        with self.assertRaises(ValueError):
            pilot.validate_paths(Path('/tmp/pilot'),Path('/tmp/baseline'),pilot.ROOT/'code')

    def test_comparison_keeps_ineligible_and_normalizes_counts(self):
        diag = dict(divergences=7,max_depth_hits=0,max_rhat=1.004,
                    min_bulk_ess=1100,min_tail_ess=700,min_ebfmi=.7)
        old = dict(chains=4,samples=6000,warmup=3000,adapt_delta=.99,seed=1,
                   stage='full',data_hash='same')
        old_record = dict(job='fit-test',diagnostics=diag,inference_eligible=False)
        new = dict(old,samples=2000,warmup=4000,adapt_delta=.995,seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); job=work/'fits/fit-test';job.mkdir(parents=True)
            (job/'configuration.json').write_text(json.dumps(new))
            with patch.object(pilot.ext,'completed',return_value=old_record):
                rows=pilot.comparison([(old,old_record)],work)
                self.assertEqual(rows[0]['retained_draws'],24000)
                self.assertEqual(rows[1]['retained_draws'],8000)
                self.assertAlmostEqual(rows[1]['divergence_rate'],7/8000)
                self.assertFalse(rows[1]['inference_eligible'])
                new['data_hash']='changed';(job/'configuration.json').write_text(json.dumps(new))
                with self.assertRaises(ValueError): pilot.comparison([(old,old_record)],work)

    def test_workflow_exports_flagged_fits_and_records_failures(self):
        for fail in [False, True]:
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp).resolve(); repo=base/'repo';repo.mkdir()
                output=repo/'results/norm_learning/pilot'
                argv=['pilot','--execute','--work-root',str(base/'work'),
                      '--baseline-root',str(base/'baseline'),'--output-dir',str(output)]
                calls=[]
                def fake_run(command, log, accepted=(0,)):
                    calls.append(command)
                    if fail: raise RuntimeError('simulated sampling failure')
                    return 2 if len(calls)==1 else 0
                with patch.object(pilot,'ROOT',repo), patch.object(sys,'argv',argv), \
                     patch.object(pilot,'__file__',str(repo/'code/run_gu_bias_pilot.py')), \
                     patch.object(pilot.ext,'STAN',repo/'code/stan/gu_choice_extensions.stan'), \
                     patch.object(pilot.ext,'load_trials',return_value=([],None,None)), \
                     patch.object(pilot,'baseline_records',return_value=[({}, {})]), \
                     patch.object(pilot,'arithmetic_audit',return_value={'equivalence_passed':True}), \
                     patch.object(pilot.ext,'sha',return_value='testhash'), \
                     patch.object(pilot,'run_logged',side_effect=fake_run), \
                     patch.object(pilot,'comparison',return_value=[dict(batch='pilot',stage='full',inference_eligible=False)]):
                    status=pilot.main()
                saved=json.loads((output/'workflow_status.json').read_text())
                self.assertEqual(status,int(fail))
                self.assertEqual(saved['workflow_complete'],not fail)
                if not fail:
                    self.assertEqual(len(calls),3)
                    self.assertEqual(saved['diagnostic_review_required'],['full'])
                    self.assertFalse(saved['parameter_recovery_validated'])
                    for command in calls[1:]:
                        self.assertEqual(command[command.index('--expected-jobs')+1],'2')
                else:
                    self.assertEqual(len(calls),1)
                    self.assertFalse((output/'baseline_pilot_comparison.tsv').exists())


if __name__ == '__main__':
    unittest.main()
