"""Exercise the Linux launch contract without starting Stan fits."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ValidationWrapperTests(unittest.TestCase):
    def run_wrapper(self, mode, *, execute=True, status=0, export_status=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'code').mkdir()
            (root/'bin').mkdir()
            shutil.copyfile(ROOT/'code/run_gu_validation.sh', root/'code/run_gu_validation.sh')
            fake = root/'bin/python'
            fake.write_text('#!/bin/sh\n'
                'echo "$*" >> "$TEST_CALLS"\n'
                'case "$*" in\n'
                ' *run_gu_hierarchical*) echo "fit log"; exit "$TEST_STATUS" ;;\n'
                ' *) echo "export log"; exit "$TEST_EXPORT_STATUS" ;;\n'
                'esac\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(root/'bin')+os.pathsep+os.environ['PATH'],
                TEST_CALLS=str(root/'calls.txt'), TEST_STATUS=str(status),
                TEST_EXPORT_STATUS=str(export_status), SRNDNA_STAN_SCRATCH_BASE=str(root/'scratch'))
            args = ['bash', str(root/'code/run_gu_validation.sh'), mode]
            if execute:
                args.append('--execute')
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            calls = (root/'calls.txt').read_text().splitlines() if (root/'calls.txt').exists() else []
            logs = {p.name:p.read_text() for p in root.glob('results/norm_learning/*/*.txt')}
            return result.returncode, calls, logs

    def test_prior_plan_and_logs(self):
        status, calls, logs = self.run_wrapper('priors')
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)
        for token in ['--models rw_free', '--age-mode none', '--prior-scales .5 1 2',
                      '--jobs 40 --chains 4 --threads-per-chain 2',
                      '--warmup 3000 --samples 6000', '--execute']:
            self.assertIn(token, calls[0])
        self.assertIn('--phase fit --expected-jobs 6', calls[1])
        self.assertIn('fit log', logs['run.txt'])
        self.assertIn('export log', logs['export.txt'])

    def test_recovery_plan(self):
        status, calls, _ = self.run_wrapper('recovery')
        self.assertEqual(status, 0)
        self.assertIn('--phase recovery', calls[0])
        self.assertIn('--recovery-reps 10', calls[0])
        self.assertIn('--phase recovery --expected-jobs 10', calls[1])

    def test_dry_run_never_exports(self):
        status, calls, logs = self.run_wrapper('priors', execute=False)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 1)
        self.assertNotIn('--execute', calls[0])
        self.assertIn('preflight.txt', logs)
        self.assertNotIn('run.txt', logs)

    def test_flags_preserved_and_exported(self):
        status, calls, _ = self.run_wrapper('recovery', status=2)
        self.assertEqual(status, 2)
        self.assertEqual(len(calls), 2)

    def test_failure_stops_export(self):
        status, calls, _ = self.run_wrapper('priors', status=1)
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 1)

    def test_export_failure_is_not_hidden(self):
        status, calls, _ = self.run_wrapper('priors', export_status=3)
        self.assertEqual(status, 3)
        self.assertEqual(len(calls), 2)

    def test_unknown_mode_fails_without_python(self):
        status, calls, _ = self.run_wrapper('other')
        self.assertEqual(status, 1)
        self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
