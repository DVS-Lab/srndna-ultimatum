"""The launch wrapper must export flagged fits without hiding actual failures."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FollowupTests(unittest.TestCase):
    def run_wrapper(self, status):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'code').mkdir()
            (root/'bin').mkdir()
            shutil.copyfile(ROOT/'code/run_gu_followup.sh', root/'code/run_gu_followup.sh')
            fake = root/'bin/python'
            fake.write_text('#!/bin/sh\n'
                'echo "$*" >> "$TEST_CALLS"\n'
                'case "$*" in\n'
                '  *run_gu_hierarchical*) echo "fit log"; exit "$TEST_STATUS" ;;\n'
                '  *) echo "export log"; exit 0 ;;\n'
                'esac\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(root/'bin')+os.pathsep+os.environ['PATH'],
                TEST_CALLS=str(root/'calls.txt'), TEST_STATUS=str(status),
                SRNDNA_STAN_SCRATCH_BASE=str(root/'scratch'))
            result = subprocess.run(['bash', str(root/'code/run_gu_followup.sh'), 'age'],
                                    env=env, capture_output=True, text=True)
            calls = (root/'calls.txt').read_text().splitlines()
            self.assertTrue((root/'results/norm_learning/stan-agegroup-v1/run.log').exists())
            self.assertIn('--age-mode group', calls[0])
            self.assertIn('--jobs 40', calls[0])
            return result.returncode, calls

    def test_flagged_fits_are_exported(self):
        status, calls = self.run_wrapper(2)
        self.assertEqual(status, 2)
        self.assertEqual(len(calls), 2)
        self.assertIn('export_gu_stan_diagnostics.py', calls[1])

    def test_actual_failure_stops_export(self):
        status, calls = self.run_wrapper(1)
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 1)

    def test_success_exports(self):
        status, calls = self.run_wrapper(0)
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
