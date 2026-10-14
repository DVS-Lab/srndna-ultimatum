"""Exercise the Linux handoff without starting FSL jobs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BatchTests(unittest.TestCase):
    def run_fixture(self, fail_stage):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            binaries = base / 'bin'
            binaries.mkdir()
            fsl = base / 'fsl'
            (fsl / 'etc').mkdir(parents=True)
            (fsl / 'etc/fslversion').write_text('fixture\n')
            work, report = base / 'work', base / 'report'
            (work / 'logs').mkdir(parents=True)
            (work / 'logs/model.log').write_text('fixture FEAT diagnostic\n')
            for name in ('feat', 'feat_model'):
                tool = binaries / name
                tool.write_text('#!/bin/sh\nexit 0\n')
                tool.chmod(0o755)
            fake_python = binaries / 'python3'
            fake_python.write_text('''#!/bin/bash
if [[ "$1" == --version ]]; then echo fixture-python; exit 0; fi
phase=prepare
for arg in "$@"; do [[ "$arg" == --execute ]] && phase=execute; done
echo "$phase" >> "$BATCH_TEST_CALLS"
[[ "$phase" == "$BATCH_TEST_FAIL" ]] && exit 17
exit 0
''')
            fake_python.chmod(0o755)
            calls = base / 'calls.txt'
            env = dict(os.environ, PATH=str(binaries) + ':' + os.environ['PATH'],
                       FSLDIR=str(fsl), BATCH_TEST_CALLS=str(calls),
                       BATCH_TEST_FAIL=fail_stage)
            result = subprocess.run(['bash', str(ROOT / 'code/run_crossed_behavior_batch.sh'),
                                     str(work), str(report), '2'],
                                    env=env, capture_output=True, text=True)
            attempt, = report.glob('attempt-*')
            self.assertEqual((attempt / 'exit_status.txt').read_text().strip(),
                             str(result.returncode))
            self.assertTrue((attempt / 'feat_logs/model.txt').is_file())
            self.assertTrue((attempt / 'run.txt').is_file())
            return result, calls.read_text().splitlines()

    def test_success(self):
        result, calls = self.run_fixture('none')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ['prepare', 'execute'])

    def test_preflight_failure_stops_launch(self):
        result, calls = self.run_fixture('prepare')
        self.assertEqual(result.returncode, 17)
        self.assertEqual(calls, ['prepare'])
        self.assertNotIn('PASS: all 18', result.stdout)

    def test_execution_failure_is_not_hidden_by_tee(self):
        result, calls = self.run_fixture('execute')
        self.assertEqual(result.returncode, 17)
        self.assertEqual(calls, ['prepare', 'execute'])
        self.assertNotIn('PASS: all 18', result.stdout)
