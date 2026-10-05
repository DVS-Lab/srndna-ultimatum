"""Launch contract without compilation/sampling; logs survive both flags and failures."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class WrapperTests(unittest.TestCase):
    def launch(self, mode='fit', execute=True, status=0, export_status=0):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'code').mkdir(); (root/'bin').mkdir()
            script=root/'code/run_gu_choice_extensions.sh'
            shutil.copyfile(ROOT/'code/run_gu_choice_extensions.sh',script)
            fake=root/'bin/python'
            fake.write_text('#!/bin/sh\necho "$*" >> "$TEST_CALLS"\n'
                'case "$*" in\n *run_gu_choice_extensions.py*) echo fit; exit "$FIT_STATUS";;\n'
                ' *) echo export; exit "$EXPORT_STATUS";;\nesac\n')
            fake.chmod(0o755)
            env=dict(os.environ,PATH=str(root/'bin')+os.pathsep+os.environ['PATH'],
                     TEST_CALLS=str(root/'calls'),FIT_STATUS=str(status),
                     EXPORT_STATUS=str(export_status),SRNDNA_STAN_SCRATCH_BASE=str(root/'scratch'))
            cmd=['bash',str(script),mode]+(['--execute'] if execute else [])
            result=subprocess.run(cmd,env=env,text=True,capture_output=True)
            calls=(root/'calls').read_text().splitlines()
            logs={p.name:p.read_text() for p in root.glob('results/norm_learning/*/*.txt')}
            return result.returncode,calls,logs

    def test_fit_plan_and_logging(self):
        status,calls,logs=self.launch()
        self.assertEqual(status,0)
        self.assertEqual(len(calls),2)
        self.assertIn('positive-bias-v2',calls[0])
        for value in ('--models rw_positive rw_positive_bias','--stages full run1',
                      '--jobs 40 --chains 4 --threads-per-chain 2',
                      '--warmup 3000 --samples 6000','--execute'):
            self.assertIn(value,calls[0])
        self.assertIn('--phase fit --expected-jobs 4',calls[1])
        self.assertIn('fit',logs['run.txt']); self.assertIn('export',logs['export.txt'])

    def test_smoke_plan(self):
        status,calls,_=self.launch('smoke')
        self.assertEqual(status,0)
        self.assertIn('positive-bias-smoke-v2',calls[0])
        self.assertIn('--phase smoke',calls[0]); self.assertIn('--samples 100',calls[0])
        self.assertIn('--phase smoke --expected-jobs 4',calls[1])

    def test_preflight_never_exports(self):
        status,calls,logs=self.launch(execute=False)
        self.assertEqual(status,0); self.assertEqual(len(calls),1)
        self.assertNotIn('--execute',calls[0]); self.assertIn('preflight.txt',logs)

    def test_flags_still_export(self):
        status,calls,_=self.launch(status=2)
        self.assertEqual(status,2); self.assertEqual(len(calls),2)

    def test_failure_stops_export(self):
        status,calls,_=self.launch(status=1)
        self.assertEqual(status,1); self.assertEqual(len(calls),1)

    def test_export_failure_propagates(self):
        status,_,_=self.launch(export_status=3)
        self.assertEqual(status,3)


if __name__=='__main__': unittest.main()
