from concurrent.futures import wait as wait_all
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import scale_palm_recovery as scale


class ScaleTests(unittest.TestCase):
    def test_driver_selection_rejects_other_users_repositories_and_matlab(self):
        repo=Path('/repo')
        info=dict(pid=os.getpid()+10000,uid=os.getuid(),cwd=repo,
                  argv=['python3','-u','code/resume_palm_headless.py'])
        self.assertTrue(scale.is_driver(info,repo))
        self.assertFalse(scale.is_driver(dict(info,uid=os.getuid()+1),repo))
        self.assertFalse(scale.is_driver(dict(info,cwd=Path('/other')),repo))
        self.assertFalse(scale.is_driver(dict(info,argv=['matlab','code/resume_palm_headless.py']),repo))
        self.assertFalse(scale.is_driver(dict(info,argv=['python3','-c','x'*10000]),repo))
        custom=dict(info,argv=[*info['argv'],'--work-root','/other-scratch'])
        self.assertFalse(scale.is_driver(custom,repo))
        self.assertTrue(scale.is_driver(custom,repo,Path('/other-scratch')))

    def test_capacity_reserves_ram_and_honors_request_and_cpu_limits(self):
        budget=dict(total=128*scale.GIB,available=80*scale.GIB)
        choice=scale.choose_jobs(8,budget,2*scale.GIB,48)
        self.assertEqual(choice['selected_jobs'],8)
        self.assertEqual(choice['estimated_bytes_per_job'],4*scale.GIB)
        self.assertEqual(scale.choose_jobs(40,budget,8*scale.GIB,48)['selected_jobs'],5)
        self.assertEqual(scale.choose_jobs(8,budget,0,4)['selected_jobs'],2)
        with self.assertRaisesRegex(RuntimeError,'insufficient memory'):
            scale.choose_jobs(8,dict(total=128*scale.GIB,available=scale.GIB),0,48)

    def test_eight_workers_can_run_without_a_large_queued_backlog(self):
        barrier=threading.Barrier(8);seen=[]
        tasks=[(dict(name=str(i)),'tfce') for i in range(8)]
        def worker(item):
            seen.append(item[0]['name']);barrier.wait(timeout=5)
            return [dict(model=item[0]['name'])]
        rows,failures,pending=scale.parallel_jobs(tasks,worker,8)
        self.assertEqual(len(rows),8);self.assertEqual(len(seen),8)
        self.assertFalse(failures);self.assertFalse(pending)

    def test_first_failure_drains_active_workers_and_does_not_queue_more(self):
        barrier=threading.Barrier(4);seen=[]
        tasks=[(dict(name=str(i)),'tfce') for i in range(12)]
        def worker(item):
            name=item[0]['name'];seen.append(name);barrier.wait(timeout=5)
            if name=='0':raise RuntimeError('engine failure')
            return [dict(model=name)]
        with patch.object(scale,'wait',side_effect=lambda active,**kwargs:wait_all(active)):
            rows,failures,pending=scale.parallel_jobs(tasks,worker,4)
        self.assertEqual(len(seen),4);self.assertEqual(len(rows),3)
        self.assertEqual(len(failures),1);self.assertEqual(len(pending),8)

    def test_bad_signal_handler_or_changed_pid_is_never_signalled(self):
        info=dict(pid=12345,start='10',uid=os.getuid(),caught=0,ignored=0,blocked=0,
                  argv=['python3','driver.py'],cwd=Path('/repo'),state='S')
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(scale,'inspect_process',return_value=info),patch.object(scale.os,'kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'SIGINT handler'):
                    scale.drain_driver(info,Path(tmp)/'request.json')
                kill.assert_not_called()
            with patch.object(scale,'inspect_process',return_value=dict(info,start='11')),patch.object(scale.os,'kill') as kill:
                scale.drain_driver(info,Path(tmp)/'request.json')
                kill.assert_not_called()

    @unittest.skipUnless(os.name=='posix','requires POSIX process signals')
    def test_sigint_to_python_driver_allows_both_subprocess_jobs_to_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            # Match the running driver's executor/subprocess structure. Only
            # the driver receives SIGINT; its two jobs must both save receipts.
            program="""import concurrent.futures,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1])
def worker(i):
    subprocess.run([sys.executable,'-c','import time;time.sleep(0.6)'],check=True)
    (root/('done'+str(i))).write_text('complete')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    tasks=[pool.submit(worker,i) for i in range(2)]
    (root/'ready').write_text('ready')
    concurrent.futures.wait(tasks)
"""
            with (root/'driver.txt').open('w') as log:
                process=subprocess.Popen([sys.executable,'-c',program,str(root)],stdout=log,stderr=log)
                try:
                    deadline=time.monotonic()+10
                    while not (root/'ready').exists() and process.poll() is None and time.monotonic()<deadline:
                        time.sleep(.02)
                    self.assertTrue((root/'ready').exists())
                    info=dict(pid=process.pid,start='fixture',uid=os.getuid(),
                        caught=1 << (signal.SIGINT-1),ignored=0,blocked=0,
                        argv=['python3','fixture'],cwd=root,state='S')
                    def inspect(pid):
                        return info if process.poll() is None else None
                    with patch.object(scale,'inspect_process',side_effect=inspect):
                        scale.drain_driver(info,root/'request.json',poll_seconds=.02)
                    self.assertTrue((root/'done0').is_file())
                    self.assertTrue((root/'done1').is_file())
                    self.assertNotEqual(process.returncode,0)
                finally:
                    if process.poll() is None:
                        process.terminate();process.wait(timeout=5)

    def test_validated_runtime_is_reused_and_tampering_blocks_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);(work/'contract.json').write_text('{}')
            profile=work/'runtime-headless/validated';profile.mkdir(parents=True)
            scale.palm.pa.write_json(profile/'contract.json',dict(
                runtime_code_sha256=scale.palm.pa.digest(Path(scale.headless.__file__)),
                scheduler_sha256=scale.palm.pa.digest(Path(scale.recovery.__file__)),
                original_contract_sha256=scale.palm.pa.digest(work/'contract.json')))
            artifact=profile/'verified';artifact.write_text('original')
            for name in ('startup_selection.json','smoke_complete.json','canary_complete.json'):
                scale.palm.pa.write_json(profile/name,dict(outputs={str(artifact):scale.palm.pa.digest(artifact)}))
            self.assertEqual(scale.validated_runtime(work),profile)
            artifact.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                scale.validated_runtime(work)

    def test_original_runtime_and_statistical_files_remain_at_validated_hashes(self):
        repo=Path(__file__).resolve().parents[1]
        contract=json.loads((repo/'results/dmn_followup/v1/palm/contract.json').read_text())
        self.assertEqual(scale.palm.pa.digest(repo/'code/run_dmn_heteroscedasticity.py'),contract['code'])
        self.assertEqual(scale.palm.pa.digest(repo/'code/run_permutation_audit.py'),contract['helper'])
        for filename in ('code/resume_palm_headless.py','code/resume_palm_conservative.py'):
            expected=subprocess.check_output(['git','-C',str(repo),'show','e271790:'+filename])
            self.assertEqual((repo/filename).read_bytes(),expected)


if __name__=='__main__':unittest.main()
