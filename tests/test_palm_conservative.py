import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import collect_palm_failure_logs as logs
import resume_palm_conservative as recovery


class PalmConservativeTests(unittest.TestCase):
    def test_collect_preserves_originals_and_skips_completed_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/'work';report=Path(tmp)/'report'
            work.mkdir();report.mkdir()
            failed=work/'jobs/model/tfce/attempt-failed';failed.mkdir(parents=True)
            full=b'x'*(logs.TAIL_BYTES+20)+b'failure at end\n'
            (failed/'run.txt').write_bytes(full)
            (failed/'command.json').write_text('["matlab"]')
            good=work/'jobs/model/tfce/attempt-good';good.mkdir()
            (good/'run.txt').write_text('done')
            receipt=good.parent/'complete.json'
            receipt.write_text(json.dumps(dict(attempt=str(good))))
            (report/'status.json').write_text('{"complete":false}')
            original_receipt=receipt.read_bytes()
            out=logs.collect(work,report)
            data=json.loads((out/'collection.json').read_text())
            self.assertEqual(data['failed_or_incomplete_attempts'],1)
            row=data['logs'][0]
            self.assertTrue(row['truncated'])
            self.assertEqual((out/row['excerpt']).stat().st_size,logs.TAIL_BYTES)
            self.assertEqual((failed/'run.txt').read_bytes(),full)
            self.assertEqual(receipt.read_bytes(),original_receipt)
            self.assertEqual((out/'status.json').read_bytes(),(report/'status.json').read_bytes())
            self.assertNotEqual(out,logs.collect(work,report))

    def test_collector_refuses_active_batch(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            with (work/'batch.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,'active'):
                    logs.collect(work,work/'report')
            self.assertFalse((work/'report').exists())

    def test_serial_stops_after_first_failure(self):
        tasks=[({'name':str(i)},'tfce') for i in range(5)];seen=[]
        def worker(item):
            seen.append(item[0]['name'])
            if item[0]['name']=='1':raise RuntimeError('engine failed')
            return [dict(model=item[0]['name'])]
        rows,failures,pending=recovery.bounded_run(tasks,worker,1)
        self.assertEqual(seen,['0','1'])
        self.assertEqual(len(rows),1);self.assertEqual(len(failures),1)
        self.assertEqual([r['model'] for r in pending],['2','3','4'])

    def test_two_jobs_drain_without_starting_a_third(self):
        started=threading.Barrier(2);failed=threading.Event();seen=[]
        tasks=[({'name':str(i)},'tfce') for i in range(5)]
        def worker(item):
            name=item[0]['name'];seen.append(name)
            started.wait(timeout=5)
            if name=='0':
                failed.set();raise RuntimeError('failure')
            failed.wait(timeout=5)
            return [dict(model=name)]
        # Ensure scheduler sees failure before treating the second as done.
        original_wait=recovery.wait
        def wait_all(active,**kwargs):
            return original_wait(active)
        with patch.object(recovery,'wait',side_effect=wait_all):
            rows,failures,pending=recovery.bounded_run(tasks,worker,2)
        self.assertEqual(set(seen),{'0','1'})
        self.assertEqual(len(rows),1);self.assertEqual(len(failures),1)
        self.assertEqual(len(pending),3)

    def test_all_success_and_concurrency_limit(self):
        tasks=[({'name':str(i)},'tfce') for i in range(7)]
        rows,failures,pending=recovery.bounded_run(tasks,lambda item:[item[0]],2)
        self.assertEqual(len(rows),7);self.assertFalse(failures);self.assertFalse(pending)
        with self.assertRaises(ValueError):recovery.bounded_run(tasks,lambda x:x,40)

    def test_ignore_exception_is_narrow(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp)
            subprocess.run(['git','init','-q',str(temp)],check=True)
            (temp/'.gitignore').write_bytes((root/'.gitignore').read_bytes())
            for name,ignored in [('participants.tsv',True),('bids/participants.tsv',True),
                ('results/other/participants.tsv',True),
                ('results/dmn_followup/v1/palm/designs/full-rt__dmn-age/participants.tsv',False)]:
                result=subprocess.run(['git','-C',str(temp),'check-ignore','-q',name])
                self.assertEqual(result.returncode==0,ignored,name)

    def test_participant_table_alignment(self):
        subjects=[f'sub-{i}' for i in range(47)]
        sample={s:{'older':int(i>=25)} for i,s in enumerate(subjects)}
        models=[dict(name='test',subjects=subjects)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'designs/test/participants.tsv';p.parent.mkdir(parents=True)
            rows=[dict(subject=s,variance_group=str(1+int(sample[s]['older']))) for s in subjects]
            recovery.palm.pa.write_rows(p,rows)
            recovery.validate_participants(models,root,sample)
            recovery.palm.pa.write_rows(p,rows[::-1])
            with self.assertRaises(ValueError):recovery.validate_participants(models,root,sample)

    def test_recovery_retains_complete_job_and_records_new_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';report=root/'report';source=root/'source'
            work.mkdir();report.mkdir();source.mkdir()
            (work/'smoke').mkdir()
            smoke=work/'smoke/run.txt';smoke.write_text('passed')
            (work/'smoke/complete.json').write_text(json.dumps(dict(outputs={str(smoke):logs.digest(smoke)})))
            model=dict(name='test');models=[model]
            contract=dict(models=['test'],engine='unchanged-matlab',shuffles=10000)
            (work/'contract.json').write_text(json.dumps(contract))
            complete=work/'jobs/test/tfce/complete.json';complete.parent.mkdir(parents=True)
            complete.write_text('{"fixture":"original-completion"}')
            original=complete.read_bytes();calls=[]
            def worker(m,method,inputs,w,r,engine,palm,n):
                calls.append(method)
                self.assertEqual(engine,'unchanged-matlab');self.assertEqual(n,10000)
                if method!='tfce':raise RuntimeError('new engine failure')
                return [dict(model='test',method=method,contrast=1)]
            with patch.object(recovery.subprocess,'run') as prepare, \
                 patch.object(recovery.palm.pa,'standardized_dmn_catalog',return_value=(models,{},None)), \
                 patch.object(recovery,'validate_participants'), \
                 patch.object(recovery.palm,'job',side_effect=worker):
                with self.assertRaisesRegex(RuntimeError,'recovery stopped'):
                    recovery.recover(work,report,source,root/'palm',1,True)
            self.assertEqual(prepare.call_count,2)
            self.assertEqual(calls,['tfce','cluster-z2.6'])
            self.assertEqual(complete.read_bytes(),original)
            self.assertEqual(json.loads((work/'contract.json').read_text()),contract)
            state=json.loads((report/'status.json').read_text())
            self.assertFalse(state['complete']);self.assertEqual(len(state['failures']),1)
            self.assertEqual(len(state['not_attempted']),1)
            self.assertEqual(len(recovery.palm.pa.read_rows(report/'summary.tsv')),1)

    def test_damaged_completed_receipt_blocks_all_new_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);work=root/'work';report=root/'report';work.mkdir();report.mkdir()
            (work/'smoke').mkdir();smoke=work/'smoke/run.txt';smoke.write_text('passed')
            (work/'smoke/complete.json').write_text(json.dumps(dict(outputs={str(smoke):logs.digest(smoke)})))
            (work/'contract.json').write_text(json.dumps(dict(models=['test'],engine='matlab',shuffles=10000)))
            receipt=work/'jobs/test/tfce/complete.json';receipt.parent.mkdir(parents=True);receipt.write_text('{}')
            with patch.object(recovery.subprocess,'run'), \
                 patch.object(recovery.palm.pa,'standardized_dmn_catalog',return_value=([dict(name='test')],{},None)), \
                 patch.object(recovery,'validate_participants'), \
                 patch.object(recovery.palm,'job',side_effect=ValueError('checksum mismatch')) as job:
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                    recovery.recover(work,report,root/'source',root/'palm',2,True)
                self.assertEqual(job.call_count,1)


if __name__=='__main__':unittest.main()
