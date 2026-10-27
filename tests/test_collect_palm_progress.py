import fcntl
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import collect_palm_progress as collector


class PalmProgressTests(unittest.TestCase):
    def fixture(self, root):
        work, report = root / 'work', root / 'report'
        work.mkdir(); report.mkdir()
        (work / 'contract.json').write_text(json.dumps(dict(models=['model'], methods=['tfce', 'cluster'])))
        attempt = work / 'jobs/model/tfce/attempt-original'
        attempt.mkdir(parents=True)
        artifact = attempt / 'map.nii.gz'
        artifact.write_bytes(b'original completed output')
        receipt = attempt.parent / 'complete.json'
        receipt.write_text(json.dumps(dict(attempt=str(attempt), outputs={str(artifact): collector.digest(artifact)})))
        pending = work / 'jobs/model/cluster/attempt-running/run.txt'
        pending.parent.mkdir(parents=True)
        pending.write_text('still running')
        (report / 'status.json').write_text('{"complete": false}')
        return work, report, receipt, artifact

    def test_active_collection_preserves_sources_and_collects_only_receipted_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, receipt, artifact = self.fixture(Path(tmp))
            before = {p: p.read_bytes() for p in [receipt, artifact, report / 'status.json']}
            with (work / 'batch.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                output = collector.collect(work, report)
            state = json.loads((output / 'collection_status.json').read_text())
            self.assertTrue(state['batch_lock_held_at_start'])
            self.assertTrue(state['collection_complete'])
            self.assertFalse(state['all_jobs_complete'])
            self.assertEqual(state['verified_completed_jobs'], 1)
            self.assertEqual(state['unfinished_jobs'], 1)
            self.assertEqual((output / 'completed/model/tfce/map.nii.gz').read_bytes(), artifact.read_bytes())
            self.assertEqual(len(list(output.glob('unfinished_logs/*/*/*/run.txt'))), 1)
            for path, original in before.items():
                self.assertEqual(path.read_bytes(), original)
            second = collector.collect(work, report)
            self.assertNotEqual(output, second)

    def test_tampered_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, receipt, artifact = self.fixture(Path(tmp))
            artifact.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'output changed'):
                collector.collect(work, report)
            states = list(report.glob('collections/*/collection_status.json'))
            self.assertFalse(json.loads(states[0].read_text())['collection_complete'])
            self.assertEqual(artifact.read_bytes(), b'tampered')

    def test_metadata_only_verifies_outputs_without_duplicating_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, receipt, artifact = self.fixture(Path(tmp))
            output = collector.collect(work, report, metadata_only=True)
            self.assertTrue(json.loads((output / 'collection_status.json').read_text())['metadata_only'])
            self.assertTrue((output / 'completed/model/tfce/complete.json').is_file())
            self.assertFalse((output / 'completed/model/tfce/map.nii.gz').exists())
            self.assertTrue(artifact.is_file())
            artifact.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'output changed'):
                collector.collect(work, report, metadata_only=True)

    def test_receipt_cannot_collect_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, receipt, artifact = self.fixture(Path(tmp))
            saved = json.loads(receipt.read_text())
            saved['attempt'] = str(Path(tmp))
            receipt.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'invalid completed attempt'):
                collector.collect(work, report)

    def test_all_complete_is_counted_independently_of_stale_batch_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, receipt, artifact = self.fixture(Path(tmp))
            (work / 'contract.json').write_text(json.dumps(dict(models=['model'], methods=['tfce'])))
            output = collector.collect(work, report)
            state = json.loads((output / 'collection_status.json').read_text())
            self.assertTrue(state['all_jobs_complete'])
            self.assertEqual(state['expected_jobs'], 1)
            self.assertFalse(json.loads((report / 'status.json').read_text())['complete'])

    def test_collection_participant_order_table_is_not_ignored(self):
        repo = Path(__file__).resolve().parents[1]
        for name, ignored in [
            ('results/dmn_followup/v1/palm/collections/example/designs/model/participants.tsv', False),
            ('results/other/participants.tsv', True),
        ]:
            result = subprocess.run(['git', '-C', str(repo), 'check-ignore', '-q', name])
            self.assertEqual(result.returncode == 0, ignored)


if __name__ == '__main__':
    unittest.main()
