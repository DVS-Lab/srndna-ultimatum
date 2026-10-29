import csv
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import collect_palm_progress as collector
import run_dmn_heteroscedasticity as palm


class PalmInterimTests(unittest.TestCase):
    def fixture(self, root):
        work, report = root/'work', root/'report'
        work.mkdir(); report.mkdir()
        config = root/'config.json'
        config.write_text(json.dumps({'models': [dict(name='model', block='block',
            c=[[1], [-1]], labels=['positive', 'negative'])]}))
        contract = dict(models=['model'], methods=['tfce', 'cluster-z2.6'],
            source=collector.digest(config), code=collector.digest(Path(palm.__file__)),
            helper=collector.digest(Path(palm.pa.__file__)))
        (work/'contract.json').write_text(json.dumps(contract))
        block = work/'blocks/block'; block.mkdir(parents=True)
        mask = block/'mask.nii.gz'
        nib.save(nib.Nifti1Image(np.array([1, 1, 1, 0], dtype=np.uint8).reshape(2, 2, 1), np.eye(4)), mask)
        (block/'complete.json').write_text(json.dumps(dict(outputs={str(mask): collector.digest(mask)})))
        design = work/'designs/model'; design.mkdir(parents=True)
        (design/'mask.nii.gz').symlink_to(mask)
        attempt = work/'jobs/model/tfce/attempt-complete'; attempt.mkdir(parents=True)
        artifacts = []
        for c, values in [(1, [.96, .95, 0, 1]), (2, [.6, .2, 0, 1])]:
            path = attempt/f'palm_tfce_tstat_fwep_c{c}.nii.gz'
            nib.save(nib.Nifti1Image(np.array(values, dtype=np.float64).reshape(2, 2, 1), np.eye(4)), path)
            artifacts.append(path)
        receipt = attempt.parent/'complete.json'
        receipt.write_text(json.dumps(dict(attempt=str(attempt),
            outputs={str(p): collector.digest(p) for p in artifacts})))
        pending = work/'jobs/model/cluster-z2.6/attempt-active'
        pending.mkdir(parents=True)
        (pending/'matlab_console.txt').write_text(
            '4.28%\t [Design 1/1, Contrast 1/4, Shuffling 1711/10000, Modality 1/1]\n'
            '25.50%\t [Design 1/1, Contrast 2/4, Shuffling 200/10000, Modality 1/1]\n')
        # A partial map without a completion receipt must never enter results.
        nib.save(nib.Nifti1Image(np.ones((2, 2, 1)), np.eye(4)), pending/'palm_clustere_tstat_fwep_c1.nii.gz')
        (report/'summary.tsv').write_text('old canonical summary\n')
        return work, report, config, receipt, artifacts, mask, pending

    def test_running_snapshot_only_summarizes_verified_completions(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, config, receipt, artifacts, mask, pending = self.fixture(Path(tmp))
            before = {p: p.read_bytes() for p in [receipt, mask, *artifacts, report/'summary.tsv']}
            with (work/'batch.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                output = collector.collect(work, report, True, True, config)
            state = json.loads((output/'collection_status.json').read_text())
            self.assertTrue(state['batch_lock_held_at_start'])
            self.assertEqual(state['interim_summary_rows'], 2)
            with (output/'interim_results.tsv').open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual([r['contrast_name'] for r in rows], ['positive', 'negative'])
            self.assertAlmostEqual(float(rows[0]['minimum_spatial_fwe_p']), .04)
            self.assertAlmostEqual(float(rows[1]['minimum_spatial_fwe_p']), .4)
            self.assertEqual([r['significant_voxels'] for r in rows], ['1', '0'])
            self.assertFalse(list((output/'completed').rglob('*.nii.gz')))
            for path, original in before.items():
                self.assertEqual(path.read_bytes(), original)

    def test_progress_uses_overall_percent_not_current_contrast_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, config, receipt, artifacts, mask, pending = self.fixture(Path(tmp))
            row = collector.progress_record(pending.parent, 'model', 'cluster-z2.6', False)
            self.assertEqual(row['overall_percent'], 25.5)
            self.assertEqual(row['shuffle_current'], 200)
            self.assertEqual(row['shuffles_per_contrast'], 10000)
            self.assertEqual(row['contrast_current'], 2)
            self.assertEqual(row['status'], 'unfinished')
            (pending/'exit_status.json').write_text(json.dumps(dict(returncode=1, statement_reached=True)))
            self.assertEqual(collector.progress_record(pending.parent, 'model', 'cluster', False)['status'], 'latest_attempt_failed')
            self.assertEqual(collector.progress_record(receipt.parent, 'model', 'tfce', True)['overall_percent'], 100)
            self.assertEqual(collector.progress_record(work/'absent', 'model', 'tfce', False)['status'], 'not_started')

    def test_changed_source_config_and_mask_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, config, receipt, artifacts, mask, pending = self.fixture(Path(tmp))
            original = config.read_bytes()
            config.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                collector.collect(work, report, True, True, config)
            config.write_bytes(original)
            nib.save(nib.Nifti1Image(np.ones((2, 2, 1)), np.eye(4)), mask)
            with self.assertRaisesRegex(ValueError, 'mask is not the verified original'):
                collector.collect(work, report, True, True, config)

    def test_unreceipted_probability_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, config, receipt, artifacts, mask, pending = self.fixture(Path(tmp))
            saved = json.loads(receipt.read_text())
            del saved['outputs'][str(artifacts[1])]
            receipt.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'not in the verified completion receipt'):
                collector.collect(work, report, True, True, config)

    def test_zero_completed_writes_empty_results_with_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, report, config, receipt, artifacts, mask, pending = self.fixture(Path(tmp))
            receipt.unlink()
            output = collector.collect(work, report, True, True, config)
            self.assertEqual(len((output/'interim_results.tsv').read_text().splitlines()), 1)
            self.assertEqual(json.loads((output/'collection_status.json').read_text())['verified_completed_jobs'], 0)


if __name__ == '__main__':
    unittest.main()
