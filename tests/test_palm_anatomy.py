import csv
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import collect_palm_progress as cp
import plot_completed_palm_anatomy as pa
import run_dmn_heteroscedasticity as palm


class PalmAnatomyTests(unittest.TestCase):
    def fixture(self, root, methods=('tfce', 'cluster-z2.6', 'cluster-z3.1')):
        work, report = root/'work', root/'report'
        work.mkdir(); report.mkdir()
        shape = (18, 20, 16)
        affine = np.diag([2., 2., 2., 1.]); affine[:3, 3] = [-18, -20, -16]
        mask = np.zeros(shape, dtype=np.uint8); mask[2:-2, 2:-2, 2:-2] = 1
        selected = np.zeros(shape, dtype=bool); selected[9:11, 11:13, 8:10] = True
        config = root/'config.json'
        models = [dict(name=t+'__dmn-age', block=t, labels=['younger', 'older', 'younger-older', 'older-younger'],
                       c=[[1], [-1], [1], [-1]], subjects=[f'sub-{i}' for i in range(47)]) for t in pa.TIMINGS]
        config.write_text(json.dumps(dict(models=models)))
        contract = dict(models=[m['name'] for m in models], methods=list(pa.METHODS),
            source=cp.digest(config), code=cp.digest(Path(palm.__file__)), helper=cp.digest(Path(palm.pa.__file__)))
        (work/'contract.json').write_text(json.dumps(contract))
        for m in models:
            block = work/'blocks'/m['block']; block.mkdir(parents=True)
            mask_path = block/'mask.nii.gz'; nib.save(nib.Nifti1Image(mask, affine), mask_path)
            (block/'complete.json').write_text(json.dumps(dict(outputs={str(mask_path): cp.digest(mask_path)})))
            design = work/'designs'/m['name']; design.mkdir(parents=True)
            (design/'mask.nii.gz').symlink_to(mask_path)
        for method in methods:
            folder = work/'jobs/full-rt__dmn-age'/method/'attempt-complete'; folder.mkdir(parents=True)
            artifacts = []
            for c in range(1, 5):
                z = np.ones(shape)*.1; z[selected] = 3.5; z[10, 12, 9] = 4.2
                values = np.ones(shape)*.8
                if c == 1:
                    values[selected] = .98
                if c == 3 and method == 'cluster-z2.6':
                    values[selected] = .9566
                # Ensure off-mask maxima do not determine corrected p/counts.
                values[0, 0, 0] = 1
                unit = 'tfce' if method == 'tfce' else 'clustere'
                for name, data in ((f'palm_vox_zvstat_c{c}.nii.gz', z),
                                   (f'palm_{unit}_zvstat_fwep_c{c}.nii.gz', values)):
                    path = folder/name; nib.save(nib.Nifti1Image(data, affine), path); artifacts.append(path)
            (folder.parent/'complete.json').write_text(json.dumps(dict(attempt=str(folder),
                outputs={str(p): cp.digest(p) for p in artifacts})))
        # Partial, apparently significant maps must remain unrendered.
        pending = work/'jobs/partner-post__dmn-age/tfce/attempt-partial'; pending.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.ones(shape), affine), pending/'palm_tfce_zvstat_fwep_c3.nii.gz')
        (pending/'run.txt').write_text('40% [Design 1/1, Contrast 2/4, Shuffling 6000/10000, Modality 1/1]')
        standard = root/'reference.nii.gz'
        xyz = np.indices(shape)
        anatomy = np.exp(-sum(((xyz[i]-(shape[i]-1)/2)/(shape[i]/3))**2 for i in range(3)))*1000
        nib.save(nib.Nifti1Image(anatomy, affine), standard)
        with (work/'batch.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            collection = cp.collect(work, report, True, True, config)
        return work, report, collection, config, standard, selected

    def test_complete_export_does_not_launch_jobs_or_modify_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, report, collection, config, standard, selected = self.fixture(root)
            before = {p: cp.digest(p) for p in work.rglob('*') if p.is_file()}
            output = root/'outputs'
            with patch.object(palm.subprocess, 'run', side_effect=AssertionError('must not launch anything')):
                pa.build(collection, work, standard, output, config)
            status = json.loads((output/'status.json').read_text())
            self.assertTrue(status['complete']); self.assertTrue(status['batch_lock_held_at_collection'])
            self.assertEqual(status['completed_age_jobs'], 3); self.assertEqual(status['signed_contrasts'], 12)
            self.assertEqual(len(list(output.glob('*_methods.png'))), 6)
            self.assertEqual(len(list(output.glob('*_age_components_page*.png'))), 1)
            maps = output/'maps/full-rt__dmn-age/cluster-z2.6'
            self.assertTrue(np.array_equal(nib.load(maps/'contrast3_significant_mask.nii.gz').get_fdata(), selected))
            rows = pa.read_rows(output/'native_summary.tsv')
            focal = next(r for r in rows if r['method'] == 'cluster-z2.6' and r['contrast'] == '3')
            self.assertAlmostEqual(float(focal['minimum_spatial_fwe_p']), .0434)
            self.assertEqual(int(focal['significant_voxels']), 8)
            parts = pa.read_rows(output/'native_components.tsv')
            part = next(r for r in parts if r['method'] == 'cluster-z2.6' and r['contrast'] == '3')
            self.assertEqual([float(part[f'peak_{a}_mni_mm']) for a in 'xyz'], [2, 4, 2])
            self.assertAlmostEqual(float(part['volume_mm3']), 64)
            self.assertEqual(len(pa.read_rows(output/'pending_jobs.tsv')), 6)
            self.assertFalse((output/'maps/partner-post__dmn-age').exists())
            for path, expected in before.items(): self.assertEqual(cp.digest(path), expected)
            for row in pa.read_rows(output/'inventory.tsv'):
                self.assertEqual(cp.digest(output/row['path']), row['sha256'])
            with self.assertRaises(FileExistsError): pa.build(collection, work, standard, output, config)

    def test_tampered_completed_map_or_collection_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, report, collection, config, standard, selected = self.fixture(root, ('cluster-z2.6',))
            raw = next(work.glob('jobs/full-rt__dmn-age/cluster-z2.6/attempt-*/palm_vox_zvstat_c3.nii.gz'))
            raw.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'differs from receipt'):
                pa.load_completed(collection, work, config, root/'bad')
            (collection/'interim_results.tsv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'collection checksum mismatch'):
                pa.verify_collection(collection)

    def test_plane_orientation_and_native_components(self):
        affine = np.diag([2., 3., 4., 1.]); affine[:3, 3] = [-4, -6, -8]
        values = np.arange(125).reshape(5, 5, 5)
        plane, extent, actual = pa.plane(values, affine, 0, 0)
        self.assertTrue(np.array_equal(plane, values[2, :, :].T))
        self.assertEqual(extent, [-7.5, 7.5, -10, 10]); self.assertEqual(actual, 0)
        oblique = affine.copy(); oblique[0, 1] = .1
        with self.assertRaisesRegex(ValueError, 'canonical'): pa.plane(values, oblique, 0, 0)
        with self.assertRaisesRegex(ValueError, 'outside'): pa.plane(values, affine, 0, 1000)
        sig = np.zeros((5, 5, 5), bool); sig[1, 1, 1] = True; sig[2, 2, 2] = True; sig[4, 4, 4] = True
        parts = pa.native_components(sig, np.ones(sig.shape)*3, np.ones(sig.shape)*.97, affine)
        self.assertEqual([p['native_voxels'] for p in parts], [2, 1])
        self.assertAlmostEqual(parts[0]['volume_mm3'], 48)

    def test_no_significant_cut_does_not_use_unthresholded_peak(self):
        payload = {('full-rt', 'tfce', 3): dict(sig=np.zeros((3, 3, 3), bool), z=np.ones((3, 3, 3))*9)}
        cut, rule = pa.select_cut(payload, 'full-rt', 3, [1, 2, 3])
        self.assertEqual(cut.tolist(), [1, 2, 3]); self.assertIn('No significant', rule)

    def test_probability_threshold_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, report, collection, config, standard, selected = self.fixture(root, ('cluster-z2.6',))
            folder = work/'jobs/full-rt__dmn-age/cluster-z2.6'
            receipt = json.loads((folder/'complete.json').read_text())
            path = next(Path(p) for p in receipt['outputs'] if 'fwep_c3.' in p)
            im = nib.load(path); values = im.get_fdata(); values[selected] = .95
            values[10, 12, 9] = .9500001
            nib.save(nib.Nifti1Image(values, im.affine), path)
            receipt['outputs'][str(path)] = cp.digest(path)
            (folder/'complete.json').write_text(json.dumps(receipt))
            collection = cp.collect(work, report, True, True, config)
            payload, *_ = pa.load_completed(collection, work, config, root/'strict')
            self.assertEqual(payload['full-rt', 'cluster-z2.6', 3]['significant_voxels'], 1)

    def test_grid_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, report, collection, config, standard, selected = self.fixture(root, ('cluster-z2.6',))
            # Change raw geometry, update its receipt and recollect. Hashes alone
            # must not allow displaying a geometrically inconsistent map.
            folder = work/'jobs/full-rt__dmn-age/cluster-z2.6'
            receipt = json.loads((folder/'complete.json').read_text())
            path = next(Path(p) for p in receipt['outputs'] if 'palm_vox_zvstat_c1.' in p)
            im = nib.load(path); affine = im.affine.copy(); affine[0, 3] += 1
            nib.save(nib.Nifti1Image(im.get_fdata(), affine), path)
            receipt['outputs'][str(path)] = cp.digest(path)
            (folder/'complete.json').write_text(json.dumps(receipt))
            collection = cp.collect(work, report, True, True, config)
            with self.assertRaisesRegex(ValueError, 'grids differ'):
                pa.load_completed(collection, work, config, root/'bad')


if __name__ == '__main__':
    unittest.main()
