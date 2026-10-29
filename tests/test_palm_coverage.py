import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import audit_palm_varcope_coverage as audit
import collect_palm_progress as cp
import test_palm_partner_motion as fixtures


class CoverageTests(unittest.TestCase):
    def test_variance_hypotheses_separate_two_one_and_no_supported_runs(self):
        result = audit.variance_comparison([[2, 99, 2, 0], [6, 6, 6, 0]],
            [[1, 0, -1, np.nan], [3, 3, 0, np.inf]], [3, 6, 0, 0], [.75, 3, 0, 0])
        np.testing.assert_array_equal(result['positive_variance_runs'], [2, 1, 0, 0])
        np.testing.assert_allclose(result['expected_if_only_positive_runs_used'][:2], [3, 6])
        np.testing.assert_array_equal(result['existing_l2_cope_agrees'], [1, 1, 0, 0])
        np.testing.assert_array_equal(result['existing_l2_variance_agrees'], [1, 1, 0, 0])
        self.assertTrue(np.isnan(result['expected_if_only_positive_runs_used'][2:]).all())
        # A legitimate zero COPE is not a missing observation.
        zero = audit.variance_comparison([[0], [0]], [[1], [1]], [0], [.5])
        self.assertTrue(zero['existing_l2_cope_agrees'][0])

    def test_classification_does_not_guess_coverage_from_zero_variance(self):
        cases = [(np.nan, 1, 1, 'nonfinite_variance'), (-1, 1, 1, 'negative_variance'),
            (1, 0, 1, 'positive_variance_outside_run_mask'), (1, 1, 1, 'positive_variance'),
            (0, None, 1, 'zero_variance_mask_unavailable'), (0, 0, 1, 'zero_variance_outside_run_mask'),
            (0, np.nan, 1, 'zero_variance_nonfinite_run_mask'),
            (0, 1, None, 'zero_variance_inside_run_mask_signal_unavailable'),
            (0, 1, np.nan, 'zero_variance_inside_run_mask_nonfinite_signal'),
            (0, 1, 0, 'zero_variance_inside_run_mask_flat_signal'),
            (0, 1, 1, 'zero_variance_inside_run_mask_variable_signal')]
        for v, m, sd, expected in cases:
            self.assertEqual(audit.classify(v, m, sd), expected)

    def test_reader_checks_hash_and_geometry_but_preserves_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); reference = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
            path = root/'image.nii.gz'
            nib.save(nib.Nifti1Image(np.full((2, 2, 2), np.nan), np.eye(4)), path)
            reader = audit.Inputs(reference, {str(path): cp.digest(path)})
            self.assertTrue(np.isnan(reader.load(path, 'test')).all())
            self.assertIsNone(reader.load(root/'absent.nii.gz', 'optional', optional=True))
            with self.assertRaises(FileNotFoundError): reader.load(root/'absent.nii.gz', 'required')
            nib.save(reference, path)
            with self.assertRaisesRegex(ValueError, 'provenance'): reader.load(path, 'changed')
            shifted = np.eye(4); shifted[0, 3] = 1
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), shifted), path)
            with self.assertRaisesRegex(ValueError, 'grid mismatch'):
                audit.Inputs(reference, {}).load(path, 'geometry')

    def test_temporal_summary_includes_zero_signal_and_reports_nonfinite(self):
        data = np.array([[[[0, 0, 0], [0, 1, 2], [0, np.nan, 2]]]])
        sd, missing = audit.temporal_summary(data, np.ones((1, 1, 3), bool))
        np.testing.assert_allclose(sd[:2], [0, 1]); self.assertTrue(np.isnan(sd[2]))
        np.testing.assert_array_equal(missing, [0, 0, 1])

    def test_full_read_only_audit_accepts_partial_partner_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model, baseline = fixtures.PartnerMotionTests().fixture(root)
            shape = (4, 4, 4)
            def save(path, values):
                path.parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(np.asarray(values, float), np.eye(4)), path)
            flagged = tuple(model['subjects'][:3])
            for index, image_name in enumerate(model['images']):
                image = Path(image_name); save(image.parent/'varcope1.nii.gz', np.full(shape, .8))
                save(image.parent.parent/'mask.nii.gz', np.ones(shape))
                for run in ('01', '02'):
                    folder = image.parents[3]/f'L1_run-{run}.feat'
                    mask = np.ones(shape)
                    if index < 3:
                        variance = nib.load(folder/'stats/varcope7.nii.gz').get_fdata()
                        if run == '01':
                            variance.flat[index] = 0
                            if index == 0: mask.flat[index] = 0
                            save(folder/'stats/varcope7.nii.gz', variance)
                            # Existing L2 follows the sole positive run at this voxel.
                            cope = nib.load(image).get_fdata()
                            other = image.parents[3]/'L1_run-02.feat/stats'
                            cope.flat[index] = nib.load(other/'cope7.nii.gz').get_fdata().flat[index]
                            save(image, cope)
                            l2v = nib.load(image.parent/'varcope1.nii.gz').get_fdata(); l2v.flat[index] = 4
                            save(image.parent/'varcope1.nii.gz', l2v)
                        signal = np.broadcast_to(np.arange(4.), shape+(4,)).copy()
                        if index in (0, 1): signal.reshape(-1, 4)[index] = 0
                        save(folder/'filtered_func_data.nii.gz', signal)
                        save(folder/'mean_func.nii.gz', signal.mean(axis=-1))
                        save(folder/'stats/sigmasquareds.nii.gz', np.ones(shape))
                    save(folder/'mask.nii.gz', mask)
            config = root/'results/dmn_covariate_standardized/v1/randomise/config.json'
            hashes = config.with_name('input_provenance.json')
            hashes.write_text(json.dumps({p: cp.digest(Path(p)) for p in model['images']}))
            source = root/'partial'; source.mkdir()
            cp.write_json(source/'status.json', dict(complete=False, native_diagnostics_complete=True))
            cp.write_json(source/'provenance.json', dict(sources={str(p): cp.digest(p) for p in (config, hashes)}))
            rows = [dict(subject=s, verified=int(s not in flagged)) for s in model['subjects']]
            cp.write_rows(source/'run_weight_reconstruction.tsv', rows, ['subject', 'verified'])
            cp.write_rows(source/'inventory.tsv', [dict(path=p.name, sha256=cp.digest(p)) for p in source.iterdir()], ['path', 'sha256'])
            standard = root/'synthetic_template.nii.gz'; save(standard, np.ones(shape))
            before = {p: cp.digest(p) for p in root.rglob('*') if p.is_file()}
            output = root/'audit'; audit.build(root, source, baseline, standard, output, flagged)
            state = json.loads((output/'status.json').read_text())
            self.assertTrue(state['complete']); self.assertFalse(state['analysis_changed'])
            self.assertEqual(state['exclusions'], 0); self.assertEqual(state['new_fits'], 0)
            self.assertEqual(len(list(output.glob('*.png'))), 3)
            read = audit.pm.bars.anatomy.read_rows
            self.assertEqual(len(read(output/'run_coverage.tsv')), 94)
            self.assertEqual(len(read(output/'cohort_voxel_support.tsv')), 47*51)
            detailed = read(output/'flagged_voxels.tsv'); self.assertEqual(len(detailed), 3*51)
            kinds = {r['run01_classification'] for r in detailed}
            self.assertTrue({'zero_variance_outside_run_mask', 'zero_variance_inside_run_mask_flat_signal',
                'zero_variance_inside_run_mask_variable_signal'} <= kinds)
            for row in read(output/'l2_handling.tsv')[:3]:
                self.assertEqual(row['one_positive_variance_run'], '1')
                self.assertEqual(row['l2_variance_matches_single_positive_run'], '1')
            for p, digest in before.items(): self.assertEqual(cp.digest(p), digest)
            for r in read(output/'inventory.tsv'): self.assertEqual(cp.digest(output/r['path']), r['sha256'])
            with self.assertRaises(FileExistsError): audit.build(root, source, baseline, standard, output, flagged)


if __name__ == '__main__':
    unittest.main()
