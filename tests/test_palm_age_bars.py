import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import collect_palm_progress as cp
import plot_palm_age_bars as bars


class PalmAgeBarTests(unittest.TestCase):
    def design(self):
        rng = np.random.default_rng(31)
        younger = np.r_[np.ones(25), np.zeros(22)]
        x = np.c_[younger, 1-younger, rng.normal(size=(47, 4))]
        c = np.zeros((4, 6)); c[0, 0] = 1; c[1, 1] = 1
        c[2] = c[0]-c[1]; c[3] = -c[2]
        y = x@np.array([4, -2, .5, -.4, 1, -.8])+rng.normal(size=47)
        return x, c, y

    def test_adjusted_bars_match_group_coefficients_and_conditional_sem(self):
        x, c, y = self.design()
        rows, adjusted, groups, difference = bars.adjusted_bars(x, c, y)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        self.assertEqual([r['n'] for r in rows], [25, 22])
        self.assertAlmostEqual(rows[0]['adjusted_mean'], beta[0])
        self.assertAlmostEqual(rows[1]['adjusted_mean'], beta[1])
        self.assertAlmostEqual(difference, beta[0]-beta[1])
        self.assertAlmostEqual(rows[0]['conditional_participant_sem'], adjusted[:25].std(ddof=1)/5)
        self.assertEqual(groups[:25], ['younger']*25)
        perm = [3, 5, 1, 4, 0, 2]
        result = bars.adjusted_bars(x[:, perm], c[:, perm], y)
        self.assertTrue(np.allclose(adjusted, result[1]))
        self.assertAlmostEqual(difference, result[3])

    def test_invalid_membership_or_nonfinite_values_stop(self):
        x, c, y = self.design()
        x[0, 0] = .5
        with self.assertRaisesRegex(ValueError, 'one-hot'): bars.adjusted_bars(x, c, y)
        x, c, y = self.design(); y[0] = np.nan
        with self.assertRaisesRegex(ValueError, 'invalid'): bars.adjusted_bars(x, c, y)

    def test_extraction_includes_zero_values_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mask = root/'mask.nii.gz'; image = root/'cope.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), mask)
            data = np.zeros((2, 2, 2)); data[0, 0, 0] = 8
            nib.save(nib.Nifti1Image(data, np.eye(4)), image)
            model = dict(subjects=['sub-104'], images=[str(image)])
            hashes = {str(image): cp.digest(image)}
            values, _ = bars.extract(model, mask, hashes)
            self.assertEqual(values.tolist(), [1.])
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'differs'): bars.extract(model, mask, hashes)

    def test_full_export_records_exact_roi_no_independent_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); anatomical = root/'anatomy'; anatomical.mkdir()
            x, c, y = self.design(); subjects = [f'sub-{i}' for i in range(47)]
            images = []; hashes = {}
            for subject, value in zip(subjects, y):
                path = root/f'{subject}.nii.gz'
                nib.save(nib.Nifti1Image(np.full((4, 4, 4), value), np.eye(4)), path)
                images.append(str(path)); hashes[str(path)] = cp.digest(path)
            model = dict(name='full-rt__dmn-age', x=x.tolist(), c=c.tolist(), subjects=subjects, images=images,
                         labels=['younger', 'older', 'younger-older', 'older-younger'])
            directory = root/'results/dmn_covariate_standardized/v1/randomise'; directory.mkdir(parents=True)
            config = directory/'config.json'; config.write_text(json.dumps(dict(models=[model])))
            source = directory/'input_provenance.json'; source.write_text(json.dumps(hashes))
            collection = root/'collection'; collection.mkdir()
            (collection/'contract.json').write_text(json.dumps(dict(source=cp.digest(config), input_provenance=cp.digest(source))))
            (anatomical/'status.json').write_text('{"complete":true}')
            (anatomical/'provenance.json').write_text(json.dumps(dict(collection=str(collection))))
            mask = anatomical/'maps/full-rt__dmn-age/cluster-z2.6/contrast3_significant_mask.nii.gz'
            mask.parent.mkdir(parents=True)
            keep = np.zeros(64, np.uint8); keep[:51] = 1
            nib.save(nib.Nifti1Image(keep.reshape(4, 4, 4), np.eye(4)), mask)
            cp.write_rows(anatomical/'native_summary.tsv', [dict(model=model['name'], method='cluster-z2.6', contrast=3,
                significant_voxels=51, minimum_spatial_fwe_p=.0434)],
                ['model', 'method', 'contrast', 'significant_voxels', 'minimum_spatial_fwe_p'])
            files = [dict(path=str(p.relative_to(anatomical)), sha256=cp.digest(p))
                     for p in anatomical.rglob('*') if p.is_file()]
            cp.write_rows(anatomical/'inventory.tsv', files, ['path', 'sha256'])
            before = {Path(p): cp.digest(Path(p)) for p in images}
            output = root/'bars'
            bars.build(root, anatomical, output)
            self.assertTrue((output/'dmn_age_selected_roi_bars.png').is_file())
            self.assertEqual(len(bars.anatomy.read_rows(output/'participant_estimates.tsv')), 47)
            recorded = json.loads((output/'provenance.json').read_text())
            self.assertIn('circular', recorded['interpretation'])
            self.assertIn('no independent ROI p-value', recorded['interpretation'])
            self.assertIn('not inferential', recorded['error_bars'])
            for path, digest in before.items(): self.assertEqual(cp.digest(path), digest)
            with self.assertRaises(FileExistsError): bars.build(root, anatomical, output)


if __name__ == '__main__':
    unittest.main()
