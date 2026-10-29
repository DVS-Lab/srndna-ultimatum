import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import audit_palm_partner_motion as pm
import collect_palm_progress as cp
import plot_palm_age_bars as bars


class PartnerMotionTests(unittest.TestCase):
    def model(self):
        rng = np.random.default_rng(31)
        young = np.r_[np.ones(25), np.zeros(22)]
        x = np.c_[young, 1-young, rng.normal(size=(47, 4))]
        # FD and tSNR are correlated with age and each other, deliberately.
        x[:, 4] += 1.4*(1-young); x[:, 3] += .6*x[:, 4]
        c = np.zeros((4, 6)); c[0, 0] = 1; c[1, 1] = 1
        c[2] = c[0]-c[1]; c[3] = -c[2]
        return dict(name='full-rt__dmn-age', x=x.tolist(), c=c.tolist(),
            labels=['younger', 'older', 'younger-older', 'older-younger'],
            subjects=[f'sub-{i}' for i in range(47)],
            nuisance={name: x[:, j].tolist() for name, j in [('ismale', 2), ('tsnr', 3), ('fd_mean', 4), ('RT', 5)]})

    def test_common_weight_decomposition_is_exact_and_validates_inputs(self):
        conditions = np.array([[[3, 8], [8, 11], [2, 4]], [[4, 6], [10, 8], [3, 2]]], float)
        contrast = conditions[:, 1]-conditions[:, 2]
        variances = np.array([[1, 3], [4, 2]], float)
        combined, reconstructed, weights = pm.common_weights(conditions, contrast, variances)
        self.assertTrue(np.allclose(combined[1]-combined[2], reconstructed))
        self.assertTrue(np.allclose(weights.sum(axis=0), 1))
        self.assertTrue(np.allclose(combined[0], (conditions[:, 0]*weights).sum(axis=0)))
        variances[0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'nonpositive'): pm.common_weights(conditions, contrast, variances)
        with self.assertRaisesRegex(ValueError, 'within-run'): pm.common_weights(conditions, contrast+1, np.ones((2, 2)))

    def test_policies_remove_only_requested_columns_and_are_canonical_invariant(self):
        model = self.model(); plans, columns = pm.designs(model)
        self.assertEqual([plans[p][0].shape[1] for p in pm.POLICIES], [2, 5, 4, 6])
        self.assertTrue(np.array_equal(plans['no-group-FD'][0], np.delete(np.array(model['x']), columns['fd_mean'], axis=1)))
        perm = [3, 5, 1, 4, 0, 2]
        other = dict(model, x=np.array(model['x'])[:, perm].tolist(), c=np.array(model['c'])[:, perm].tolist())
        y = np.random.default_rng(12).normal(size=47)
        altered, _ = pm.designs(other)
        for p in plans:
            a = pm.fit_diagnostic(*plans[p], y)[1]; b = pm.fit_diagnostic(*altered[p], y)[1]
            for key in a: self.assertAlmostEqual(a[key], b[key])

    def test_omitted_fd_identity_refits_all_remaining_slopes(self):
        model = self.model(); x = np.array(model['x']); c = np.array(model['c'])
        y = x@np.array([4, -2, 1, -1, 3, .5])+np.random.default_rng(45).normal(size=47)
        result = pm.motion_identity(model, y, 4)
        reduced = np.delete(x, 4, axis=1); reduced_c = np.delete(c[2], 4)
        self.assertAlmostEqual(result['refitted_no_group_fd_age_gap'], reduced_c@np.linalg.lstsq(reduced, y, rcond=None)[0])
        self.assertAlmostEqual(result['refitted_no_group_fd_age_gap']-result['full_age_gap'], result['omitted_fd_product'])

    def fixture(self, root):
        model = self.model(); model['images'] = []; hashes = {}
        affine = np.eye(4); shape = (4, 4, 4); roi = np.zeros(64, np.uint8); roi[:51] = 1; roi = roi.reshape(shape)
        rng = np.random.default_rng(22); y = np.array(model['x'])@np.array([4, -2, .5, -.4, 1, -.8])+rng.normal(size=47)
        expected = []
        def save(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.asarray(data, float), affine), path)
        for subject, value in zip(model['subjects'], y):
            subject_root = root/'derivatives/nppi-dmn'/subject
            run_values = []
            for run, delta in [('01', 2), ('02', -1)]:
                folder = subject_root/f'L1_run-{run}.feat'; folder.mkdir(parents=True)
                text = 'set fmri(evs_real) 28\n'+''.join(f'set fmri(evtitle{j}) "{name}"\n' for j, name in [(12, 'comp_p'), (14, 'in_p'), (16, 'out_p')])
                (folder/'design.fsf').write_text(text)
                con = np.zeros((7, 28)); con[1, 11] = 1; con[3, 13] = 1; con[5, 15] = 1; con[6] = con[3]-con[5]
                with (folder/'design.con').open('w') as stream:
                    stream.write('/NumWaves 28\n/NumContrasts 7\n/Matrix\n'); np.savetxt(stream, con)
                conditions = np.array([np.full(shape, 3+delta), np.full(shape, value/2+4+delta), np.full(shape, -value/2+4)])
                contrast = conditions[1]-conditions[2]
                variance = np.full(shape, 1 if run == '01' else 4)
                for number, values in zip(pm.PARTNERS.values(), conditions): save(folder/'stats'/f'cope{number}.nii.gz', values)
                save(folder/'stats/cope7.nii.gz', contrast); save(folder/'stats/varcope7.nii.gz', variance)
                run_values.append(conditions)
            a, b = run_values
            contrast = .8*(a[1]-a[2])+.2*(b[1]-b[2])
            image = subject_root/'L2.gfeat/cope7.feat/stats/cope1.nii.gz'; save(image, contrast)
            model['images'].append(str(image)); hashes[str(image)] = cp.digest(image)
            for j, number in enumerate(pm.PARTNERS.values()):
                save(subject_root/f'L2.gfeat/cope{number}.feat/stats/cope1.nii.gz', .25*a[j]+.75*b[j])
            expected.append(float(contrast[roi > 0].mean()))
        config_dir = root/'results/dmn_covariate_standardized/v1/randomise'; config_dir.mkdir(parents=True)
        config = config_dir/'config.json'; config.write_text(json.dumps(dict(models=[model])))
        input_hashes = config_dir/'input_provenance.json'; input_hashes.write_text(json.dumps(hashes))
        baseline = root/'baseline'; baseline.mkdir()
        save(baseline/'selected_age_roi.nii.gz', roi)
        summaries, adjusted, groups, _ = bars.adjusted_bars(model['x'], model['c'], np.array(expected))
        cp.write_rows(baseline/'bar_summary.tsv', summaries, list(summaries[0]))
        people = [dict(subject=s, age_group=g, raw_roi_cope7=v, adjusted_roi_cope7=float(a))
            for s, g, v, a in zip(model['subjects'], groups, expected, adjusted)]
        cp.write_rows(baseline/'participant_estimates.tsv', people, list(people[0]))
        cp.write_json(baseline/'provenance.json', dict(sources={str(p): cp.digest(p) for p in [config, input_hashes]}))
        cp.write_json(baseline/'status.json', dict(complete=True))
        files = [dict(path=str(p.relative_to(baseline)), sha256=cp.digest(p)) for p in baseline.iterdir()]
        cp.write_rows(baseline/'inventory.tsv', files, ['path', 'sha256'])
        return model, baseline

    def test_full_three_partner_export_preserves_inputs_and_reconstructs_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model, baseline = self.fixture(root)
            before = {p: cp.digest(p) for p in (root/'derivatives').rglob('*') if p.is_file()}
            output = root/'outputs'; pm.build(root, baseline, output)
            state = json.loads((output/'status.json').read_text()); self.assertTrue(state['complete'])
            self.assertEqual(state['excluded_participants'], 0)
            self.assertEqual(len(list(output.glob('*.png'))), 4)
            rows = bars.anatomy.read_rows(output/'bar_summary.tsv')
            self.assertEqual({r['metric'] for r in rows}, {'computer', 'similar', 'dissimilar', 'similar-minus-dissimilar'})
            self.assertEqual(len(rows), 2*4*4*2)
            self.assertTrue(all(r['verified'] == '1' for r in bars.anatomy.read_rows(output/'run_weight_reconstruction.tsv')))
            for path, expected in before.items(): self.assertEqual(cp.digest(path), expected)
            with self.assertRaises(FileExistsError): pm.build(root, baseline, output)

    def test_failed_exact_reconstruction_saves_native_diagnostics_without_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model, baseline = self.fixture(root)
            image = Path(model['images'][0]).parents[3]/'L1_run-01.feat/stats/varcope7.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4)), image)
            output = root/'outputs'
            with self.assertRaisesRegex(RuntimeError, 'native plots saved'):
                pm.build(root, baseline, output)
            state = json.loads((output/'status.json').read_text()); self.assertFalse(state['complete'])
            self.assertTrue(state['native_diagnostics_complete']); self.assertEqual(state['excluded_participants'], 0)
            self.assertTrue((output/'native_three_partner_bars.png').is_file())
            self.assertFalse((output/'matched_three_partner_bars.png').exists())
            audit = bars.anatomy.read_rows(output/'run_weight_reconstruction.tsv')
            self.assertEqual(sum(r['verified'] == '0' for r in audit), 1)


if __name__ == '__main__':
    unittest.main()
