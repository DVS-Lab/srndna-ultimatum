import itertools
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import audit_age_figures as audit


class AgeFigureTests(unittest.TestCase):
    def crossed(self):
        rng = np.random.default_rng(92)
        young = np.r_[np.ones(25), np.zeros(22)]
        b = rng.normal(size=47); b -= b.mean()
        x = np.c_[rng.normal(size=(47, 4)), young, 1-young, b*young, b*(1-young)]
        y = x@np.array([.2, -.3, .5, -.4, 1, 2, -2, 1])+rng.normal(size=47)
        return x, y, np.array([0, 0, 0, 0, 0, 0, -1, 1]), [f'sub-{i}' for i in range(47)]

    def save(self, path, values, affine=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=float), np.eye(4) if affine is None else affine), path)
        return path

    def test_exact_49_voxel_selection_and_checksum_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); root = repo/'results/dmn_followup/v1/coverage_check/v1'
            shape = (4, 4, 4); old = np.zeros(64); old[:51] = 1; old = old.reshape(shape)
            valid = old.copy(); valid.ravel()[49:51] = 0
            folder = root/'maps/palm/cluster-z2.6'
            self.save(folder/'palm_vox_zvstat_c3.nii', valid*3)
            self.save(folder/'palm_clustere_zvstat_fwep_c3.nii', valid*.96)
            self.save(root/'preparation/previous_roi.nii.gz', old)
            self.save(root/'preparation/mask.nii.gz', valid)
            self.save(root/'maps/randomise-contrast3/cluster-t3.1/randomise_clustere_corrp_tstat1.nii.gz', valid*.96)
            rows = [dict(path=str(p.relative_to(root)), sha256=audit.cp.digest(p)) for p in root.rglob('*') if p.is_file()]
            audit.write(root/'inventory.tsv', rows)
            selected = audit.dmn_roi(repo, {})
            self.assertEqual(int(selected.get_fdata().sum()), 49)
            self.save(folder/'palm_vox_zvstat_c3.nii', valid*4)
            with self.assertRaisesRegex(ValueError, 'changed inventoried'):
                audit.dmn_roi(repo, {})

    def test_all_deletions_refit_and_include_rank_loss(self):
        x, y, c, subjects = self.crossed()
        rows = audit.deletion_checks(x, y, c, subjects)
        self.assertEqual(len(rows), 47+1081)
        target = next(r for r in rows if r['omitted_subjects'] == 'sub-0;sub-3')
        keep = np.ones(47, bool); keep[[0, 3]] = False
        self.assertAlmostEqual(target['estimate'], c@np.linalg.lstsq(x[keep], y[keep], rcond=None)[0])
        # Only the first two participants identify this nuisance column.
        x[:, 0] = 0; x[:2, 0] = [1, 2]
        rows = audit.deletion_checks(x, y, c, subjects)
        target = next(r for r in rows if r['omitted_subjects'] == 'sub-0;sub-1')
        self.assertEqual(target['full_rank'], 0); self.assertEqual(target['estimate'], '')

    def test_hc3_and_original_group_coding(self):
        x, y, c, subjects = self.crossed()
        summary, detail, deleted, _, _ = audit.influence_audit(x, y, c, subjects)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]; inverse = np.linalg.pinv(x)
        residual = y-x@beta; h = np.sum(x*inverse.T, axis=1)
        covariance = inverse@np.diag((residual/(1-h))**2)@inverse.T
        self.assertAlmostEqual(summary[0]['older_minus_younger'], c@beta)
        self.assertAlmostEqual(summary[0]['interaction_hc3_se'], np.sqrt(c@covariance@c))
        self.assertEqual(len(detail), 47); self.assertEqual(len(deleted), 1128)
        with self.assertRaisesRegex(ValueError, 'unexpected older'):
            audit.influence_audit(x, y, -c, subjects)

    def test_huber_limits_response_outlier_without_deletion(self):
        rng = np.random.default_rng(7)
        x = np.c_[np.ones(80), np.linspace(-2, 2, 80)]
        y = x@np.array([1., 2.])+rng.normal(0, .1, len(x)); y[-1] += 100
        ols = np.linalg.lstsq(x, y, rcond=None)[0]
        beta, weights, iterations = audit.huber_fit(x, y)
        self.assertLess(abs(beta[1]-2), abs(ols[1]-2))
        self.assertEqual(len(weights), 80); self.assertTrue(np.all(weights > 0))
        self.assertLess(weights[-1], .1); self.assertGreater(iterations, 0)

    def test_pooled_and_group_bars_keep_original_nuisance_adjustment(self):
        rng = np.random.default_rng(4); old = np.r_[np.zeros(25), np.ones(22)]
        x = rng.normal(size=(47, 6)); x[:, 1] = old-old.mean(); x[:, 5] = 1
        # Deliberately non-zero nuisance means: do not silently recenter.
        x[:, 0] += 2
        c = np.zeros((4, 6)); c[0, 5] = 1; c[1, 5] = -1; c[2, 1] = 1; c[3, 1] = -1
        y = x@np.arange(6)+rng.normal(size=47)
        rows, adjusted = audit.adjusted_sample_bars(dict(x=x.tolist(), c=c.tolist()), y)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        expected = y-x[:, [0, 2, 3, 4]]@beta[[0, 2, 3, 4]]
        np.testing.assert_allclose(adjusted, expected)
        self.assertEqual([r['n'] for r in rows], [47, 25, 22])
        self.assertAlmostEqual(rows[0]['conditional_participant_sem'], expected.std(ddof=1)/np.sqrt(47))
        self.assertAlmostEqual(rows[2]['adjusted_mean']-rows[1]['adjusted_mean'], beta[1])
        negative, _ = audit.adjusted_sample_bars(dict(x=x.tolist(), c=c.tolist()), -y)
        self.assertAlmostEqual(negative[0]['adjusted_mean'], -rows[0]['adjusted_mean'])

    def test_extraction_flags_variance_and_includes_real_zero_cope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image = root/'cope7.feat/stats/cope1.nii.gz'
            values = np.ones((2, 2, 2)); values[0, 0, 0] = 0
            self.save(image, values)
            variance = np.ones_like(values); variance[0, 0, 0] = 0
            self.save(image.with_name('varcope1.nii.gz'), variance)
            self.save(image.parent.parent/'mask.nii.gz', np.ones_like(values))
            model = dict(name='fixture', subjects=['sub-104'], images=[str(image)])
            roi = nib.Nifti1Image(np.ones_like(values), np.eye(4)); coverage = []
            result = audit.extract_checked(model, roi, {str(image): audit.cp.digest(image)}, {}, coverage)
            self.assertAlmostEqual(result[0], 7/8)
            self.assertEqual(coverage[0]['coverage_complete'], 0)
            self.assertEqual(coverage[0]['valid_variance_and_mask_voxels'], 7)
            with self.assertRaisesRegex(ValueError, 'COPE changed'):
                audit.extract_checked(model, roi, {str(image): 'bad'}, {}, [])

    def test_probability_sampling_and_xml_volume_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp)/'labels.xml'
            xml.write_text('<atlas><data><label index="0">Left Cerebral White Matter</label>'
                '<label index="1">Left Cerebral Cortex</label><label index="2">Left Lateral Ventricle</label>'
                '<label index="3">Brain-Stem</label></data></atlas>')
            labels = audit.atlas_labels(xml); categories = audit.tissue_categories(labels)
            self.assertEqual(categories[0], 'white_matter'); self.assertEqual(categories[2], 'ventricular_csf')
            data = np.zeros((3, 3, 3, 4)); data[:] = [10, 60, 20, 0]
            atlas = nib.Nifti1Image(data, np.eye(4))
            roi = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
            samples = audit.atlas_samples(atlas, roi, 4)
            np.testing.assert_allclose(samples, np.tile([10, 60, 20, 0], (8, 1)))
            self.assertAlmostEqual(samples.sum(1).mean(), 90)  # no renormalization
            with self.assertRaisesRegex(ValueError, 'volume/label'):
                audit.atlas_samples(atlas, roi, 3)

    def test_batch_preserves_failed_phase_and_runs_independent_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'outputs'
            with patch.object(sys, 'argv', ['audit', '--output-root', str(output)]), \
                 patch.object(audit, 'run_dmn', side_effect=RuntimeError('fixture failure')), \
                 patch.object(audit, 'run_effects') as effects, \
                 patch.object(audit, 'run_influence') as influence, \
                 patch.object(audit, 'run_localization') as localization, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit.main(), 1)
                effects.assert_called_once(); influence.assert_called_once(); localization.assert_called_once()
                status = json.loads((output/'status.json').read_text())
                self.assertFalse(status['complete']); self.assertEqual(status['exclusions'], 0)
                self.assertTrue((output/'dmn/inventory.tsv').is_file())
                with self.assertRaises(FileExistsError): audit.main()

    def supplemental_fixture(self, root):
        rng = np.random.default_rng(4); old = np.r_[np.zeros(25), np.ones(22)]
        x = rng.normal(size=(47, 6)); x[:, 1] = old-old.mean(); x[:, 5] = 1
        c = np.zeros((4, 6)); c[0, 5] = 1; c[1, 5] = -1; c[2, 1] = 1; c[3, 1] = -1
        images = []; hashes = {}
        for i in range(47):
            path = root/f'sub-{i}/cope7.feat/stats/cope1.nii.gz'
            self.save(path, np.full((2, 2, 2), float(x[i]@np.arange(6))))
            self.save(path.with_name('varcope1.nii.gz'), np.ones((2, 2, 2)))
            self.save(path.parent.parent/'mask.nii.gz', np.ones((2, 2, 2)))
            images.append(str(path)); hashes[str(path)] = audit.cp.digest(path)
        model = dict(name='fixture', subjects=[f'sub-{i}' for i in range(47)], images=images,
            x=x.tolist(), c=c.tolist(), source_hashes={str(root/'sub-0/cope7.feat/stats/cope1.nii.gz'): hashes[images[0]]})
        return model, hashes

    def test_full_supplemental_extraction_bars_sign_and_coverage_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model, hashes = self.supplemental_fixture(root)
            roi = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
            items = [dict(name=name, title=title, model=model, contrast=contrast, roi=roi, p=.02, hashes=hashes)
                for name, _, contrast, title in audit.SUPPLEMENTAL]
            output = root/'outputs'; output.mkdir()
            before = {p: audit.cp.digest(p) for p in root.rglob('*.nii.gz')}
            with patch.object(audit, 'supplemental_rois', return_value=items):
                audit.run_effects(root, output, {})
            self.assertEqual(len(list(output.glob('*_bars.png'))), 5)
            self.assertEqual(len(audit.read(output/'participant_estimates.tsv')), 5*47)
            self.assertEqual(len(audit.read(output/'coverage.tsv')), 5*47)
            rows = audit.read(output/'bar_summary.tsv')
            positive = next(r for r in rows if r['target'] == 'human-computer-mean' and r['age_group'] == 'all')
            negative = next(r for r in rows if r['target'] == 'computer-human-mean' and r['age_group'] == 'all')
            self.assertAlmostEqual(float(positive['adjusted_mean']), -float(negative['adjusted_mean']))
            for path, checksum in before.items(): self.assertEqual(audit.cp.digest(path), checksum)

    def test_full_influence_export_uses_original_not_canonical_column_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); x, y, contrast, subjects = self.crossed()
            permutation = [3, 1, 7, 5, 0, 6, 2, 4]
            models = {}; hashes = {}
            for k, timing in enumerate(audit.pf.TIMINGS, start=1):
                images = []
                for i, subject in enumerate(subjects):
                    for cope in (2, 4, 6, 7):
                        p = root/timing/subject/f'cope{cope}.feat/stats/cope1.nii.gz'
                        self.save(p, np.full((2, 2, 2), y[i]*k+cope))
                        self.save(p.with_name('varcope1.nii.gz'), np.ones((2, 2, 2)))
                        self.save(p.parent.parent/'mask.nii.gz', np.ones((2, 2, 2)))
                        if cope == 7:
                            images.append(str(p)); hashes[str(p)] = audit.cp.digest(p)
                name = f'crossed__{timing}_act_acceptance-intercept'
                models[name] = dict(name=name, images=images, subjects=subjects,
                    x=x[:, permutation].tolist(), c=[contrast[permutation].tolist()], source_hashes={})
            roi = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)); output = root/'outputs'; output.mkdir()
            with patch.object(audit, 'catalog', return_value=(None, models, hashes)), \
                 patch.object(audit, 'parietal_roi', return_value=roi), \
                 patch.object(audit, 'validate_activation_partners'), \
                 patch.object(audit.pf, 'aligned_design', return_value=(x, np.array([contrast]))):
                audit.run_influence(root, output, {})
            self.assertEqual(len(list(output.glob('*.png'))), 4)
            self.assertEqual(len(audit.read(output/'all_single_pair_deletions.tsv')), 3*(47+1081))
            self.assertEqual(len(audit.read(output/'coverage.tsv')), 3*47*4)
            slopes = audit.read(output/'slope_comparison.tsv')
            first = next(r for r in slopes if r['timing'] == 'full-rt' and r['method'] == 'OLS')
            self.assertAlmostEqual(float(first['older_minus_younger']), contrast@np.linalg.lstsq(x, y, rcond=None)[0])

    def test_localization_exports_atlas_probabilities_without_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); atlas = root/'data/atlases'; folder = atlas/'HarvardOxford'
            sub = np.zeros((3, 3, 3, 4)); sub[:] = [10, 60, 20, 0]
            self.save(folder/'HarvardOxford-sub-prob-2mm.nii.gz', sub)
            cortical = np.zeros((3, 3, 3, 2)); cortical[:] = [30, 10]
            self.save(folder/'HarvardOxford-cort-prob-2mm.nii.gz', cortical)
            (atlas/'HarvardOxford-Subcortical.xml').write_text('<atlas><data>'+
                ''.join(f'<label index="{i}">{name}</label>' for i, name in enumerate(
                    ['Left Cerebral White Matter', 'Left Cerebral Cortex', 'Left Lateral Ventricle', 'Brain-Stem']))+'</data></atlas>')
            (atlas/'HarvardOxford-Cortical.xml').write_text('<atlas><data><label index="0">Region A</label><label index="1">Region B</label></data></atlas>')
            standard = self.save(root/'standard.nii.gz', np.ones((3, 3, 3)))
            roi = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)); output = root/'outputs'; output.mkdir()
            item = dict(name='fixture', title='Synthetic localization', model=dict(name='fixture'), contrast=1, roi=roi, p=.02)
            with patch.object(audit, 'supplemental_rois', return_value=[item]), \
                 patch.object(audit, 'dmn_roi', return_value=roi), patch.object(audit, 'parietal_roi', return_value=roi):
                audit.run_localization(root, output, {}, root, standard)
            self.assertEqual(len(list(output.glob('*_localization.png'))), 3)
            rows = audit.read(output/'tissue_summary.tsv')
            gm = next(r for r in rows if r['target'] == 'fixture' and r['tissue'] == 'gray_matter')
            self.assertEqual(float(gm['mean_atlas_probability_percent']), 60.)
            self.assertEqual(float(gm['dominant_fraction']), 1.)
            self.assertEqual(len(audit.read(output/'voxel_tissue_probabilities.tsv')), 3*8)

    def test_dmn_49_export_preserves_distinct_masks_and_matched_decomposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original, _, _, subjects = self.crossed()
            x = np.c_[original[:, 4:6], original[:, :4]]
            c = np.zeros((4, 6)); c[0, 0] = 1; c[1, 1] = 1; c[2] = c[0]-c[1]; c[3] = -c[2]
            model = dict(name='full-rt__dmn-age', x=x.tolist(), c=c.tolist(), subjects=subjects,
                images=[f'/unused/{s}' for s in subjects], source_hashes={},
                nuisance={name: x[:, j].tolist() for j, name in enumerate(['ismale', 'tsnr', 'fd_mean', 'RT'], 2)})
            roi = np.zeros(64); roi[:49] = 1; roi = nib.Nifti1Image(roi.reshape((4, 4, 4)), np.eye(4))
            other = np.zeros(64); other[3:52] = .96
            self.save(root/'results/dmn_followup/v1/coverage_check/v1/maps/randomise-contrast3/cluster-t3.1/randomise_clustere_corrp_tstat1.nii.gz', other.reshape((4, 4, 4)))
            rng = np.random.default_rng(8); matched = rng.normal(size=(47, 4)); matched[:, 3] = matched[:, 1]-matched[:, 2]
            native = matched.copy(); native[:, :3] += .5
            checks = [dict(subject=s, verified=1) for s in subjects]
            output = root/'outputs'; output.mkdir()
            with patch.object(audit, 'dmn_roi', return_value=roi), \
                 patch.object(audit, 'catalog', return_value=(None, {model['name']: model}, {})), \
                 patch.object(audit.pm, 'extract_partners', return_value=(native, matched, checks)):
                audit.run_dmn(root, output, {})
            self.assertEqual(len(list(output.glob('*.png'))), 4)
            self.assertEqual(len(audit.read(output/'bar_summary.tsv')), 2*4*4*2)
            selection = json.loads((output/'selection.json').read_text())
            self.assertEqual(selection['overlap_voxels'], 46); self.assertEqual(selection['union_voxels'], 52)


if __name__ == '__main__':
    unittest.main()
