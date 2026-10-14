from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import run_crossed_behavior_l3 as crossed
from prepare_sub144_imaging_repair import fsf_value
from audit_l1_designs import read_vest_matrix


class CrossedBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans, cls.provenance = crossed.build_plan(
            ROOT, list(crossed.REPORTS), Path('/tmp/crossed-design-test'), Path('/tmp/standard.nii.gz'))

    def test_complete_cross_and_calibrations(self):
        self.assertEqual(len(self.plans), 18)
        self.assertEqual(sum(j['calibration'] for j, _ in self.plans), 6)
        self.assertEqual(len({j['run'] for j, _ in self.plans}), 18)
        for timing in crossed.REPORTS:
            self.assertEqual({(j['model'], j['behavior']) for j, _ in self.plans if j['timing'] == timing},
                             {(f, m) for f in crossed.FAMILIES for m in crossed.METRICS})

    def test_rank_sample_contrasts_and_copecoverage(self):
        for job, text in self.plans:
            with self.subTest(model=job['run']):
                design = crossed.matrix(text)
                self.assertEqual(design.shape, (47, 8))
                self.assertEqual(np.linalg.matrix_rank(design), 8)
                np.testing.assert_equal(design[:, 4:6].sum(axis=0), [25, 22])
                self.assertEqual(len(crossed.members(text)), 47)
                self.assertIn('sub-143', [s for _, s, _ in crossed.members(text)])
                self.assertIn('sub-144', [s for _, s, _ in crossed.members(text)])
                for _, _, name in crossed.members(text):
                    self.assertTrue(name.endswith('/L2.gfeat/cope7.feat/stats/cope1.nii.gz'))
                    self.assertIn('/' + job['model'] + '/', name)
                for i, (_, weights) in enumerate(crossed.CONTRASTS, 1):
                    self.assertEqual([float(fsf_value(text, f'con_real{i}.{c}')) for c in range(1, 9)], weights)

    def test_no_silent_nuisance_or_behavior_changes(self):
        for job, text in self.plans:
            retained = (ROOT / crossed.REPORTS[job['timing']] /
                        crossed.FAMILIES[job['model']] / 'design/design.fsf').read_text()
            old = crossed.matrix(retained)
            nuisance = old[:, [2, 3, 4, 5, 0, 1]] if job['model'] == 'nppi-dmn' else old[:, :6]
            np.testing.assert_allclose(crossed.matrix(text)[:, :6], nuisance, atol=1e-12, rtol=0)
            if job['calibration']:
                np.testing.assert_allclose(crossed.matrix(text), old, atol=1e-8, rtol=0)
        # Same behavioral values in all three imaging families, without claiming
        # that their retained tSNR/motion values are identical.
        for metric in crossed.METRICS:
            matrices = [crossed.matrix(t) for j, t in self.plans if j['behavior'] == metric]
            for m in matrices[1:]:
                np.testing.assert_equal(m[:, 6:], matrices[0][:, 6:])
        dmn = [j for j, _ in self.plans if j['model'] == 'nppi-dmn']
        self.assertTrue(all(j['tsnr_max_abs_vs_ecn'] > 0 for j in dmn))

    def test_duplicate_or_changed_behavior_rejected(self):
        original = crossed.read_rows
        for mode in ('duplicate', 'changed'):
            def read(path):
                rows = original(path)
                if path == ROOT / crossed.COVARIATES:
                    if mode == 'duplicate':
                        rows[-1] = dict(rows[0])
                    else:
                        rows[0]['corrected_sensitivity_young'] = '999'
                return rows
            with patch.object(crossed, 'read_rows', side_effect=read), self.assertRaises(ValueError):
                crossed.build_plan(ROOT, ['partner-post'], Path('/tmp/test'), Path('/tmp/standard'))

    def test_source_inventory_hash_guard(self):
        original = crossed.digest
        def altered(path):
            return 'changed' if path.name == 'design.fsf' else original(path)
        with patch.object(crossed, 'digest', side_effect=altered), self.assertRaisesRegex(ValueError, 'inventory'):
            crossed.build_plan(ROOT, ['partner-post'], Path('/tmp/test'), Path('/tmp/standard'))

    def test_prepare_resume_and_manifest_tamper_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'scratch'
            standard = Path(tmp) / 'standard.nii.gz'
            standard.write_bytes(b'fixture')
            with patch.object(crossed, 'check_images', return_value={}):
                manifest = crossed.prepare(ROOT, ['partner-post'], root, standard)
                self.assertEqual(manifest, crossed.prepare(ROOT, ['partner-post'], root, standard))
                self.assertEqual(json.loads((root / 'preparation_config.json').read_text())['jobs'], 6)
                manifest.write_text(manifest.read_text().replace('nppi-dmn', 'nppi-UNKNOWN', 1))
                with self.assertRaisesRegex(ValueError, 'manifest changed'):
                    crossed.prepare(ROOT, ['partner-post'], root, standard)

    def test_missing_images_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'scratch'
            with patch.object(crossed, 'check_images', side_effect=FileNotFoundError('missing COPE')):
                with self.assertRaises(FileNotFoundError):
                    crossed.prepare(ROOT, ['partner-post'], root, Path('/tmp/standard'))
            self.assertFalse(root.exists())

    @unittest.skipUnless(os.environ.get('SRNDNA_TEST_FEAT_MODEL'), 'optional FSL compilation integration test')
    def test_actual_feat_model_compiles_all_18_designs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for job, text in self.plans:
                root = Path(tmp) / job['run']
                root.with_suffix('.fsf').write_text(text)
                result = subprocess.run([os.environ['SRNDNA_TEST_FEAT_MODEL'], str(root)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                np.testing.assert_allclose(read_vest_matrix(root.with_suffix('.mat')),
                                           crossed.matrix(text), atol=1e-5, rtol=0)
                np.testing.assert_equal(read_vest_matrix(root.with_suffix('.con')),
                                        [weights for _, weights in crossed.CONTRASTS])
                np.testing.assert_equal(read_vest_matrix(root.with_suffix('.grp')), np.ones((47, 1)))


if __name__ == '__main__':
    unittest.main()
