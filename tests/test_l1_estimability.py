from __future__ import annotations

import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
SPEC = importlib.util.spec_from_file_location('audit_l1_estimability', ROOT / 'code/audit_l1_estimability.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EstimabilityTests(unittest.TestCase):
    def test_empty_nuisance_does_not_invalidate_task_contrast(self):
        x = np.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        result, checks = MODULE.diagnostics(x, np.array([[1., -1., 0.], [0., 0., 1.]]))
        self.assertEqual(result['zero_columns'], [3])
        self.assertEqual(result['remaining_deficiency_after_zero_removal'], 0)
        self.assertEqual([r['estimable'] for r in checks], [1, 0])

    def test_duplicate_columns_sum_estimable_difference_not(self):
        x = np.array([[1., 1.], [2., 2.], [-1., -1.]])
        result, checks = MODULE.diagnostics(x, np.array([[1., 1.], [1., -1.]]))
        self.assertEqual(result['remaining_deficiency_after_zero_removal'], 1)
        self.assertEqual(result['nonzero_columns_in_nullspace'], [1, 2])
        self.assertEqual([r['estimable'] for r in checks], [1, 0])

    def test_scaling_contrast_is_essential(self):
        x = np.array([[1., 200.], [2., 400.], [-1., -200.]])
        _, checks = MODULE.diagnostics(x, np.array([[1., 200.], [1., 1.]]))
        self.assertEqual([r['estimable'] for r in checks], [1, 0])

    def test_constant_column_is_not_empty(self):
        result, checks = MODULE.diagnostics(np.array([[1., -1.], [1., 0.], [1., 1.]]), np.eye(2))
        self.assertEqual(result['zero_columns'], [])
        self.assertEqual(result['constant_nonzero_columns'], [1])
        self.assertTrue(all(r['estimable'] for r in checks))

    def test_near_dependency_is_only_sensitivity_flag(self):
        x = np.array([[1., 1.], [0., 1e-10], [-1., -1.]])
        result, checks = MODULE.diagnostics(x, np.array([[1., -1.]]))
        self.assertEqual(result['scaled_rank'], 2)
        self.assertEqual(checks[0]['estimable'], 1)
        self.assertEqual(checks[0]['sensitivity_estimable'], 0)

    def test_wide_matrix_and_zero_contrast(self):
        _, checks = MODULE.diagnostics(np.array([[1., 1., 0.], [0., 0., 1.]]), np.array([[1., 1., 0.], [0., 0., 0.]]))
        self.assertEqual([r['estimable'] for r in checks], [1, 0])

    def test_nonfinite_and_mismatched_inputs_fail(self):
        for x, c in [(np.array([[np.nan]]), np.ones((1, 1))), (np.eye(2), np.ones((1, 1)))]:
            with self.assertRaises(ValueError):
                MODULE.diagnostics(x, c)

    def test_confounds_padding_and_verified_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'design.mat').write_text('/Matrix\n1 0 1\n0 1 1\n1 1 0\n')
            (root / 'design.con').write_text('/Matrix\n1 -1\n')
            (root / 'design.fsf').write_text('set fmri(evs_orig) 2\nset fmri(evs_real) 2\nset fmri(evtitle1) "in"\nset fmri(evtitle2) "out"\n')
            result, checks = MODULE.inspect_design(root)
            self.assertEqual(result['contrast_padding_nuisance_zeros'], 1)
            self.assertEqual(result['column_labels'], '1:in[EV]|2:out[EV]|3:appended_nuisance')
            self.assertEqual(checks[0]['estimable'], 1)
            (root / 'design.con').write_text('/Matrix\n1\n')
            with self.assertRaises(ValueError):
                MODULE.inspect_design(root)

    def test_missing_inputs_never_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / 'sample.csv'
            sample.write_text('participant_id\nsub-144\n')
            self.assertEqual(MODULE.audit(root/'original', root/'repair', sample, root/'audit'), 2)

    def test_corrected_summary_substitutes_repaired_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / 'sample.csv'
            sample.write_text('participant_id\nsub-144\n')
            for source in ('original', 'repair'):
                for run in ('01', '02'):
                    for model in ('act', 'nppi-dmn', 'nppi-ecn'):
                        feat = root/source/'sub-144'/f'L1_task-ultimatum_model-02_type-{model}_run-{run}_sm-6.feat'
                        feat.mkdir(parents=True)
                        # Only the historical design has an empty second EV.
                        (feat/'design.mat').write_text('/Matrix\n1 0\n0 '+('0' if source == 'original' else '1')+'\n')
                        (feat/'design.con').write_text('/Matrix\n' + '0 1\n'*7)
                        (feat/'design.fsf').write_text('set fmri(evs_orig) 2\nset fmri(evs_real) 2\n')
            self.assertEqual(MODULE.audit(root/'original', root/'repair', sample, root/'audit'), 1)
            with (root/'audit/summary.tsv').open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertTrue(all(r['focal_nonestimable'] == '6' for r in rows if r['scope'] == 'production'))
            self.assertTrue(all(r['focal_nonestimable'] == '0' for r in rows if r['scope'] == 'corrected'))


if __name__ == '__main__':
    unittest.main()
