import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from phase_design_diagnostics import diagnose_design, pad_contrasts


class PhaseDesignDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        # Exactly centered orthonormal columns, avoiding random tolerances.
        self.u = np.array([1., -1., 1., -1.]) / 2
        self.v = np.array([1., 1., -1., -1.]) / 2

    def test_known_correlated_difference_and_sum(self):
        rho = .75
        x = np.column_stack([self.u, rho * self.u + np.sqrt(1 - rho**2) * self.v])
        checks = diagnose_design(x, [[1, -1], [1, 1], [1, 0]])["contrasts"]
        self.assertAlmostEqual(checks[0]["contrast_vif"], 1 / (1 - rho))
        self.assertAlmostEqual(checks[1]["contrast_vif"], 1 / (1 + rho))
        self.assertLess(checks[1]["contrast_vif"], 1)
        self.assertAlmostEqual(checks[2]["contrast_vif"], 1 / (1 - rho**2))

    def test_empty_nuisance_and_intercept_are_not_rank_failures(self):
        x = np.column_stack([self.u + 4, self.v - 2, np.zeros(4), np.ones(4)])
        result = diagnose_design(x, [1, -1])
        self.assertEqual(result["design"]["zero_columns"], [3])
        self.assertEqual(result["design"]["constant_nonzero_columns"], [4])
        self.assertEqual(result["design"]["contrast_padding_nuisance_zeros"], 2)
        self.assertTrue(result["design"]["full_rank_after_constant_removal"])
        self.assertAlmostEqual(result["contrasts"][0]["contrast_vif"], 1)
        self.assertAlmostEqual(result["contrasts"][0]["iid_contrast_variance"], 2)
        json.dumps(result, allow_nan=False)

    def test_empty_focal_is_not_estimable(self):
        result = diagnose_design(np.column_stack([self.u, np.zeros(4)]), [0, 1])
        self.assertEqual(result["contrasts"][0]["status"], "nonestimable_empty_column")
        self.assertFalse(result["contrasts"][0]["estimable"])

    def test_rank_deficiency_tests_contrast_not_just_rank(self):
        result = diagnose_design(np.column_stack([self.u, self.u]), [[1, -1], [1, 1]])
        self.assertEqual(result["design"]["remaining_rank_deficiency"], 1)
        self.assertEqual(result["contrasts"][0]["status"], "nonestimable_rank_deficiency")
        self.assertTrue(result["contrasts"][1]["estimable"])
        self.assertAlmostEqual(result["contrasts"][1]["iid_contrast_variance"], 1)
        self.assertIsNone(result["contrasts"][1]["upstream_unit_norm_contrast_vif"])

    def test_physical_contrast_is_invariant_to_reparameterization(self):
        x = np.column_stack([self.u, .6 * self.u + .8 * self.v])
        c = np.array([1., -1.])
        scale = np.array([2., 7.])
        original = diagnose_design(x, c)["contrasts"][0]
        # X*scale has beta/scale, so c*scale preserves c beta.
        rescaled = diagnose_design(x * scale, c * scale)["contrasts"][0]
        for field in ["iid_contrast_variance", "iid_orthogonal_reference_variance",
                      "actual_basis_variance_ratio", "iid_efficiency"]:
            self.assertAlmostEqual(original[field], rescaled[field])
        self.assertNotAlmostEqual(original["upstream_unit_norm_contrast_vif"],
                                  rescaled["upstream_unit_norm_contrast_vif"])

    def test_matches_original_units_gram_formula(self):
        x = np.column_stack([self.u * 3, (.4 * self.u + np.sqrt(.84) * self.v) * 7])
        c = np.array([2., -1.])
        actual = diagnose_design(x, c)["contrasts"][0]
        gram = x.T @ x
        variance = c @ np.linalg.inv(gram) @ c
        ideal = c @ np.linalg.inv(np.diag(np.diag(gram))) @ c
        self.assertAlmostEqual(actual["iid_contrast_variance"], variance)
        self.assertAlmostEqual(actual["contrast_vif"], variance / ideal)

    def test_orthogonalized_basis_not_claimed_as_original_stimulus_cvif(self):
        result = diagnose_design(np.column_stack([self.u, self.v]), [1, -1],
                                 orthogonalized=True)
        self.assertIsNone(result["contrasts"][0]["contrast_vif"])
        self.assertIsNone(result["contrasts"][0]["upstream_unit_norm_contrast_vif"])
        self.assertAlmostEqual(result["contrasts"][0]["actual_basis_variance_ratio"], 1)
        self.assertIn("not_original_stimulus", result["design"]["vif_interpretation"])

    def test_unsupported_intercept_and_zero_contrasts(self):
        result = diagnose_design(np.column_stack([self.u, np.ones(4)]), [[0, 1], [0, 0]])
        self.assertEqual(result["contrasts"][0]["status"],
                         "unsupported_constant_contrast_after_centering")
        self.assertEqual(result["contrasts"][1]["status"], "zero_contrast")
        result = diagnose_design(np.zeros((4, 2)), [1, 0])
        self.assertFalse(result["contrasts"][0]["estimable"])
        json.dumps(result, allow_nan=False)

    def test_invalid_inputs_fail_clearly(self):
        with self.assertRaises(ValueError):
            pad_contrasts([1, 2, 3], 2)
        with self.assertRaises(ValueError):
            diagnose_design([[1, 0], [2, float("nan")]], [1, 0])
        with self.assertRaises(ValueError):
            diagnose_design([[1, 0], [2, 1]], [float("inf"), 0])
        with self.assertRaises(ValueError):
            diagnose_design([[1, 0]], [1, 0])
        with self.assertRaises(ValueError):
            diagnose_design([[1, 0], [2, 1]], [1, 0], rcond=0)


if __name__ == "__main__":
    unittest.main()
