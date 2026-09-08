from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_l1_designs", ROOT / "code/audit_l1_designs.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class L1DesignAuditTests(unittest.TestCase):
    @unittest.skipUnless(np is not None, "NumPy is not installed")
    def test_full_rank_matrix_diagnostics(self) -> None:
        matrix = np.asarray(
            [
                [1.0, -1.0, 0.0],
                [1.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        result = MODULE.matrix_diagnostics(matrix)
        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["rank_deficient"], 0)
        self.assertEqual(result["zero_variance_columns"], 1)
        self.assertLess(result["max_abs_column_correlation"], 0.01)

    @unittest.skipUnless(np is not None, "NumPy is not installed")
    def test_rank_deficiency_is_flagged(self) -> None:
        matrix = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        result = MODULE.matrix_diagnostics(matrix)
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["rank_deficient"], 1)
        self.assertGreater(result["max_abs_column_correlation"], 0.999)


if __name__ == "__main__":
    unittest.main()
