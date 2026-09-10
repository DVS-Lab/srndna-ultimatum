from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_prepared_ultimatum_designs import block_difference


class PreparedDesignAuditTests(unittest.TestCase):
    def test_block_difference_selects_only_requested_columns(self) -> None:
        production = [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]]
        candidate = [[0.0, 2.0, 10.0], [2.0, 4.0, 20.0]]
        task_max, task_rmse = block_difference(production, candidate, 0, 2)
        tail_max, tail_rmse = block_difference(production, candidate, 2, 3)
        self.assertEqual(task_max, 1.0)
        self.assertAlmostEqual(task_rmse, 2 ** -0.5)
        self.assertEqual((tail_max, tail_rmse), (0.0, 0.0))

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "row counts differ"):
            block_difference([[1.0]], [[1.0], [2.0]], 0, 1)
        with self.assertRaisesRegex(ValueError, "column counts differ"):
            block_difference([[1.0]], [[1.0, 2.0]], 0, 1)


if __name__ == "__main__":
    unittest.main()
