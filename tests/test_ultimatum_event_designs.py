from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_ultimatum_event_designs import matrix_metrics, parse_revision


class UltimatumEventDesignAuditTests(unittest.TestCase):
    def test_identical_first_nine_columns_are_exact_match(self) -> None:
        matrix = [[float(row * 9 + column) for column in range(9)] for row in range(1, 5)]
        production = [row + [100.0, 200.0] for row in matrix]
        result = matrix_metrics(production, matrix)
        self.assertEqual(result["max_absolute_difference"], 0)
        self.assertEqual(result["relative_rmse"], 0)
        self.assertAlmostEqual(result["minimum_column_correlation"], 1)

    def test_difference_is_detected(self) -> None:
        production = [[float(row + column) for column in range(9)] for row in range(4)]
        candidate = [row.copy() for row in production]
        candidate[0][3] += 2
        result = matrix_metrics(production, candidate)
        self.assertGreater(result["relative_rmse"], 0)
        self.assertGreater(result["max_absolute_difference"], 0)

    def test_revision_parser(self) -> None:
        self.assertEqual(parse_revision("corrected=WORKTREE"), ("corrected", "WORKTREE"))
        with self.assertRaises(ValueError):
            parse_revision("unsafe label=HEAD")


if __name__ == "__main__":
    unittest.main()
