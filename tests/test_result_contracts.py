from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "reviewer" / "tables"


def rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class ResultContractTests(unittest.TestCase):
    def test_task_sample_contract(self) -> None:
        metrics = {row["metric"]: row["value"] for row in rows("task_event_summary.tsv")}
        self.assertEqual(metrics["participants"], "47")
        self.assertEqual(metrics["runs"], "94")
        self.assertEqual(metrics["trials"], "6768")
        self.assertEqual(metrics["missed_trials"], "113")

    def test_primary_model_contract(self) -> None:
        target = next(
            row
            for row in rows("primary_acceptance_models.tsv")
            if row["model_id"] == "primary_maximal"
            and row["term"] == "offer_c:age_groupolder:similaritysimilar"
        )
        self.assertEqual(target["n_participants"], "47")
        self.assertEqual(target["n_trials"], "4439")
        self.assertEqual(target["singular"], "FALSE")
        self.assertAlmostEqual(float(target["estimate"]), 0.06665, places=4)

        optimizers = rows("primary_optimizer_diagnostics.tsv")
        self.assertEqual(len(optimizers), 5)
        self.assertTrue(all(row["converged"] == "TRUE" for row in optimizers))
        self.assertTrue(all(row["singular"] == "FALSE" for row in optimizers))

    def test_focal_cluster_contract(self) -> None:
        clusters = {row["result_id"]: row for row in rows("focal_cluster_inventory.tsv")}
        self.assertEqual(clusters["dmn_age"]["nonzero_voxels"], "26")
        self.assertEqual(clusters["ecn_sensitivity"]["nonzero_voxels"], "23")
        self.assertEqual(clusters["dmn_age"]["voxel_z_mm"], "3.220000")

    def test_tracked_l3_designs_are_full_rank_with_unique_inputs(self) -> None:
        designs = rows("l3_design_summary.tsv")
        self.assertEqual(len(designs), 2)
        for design in designs:
            self.assertEqual(design["n_points"], "47")
            self.assertEqual(design["n_evs"], design["matrix_rank"])
            self.assertEqual(design["duplicate_input_count"], "0")


if __name__ == "__main__":
    unittest.main()
