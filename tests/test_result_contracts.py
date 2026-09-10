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
        self.assertEqual(metrics["missed_trials"], "114")

    def test_primary_model_contract(self) -> None:
        target = next(
            row
            for row in rows("primary_acceptance_models.tsv")
            if row["model_id"] == "primary_maximal"
            and row["term"] == "offer_c:age_groupolder:similaritysimilar"
        )
        self.assertEqual(target["n_participants"], "47")
        self.assertEqual(target["n_trials"], "4438")
        self.assertEqual(target["singular"], "FALSE")
        self.assertAlmostEqual(float(target["estimate"]), 0.11358, places=4)

        optimizers = rows("primary_optimizer_diagnostics.tsv")
        self.assertEqual(len(optimizers), 5)
        converged = [row for row in optimizers if row["converged"] == "TRUE"]
        self.assertEqual(len(converged), 3)
        self.assertTrue(all(row["singular"] == "FALSE" for row in optimizers))
        self.assertTrue(
            all(abs(float(row["estimate"]) - float(target["estimate"])) < 1e-4 for row in converged)
        )

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

    def test_corrected_l3_covariates_are_complete_and_centered(self) -> None:
        covariates = {
            row["covariate"]: row
            for row in rows("l3_covariate_correction_summary.tsv")
        }
        self.assertEqual(
            set(covariates), {"fairness_sensitivity", "fairness_norm_proxy"}
        )
        for row in covariates.values():
            self.assertEqual(row["participants_changed_gt_1e_8"], "47")
            self.assertGreater(
                float(row["production_historical_refit_correlation"]), 0.999999
            )
            self.assertLess(
                float(row["production_historical_refit_max_abs_difference"]),
                2e-5,
            )
            self.assertGreater(float(row["submitted_corrected_correlation"]), 0.98)

        diagnostics = rows("l3_covariate_model_diagnostics.tsv")
        self.assertEqual(len(diagnostics), 8)
        self.assertTrue(all(row["singular"] == "FALSE" for row in diagnostics))
        self.assertTrue(all(row["convergence_message"] == "NA" for row in diagnostics))

        l3_rows = rows("l3_event_corrected_covariates.tsv")
        self.assertEqual(len(l3_rows), 47)
        self.assertEqual(
            set(l3_rows[0]),
            {
                "subjID",
                "corrected_sensitivity_young",
                "corrected_sensitivity_old",
                "corrected_norm_young",
                "corrected_norm_old",
                "corrected_mean_rt_z",
            },
        )
        sensitivity_sum = sum(
            float(row["corrected_sensitivity_young"])
            + float(row["corrected_sensitivity_old"])
            for row in l3_rows
        )
        norm_sum = sum(
            float(row["corrected_norm_young"])
            + float(row["corrected_norm_old"])
            for row in l3_rows
        )
        mean_rt_z = [float(row["corrected_mean_rt_z"]) for row in l3_rows]
        self.assertAlmostEqual(sensitivity_sum, 0.0, places=10)
        self.assertAlmostEqual(norm_sum, 0.0, places=10)
        self.assertAlmostEqual(sum(mean_rt_z), 0.0, places=10)
        self.assertAlmostEqual(sum(value * value for value in mean_rt_z), 46.0, places=8)


if __name__ == "__main__":
    unittest.main()
