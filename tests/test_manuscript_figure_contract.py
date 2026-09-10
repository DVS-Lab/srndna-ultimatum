import csv
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_export_module():
    path = ROOT / "code/export_corrected_dmn_figure_data.py"
    spec = importlib.util.spec_from_file_location("export_corrected_dmn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ManuscriptFigureContractTests(unittest.TestCase):
    def test_corrected_dmn_design_has_expected_order_and_repaired_sub144(self):
        module = load_export_module()
        design = ROOT / module.DESIGN_RELATIVE
        inputs = module.parse_feat_inputs(design)
        participants = [module.participant_from_path(path) for path in inputs]
        self.assertEqual(len(inputs), 47)
        self.assertEqual(len(set(participants)), 47)
        self.assertEqual(participants[33:36], ["sub-143", "sub-144", "sub-147"])
        self.assertIn("srndna-ultimatum-sub144-repair-v2", str(inputs[34]))
        self.assertTrue(
            all("cope7.feat/stats/cope1.nii.gz" in str(path) for path in inputs)
        )

    def test_participant_is_parsed_from_difference_path(self):
        module = load_export_module()
        example = Path("/tmp/sub-101/example.gfeat/cope7.feat/stats/cope1.nii.gz")
        self.assertEqual(module.participant_from_path(example), "sub-101")

    def test_corrected_dmn_matrix_is_47_by_6(self):
        path = ROOT / (
            "results/reviewer/l3_repair_designs/dmn-age/"
            "reported-covariates-corrected/design.mat"
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        matrix = [
            line.split()
            for line in lines[lines.index("/Matrix") + 1 :]
            if line.strip()
        ]
        self.assertEqual(len(matrix), 47)
        self.assertTrue(all(len(row) == 6 for row in matrix))

    def test_final_result_set_has_only_supported_revision_dispositions(self):
        path = ROOT / "results/manuscript/tables/final_result_set.tsv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = {
                row["result_id"]: row
                for row in csv.DictReader(stream, delimiter="\t")
            }
        self.assertIn("Retain", rows["dmn_age_similarity"]["revision_disposition"])
        self.assertIn("Remove", rows["ecn_fairness_sensitivity"]["revision_disposition"])
        self.assertIn("do not add", rows["activation_norm_proxy"]["revision_disposition"])

    def test_corrected_dmn_bar_data_has_complete_age_by_partner_cells(self):
        path = ROOT / "results/manuscript/source_data/figure3_dmn_flame_bar_summary.tsv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["age_group"], row["condition"]) for row in rows},
            {
                ("younger", "similar"),
                ("younger", "dissimilar"),
                ("older", "similar"),
                ("older", "dissimilar"),
            },
        )
        for row in rows:
            estimate = float(row["flame_cluster_mean_estimate"])
            standard_error = float(row["mean_voxelwise_standard_error"])
            self.assertGreater(standard_error, 0)
            self.assertLess(float(row["display_conf_low"]), estimate)
            self.assertGreater(float(row["display_conf_high"]), estimate)

    def test_corrected_dmn_condition_inputs_have_positive_varcopes(self):
        path = ROOT / "results/manuscript/source_data/figure3_dmn_condition_input_roi.tsv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 94)
        for condition in ("similar", "dissimilar"):
            condition_rows = [row for row in rows if row["condition"] == condition]
            self.assertEqual(len(condition_rows), 47)
            self.assertEqual(len({row["participant"] for row in condition_rows}), 47)
            self.assertEqual(
                sorted(int(row["design_index"]) for row in condition_rows),
                list(range(1, 48)),
            )
        self.assertTrue(
            all(float(row["cluster_mean_varcope"]) > 0 for row in rows)
        )
        self.assertTrue(
            all(float(row["inverse_varcope_weight"]) > 0 for row in rows)
        )

    def test_corrected_dmn_bar_pattern_matches_retained_interaction(self):
        path = ROOT / "results/manuscript/source_data/figure3_dmn_flame_bar_summary.tsv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        estimates = {
            (row["age_group"], row["condition"]): float(
                row["flame_cluster_mean_estimate"]
            )
            for row in rows
        }
        younger_difference = estimates[("younger", "similar")] - estimates[
            ("younger", "dissimilar")
        ]
        older_difference = estimates[("older", "similar")] - estimates[
            ("older", "dissimilar")
        ]
        self.assertGreater(younger_difference, older_difference)


if __name__ == "__main__":
    unittest.main()
