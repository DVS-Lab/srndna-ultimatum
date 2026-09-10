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

    def test_corrected_dmn_plot_data_has_expected_direction(self):
        path = ROOT / "results/manuscript/source_data/figure3_dmn_plot_data.tsv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 47)
        by_group = {
            group: [
                float(row["nuisance_adjusted_for_display"])
                for row in rows
                if row["age_group"] == group
            ]
            for group in ("younger", "older")
        }
        self.assertEqual(len(by_group["younger"]), 25)
        self.assertEqual(len(by_group["older"]), 22)
        younger_mean = sum(by_group["younger"]) / len(by_group["younger"])
        older_mean = sum(by_group["older"]) / len(by_group["older"])
        self.assertGreater(younger_mean, older_mean)


if __name__ == "__main__":
    unittest.main()
