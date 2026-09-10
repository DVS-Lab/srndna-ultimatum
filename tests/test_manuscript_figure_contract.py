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

    def test_condition_paths_are_derived_from_difference_path(self):
        module = load_export_module()
        example = Path("/tmp/sub-101/example.gfeat/cope7.feat/stats/cope1.nii.gz")
        self.assertIn("/cope4.feat/", str(module.condition_input(example, 4)))
        self.assertIn("/cope6.feat/", str(module.condition_input(example, 6)))

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


if __name__ == "__main__":
    unittest.main()
