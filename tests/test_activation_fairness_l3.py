import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "code/prepare_activation_fairness_l3.py"
    spec = importlib.util.spec_from_file_location("prepare_activation_fairness", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ActivationFairnessL3Tests(unittest.TestCase):
    def test_activation_input_changes_model_and_cope(self):
        module = load_module()
        source = (
            "/tmp/sub-101/L2_task-ultimatum_model-02_type-nppi-dmn_sm-6.gfeat/"
            "cope7.feat/stats/cope1.nii.gz"
        )
        result = module.activation_input(source, 4)
        self.assertIn("type-act_sm-6.gfeat/cope4.feat", result)
        self.assertNotIn("nppi-dmn", result)

    def test_preparer_writes_full_rank_intercept_models(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            work_root = Path(temporary) / "work"
            manifest = module.prepare(ROOT, work_root)
            self.assertTrue(manifest.is_file())
            for condition, cope in (("similar", 4), ("dissimilar", 6)):
                fsf = work_root / "fsf" / f"activation-offer-slope_{condition}.fsf"
                lines = fsf.read_text(encoding="utf-8").splitlines()
                inputs = module.parse_inputs(lines)
                self.assertEqual(len(inputs), 47)
                self.assertTrue(
                    all(
                        "type-act_sm-6.gfeat" in path
                        and f"/cope{cope}.feat/" in path
                        for _, _, path in inputs
                    )
                )
                sub144 = [path for _, subject, path in inputs if subject == "sub-144"]
                self.assertEqual(len(sub144), 1)
                self.assertIn("srndna-ultimatum-sub144-repair-v2", sub144[0])
                evs = module.parse_evs(lines)
                matrix = [
                    [evs[(row, column)] for column in range(1, 7)]
                    for row in range(1, 48)
                ]
                self.assertTrue(all(row[0] == 1.0 for row in matrix))
                self.assertAlmostEqual(sum(row[1] for row in matrix), 0.0)
                self.assertEqual(module.matrix_rank(matrix), 6)
                self.assertIn(
                    'set fmri(conname_real.1) "adjusted-mean-positive"', lines
                )
                self.assertIn('set fmri(con_real1.1) 1.0', lines)


if __name__ == "__main__":
    unittest.main()
