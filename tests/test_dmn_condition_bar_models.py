import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "code/prepare_dmn_condition_bar_models.py"
    spec = importlib.util.spec_from_file_location("prepare_dmn_bars", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DmnConditionBarModelTests(unittest.TestCase):
    def test_condition_paths_replace_only_cope_directory(self):
        module = load_module()
        source = "/tmp/sub-101/model.gfeat/cope7.feat/stats/cope1.nii.gz"
        self.assertEqual(
            module.condition_input(source, 4),
            "/tmp/sub-101/model.gfeat/cope4.feat/stats/cope1.nii.gz",
        )
        self.assertEqual(
            module.condition_input(source, 6),
            "/tmp/sub-101/model.gfeat/cope6.feat/stats/cope1.nii.gz",
        )

    def test_preparer_writes_two_models_with_repaired_sub144(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            work_root = Path(temporary) / "work"
            manifest = module.prepare(ROOT, work_root)
            text = manifest.read_text(encoding="utf-8")
            self.assertIn("dmn-age_condition-similar", text)
            self.assertIn("dmn-age_condition-dissimilar", text)
            for condition, cope in (("similar", 4), ("dissimilar", 6)):
                fsf = work_root / "fsf" / f"dmn-age_condition-{condition}.fsf"
                inputs = module.parse_inputs(fsf.read_text().splitlines())
                self.assertEqual(len(inputs), 47)
                self.assertTrue(all(f"/cope{cope}.feat/" in path for _, _, path in inputs))
                sub144 = [path for _, subject, path in inputs if subject == "sub-144"]
                self.assertEqual(len(sub144), 1)
                self.assertIn("srndna-ultimatum-sub144-repair-v2", sub144[0])


if __name__ == "__main__":
    unittest.main()
