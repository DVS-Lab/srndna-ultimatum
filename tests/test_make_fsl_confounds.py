import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from make_fsl_confounds import ACOMPCOR, MOTION, convert, selected_columns


class MakeFslConfoundsTests(unittest.TestCase):
    def test_converter_writes_historical_column_order_without_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "confounds.tsv"
            destination = root / "fsl.tsv"
            columns = [
                "cosine00",
                "non_steady_state_outlier00",
                *MOTION,
                *ACOMPCOR,
                "framewise_displacement",
                "unused",
            ]
            source.write_text(
                "\t".join(columns)
                + "\n"
                + "\t".join(
                    [
                        "0.1",
                        "1",
                        *("0" for _ in MOTION),
                        *("0.2" for _ in ACOMPCOR),
                        "n/a",
                        "99",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(convert(source, destination), 1)
            values = destination.read_text(encoding="utf-8").strip().split("\t")
            self.assertEqual(len(values), 15)
            self.assertEqual(values[-1], "0")
            self.assertNotIn("99", values)

    def test_missing_required_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required confound columns"):
            selected_columns(["trans_x"])


if __name__ == "__main__":
    unittest.main()
