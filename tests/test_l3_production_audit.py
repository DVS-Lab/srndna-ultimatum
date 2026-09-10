from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_l3_production import audit


class L3ProductionAuditTests(unittest.TestCase):
    def test_exact_mask_match_traces_design_and_smoothness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            search = base / "fsl"
            gfeat = search / "L3_SANS/example.gfeat"
            cope = gfeat / "cope1.feat"
            (cope / "stats").mkdir(parents=True)
            mask = base / "paper-mask.nii.gz"
            mask.write_bytes(b"mask")
            (cope / "cluster_mask_zstat3.nii.gz").write_bytes(b"mask")
            (cope / "cluster_zstat3.txt").write_text("cluster\n", encoding="utf-8")
            (cope / "stats/smoothness").write_text(
                "DLH 0.12\nVOLUME 100\nRESELS 25\n", encoding="utf-8"
            )
            gfeat.joinpath("design.fsf").write_text(
                "\n".join(
                    [
                        "set fmri(npts) 2",
                        "set fmri(multiple) 2",
                        "set fmri(mixed_yn) 3",
                        "set fmri(z_thresh) 3.1",
                        "set fmri(prob_thresh) 0.05",
                        'set feat_files(1) "/x/sub-143/L2/cope7.feat/stats/cope1.nii.gz"',
                        'set feat_files(2) "/x/sub-144/L2/cope7.feat/stats/cope1.nii.gz"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            for name in ("design.mat", "design.con", "design.grp"):
                gfeat.joinpath(name).write_text(name, encoding="utf-8")

            group_rows, trace_rows = audit(
                [("working", search)],
                [("focal", mask)],
                base / "output",
                base / "tracked.tsv",
            )
            self.assertEqual(len(group_rows), 1)
            self.assertEqual(len(trace_rows), 1)
            row = trace_rows[0]
            self.assertEqual(row["exact_match_count"], 1)
            self.assertEqual(row["zstat_index"], "3")
            self.assertEqual(row["sub144_input_indices"], "2")
            self.assertEqual(row["source_copes"], "7")
            self.assertEqual(row["dlh"], "0.12")
            self.assertEqual(row["volume"], "100")
            self.assertEqual(row["resels"], "25")


if __name__ == "__main__":
    unittest.main()
