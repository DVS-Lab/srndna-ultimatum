from __future__ import annotations

import gzip
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_l3_production import audit, nifti_support, support_match


def write_nifti(path: Path, values: list[float]) -> None:
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, 2, 2, 1, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, 1, 2, 2, 2, 1, 1, 1, 1)
    struct.pack_into("<f", header, 108, 352)
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<12f", header, 280, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0)
    header[344:348] = b"n+1\x00"
    payload = struct.pack(f"<{len(values)}f", *values)
    with gzip.open(path, "wb") as stream:
        stream.write(header)
        stream.write(payload)


class L3ProductionAuditTests(unittest.TestCase):
    def test_binary_mask_matches_one_label_in_multicluster_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            focal_path = base / "focal.nii.gz"
            cluster_path = base / "cluster_mask_zstat1.nii.gz"
            write_nifti(focal_path, [0, 1, 1, 0])
            write_nifti(cluster_path, [0, 2, 2, 3])

            match = support_match(
                nifti_support(focal_path), nifti_support(cluster_path)
            )
            self.assertIsNotNone(match)
            assert match is not None
            self.assertEqual(match["match_method"], "exact_cluster_support")
            self.assertEqual(match["cluster_label"], 2)
            self.assertEqual(match["focal_voxels"], 2)
            self.assertEqual(match["cluster_voxels"], 2)
            self.assertEqual(match["dice"], "1.000000")

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
            self.assertEqual(row["match_method"], "exact_sha256")
            self.assertEqual(row["zstat_index"], "3")
            self.assertEqual(row["sub144_input_indices"], "2")
            self.assertEqual(row["source_copes"], "7")
            self.assertEqual(row["dlh"], "0.12")
            self.assertEqual(row["volume"], "100")
            self.assertEqual(row["resels"], "25")

    def test_support_mode_traces_binarized_cluster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            search = base / "fsl"
            gfeat = search / "L3_SANS/example.gfeat"
            cope = gfeat / "cope1.feat"
            (cope / "stats").mkdir(parents=True)
            mask = base / "paper-mask.nii.gz"
            cluster = cope / "cluster_mask_zstat2.nii.gz"
            write_nifti(mask, [0, 1, 1, 0])
            write_nifti(cluster, [0, 4, 4, 2])
            gfeat.joinpath("design.fsf").write_text(
                "set fmri(npts) 1\n"
                'set feat_files(1) "/x/sub-144/L2/cope7.feat/stats/cope1.nii.gz"\n',
                encoding="utf-8",
            )

            _, trace_rows = audit(
                [("working", search)],
                [("focal", mask)],
                base / "output",
                None,
                match_mode="support",
            )
            self.assertEqual(len(trace_rows), 1)
            row = trace_rows[0]
            self.assertEqual(row["exact_match_count"], 0)
            self.assertEqual(row["support_match_count"], 1)
            self.assertEqual(row["match_method"], "exact_cluster_support")
            self.assertEqual(row["cluster_label"], 4)
            self.assertEqual(row["zstat_index"], "2")


if __name__ == "__main__":
    unittest.main()
