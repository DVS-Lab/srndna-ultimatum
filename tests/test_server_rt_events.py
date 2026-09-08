from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_server_rt_events import audit, correlation, parse_path_mappings, parse_vest_matrix, remap_path


class ServerRTEventAuditTests(unittest.TestCase):
    def test_stale_fsf_path_can_be_remapped_without_editing_fsf(self) -> None:
        mappings = parse_path_mappings(["/data/projects/old=/ZPOOL/data/projects/old"])
        self.assertEqual(
            remap_path(Path("/data/projects/old/derivatives/file.txt"), mappings),
            Path("/ZPOOL/data/projects/old/derivatives/file.txt"),
        )
        self.assertEqual(remap_path(Path("/other/file.txt"), mappings), Path("/other/file.txt"))

    def test_vest_parser_and_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "design.mat"
            path.write_text("/NumWaves 2\n/NumPoints 3\n/Matrix\n1 3\n2 2\n3 1\n", encoding="utf-8")
            matrix = parse_vest_matrix(path)
            self.assertAlmostEqual(correlation(matrix, 0, 1), -1.0)

    def test_audit_follows_rendered_paths_and_retained_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            participant = "sub-999"
            sample = base / "sample.csv"
            sample.write_text("subjID,younger,older\nsub-999,1,0\n", encoding="utf-8")
            bids_func = base / "bids" / participant / "func"
            bids_func.mkdir(parents=True)
            ev_dir = base / "ev" / participant / "ultimatum-rt"
            ev_dir.mkdir(parents=True)
            l1_root = base / "l1"
            for run in ("01", "02"):
                event_path = bids_func / f"{participant}_task-ultimatum_run-{run}_events.tsv"
                event_path.write_text(
                    "onset\tduration\ttrial_type\tresponse_time\tOffer\n"
                    "1\t3.5\tevent_accept_ingroup\t1.2\t7\n"
                    "5\t3.5\tevent_reject_outgroup\t1.1\t2\n"
                    "5\t0\tevent_RT\t1.1\t2\n",
                    encoding="utf-8",
                )
                rt = ev_dir / f"run-{run}_event_RT.txt"
                rt_pmod = ev_dir / f"run-{run}_event_RT_pmod.txt"
                rt.write_text("5 0 1\n", encoding="utf-8")
                rt_pmod.write_text("5 0 1.1\n", encoding="utf-8")
                feat = l1_root / participant / f"L1_task-ultimatum_model-02_type-act_run-{run}_sm-6.feat"
                feat.mkdir(parents=True)
                main_paths = []
                for index in (1, 3, 5):
                    main = base / f"main-{run}-{index}.txt"
                    main.write_text("1 3.5 1\n", encoding="utf-8")
                    main_paths.append((index, main))
                fsf_lines = [f'set fmri(custom{index}) "{path}"' for index, path in main_paths]
                fsf_lines.extend((f'set fmri(custom8) "{rt}"', f'set fmri(custom9) "{rt_pmod}"'))
                (feat / "design.fsf").write_text("\n".join(fsf_lines) + "\n", encoding="utf-8")
                matrix_rows = [" ".join(str((row + 1) * (column + 1)) for column in range(9)) for row in range(4)]
                (feat / "design.mat").write_text(
                    "/NumWaves 9\n/NumPoints 4\n/Matrix\n" + "\n".join(matrix_rows) + "\n",
                    encoding="utf-8",
                )
            output = base / "output"
            tracked_summary = base / "tracked" / "summary.tsv"
            result = audit(base / "bids", base / "ev", l1_root, sample, output, tracked_summary)
            self.assertEqual(result["analysis_sample_runs_expected"], 2)
            self.assertEqual(result["rt_files_found_now"], 2)
            with (output / "rt_production_by_run.tsv").open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(rows[0]["source_responded_trials"], "2")
            self.assertEqual(rows[0]["rt_file_rows_now"], "1")
            with tracked_summary.open(newline="") as stream:
                summary = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(summary[0]["metric"], "analysis_sample_runs_expected")
            self.assertNotIn("participant", summary[0])
            self.assertNotIn("rt_file_path_from_fsf", summary[0])


if __name__ == "__main__":
    unittest.main()
