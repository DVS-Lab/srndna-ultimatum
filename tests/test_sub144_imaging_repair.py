from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from make_ultimatum_3col import generate
from prepare_sub144_imaging_repair import (
    fsf_value,
    remap_recorded_path,
    render_l1,
)
from run_ultimatum_repair_jobs import complete, read_manifest, validate_job


class Sub144ImagingRepairTests(unittest.TestCase):
    def test_rendered_activation_uses_corrected_companion_evs(self) -> None:
        events = (
            ROOT
            / "source_data/bids/sub-144/func"
            / "sub-144_task-ultimatum_run-01_events.tsv"
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bold = base / "bold.nii.gz"
            confound = base / "confounds.tsv"
            bold.write_bytes(b"bold")
            confound.write_text("0\n", encoding="utf-8")
            prefix = base / "ev" / "run-01"
            counts = generate(events, prefix, "companion")
            destination = base / "rendered.fsf"
            render_l1(
                ROOT / "templates/L1_task-ultimatum_model-02_type-act.fsf",
                destination,
                base / "output",
                bold,
                confound,
                prefix,
                counts,
                base / "production",
                base / "dataset",
            )
            text = destination.read_text(encoding="utf-8")
            self.assertEqual(fsf_value(text, "shape7"), "3")
            self.assertEqual(fsf_value(text, "feat_files(1)"), str(bold))
            self.assertEqual(fsf_value(text, "confoundev_files(1)"), str(confound))
            self.assertEqual(fsf_value(text, "custom8"), f"{prefix}_event_RT.txt")
            self.assertEqual(counts["event_RT"], 63)
            self.assertEqual(counts["missed_trial"], 1)

    def test_recorded_mounts_are_remapped(self) -> None:
        self.assertEqual(
            remap_recorded_path(
                "/data/projects/srndna-data/derivatives/file.tsv",
                Path("/production"),
                Path("/dataset"),
            ),
            Path("/dataset/derivatives/file.tsv"),
        )

    def test_runner_refuses_existing_incomplete_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fsf = base / "job.fsf"
            source = base / "source.txt"
            output = base / "output.feat"
            fsf.write_text("fsf\n", encoding="utf-8")
            source.write_text("input\n", encoding="utf-8")
            output.mkdir()
            row = {
                "stage": "l1",
                "model": "act",
                "run": "01",
                "fsf": str(fsf),
                "output": str(output),
                "inputs": str(source),
            }
            with self.assertRaises(FileExistsError):
                validate_job(row, resume=False)
            with self.assertRaises(FileExistsError):
                validate_job(row, resume=True)
            self.assertFalse(complete(row))

    def test_manifest_stage_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.tsv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("stage", "model", "run", "fsf", "output", "inputs"),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {"stage": "l1", "model": "act", "run": "01", "fsf": "a", "output": "b", "inputs": "c"}
                )
                writer.writerow(
                    {"stage": "l2", "model": "act", "run": "", "fsf": "d", "output": "e", "inputs": "f"}
                )
            self.assertEqual(len(read_manifest(path, "l2")), 1)


if __name__ == "__main__":
    unittest.main()
