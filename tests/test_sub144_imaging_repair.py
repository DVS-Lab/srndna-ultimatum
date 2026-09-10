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
    discover_confound,
    fmriprep_version,
    fsf_value,
    remap_recorded_path,
    render_l1,
)
from run_ultimatum_repair_jobs import complete, read_manifest, validate_job


class Sub144ImagingRepairTests(unittest.TestCase):
    def test_missing_derived_confound_is_rebuilt_from_fmriprep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            source = (
                dataset
                / "derivatives/fmriprep/sub-144/func"
                / "sub-144_task-ultimatum_run-1_desc-confounds_timeseries.tsv"
            )
            source.parent.mkdir(parents=True)
            source.write_text(
                "\t".join(
                    [
                        "cosine00",
                        "trans_x",
                        "trans_y",
                        "trans_z",
                        "rot_x",
                        "rot_y",
                        "rot_z",
                        *(f"a_comp_cor_{index:02d}" for index in range(6)),
                        "framewise_displacement",
                    ]
                )
                + "\n"
                + "\t".join(["0.1", *("0" for _ in range(12)), "n/a"])
                + "\n",
                encoding="utf-8",
            )
            destination = root / "work/confounds.tsv"
            actual = discover_confound(
                dataset,
                root / "production",
                "sub-144",
                "01",
                None,
                destination,
            )
            self.assertEqual(actual, destination.resolve())
            self.assertTrue(destination.is_file())
            self.assertEqual(len(destination.read_text().strip().split("\t")), 14)

    def test_fmriprep_version_comes_from_derivative_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            description = root / "derivatives/fmriprep/dataset_description.json"
            description.parent.mkdir(parents=True)
            description.write_text(
                '{"GeneratedBy": [{"Name": "fMRIPrep", "Version": "21.0.2"}]}',
                encoding="utf-8",
            )
            self.assertEqual(fmriprep_version(root), "21.0.2")

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
