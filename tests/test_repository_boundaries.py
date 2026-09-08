from __future__ import annotations

import subprocess
import unittest
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


class RepositoryBoundaryTests(unittest.TestCase):
    def test_full_bids_tree_is_not_tracked(self) -> None:
        paths = tracked_paths()
        self.assertFalse(any(path.startswith("bids/") for path in paths))

    def test_unrelated_computational_models_are_not_tracked(self) -> None:
        paths = tracked_paths()
        excluded = (
            "behavioral_analyses/codes/computation/",
            "behavioral_analyses/codes/RLtutorial_codeNdata/",
        )
        self.assertFalse(any(path.startswith(excluded) for path in paths))

    def test_curated_source_inputs_are_present(self) -> None:
        paths = tracked_paths()
        event_files = [
            path
            for path in paths
            if path.startswith("source_data/bids/") and path.endswith("_events.tsv")
        ]
        rating_files = [
            path
            for path in paths
            if path.startswith("source_data/partner_ratings/") and path.endswith(".csv")
        ]
        with (ROOT / "behavioral_analyses/data/participant_L3_47.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            participants = {row["subjID"] for row in csv.DictReader(stream)}

        expected_events = {
            f"source_data/bids/{participant}/func/"
            f"{participant}_task-ultimatum_run-{run}_events.tsv"
            for participant in participants
            for run in ("01", "02")
        }
        self.assertEqual(set(event_files), expected_events)

        rating_participants = {
            f"sub-{Path(path).parent.name}" for path in rating_files
        }
        self.assertTrue(rating_files)
        self.assertLessEqual(rating_participants, participants)

    def test_curated_bids_tree_has_no_imaging_payloads(self) -> None:
        paths = tracked_paths()
        payload_suffixes = (".nii", ".nii.gz", ".dcm")
        self.assertFalse(
            any(
                path.startswith("source_data/bids/") and path.endswith(payload_suffixes)
                for path in paths
            )
        )

    def test_cited_snapshot_records_slice_thickness_and_spacing(self) -> None:
        with (ROOT / "source_data/bids/task-ultimatum_bold.json").open(
            encoding="utf-8"
        ) as stream:
            metadata = json.load(stream)
        self.assertEqual(metadata["SliceThickness"], 2.8)
        self.assertEqual(metadata["SpacingBetweenSlices"], 3.22)
        self.assertAlmostEqual(
            metadata["SpacingBetweenSlices"] / metadata["SliceThickness"],
            1.15,
        )


if __name__ == "__main__":
    unittest.main()
