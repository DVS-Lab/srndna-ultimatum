from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "build_event_corrected_trials.py"
SPEC = importlib.util.spec_from_file_location("build_event_corrected_trials", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EventCorrectedTrialsTests(unittest.TestCase):
    def test_corrected_sub144_event_hashes(self) -> None:
        expected = {
            "01": "d91318585db26f0b43fb08248ec9f39d73b0665579472b95768f153d73e5d500",
            "02": "e82949ec83f349085064019f3cc8f2487edae26b46e4adcdac3206bf2b61a7c1",
        }
        for run, digest in expected.items():
            path = (
                ROOT
                / "source_data/bids/sub-144/func"
                / f"sub-144_task-ultimatum_run-{run}_events.tsv"
            )
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_overlay_changes_only_sub144_behavioral_columns(self) -> None:
        source = ROOT / "behavioral_analyses/data/all_trials_brains.csv"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corrected.csv"
            summary = MODULE.build(
                source,
                ROOT / "source_data/bids",
                output,
            )

            self.assertEqual(summary["rows_overlaid"], 144)
            self.assertGreater(summary["changed_behavior_cells"], 0)
            self.assertEqual(summary["historical_missed_trials"], 0)
            self.assertEqual(summary["corrected_missed_trials"], 1)
            self.assertEqual(summary["historical_responded_human_trials"], 96)
            self.assertEqual(summary["corrected_responded_human_trials"], 95)
            self.assertNotEqual(
                summary["historical_profile_sha256"],
                summary["corrected_profile_sha256"],
            )

            _, original = MODULE.read_rows(source)
            _, corrected = MODULE.read_rows(output)
            self.assertEqual(len(original), len(corrected))
            for old, new in zip(original, corrected):
                if old["subjID"] != "sub-144":
                    self.assertEqual(old, new)
                else:
                    for column in old:
                        if column not in MODULE.BEHAVIOR_COLUMNS:
                            self.assertEqual(old[column], new[column])

            profiles = {}
            for subject in ("sub-143", "sub-144"):
                profiles[subject] = MODULE.profile(
                    row for row in corrected if row["subjID"] == subject
                )
            self.assertNotEqual(profiles["sub-143"], profiles["sub-144"])

    def test_first_corrected_run_contains_the_recovered_miss(self) -> None:
        path = (
            ROOT
            / "source_data/bids/sub-144/func"
            / "sub-144_task-ultimatum_run-01_events.tsv"
        )
        trials = MODULE.event_trials(path, response_delay=1.0)
        self.assertEqual(len(trials), 72)
        self.assertEqual(trials[0]["missed"], "1")
        self.assertEqual(trials[0]["human"], "1")
        self.assertEqual(trials[0]["ingroup"], "1")
        self.assertEqual(trials[1]["response_time"], "1.58989")


if __name__ == "__main__":
    unittest.main()
