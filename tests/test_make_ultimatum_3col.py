from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from make_ultimatum_3col import ev_rows, generate, read_events


class MakeUltimatum3ColTests(unittest.TestCase):
    def test_legacy_and_current_trial_labels_are_not_double_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.tsv"
            events.write_text(
                "onset\tduration\ttrial_type\tresponse_time\tOffer\tIsFairBlock\n"
                "1\t3.5\tevent_accept_ingroup\t2.5\t8\t1\n"
                "1\t3.5\tevent_ingroup\t2.5\t8\t1\n"
                "1\t0\tevent_RT\t2.5\t8\t1\n"
                "5\t3.5\tevent_reject_outgroup_unfair\t1.2\t2\t0\n"
                "9\t3.5\tmissed_trial\tn/a\t5\t1\n",
                encoding="utf-8",
            )
            rows = ev_rows(read_events(events), "companion")
            self.assertEqual(len(rows["event_ingroup"]), 1)
            self.assertEqual(len(rows["event_outgroup"]), 1)
            self.assertEqual(rows["event_ingroup_pmod"][0][2], 8)
            self.assertEqual(len(rows["missed_trial"]), 1)
            self.assertEqual(len(rows["event_RT"]), 1)

            substantive = ev_rows(read_events(events), "substantive")
            self.assertEqual(len(substantive["event_RT"]), 2)
            self.assertEqual(substantive["event_RT_pmod"][1][2], 1.2)

    def test_generator_removes_stale_empty_missed_file(self) -> None:
        path = ROOT / "source_data/bids/sub-144/func/sub-144_task-ultimatum_run-02_events.tsv"
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "run-02"
            stale = Path(f"{prefix}_missed_trial.txt")
            stale.write_text("stale\n", encoding="utf-8")
            counts = generate(path, prefix, "companion")
            self.assertEqual(counts["missed_trial"], 0)
            self.assertFalse(stale.exists())
            self.assertEqual(counts["event_RT"], 63)
            self.assertEqual(
                sum(counts[f"event_{partner}"] for partner in ("computer", "ingroup", "outgroup")),
                72,
            )


if __name__ == "__main__":
    unittest.main()
