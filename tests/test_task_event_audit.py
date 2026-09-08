from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_task_events import audit, parse_event_file, partner_from_block


class TaskEventAuditTests(unittest.TestCase):
    def test_partner_from_block_is_strict(self) -> None:
        self.assertEqual(partner_from_block("block_ingroup_fair"), "ingroup")
        with self.assertRaises(ValueError):
            partner_from_block("block_unknown")

    def test_miss_recovers_partner_from_enclosing_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sub-999_task-ultimatum_run-01_events.tsv"
            path.write_text(
                "onset\tduration\ttrial_type\tresponse_time\tOffer\tIsFairBlock\n"
                "1\t10\tblock_outgroup_fair\tn/a\tn/a\tn/a\n"
                "2\t3.5\tevent_accept_outgroup\t1.2\t7\t1\n"
                "6.25\t3.5\tmissed_trial\tn/a\t3\t1\n",
                encoding="utf-8",
            )
            trials, blocks = parse_event_file(path, "sub-999", "01")
            self.assertEqual(blocks, 1)
            self.assertEqual(len(trials), 2)
            self.assertEqual(trials[1].partner, "outgroup")
            self.assertEqual(trials[1].missed, 1)
            self.assertFalse(trials[0].rt_event_present)

    def test_companion_rt_event_is_matched_by_onset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sub-999_task-ultimatum_run-01_events.tsv"
            path.write_text(
                "onset\tduration\ttrial_type\tresponse_time\tOffer\tIsFairBlock\n"
                "1\t10\tblock_outgroup_fair\tn/a\tn/a\tn/a\n"
                "1\t3.5\tevent_accept_outgroup\t1.2\t7\t1\n"
                "1\t0\tevent_RT\t1.2\t7\t1\n",
                encoding="utf-8",
            )
            trials, _ = parse_event_file(path, "sub-999", "01")
            self.assertTrue(trials[0].rt_event_present)

    def test_synthetic_audit_writes_expected_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bids = base / "bids" / "sub-999" / "func"
            bids.mkdir(parents=True)
            sample = base / "sample.csv"
            sample.write_text("subjID,younger,older\nsub-999,1,0\n", encoding="utf-8")
            events = bids / "sub-999_task-ultimatum_run-01_events.tsv"
            events.write_text(
                "onset\tduration\ttrial_type\tresponse_time\tOffer\tIsFairBlock\n"
                "1\t10\tblock_ingroup_fair\tn/a\tn/a\tn/a\n"
                "2\t3.5\tevent_reject_ingroup\t1.4\t2\t1\n"
                "6.25\t3.5\tmissed_trial\tn/a\t4\t1\n",
                encoding="utf-8",
            )
            output = base / "output"
            private = base / "private"
            result = audit(base / "bids", sample, output, private)
            self.assertEqual(result, {"participants": 1, "runs": 1, "trials": 2, "misses": 1})
            with (private / "missed_trials_by_participant.tsv").open(newline="") as stream:
                row = next(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(row["missed_similar"], "1")
            with (output / "task_event_summary.tsv").open(newline="") as stream:
                summary = {row["metric"]: row for row in csv.DictReader(stream, delimiter="\t")}
            self.assertEqual(summary["first_block_trials_missing_event_RT_row"]["value"], "1")
            self.assertEqual(summary["nonfirst_trials_missing_event_RT_row"]["value"], "0")
            with (output / "rt_event_construction_summary.tsv").open(newline="") as stream:
                rt_rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(rt_rows[0]["evidence_level"], "curated_BIDS_events_only")
            self.assertEqual(rt_rows[1]["missing_percent"], "100.000000")


if __name__ == "__main__":
    unittest.main()
