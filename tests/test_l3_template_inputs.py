from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_l3_template_inputs import audit, template_inventory


class L3TemplateInputAuditTests(unittest.TestCase):
    def test_extracts_sub143_and_flags_matching_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            template = Path(temporary) / "L3_test.fsf"
            template.write_text(
                "set fmri(npts) 2\n"
                "set fmri(multiple) 2\n"
                'set feat_files(1) "/root/sub-142/L2_task-ultimatum_model-02_type-nppi-dmn_sm-6.gfeat/cope7.feat/stats/cope1.nii.gz"\n'
                'set feat_files(2) "/root/sub-143/L2_task-ultimatum_model-02_REPLACEME_sm-6.gfeat/copeCOPENUM.feat/stats/cope1.nii.gz"\n',
                encoding="utf-8",
            )
            summary, rows = template_inventory(template)
            self.assertEqual(summary["declared_count_matches_inputs"], 1)
            self.assertEqual(summary["unique_participants"], 2)
            self.assertEqual(summary["unresolved_placeholders"], "COPENUM,REPLACEME")
            self.assertEqual(rows[0]["input_index"], 2)
            self.assertEqual(rows[0]["cope"], "COPENUM")

    def test_writes_inventory_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            templates = base / "templates"
            templates.mkdir()
            (templates / "L3_test.fsf").write_text(
                "set fmri(npts) 1\n"
                "set fmri(multiple) 1\n"
                'set feat_files(1) "/root/sub-143/L2/cope7.feat/stats/cope1.nii.gz"\n',
                encoding="utf-8",
            )
            output = base / "output"
            audit(templates, output)
            with (output / "l3_template_sub143_inputs.tsv").open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(rows[0]["input_index"], "1")


if __name__ == "__main__":
    unittest.main()
