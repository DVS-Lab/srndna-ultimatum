from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_image_headers import parse_header, parse_result


class ImageAuditTests(unittest.TestCase):
    def test_header_parser_extracts_geometry(self) -> None:
        parsed = parse_header(
            "dim1 66\ndim2 78\ndim3 61\npixdim1 2.973\npixdim2 2.973\npixdim3 3.220\nintent Z-score\n"
        )
        self.assertEqual(parsed["dim3"], "61")
        self.assertEqual(parsed["pixdim3"], "3.220")
        self.assertEqual(parsed["intent"], "Z-score")

    def test_result_argument_requires_safe_label(self) -> None:
        label, path = parse_result("dmn=/tmp/result.nii.gz")
        self.assertEqual(label, "dmn")
        self.assertEqual(path, Path("/tmp/result.nii.gz"))
        with self.assertRaises(Exception):
            parse_result("bad label=/tmp/result.nii.gz")


if __name__ == "__main__":
    unittest.main()
