from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from prepare_ultimatum_l3_repair import parse_evs, parse_inputs, render
from audit_prepared_ultimatum_l3_designs import read_header_value, read_matrix
from collect_ultimatum_l3_repair_designs import collect
from run_ultimatum_repair_jobs import complete, read_manifest


class UltimatumL3RepairTests(unittest.TestCase):
    def test_compiled_design_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            design = Path(directory) / "design.mat"
            design.write_text(
                "/NumWaves\t2\n/NumPoints\t2\n/Matrix\n1 0\n0 1\n",
                encoding="utf-8",
            )
            self.assertEqual(read_header_value(design, "NumWaves"), 2)
            self.assertEqual(read_matrix(design), [[1.0, 0.0], [0.0, 1.0]])

    def test_image_only_render_preserves_design_and_replaces_sub144_input(self) -> None:
        source = ROOT / "results/reviewer/production_audits/dmn-age/design/design.fsf"
        source_lines = source.read_text(encoding="utf-8").splitlines()
        participants = [subject for _, subject, _ in parse_inputs(source_lines)]
        covariates = {subject: {"subjID": subject} for subject in participants}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            destination = base / "rendered.fsf"
            repaired = base / "repair/sub-144/cope1.nii.gz"
            standard = base / "MNI152_T1_2mm_brain.nii.gz"
            audit = render(
                source,
                destination,
                base / "result",
                base / "production",
                repaired,
                standard,
                covariates,
                None,
                None,
            )
            rendered_lines = destination.read_text(encoding="utf-8").splitlines()
            self.assertEqual(parse_evs(source_lines), parse_evs(rendered_lines))
            rendered_inputs = parse_inputs(rendered_lines)
            sub144 = [path for _, subject, path in rendered_inputs if subject == "sub-144"]
            self.assertEqual(sub144, [str(repaired)])
            self.assertIn(f'set fmri(regstandard) "{standard}"', rendered_lines)
            self.assertEqual(audit["inputs"].split("|")[-1], str(standard))
            self.assertEqual(audit["rank"], 6)
            self.assertEqual(audit["changed_ev_columns"], "")

    def test_fairness_corrected_render_changes_only_group_covariate_columns(self) -> None:
        source = ROOT / "results/reviewer/production_audits/ecn-sensitivity/design/design.fsf"
        source_lines = source.read_text(encoding="utf-8").splitlines()
        inputs = parse_inputs(source_lines)
        source_evs = parse_evs(source_lines)
        covariates = {}
        for index, subject, _ in inputs:
            covariates[subject] = {
                "subjID": subject,
                "corrected_sensitivity_young": str(source_evs[(index, 7)]),
                "corrected_sensitivity_old": str(source_evs[(index, 8)]),
            }
        covariates["sub-144"]["corrected_sensitivity_old"] = str(
            float(covariates["sub-144"]["corrected_sensitivity_old"]) + 0.1
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            destination = base / "rendered.fsf"
            audit = render(
                source,
                destination,
                base / "result",
                base / "production",
                base / "repair/sub-144/cope1.nii.gz",
                base / "MNI152_T1_2mm_brain.nii.gz",
                covariates,
                "corrected_sensitivity",
                None,
            )
            rendered_evs = parse_evs(destination.read_text(encoding="utf-8").splitlines())
            changed = {
                key for key in source_evs if abs(source_evs[key] - rendered_evs[key]) > 1e-12
            }
            self.assertEqual(changed, {(35, 8)})
            self.assertEqual(audit["rank"], 8)
            self.assertEqual(audit["changed_ev_columns"], "7,8")

    def test_reported_covariate_render_changes_rt_and_fairness_columns(self) -> None:
        source = ROOT / "results/reviewer/production_audits/ecn-sensitivity/design/design.fsf"
        source_lines = source.read_text(encoding="utf-8").splitlines()
        inputs = parse_inputs(source_lines)
        source_evs = parse_evs(source_lines)
        covariates = {}
        for index, subject, _ in inputs:
            covariates[subject] = {
                "subjID": subject,
                "corrected_mean_rt_z": str(source_evs[(index, 4)]),
                "corrected_sensitivity_young": str(source_evs[(index, 7)]),
                "corrected_sensitivity_old": str(source_evs[(index, 8)]),
            }
        covariates["sub-144"]["corrected_mean_rt_z"] = str(
            float(covariates["sub-144"]["corrected_mean_rt_z"]) + 0.1
        )
        covariates["sub-144"]["corrected_sensitivity_old"] = str(
            float(covariates["sub-144"]["corrected_sensitivity_old"]) + 0.1
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            destination = base / "rendered.fsf"
            audit = render(
                source,
                destination,
                base / "result",
                base / "production",
                base / "repair/sub-144/cope1.nii.gz",
                base / "MNI152_T1_2mm_brain.nii.gz",
                covariates,
                "corrected_sensitivity",
                4,
            )
            rendered_evs = parse_evs(destination.read_text(encoding="utf-8").splitlines())
            changed = {
                key for key in source_evs if abs(source_evs[key] - rendered_evs[key]) > 1e-12
            }
            self.assertEqual(changed, {(35, 4), (35, 8)})
            self.assertEqual(audit["rank"], 8)
            self.assertEqual(audit["changed_ev_columns"], "4,7,8")

    def test_manifest_filter_selects_one_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "jobs.tsv"
            manifest.write_text(
                "stage\tmodel\trun\n"
                "l3\tecn-sensitivity\timage-only\n"
                "l3\tecn-sensitivity\treported-covariates-corrected\n"
                "l3\tactivation-norm\timage-only\n",
                encoding="utf-8",
            )
            rows = read_manifest(
                manifest,
                "l3",
                models={"ecn-sensitivity"},
                runs={"reported-covariates-corrected"},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run"], "reported-covariates-corrected")

    def test_compiled_design_collection_is_complete_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            manifest = base / "scratch/l3_repair_jobs.tsv"
            manifest.parent.mkdir()
            fields = [
                "stage",
                "model",
                "run",
                "fsf",
                "source_fsf",
                "covariate_policy",
                "changed_ev_columns",
                "n_evs",
                "design_rank",
                "expected_zstats",
                "sub144_repaired_l2",
                "standard_image",
            ]
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                for index in range(6):
                    fsf = manifest.parent / f"design-{index}.fsf"
                    for suffix in (".fsf", ".mat", ".con", ".grp"):
                        fsf.with_suffix(suffix).write_text(
                            f"artifact {index} {suffix}\n", encoding="utf-8"
                        )
                    writer.writerow(
                        {
                            "stage": "l3",
                            "model": f"model-{index}",
                            "run": "image-only",
                            "fsf": fsf,
                            "source_fsf": "production.fsf",
                            "covariate_policy": "production",
                            "changed_ev_columns": "",
                            "n_evs": "6",
                            "design_rank": "6",
                            "expected_zstats": "4",
                            "sub144_repaired_l2": "cope1.nii.gz",
                            "standard_image": "standard.nii.gz",
                        }
                    )
            output = base / "collected"
            inventory = collect(manifest, output)
            self.assertEqual(len(inventory.read_text().splitlines()), 7)
            self.assertEqual(len(list(output.glob("model-*/*/design.*"))), 24)
            with self.assertRaises(FileExistsError):
                collect(manifest, output)

    def test_l3_completion_requires_every_expected_zstat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.gfeat"
            row = {
                "stage": "l3",
                "model": "dmn-age",
                "run": "image-only",
                "fsf": "unused",
                "output": str(output),
                "inputs": "unused",
                "expected_zstats": "4",
            }
            for index in range(1, 5):
                image = output / f"cope1.feat/stats/zstat{index}.nii.gz"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"zstat")
            self.assertTrue(complete(row))
            (output / "cope1.feat/stats/zstat4.nii.gz").unlink()
            self.assertFalse(complete(row))


if __name__ == "__main__":
    unittest.main()
