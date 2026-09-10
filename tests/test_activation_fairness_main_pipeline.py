from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from prepare_activation_fairness_l3 import parse_evs, parse_inputs
from prepare_activation_fairness_main_pipeline import (
    FAIRNESS_VECTOR,
    augment_l1,
    augment_l2,
    contrast_vectors,
    render_l3,
    setting_value,
    validate_revision_templates,
)
from collect_activation_fairness_main_results import collect
from run_ultimatum_repair_jobs import complete


class ActivationFairnessMainPipelineTests(unittest.TestCase):
    def test_revision_templates_are_internally_consistent(self) -> None:
        validate_revision_templates(ROOT)

    def test_l1_augmentation_only_adds_declared_contrast(self) -> None:
        source = (ROOT / "templates/L1_task-ultimatum_model-02_type-act.fsf").read_text(
            encoding="utf-8"
        )
        rendered = augment_l1(source, Path("/scratch/new-model"))
        source_lines = source.splitlines()
        rendered_lines = rendered.splitlines()
        self.assertEqual(setting_value(rendered_lines, "set fmri(ncon_orig) "), "11")
        self.assertEqual(setting_value(rendered_lines, "set fmri(ncon_real) "), "11")
        self.assertEqual(
            contrast_vectors(source_lines, "real"),
            {key: contrast_vectors(rendered_lines, "real")[key] for key in range(1, 11)},
        )
        self.assertEqual(
            contrast_vectors(source_lines, "orig"),
            {key: contrast_vectors(rendered_lines, "orig")[key] for key in range(1, 11)},
        )
        for kind in ("real", "orig"):
            actual = contrast_vectors(rendered_lines, kind)[11]
            self.assertEqual(len(actual), 9)
            for observed, expected in zip(actual, FAIRNESS_VECTOR):
                self.assertAlmostEqual(observed, expected, places=10)

    def test_l2_augmentation_carries_cope11_and_new_run_inputs(self) -> None:
        source = (ROOT / "templates/L2_task-ultimatum_model-02_type-act.fsf").read_text(
            encoding="utf-8"
        )
        rendered = augment_l2(
            source,
            Path("/scratch/new-l2"),
            (Path("/scratch/run1.feat"), Path("/scratch/run2.feat")),
        ).splitlines()
        self.assertEqual(setting_value(rendered, "set fmri(ncopeinputs) "), "11")
        self.assertEqual(setting_value(rendered, "set fmri(copeinput.11) "), "1")
        self.assertEqual(setting_value(rendered, "set feat_files(1) "), "/scratch/run1.feat")
        self.assertEqual(setting_value(rendered, "set feat_files(2) "), "/scratch/run2.feat")

    def test_l3_render_uses_cope11_and_full_rank_corrected_covariates(self) -> None:
        source = (
            ROOT
            / "results/reviewer/l3_repair_designs/dmn-age/"
            "reported-covariates-corrected/design.fsf"
        )
        participants = [
            subject
            for _, subject, _ in parse_inputs(source.read_text(encoding="utf-8").splitlines())
        ]
        l2_outputs = {subject: Path("/scratch") / subject / "new.gfeat" for subject in participants}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "design.fsf"
            inputs = render_l3(source, destination, Path("/scratch/group"), l2_outputs)
            lines = destination.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(inputs), 47)
            self.assertTrue(all(path.endswith("/cope11.feat") for path in inputs))
            evs = parse_evs(lines)
            self.assertTrue(all(evs[(row, 1)] == 1.0 for row in range(1, 48)))
            self.assertAlmostEqual(sum(evs[(row, 2)] for row in range(1, 48)), 0.0)

    def test_runner_expected_copes_supports_new_11_cope_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            l1 = base / "model.feat"
            l2 = base / "model.gfeat"
            for cope in (1, 11):
                path = l1 / "stats" / f"cope{cope}.nii.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"cope")
            self.assertTrue(
                complete(
                    {
                        "stage": "l1",
                        "model": "act-fairness-main",
                        "output": str(l1),
                        "expected_copes": "11",
                    }
                )
            )
            for cope in range(1, 12):
                path = l2 / f"cope{cope}.feat/stats/cope1.nii.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"cope")
            self.assertTrue(
                complete(
                    {
                        "stage": "l2",
                        "model": "act-fairness-main",
                        "output": str(l2),
                        "expected_copes": "11",
                    }
                )
            )
            (l2 / "cope11.feat/stats/cope1.nii.gz").unlink()
            self.assertFalse(
                complete(
                    {
                        "stage": "l2",
                        "model": "act-fairness-main",
                        "output": str(l2),
                        "expected_copes": "11",
                    }
                )
            )

    def test_collector_requires_and_records_complete_142_job_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            work = base / "work"
            work.mkdir()
            shared_fsf = work / "rendered.fsf"
            source_fsf = work / "source.fsf"
            shared_fsf.write_text("rendered\n", encoding="utf-8")
            source_fsf.write_text("source\n", encoding="utf-8")
            rows = []
            for index in range(94):
                output = work / f"l1-{index}.feat"
                for cope in (1, 11):
                    path = output / "stats" / f"cope{cope}.nii.gz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"cope")
                rows.append(
                    {
                        "stage": "l1",
                        "subject": f"sub-{index // 2:03d}",
                        "model": "act-fairness-main",
                        "run": f"{index % 2 + 1:02d}",
                        "fsf": shared_fsf,
                        "output": output,
                        "inputs": source_fsf,
                        "source_fsf": source_fsf,
                        "revision_template": "template",
                        "model_policy": "policy",
                        "expected_copes": "11",
                        "expected_zstats": "",
                    }
                )
            for index in range(47):
                output = work / f"l2-{index}.gfeat"
                for cope in range(1, 12):
                    path = output / f"cope{cope}.feat/stats/cope1.nii.gz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"cope")
                rows.append(
                    {
                        "stage": "l2",
                        "subject": f"sub-{index:03d}",
                        "model": "act-fairness-main",
                        "run": "combined",
                        "fsf": shared_fsf,
                        "output": output,
                        "inputs": source_fsf,
                        "source_fsf": source_fsf,
                        "revision_template": "template",
                        "model_policy": "policy",
                        "expected_copes": "11",
                        "expected_zstats": "",
                    }
                )
            l3_output = work / "l3.gfeat"
            for name in ("design.fsf", "design.mat", "design.con", "design.grp"):
                (l3_output / name).parent.mkdir(parents=True, exist_ok=True)
                (l3_output / name).write_text(name, encoding="utf-8")
            for contrast in range(1, 5):
                path = l3_output / f"cope1.feat/stats/zstat{contrast}.nii.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"zstat")
            for name in ("cope1.nii.gz", "varcope1.nii.gz"):
                path = l3_output / "cope1.feat/stats" / name
                path.write_bytes(b"stat")
            rows.append(
                {
                    "stage": "l3",
                    "subject": "all-47",
                    "model": "activation-fairness-main",
                    "run": "all-participants",
                    "fsf": shared_fsf,
                    "output": l3_output,
                    "inputs": source_fsf,
                    "source_fsf": source_fsf,
                    "revision_template": "template",
                    "model_policy": "policy",
                    "expected_copes": "",
                    "expected_zstats": "4",
                }
            )
            manifest = work / "activation_fairness_main_jobs.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=list(rows[0]),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            output = base / "collected"
            inventory = collect(work, output)
            self.assertEqual(len(inventory.read_text().splitlines()), 143)
            self.assertTrue((output / "design/design.mat").is_file())
            self.assertTrue((output / "focal_contrast1/zstat1.nii.gz").is_file())


if __name__ == "__main__":
    unittest.main()
