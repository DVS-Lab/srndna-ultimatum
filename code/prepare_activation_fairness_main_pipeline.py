#!/usr/bin/env python3
"""Prepare a scratch-only Ultimatum fairness-main-effect FEAT pipeline.

The historical activation model keeps separate computer, age-similar, and
age-dissimilar task EVs.  This revision preserves that fitted model and adds
contrast 11: the equal-weighted mean of the three offer-size parametric
effects.  It then carries cope 11 through fixed-effects L2 and prepares one
47-participant FLAME 1+2 model with the manuscript nuisance covariates.

This program only renders and validates FSFs plus a job manifest.  It never
runs FEAT and never overwrites the historical production analysis.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence

from make_ultimatum_3col import generate
from prepare_activation_fairness_l3 import matrix_rank, parse_evs, parse_inputs
from prepare_sub144_imaging_repair import CUSTOM_NAMES, discover_bold, discover_confound


REPO_ROOT = Path(__file__).resolve().parents[1]
L1_TEMPLATE_RELATIVE = Path(
    "templates/revision/"
    "L1_task-ultimatum_model-02_type-act_fairness-main.fsf"
)
L2_TEMPLATE_RELATIVE = Path(
    "templates/revision/"
    "L2_task-ultimatum_model-02_type-act_fairness-main.fsf"
)
L3_TEMPLATE_RELATIVE = Path(
    "templates/revision/"
    "L3_task-ultimatum_model-02_type-act-fairness-main.fsf"
)
HISTORICAL_L1_RELATIVE = Path("templates/L1_task-ultimatum_model-02_type-act.fsf")
FAIRNESS_VECTOR = (0.0, 1 / 3, 0.0, 1 / 3, 0.0, 1 / 3, 0.0, 0.0, 0.0)
CONTRAST_RE = re.compile(r"^set fmri\(con_(real|orig)(\d+)\.(\d+)\) (.*)$")


def replace_unique(lines: list[str], prefix: str, replacement: str) -> None:
    indices = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    if len(indices) != 1:
        raise ValueError(f"expected one setting beginning {prefix!r}, found {len(indices)}")
    lines[indices[0]] = replacement


def insert_before_unique(lines: list[str], marker: str, additions: Sequence[str]) -> None:
    indices = [index for index, line in enumerate(lines) if line == marker]
    if len(indices) != 1:
        raise ValueError(f"expected one marker {marker!r}, found {len(indices)}")
    lines[indices[0] : indices[0]] = list(additions)


def setting_value(lines: Sequence[str], prefix: str) -> str:
    matches = [line[len(prefix) :] for line in lines if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one setting beginning {prefix!r}, found {len(matches)}")
    value = matches[0].strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def contrast_vectors(lines: Sequence[str], kind: str) -> dict[int, tuple[float, ...]]:
    values: dict[tuple[int, int], float] = {}
    for line in lines:
        match = CONTRAST_RE.match(line)
        if match and match.group(1) == kind:
            values[(int(match.group(2)), int(match.group(3)))] = float(match.group(4))
    vectors: dict[int, tuple[float, ...]] = {}
    contrasts = sorted({contrast for contrast, _ in values})
    for contrast in contrasts:
        columns = sorted(column for item, column in values if item == contrast)
        if columns != list(range(1, max(columns) + 1)):
            raise ValueError(f"nonsequential {kind} contrast {contrast}")
        vectors[contrast] = tuple(values[(contrast, column)] for column in columns)
    return vectors


def validate_revision_templates(repository: Path) -> None:
    historical = (repository / HISTORICAL_L1_RELATIVE).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    revised = (repository / L1_TEMPLATE_RELATIVE).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    if setting_value(revised, "set fmri(ncon_orig) ") != "11":
        raise ValueError("revision L1 ncon_orig is not 11")
    if setting_value(revised, "set fmri(ncon_real) ") != "11":
        raise ValueError("revision L1 ncon_real is not 11")
    for kind in ("real", "orig"):
        old_vectors = contrast_vectors(historical, kind)
        new_vectors = contrast_vectors(revised, kind)
        if {key: new_vectors[key] for key in range(1, 11)} != old_vectors:
            raise ValueError(f"revision changed historical {kind} contrasts 1-10")
        if 11 not in new_vectors or len(new_vectors[11]) != len(FAIRNESS_VECTOR) or any(
            abs(actual - expected) > 1e-10
            for actual, expected in zip(new_vectors[11], FAIRNESS_VECTOR)
        ):
            raise ValueError(f"revision {kind} contrast 11 is not the fairness mean")
    mask_pairs = {
        (int(match.group(1)), int(match.group(2)))
        for line in revised
        if (match := re.match(r"^set fmri\(conmask(\d+)_(\d+)\) 0$", line))
    }
    expected_pairs = {(left, right) for left in range(1, 12) for right in range(1, 12) if left != right}
    if not expected_pairs.issubset(mask_pairs):
        raise ValueError("revision L1 does not define all pairwise contrast masks")

    l2 = (repository / L2_TEMPLATE_RELATIVE).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    if setting_value(l2, "set fmri(ncopeinputs) ") != "11":
        raise ValueError("revision L2 ncopeinputs is not 11")
    for cope in range(1, 12):
        if setting_value(l2, f"set fmri(copeinput.{cope}) ") != "1":
            raise ValueError(f"revision L2 does not enable cope {cope}")

    l3_path = repository / L3_TEMPLATE_RELATIVE
    l3 = l3_path.read_text(encoding="utf-8", errors="replace").splitlines()
    l3_inputs = parse_inputs(l3)
    if len(l3_inputs) != 47 or any(
        not path.endswith("/cope11.feat/stats/cope1.nii.gz")
        for _, _, path in l3_inputs
    ):
        raise ValueError("revision L3 does not reference 47 L2 cope-11 images")
    l3_evs = parse_evs(l3)
    if len(l3_evs) != 47 * 6:
        raise ValueError("revision L3 does not contain a 47 by 6 design")
    if any(l3_evs[(row, 1)] != 1.0 for row in range(1, 48)):
        raise ValueError("revision L3 first EV is not an intercept")
    if matrix_rank(
        [[l3_evs[(row, column)] for column in range(1, 7)] for row in range(1, 48)]
    ) != 6:
        raise ValueError("revision L3 design is not full rank")
    l3_contrasts = contrast_vectors(l3, "real")
    expected_l3_contrasts = {
        1: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        4: (0.0, -1.0, 0.0, 0.0, 0.0, 0.0),
    }
    if l3_contrasts != expected_l3_contrasts:
        raise ValueError("revision L3 contrasts do not match the declared model")


def validate_historical_l1_source(source_text: str, historical_text: str) -> None:
    source_lines = source_text.splitlines()
    historical_lines = historical_text.splitlines()
    for kind in ("real", "orig"):
        if contrast_vectors(source_lines, kind) != contrast_vectors(historical_lines, kind):
            raise ValueError(f"rendered source changed historical {kind} contrasts")
    for setting in ("evs_orig", "evs_real", "ncon_orig", "ncon_real"):
        prefix = f"set fmri({setting}) "
        if setting_value(source_lines, prefix) != setting_value(historical_lines, prefix):
            raise ValueError(f"rendered source changed historical {setting}")
    if float(setting_value(source_lines, "set fmri(smooth) ")) != 6.0:
        raise ValueError("rendered source does not use the historical 6 mm smoothing")


def new_contrast_block(kind: str) -> list[str]:
    label = "contrast_real" if kind == "real" else "contrast_orig"
    setting = "con_real" if kind == "real" else "con_orig"
    result = [
        f"# Display images for {label} 11",
        f"set fmri(conpic_{kind}.11) 1",
        "",
        f"# Title for {label} 11",
        f'set fmri(conname_{kind}.11) "all_partner_offer_pmod"',
        "",
        "# Equal-weighted mean offer-size slope across all partner types",
    ]
    for column, value in enumerate(FAIRNESS_VECTOR, start=1):
        rendered = "0.333333333333" if value else "0"
        result.append(f"set fmri({setting}11.{column}) {rendered}")
    result.append("")
    return result


def augment_l1(source_text: str, output_base: Path) -> str:
    lines = source_text.splitlines()
    if setting_value(lines, "set fmri(evs_orig) ") != "9":
        raise ValueError("source L1 does not have 9 original EVs")
    if setting_value(lines, "set fmri(evs_real) ") != "9":
        raise ValueError("source L1 does not have 9 real EVs")
    if setting_value(lines, "set fmri(ncon_orig) ") != "10":
        raise ValueError("source L1 does not have 10 original contrasts")
    if setting_value(lines, "set fmri(ncon_real) ") != "10":
        raise ValueError("source L1 does not have 10 real contrasts")
    replace_unique(lines, "set fmri(outputdir) ", f'set fmri(outputdir) "{output_base}"')
    replace_unique(lines, "set fmri(ncon_orig) ", "set fmri(ncon_orig) 11")
    replace_unique(lines, "set fmri(ncon_real) ", "set fmri(ncon_real) 11")
    insert_before_unique(
        lines, "# Display images for contrast_orig 1", new_contrast_block("real")
    )
    insert_before_unique(
        lines, "# Contrast masking - use >0 instead of thresholding?", new_contrast_block("orig")
    )
    mask_lines = ["# Contrast 11 does not participate in contrast masking."]
    mask_lines.extend(f"set fmri(conmask{contrast}_11) 0" for contrast in range(1, 11))
    mask_lines.extend(f"set fmri(conmask11_{contrast}) 0" for contrast in range(1, 11))
    mask_lines.append("")
    insert_before_unique(lines, "# Do contrast masking at all?", mask_lines)
    return "\n".join(lines) + "\n"


def augment_l2(source_text: str, output_base: Path, l1_inputs: Sequence[Path]) -> str:
    if len(l1_inputs) != 2:
        raise ValueError("L2 requires exactly two L1 run inputs")
    lines = source_text.splitlines()
    if setting_value(lines, "set fmri(ncopeinputs) ") != "10":
        raise ValueError("source L2 does not have 10 cope inputs")
    replace_unique(lines, "set fmri(outputdir) ", f'set fmri(outputdir) "{output_base}"')
    replace_unique(lines, "set fmri(ncopeinputs) ", "set fmri(ncopeinputs) 11")
    replace_unique(lines, "set feat_files(1) ", f'set feat_files(1) "{l1_inputs[0]}"')
    replace_unique(lines, "set feat_files(2) ", f'set feat_files(2) "{l1_inputs[1]}"')
    additions = [
        "",
        "# Use lower-level cope 11 (mean offer-size slope across all partner types)",
        "set fmri(copeinput.11) 1",
    ]
    indices = [index for index, line in enumerate(lines) if line == "set fmri(copeinput.10) 1"]
    if len(indices) != 1:
        raise ValueError("source L2 lacks unique copeinput.10")
    lines[indices[0] + 1 : indices[0] + 1] = additions
    return "\n".join(lines) + "\n"


def l1_required_inputs(lines: Sequence[str]) -> list[str]:
    values = [
        setting_value(lines, "set feat_files(1) "),
        setting_value(lines, "set confoundev_files(1) "),
    ]
    for ev in range(1, 10):
        if setting_value(lines, f"set fmri(shape{ev}) ") != "10":
            values.append(setting_value(lines, f"set fmri(custom{ev}) "))
    return [value for value in values if value and value != "dummy"]


def discover_events(
    repository: Path, dataset_root: Path, subject: str, run: int
) -> Path:
    candidates: list[Path] = []
    # A tracked corrected copy takes precedence over the downloaded snapshot.
    for run_label in (f"{run:02d}", str(run)):
        candidates.extend(
            (
                repository
                / "source_data"
                / "bids"
                / subject
                / "func"
            ).glob(f"{subject}_task-ultimatum_run-{run_label}_events.tsv")
        )
    if candidates:
        unique = sorted({path.resolve() for path in candidates if path.is_file()})
        if len(unique) != 1:
            raise ValueError(f"ambiguous tracked events for {subject} run-{run:02d}: {unique}")
        return unique[0]

    func = dataset_root / subject / "func"
    for run_label in (f"{run:02d}", str(run)):
        candidates.extend(
            func.glob(f"{subject}_task-ultimatum_run-{run_label}_events.tsv")
        )
    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected one events file for {subject} run-{run:02d}, found {unique}"
        )
    return unique[0]


def configure_run_inputs(
    text: str,
    *,
    repository: Path,
    dataset_root: Path,
    production_fsl_root: Path,
    work_root: Path,
    subject: str,
    run: int,
) -> tuple[str, Path, dict[str, int]]:
    lines = text.splitlines()
    events = discover_events(repository, dataset_root, subject, run)
    ev_prefix = work_root / "EVfiles" / subject / f"run-{run:02d}"
    counts = generate(events, ev_prefix, "substantive")
    task_trials = sum(counts[f"event_{partner}"] for partner in ("computer", "ingroup", "outgroup"))
    if task_trials < 1 or counts["event_RT"] != task_trials:
        raise ValueError(
            f"invalid substantive-trial EV construction for {subject} run-{run:02d}"
        )

    bold = discover_bold(dataset_root, subject, f"{run:02d}")
    recorded_confound = setting_value(lines, "set confoundev_files(1) ")
    confound = discover_confound(
        dataset_root,
        production_fsl_root.parent.parent,
        subject,
        f"{run:02d}",
        recorded_confound,
        work_root
        / "confounds"
        / subject
        / f"{subject}_task-ultimatum_run-{run}_desc-fslConfounds.tsv",
    )
    npts = int(float(setting_value(lines, "set fmri(npts) ")))
    confound_rows = sum(
        1 for line in confound.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if confound_rows != npts:
        raise ValueError(
            f"confound/BOLD row mismatch for {subject} run-{run:02d}: "
            f"confounds={confound_rows}, expected_volumes={npts}"
        )
    replace_unique(lines, "set feat_files(1) ", f'set feat_files(1) "{bold}"')
    replace_unique(
        lines,
        "set confoundev_files(1) ",
        f'set confoundev_files(1) "{confound}"',
    )
    replace_unique(
        lines,
        "set fmri(shape7) ",
        f"set fmri(shape7) {3 if counts['missed_trial'] else 10}",
    )
    for index, name in CUSTOM_NAMES.items():
        replace_unique(
            lines,
            f"set fmri(custom{index}) ",
            f'set fmri(custom{index}) "{ev_prefix}_{name}.txt"',
        )
    return "\n".join(lines) + "\n", events, counts


def source_l1(root: Path, subject: str, run: int) -> Path:
    return (
        root
        / subject
        / f"L1_task-ultimatum_model-02_type-act_run-{run:02d}_sm-6.feat"
        / "design.fsf"
    )


def source_l2(root: Path, subject: str) -> Path:
    return (
        root
        / subject
        / "L2_task-ultimatum_model-02_type-act_sm-6.gfeat"
        / "design.fsf"
    )


def render_l3(source: Path, destination: Path, output_base: Path, l2_outputs: dict[str, Path]) -> list[str]:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    inputs = parse_inputs(lines)
    original_evs = parse_evs(lines)
    if len(original_evs) != 47 * 6:
        raise ValueError("corrected L3 source must have 47 rows and 6 EVs")
    replace_unique(lines, "set fmri(outputdir) ", f'set fmri(outputdir) "{output_base}"')

    rendered_inputs: list[str] = []
    for index, subject, _ in inputs:
        # Higher-level FEAT consumes each participant's fixed-effects cope
        # image, not the enclosing cope*.feat directory.
        rendered = str(l2_outputs[subject] / "cope11.feat" / "stats" / "cope1.nii.gz")
        replace_unique(lines, f"set feat_files({index}) ", f'set feat_files({index}) "{rendered}"')
        rendered_inputs.append(rendered)

    older_proportion = sum(original_evs[(row, 2)] for row in range(1, 48)) / 47
    for row in range(1, 48):
        older = original_evs[(row, 2)]
        replace_unique(lines, f"set fmri(evg{row}.1) ", f"set fmri(evg{row}.1) 1.0")
        replace_unique(
            lines,
            f"set fmri(evg{row}.2) ",
            f"set fmri(evg{row}.2) {older - older_proportion:.12g}",
        )
    replace_unique(lines, "set fmri(evtitle1) ", 'set fmri(evtitle1) "intercept"')
    replace_unique(lines, "set fmri(evtitle2) ", 'set fmri(evtitle2) "older_centered"')

    specifications = (
        (1, "adjusted-mean-positive", (1, 0, 0, 0, 0, 0)),
        (2, "adjusted-mean-negative", (-1, 0, 0, 0, 0, 0)),
        (3, "older-greater-than-younger", (0, 1, 0, 0, 0, 0)),
        (4, "younger-greater-than-older", (0, -1, 0, 0, 0, 0)),
    )
    for contrast, title, vector in specifications:
        replace_unique(
            lines,
            f"set fmri(conname_real.{contrast}) ",
            f'set fmri(conname_real.{contrast}) "{title}"',
        )
        for column, value in enumerate(vector, start=1):
            replace_unique(
                lines,
                f"set fmri(con_real{contrast}.{column}) ",
                f"set fmri(con_real{contrast}.{column}) {float(value):.1f}",
            )

    rendered_evs = parse_evs(lines)
    matrix = [[rendered_evs[(row, column)] for column in range(1, 7)] for row in range(1, 48)]
    if matrix_rank(matrix) != 6:
        raise ValueError("rendered fairness-main L3 design is not full rank")
    if abs(sum(row[1] for row in matrix)) > 1e-8:
        raise ValueError("rendered L3 age-group covariate is not centered")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rendered_inputs


def ensure_empty_safe_work_root(work_root: Path, protected: Sequence[Path]) -> None:
    resolved = work_root.resolve()
    for path in protected:
        candidate = path.resolve()
        if resolved == candidate or resolved.is_relative_to(candidate):
            raise ValueError(f"work root must be outside protected tree: {candidate}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"work root is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def prepare(
    repository: Path,
    dataset_root: Path,
    production_fsl_root: Path,
    sub144_repair_root: Path,
    work_root: Path,
) -> Path:
    validate_revision_templates(repository)
    historical_l1_text = (repository / HISTORICAL_L1_RELATIVE).read_text(
        encoding="utf-8", errors="replace"
    )
    l3_source = repository / L3_TEMPLATE_RELATIVE
    if not l3_source.is_file():
        raise FileNotFoundError(l3_source)
    ensure_empty_safe_work_root(
        work_root, (repository, dataset_root, production_fsl_root, sub144_repair_root)
    )
    participants = [subject for _, subject, _ in parse_inputs(l3_source.read_text(encoding="utf-8").splitlines())]

    rows: list[dict[str, object]] = []
    l2_outputs: dict[str, Path] = {}
    for subject in participants:
        source_root = (
            sub144_repair_root / "derivatives" / "fsl"
            if subject == "sub-144"
            else production_fsl_root
        )
        l1_outputs: list[Path] = []
        for run in (1, 2):
            source = source_l1(source_root, subject, run)
            if not source.is_file():
                raise FileNotFoundError(source)
            source_text = source.read_text(encoding="utf-8", errors="replace")
            validate_historical_l1_source(source_text, historical_l1_text)
            output_base = (
                work_root
                / "derivatives"
                / "fsl"
                / subject
                / f"L1_task-ultimatum_model-02_type-act-fairness-main_run-{run:02d}_sm-6"
            )
            output = Path(f"{output_base}.feat")
            fsf = work_root / "fsf" / subject / f"L1_act-fairness-main_run-{run:02d}.fsf"
            rendered = augment_l1(source_text, output_base)
            rendered, events, counts = configure_run_inputs(
                rendered,
                repository=repository,
                dataset_root=dataset_root,
                production_fsl_root=production_fsl_root,
                work_root=work_root,
                subject=subject,
                run=run,
            )
            task_trials = sum(
                counts[f"event_{partner}"]
                for partner in ("computer", "ingroup", "outgroup")
            )
            fsf.parent.mkdir(parents=True, exist_ok=True)
            fsf.write_text(rendered, encoding="utf-8")
            inputs = [*l1_required_inputs(rendered.splitlines()), str(events)]
            for value in inputs:
                if not Path(value).exists():
                    raise FileNotFoundError(f"missing rendered L1 input: {value}")
            rows.append(
                {
                    "stage": "l1",
                    "subject": subject,
                    "model": "act-fairness-main",
                    "run": f"{run:02d}",
                    "fsf": fsf,
                    "output": output,
                    "inputs": "|".join(inputs),
                    "source_fsf": source,
                    "revision_template": repository / L1_TEMPLATE_RELATIVE,
                    "model_policy": (
                        "historical_9ev_glm;substantive_decision_rows_for_rt;"
                        "equal_weighted_offer_pmod_contrast11;"
                        f"task_trials={task_trials}"
                    ),
                    "expected_copes": 11,
                    "expected_zstats": "",
                }
            )
            l1_outputs.append(output)

        source = source_l2(source_root, subject)
        if not source.is_file():
            raise FileNotFoundError(source)
        output_base = (
            work_root
            / "derivatives"
            / "fsl"
            / subject
            / "L2_task-ultimatum_model-02_type-act-fairness-main_sm-6"
        )
        output = Path(f"{output_base}.gfeat")
        fsf = work_root / "fsf" / subject / "L2_act-fairness-main.fsf"
        fsf.write_text(
            augment_l2(source.read_text(encoding="utf-8", errors="replace"), output_base, l1_outputs),
            encoding="utf-8",
        )
        rows.append(
            {
                "stage": "l2",
                "subject": subject,
                "model": "act-fairness-main",
                "run": "combined",
                "fsf": fsf,
                "output": output,
                "inputs": "|".join(str(path) for path in l1_outputs),
                "source_fsf": source,
                "revision_template": repository / L2_TEMPLATE_RELATIVE,
                "model_policy": "fixed_effects_across_runs_carrying_all_11_copes",
                "expected_copes": 11,
                "expected_zstats": "",
            }
        )
        l2_outputs[subject] = output

    l3_base = work_root / "outputs" / "activation-fairness-main_all-participants"
    l3_fsf = work_root / "fsf" / "L3_activation-fairness-main_all-participants.fsf"
    l3_inputs = render_l3(l3_source, l3_fsf, l3_base, l2_outputs)
    rows.append(
        {
            "stage": "l3",
            "subject": "all-47",
            "model": "activation-fairness-main",
            "run": "all-participants",
            "fsf": l3_fsf,
            "output": Path(f"{l3_base}.gfeat"),
            "inputs": "|".join(l3_inputs),
            "source_fsf": l3_source,
            "revision_template": "rendered_from_corrected_47_participant_l3_design",
            "model_policy": (
                "intercept;centered_age_group;centered_sex;tsnr;mean_fd;"
                "event_corrected_taskwide_mean_rt;flame1plus2"
            ),
            "expected_copes": "",
            "expected_zstats": 4,
        }
    )

    manifest = work_root / "activation_fairness_main_jobs.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        "PASS: prepared 94 L1, 47 L2, and 1 L3 scratch-only fairness-main "
        f"jobs for {len(participants)} participants; manifest={manifest}"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--production-fsl-root", type=Path, required=True)
    parser.add_argument("--sub144-repair-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepare(
        args.repository.resolve(),
        args.dataset_root.resolve(),
        args.production_fsl_root.resolve(),
        args.sub144_repair_root.resolve(),
        args.work_root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
