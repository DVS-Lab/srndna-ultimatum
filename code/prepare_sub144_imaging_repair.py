#!/usr/bin/env python3
"""Prepare an isolated, minimal sub-144 Ultimatum L1/L2 repair workspace.

The retained production FSFs are used as the authoritative model definitions.
Only output, BOLD, confound, and corrected task-EV paths are changed. Existing
network time series are reused because the event-file repair cannot alter them.
This script renders files and a job manifest but never invokes FEAT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path
from typing import Sequence

from make_ultimatum_3col import generate


SUBJECT_RE = re.compile(r"^sub-[0-9]+$")
MODELS = ("act", "nppi-dmn", "nppi-ecn")
CUSTOM_NAMES = {
    1: "event_computer",
    2: "event_computer_pmod",
    3: "event_ingroup",
    4: "event_ingroup_pmod",
    5: "event_outgroup",
    6: "event_outgroup_pmod",
    7: "missed_trial",
    8: "event_RT",
    9: "event_RT_pmod",
}


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def fsf_value(text: str, key: str) -> str | None:
    pattern = re.compile(
        rf'^set (?:fmri\({re.escape(key)}\)|{re.escape(key)}) "?(.*?)"?$',
        re.MULTILINE,
    )
    match = pattern.search(text)
    return None if match is None else match.group(1).rstrip('"')


def replace_fsf_value(text: str, key: str, value: str, *, fmri: bool) -> str:
    pattern = re.compile(
        rf"^set (?:fmri\({re.escape(key)}\)|{re.escape(key)}) .*$", re.MULTILINE
    )
    setting = f"fmri({key})" if fmri else key
    replacement = f'set {setting} "{value}"'
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError(f"expected one FSF setting for {key}, found {count}")
    return updated


def replace_fsf_number(text: str, key: str, value: int) -> str:
    pattern = re.compile(rf"^set fmri\({re.escape(key)}\) .*$", re.MULTILINE)
    updated, count = pattern.subn(f"set fmri({key}) {value}", text, count=1)
    if count != 1:
        raise ValueError(f"expected one FSF setting for {key}, found {count}")
    return updated


def remap_recorded_path(
    value: str, production_root: Path, dataset_root: Path
) -> Path:
    mappings = (
        (Path("/data/projects/srndna-ultimatum"), production_root),
        (Path("/ZPOOL/data/projects/srndna-ultimatum"), production_root),
        (Path("/data/projects/srndna-data"), dataset_root),
        (Path("/ZPOOL/data/datasets/ds003745-work"), dataset_root),
    )
    path = Path(value)
    for old, new in mappings:
        try:
            return new / path.relative_to(old)
        except ValueError:
            continue
    return path


def unique_existing(candidates: list[Path], description: str) -> Path:
    existing: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate not in seen:
            existing.append(candidate)
            seen.add(candidate)
    if len(existing) > 1:
        hashes = {sha256(path) for path in existing}
        if len(hashes) == 1:
            return existing[0]
    if len(existing) != 1:
        rendered = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"expected exactly one {description}, found {len(existing)}; candidates:\n  {rendered}"
        )
    return existing[0]


def discover_bold(dataset_root: Path, subject: str, run: str) -> Path:
    func = dataset_root / "derivatives" / "fmriprep" / subject / "func"
    candidates: list[Path] = []
    for run_label in (str(int(run)), run):
        candidates.extend(
            func.glob(
                f"{subject}_task-ultimatum_run-{run_label}_"
                "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            )
        )
    return unique_existing(candidates, f"preprocessed BOLD for {subject} run-{run}")


def discover_confound(
    dataset_root: Path,
    production_root: Path,
    subject: str,
    run: str,
    recorded: str | None,
) -> Path:
    candidates: list[Path] = []
    if recorded:
        candidates.append(remap_recorded_path(recorded, production_root, dataset_root))
    for root in (dataset_root, production_root):
        confounds = root / "derivatives" / "fsl" / "confounds" / subject
        for run_label in (str(int(run)), run):
            candidates.append(
                confounds
                / f"{subject}_task-ultimatum_run-{run_label}_desc-fslConfounds.tsv"
            )
    return unique_existing(candidates, f"FSL confounds for {subject} run-{run}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def render_l1(
    source: Path,
    destination: Path,
    output_root: Path,
    bold: Path,
    confound: Path,
    ev_prefix: Path,
    counts: dict[str, int],
    production_root: Path,
    dataset_root: Path,
) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    text = replace_fsf_value(text, "outputdir", str(output_root), fmri=True)
    text = replace_fsf_value(text, "feat_files(1)", str(bold), fmri=False)
    text = replace_fsf_value(text, "confoundev_files(1)", str(confound), fmri=False)
    text = replace_fsf_number(text, "shape7", 3 if counts["missed_trial"] else 10)
    for index, name in CUSTOM_NAMES.items():
        text = replace_fsf_value(
            text, f"custom{index}", str(Path(f"{ev_prefix}_{name}.txt")), fmri=True
        )

    # Preserve the production physical regressors exactly, remapping only stale
    # mount prefixes. These time series depend on BOLD/network maps, not events.
    for index in (10, 20, 21, 22, 23, 24, 25, 26, 27, 28):
        key = f"custom{index}"
        recorded = fsf_value(text, key)
        if recorded is None:
            continue
        resolved = remap_recorded_path(recorded, production_root, dataset_root).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"missing retained nPPI time series for {source.name} {key}: {resolved}"
            )
        text = replace_fsf_value(text, key, str(resolved), fmri=True)
    write_atomic(destination, text)


def render_l2(
    source: Path,
    destination: Path,
    output_root: Path,
    input1: Path,
    input2: Path,
) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    text = replace_fsf_value(text, "outputdir", str(output_root), fmri=True)
    text = replace_fsf_value(text, "feat_files(1)", str(input1), fmri=False)
    text = replace_fsf_value(text, "feat_files(2)", str(input2), fmri=False)
    write_atomic(destination, text)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare(
    repository: Path,
    dataset_root: Path,
    production_root: Path,
    work_root: Path,
    subject: str,
) -> Path:
    for required in (repository, dataset_root, production_root):
        if not required.is_dir():
            raise FileNotFoundError(required)
    if not SUBJECT_RE.fullmatch(subject):
        raise ValueError(f"invalid subject label: {subject}")
    if any(is_within(work_root, root) for root in (repository, dataset_root, production_root)):
        raise ValueError("work root must be outside repository, dataset, and production roots")
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"repair work root is not empty: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)

    fsf_dir = work_root / "fsf"
    ev_dir = work_root / "EVfiles" / subject
    output_parent = work_root / "derivatives" / "fsl" / subject
    rows: list[dict[str, object]] = []
    l1_outputs: dict[tuple[str, str], Path] = {}

    for run in ("01", "02"):
        events = (
            repository
            / "source_data"
            / "bids"
            / subject
            / "func"
            / f"{subject}_task-ultimatum_run-{run}_events.tsv"
        )
        if not events.is_file():
            raise FileNotFoundError(events)
        ev_prefix = ev_dir / f"run-{run}"
        counts = generate(events, ev_prefix, "companion")
        if counts["event_RT"] != 63:
            raise ValueError(
                f"expected 63 companion RT rows for corrected {subject} run-{run}, "
                f"found {counts['event_RT']}"
            )

        act_source = production_root / subject / (
            f"L1_task-ultimatum_model-02_type-act_run-{run}_sm-6.feat/design.fsf"
        )
        if not act_source.is_file():
            raise FileNotFoundError(act_source)
        act_text = act_source.read_text(encoding="utf-8", errors="replace")
        bold = discover_bold(dataset_root, subject, run)
        confound = discover_confound(
            dataset_root,
            production_root.parent.parent,
            subject,
            run,
            fsf_value(act_text, "confoundev_files(1)"),
        )

        for model in MODELS:
            source = production_root / subject / (
                f"L1_task-ultimatum_model-02_type-{model}_run-{run}_sm-6.feat/design.fsf"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            output_root = output_parent / (
                f"L1_task-ultimatum_model-02_type-{model}_run-{run}_sm-6"
            )
            destination = fsf_dir / f"L1_{subject}_{model}_run-{run}.fsf"
            render_l1(
                source,
                destination,
                output_root,
                bold,
                confound,
                ev_prefix,
                counts,
                production_root.parent.parent,
                dataset_root,
            )
            l1_outputs[(model, run)] = Path(f"{output_root}.feat")
            rows.append(
                {
                    "stage": "l1",
                    "model": model,
                    "run": run,
                    "fsf": destination,
                    "output": Path(f"{output_root}.feat"),
                    "inputs": f"{bold}|{confound}|{events}",
                    "source_fsf": source,
                    "event_sha256": sha256(events),
                    "rt_policy": "companion_event_RT",
                }
            )

    for model in MODELS:
        source = production_root / subject / (
            f"L2_task-ultimatum_model-02_type-{model}_sm-6.gfeat/design.fsf"
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        output_root = output_parent / f"L2_task-ultimatum_model-02_type-{model}_sm-6"
        destination = fsf_dir / f"L2_{subject}_{model}.fsf"
        render_l2(
            source,
            destination,
            output_root,
            l1_outputs[(model, "01")],
            l1_outputs[(model, "02")],
        )
        rows.append(
            {
                "stage": "l2",
                "model": model,
                "run": "",
                "fsf": destination,
                "output": Path(f"{output_root}.gfeat"),
                "inputs": f"{l1_outputs[(model, '01')]}|{l1_outputs[(model, '02')]}",
                "source_fsf": source,
                "event_sha256": "",
                "rt_policy": "companion_event_RT",
            }
        )

    manifest = work_root / "repair_jobs.tsv"
    write_manifest(manifest, rows)
    print(
        f"PASS: prepared {sum(row['stage'] == 'l1' for row in rows)} L1 and "
        f"{sum(row['stage'] == 'l2' for row in rows)} L2 jobs; manifest={manifest}"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=root)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--production-fsl-root",
        type=Path,
        default=root / "derivatives" / "fsl",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--subject", default="sub-144")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepare(
        args.repository.resolve(),
        args.dataset_root.resolve(),
        args.production_fsl_root.resolve(),
        args.work_root.resolve(),
        args.subject,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
