#!/usr/bin/env python3
"""Run a transparent exploratory analysis of the Ultimatum partner ratings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


AMBIGUOUS_SESSIONS = {
    ("sub-105", "post"),
    ("sub-134", "post"),
    ("sub-144", "pre"),
    ("sub-144", "post"),
}
AMBIGUOUS_PARTICIPANTS = {participant for participant, _ in AMBIGUOUS_SESSIONS}
PARTNER_CONTRASTS = {
    "human_mean_minus_computer": {"similar": 0.5, "dissimilar": 0.5, "computer": -1.0},
    "similar_minus_computer": {"similar": 1.0, "computer": -1.0},
    "dissimilar_minus_computer": {"dissimilar": 1.0, "computer": -1.0},
    "similar_minus_dissimilar": {"similar": 1.0, "dissimilar": -1.0},
}
POLICIES = {
    "last_complete_block": "Retain the final complete block from each file.",
    "first_complete_block": "Retain the first complete block from each file.",
    "exclude_ambiguous_sessions": (
        "Exclude the four participant-session files containing changed "
        "repeated Ultimatum blocks."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fdr_bh(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment, preserving missing values and row order."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    observed = values.dropna().astype(float)
    if observed.empty:
        return result
    order = observed.sort_values().index
    ranked = observed.loc[order].to_numpy()
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def load_inputs(
    ratings_path: Path, sample_path: Path, participants_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path, sep="\t")
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    participants = pd.read_csv(participants_path, sep="\t")

    required_ratings = {
        "participant_id",
        "task",
        "timepoint",
        "source_block",
        "source_file",
        "partner",
        "rating_dimension",
        "response",
    }
    missing = required_ratings - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings table lacks columns: {sorted(missing)}")
    if "subjID" not in sample or sample["subjID"].duplicated().any():
        raise ValueError("sample table needs one unique subjID row per participant")
    if "participant_id" not in participants or participants["participant_id"].duplicated().any():
        raise ValueError("participants table needs one unique participant_id row per participant")

    sample = sample.merge(
        participants[["participant_id", "age", "sex"]],
        left_on="subjID",
        right_on="participant_id",
        how="left",
        validate="one_to_one",
    )
    sample["age_group"] = np.where(sample["younger"] == 1, "younger", "older")
    if sample[["age", "sex"]].isna().any().any():
        raise ValueError("paper sample contains participants absent from participants.tsv")

    ratings = ratings[
        (ratings["task"] == "ultimatum")
        & ratings["participant_id"].isin(sample["subjID"])
    ].copy()
    ratings["response"] = pd.to_numeric(ratings["response"], errors="raise")
    ratings["source_block"] = pd.to_numeric(
        ratings["source_block"], errors="raise"
    ).astype(int)
    return ratings, sample


def select_policy(ratings: pd.DataFrame, policy: str) -> pd.DataFrame:
    selected = ratings.copy()
    if policy == "exclude_ambiguous_sessions":
        ambiguous = pd.MultiIndex.from_frame(
            selected[["participant_id", "timepoint"]]
        ).isin(AMBIGUOUS_SESSIONS)
        selected = selected[~ambiguous]
    elif policy == "last_complete_block":
        final_block = selected.groupby(
            ["participant_id", "timepoint"]
        )["source_block"].transform("max")
        selected = selected[selected["source_block"] == final_block]
    elif policy == "first_complete_block":
        first_block = selected.groupby(
            ["participant_id", "timepoint"]
        )["source_block"].transform("min")
        selected = selected[selected["source_block"] == first_block]
    else:
        raise ValueError(f"unknown policy: {policy}")

    keys = ["participant_id", "timepoint", "rating_dimension", "partner"]
    if selected.duplicated(keys).any():
        raise ValueError(f"policy {policy} leaves duplicate rating cells")
    return selected


def build_sample_table(ratings: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in sample.sort_values("subjID").to_dict("records"):
        participant = record["subjID"]
        subject = ratings[ratings["participant_id"] == participant]
        row: dict[str, object] = {
            "participant_id": participant,
            "age": record["age"],
            "sex": record["sex"],
            "age_group": record["age_group"],
            "has_any_ultimatum_ratings": not subject.empty,
            "ambiguous_changed_repeat": participant in AMBIGUOUS_PARTICIPANTS,
            "primary_rule": "last_complete_block",
        }
        for timepoint in ("pre", "post"):
            part = subject[subject["timepoint"] == timepoint]
            row[f"{timepoint}_present"] = not part.empty
            row[f"{timepoint}_source_file"] = (
                "" if part.empty else ";".join(sorted(part["source_file"].unique()))
            )
            row[f"{timepoint}_complete_blocks"] = (
                0 if part.empty else int(part["source_block"].nunique())
            )
            row[f"{timepoint}_primary_block"] = (
                "" if part.empty else int(part["source_block"].max())
            )
        if participant == "sub-143":
            row["integrity_note"] = "no Ultimatum ratings source file exists"
        elif participant == "sub-144":
            row["integrity_note"] = (
                "ratings are unique to sub-144; both source files contain two blocks"
            )
        else:
            row["integrity_note"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def wide_cells(selected: pd.DataFrame) -> pd.DataFrame:
    return selected.pivot(
        index="participant_id",
        columns=["timepoint", "rating_dimension", "partner"],
        values="response",
    )


def contrast_series(
    wide: pd.DataFrame, timepoint: str, dimension: str, weights: dict[str, float]
) -> pd.Series:
    columns = [(timepoint, dimension, partner) for partner in weights]
    data = wide.reindex(columns=columns).dropna()
    result = sum(data[column] * weights[column[2]] for column in columns)
    return result.rename("difference")


def one_sample_row(
    values: pd.Series,
    *,
    policy: str,
    family: str,
    dimension: str,
    comparison: str,
    timepoint: str,
) -> tuple[dict[str, object], pd.Series]:
    values = values.dropna().astype(float)
    if len(values) < 2:
        statistic = p_value = standard_deviation = standard_error = effect_dz = np.nan
    else:
        statistic, p_value = stats.ttest_1samp(values, 0.0)
        standard_deviation = values.std(ddof=1)
        standard_error = standard_deviation / np.sqrt(len(values))
        effect_dz = values.mean() / standard_deviation if standard_deviation else np.nan
    row = {
        "policy": policy,
        "test_family": family,
        "dimension": dimension,
        "timepoint": timepoint,
        "comparison": comparison,
        "n": len(values),
        "mean_difference": values.mean() if len(values) else np.nan,
        "sem_difference": standard_error,
        "t": statistic,
        "df": len(values) - 1 if len(values) else np.nan,
        "p_uncorrected": p_value,
        "cohens_dz": effect_dz,
    }
    return row, values


def build_descriptives(selected_by_policy: dict[str, pd.DataFrame], sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_map = sample.set_index("subjID")["age_group"]
    for policy, selected in selected_by_policy.items():
        selected = selected.assign(age_group=selected["participant_id"].map(group_map))
        for age_group, group in [("all", selected), *selected.groupby("age_group")]:
            summary = (
                group.groupby(["timepoint", "rating_dimension", "partner"])["response"]
                .agg(n="count", mean="mean", sd="std", sem="sem", median="median")
                .reset_index()
            )
            summary.insert(0, "age_group", age_group)
            summary.insert(0, "policy", policy)
            rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def build_contrasts(
    selected_by_policy: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str, str], pd.Series]]:
    rows: list[dict[str, object]] = []
    values_by_test: dict[tuple[str, str, str, str, str], pd.Series] = {}
    observed_dimensions = {
        ("pre", "fairness"),
        ("pre", "likeability"),
        ("post", "fairness"),
        ("post", "likeability"),
        ("post", "anger"),
        ("post", "satisfaction"),
    }
    for policy, selected in selected_by_policy.items():
        wide = wide_cells(selected)
        for timepoint, dimension in sorted(observed_dimensions):
            for comparison, weights in PARTNER_CONTRASTS.items():
                values = contrast_series(wide, timepoint, dimension, weights)
                row, values = one_sample_row(
                    values,
                    policy=policy,
                    family="partner_contrasts",
                    dimension=dimension,
                    comparison=comparison,
                    timepoint=timepoint,
                )
                rows.append(row)
                values_by_test[(policy, "partner_contrasts", dimension, timepoint, comparison)] = values

        for dimension in ("fairness", "likeability"):
            for partner in ("similar", "dissimilar", "computer"):
                pre = wide.get(("pre", dimension, partner))
                post = wide.get(("post", dimension, partner))
                values = pd.concat([pre, post], axis=1, keys=["pre", "post"]).dropna()
                difference = values["post"] - values["pre"]
                comparison = f"post_minus_pre_{partner}"
                row, difference = one_sample_row(
                    difference,
                    policy=policy,
                    family="pre_post_changes",
                    dimension=dimension,
                    comparison=comparison,
                    timepoint="post-minus-pre",
                )
                rows.append(row)
                values_by_test[(policy, "pre_post_changes", dimension, "post-minus-pre", comparison)] = difference

            for comparison, weights in PARTNER_CONTRASTS.items():
                pre = contrast_series(wide, "pre", dimension, weights)
                post = contrast_series(wide, "post", dimension, weights)
                values = pd.concat([pre, post], axis=1, keys=["pre", "post"]).dropna()
                difference = values["post"] - values["pre"]
                row, difference = one_sample_row(
                    difference,
                    policy=policy,
                    family="change_in_partner_contrast",
                    dimension=dimension,
                    comparison=f"post_minus_pre_of_{comparison}",
                    timepoint="post-minus-pre",
                )
                rows.append(row)
                values_by_test[(
                    policy,
                    "change_in_partner_contrast",
                    dimension,
                    "post-minus-pre",
                    f"post_minus_pre_of_{comparison}",
                )] = difference

    table = pd.DataFrame(rows)
    table["p_fdr_bh_within_family"] = table.groupby(
        ["policy", "test_family"], group_keys=False
    )["p_uncorrected"].apply(fdr_bh)
    return table, values_by_test


def build_age_tests(
    values_by_test: dict[tuple[str, str, str, str, str], pd.Series],
    sample: pd.DataFrame,
) -> pd.DataFrame:
    sample_index = sample.set_index("subjID")
    rows: list[dict[str, object]] = []
    for key, values in values_by_test.items():
        policy, family, dimension, timepoint, comparison = key
        if policy != "last_complete_block":
            continue
        frame = values.rename("contrast").to_frame().join(
            sample_index[["age", "age_group"]], how="left"
        ).dropna()
        if len(frame) >= 3 and frame["age"].nunique() > 1:
            statistic, p_value = stats.pearsonr(frame["age"], frame["contrast"])
        else:
            statistic = p_value = np.nan
        rows.append(
            {
                "test_family": family,
                "dimension": dimension,
                "timepoint": timepoint,
                "comparison": comparison,
                "age_test": "pearson_exact_age",
                "n": len(frame),
                "estimate": statistic,
                "estimate_definition": "Pearson r",
                "statistic": statistic,
                "df": len(frame) - 2 if len(frame) else np.nan,
                "p_uncorrected": p_value,
            }
        )
        younger = frame.loc[frame["age_group"] == "younger", "contrast"]
        older = frame.loc[frame["age_group"] == "older", "contrast"]
        statistic, p_value = stats.ttest_ind(
            older, younger, equal_var=False, nan_policy="omit"
        )
        rows.append(
            {
                "test_family": family,
                "dimension": dimension,
                "timepoint": timepoint,
                "comparison": comparison,
                "age_test": "older_minus_younger_welch",
                "n": len(frame),
                "estimate": older.mean() - younger.mean(),
                "estimate_definition": "older mean minus younger mean",
                "statistic": statistic,
                "df": np.nan,
                "p_uncorrected": p_value,
            }
        )
    table = pd.DataFrame(rows)
    table["p_fdr_bh_within_age_test"] = table.groupby(
        "age_test", group_keys=False
    )["p_uncorrected"].apply(fdr_bh)
    return table


def write_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, sep="\t", index=False, float_format="%.8g", na_rep="n/a")


def main() -> int:
    args = parse_args()
    ratings, sample = load_inputs(args.ratings, args.sample, args.participants)
    selected = {policy: select_policy(ratings, policy) for policy in POLICIES}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(
        build_sample_table(ratings, sample),
        args.output_dir / "ultimatum_ratings_sample.tsv",
    )
    write_table(
        build_descriptives(selected, sample),
        args.output_dir / "ultimatum_ratings_descriptives.tsv",
    )
    contrasts, values_by_test = build_contrasts(selected)
    write_table(
        contrasts,
        args.output_dir / "ultimatum_ratings_within_subject_tests.tsv",
    )
    write_table(
        build_age_tests(values_by_test, sample),
        args.output_dir / "ultimatum_ratings_age_tests.tsv",
    )
    print(f"PASS: analyzed {len(sample)} paper participants under {len(POLICIES)} policies")
    print(f"Participants with any Ultimatum ratings: {ratings['participant_id'].nunique()}")
    print(f"Changed-repeat participants: {','.join(sorted(AMBIGUOUS_PARTICIPANTS))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
