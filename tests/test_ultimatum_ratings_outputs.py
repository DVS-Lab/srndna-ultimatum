from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reviewer" / "ultimatum_ratings"


def test_ultimatum_ratings_sample_contract():
    sample = pd.read_csv(RESULTS / "ultimatum_ratings_sample.tsv", sep="\t")
    assert len(sample) == 47
    assert sample["participant_id"].nunique() == 47
    assert sample["has_any_ultimatum_ratings"].sum() == 43

    sub143 = sample.set_index("participant_id").loc["sub-143"]
    assert not sub143["has_any_ultimatum_ratings"]
    assert sub143["integrity_note"] == "no Ultimatum ratings source file exists"

    sub144 = sample.set_index("participant_id").loc["sub-144"]
    assert sub144["ambiguous_changed_repeat"]
    assert sub144["pre_complete_blocks"] == 2
    assert sub144["post_complete_blocks"] == 2


def test_ultimatum_ratings_sensitivity_contract():
    tests = pd.read_csv(
        RESULTS / "ultimatum_ratings_within_subject_tests.tsv", sep="\t"
    )
    assert set(tests["policy"]) == {
        "last_complete_block",
        "first_complete_block",
        "exclude_ambiguous_sessions",
    }
    key = tests[
        (tests["test_family"] == "partner_contrasts")
        & (tests["comparison"] == "human_mean_minus_computer")
        & (tests["dimension"].isin(["fairness", "likeability"]))
    ]
    assert len(key) == 12
    assert (key["mean_difference"] > 0).all()
    assert (key["p_fdr_bh_within_family"] < 0.05).all()


def test_fairness_sensitivity_rating_association_contract():
    associations = pd.read_csv(
        RESULTS / "ultimatum_ratings_fairness_sensitivity.tsv", sep="\t"
    )
    assert len(associations) == 24
    assert associations.groupby("policy").size().eq(8).all()

    primary = associations[
        associations["policy"] == "last_complete_block"
    ].set_index(["rating_dimension", "rating_timepoint"])
    assert primary.loc[("fairness", "pre"), "pearson_p"] > 0.5
    assert primary.loc[("fairness", "post"), "pearson_p"] > 0.5
    assert primary.loc[("likeability", "pre"), "pearson_r"] < 0
    assert primary.loc[("anger", "post"), "pearson_r"] > 0
