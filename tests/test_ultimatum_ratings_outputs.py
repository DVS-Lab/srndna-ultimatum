import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reviewer" / "ultimatum_ratings"


def read_tsv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_bool(value):
    return value.lower() == "true"


class UltimatumRatingsOutputTests(unittest.TestCase):
    def test_ultimatum_ratings_sample_contract(self):
        sample = read_tsv("ultimatum_ratings_sample.tsv")
        self.assertEqual(len(sample), 47)
        self.assertEqual(len({row["participant_id"] for row in sample}), 47)
        self.assertEqual(
            sum(as_bool(row["has_any_ultimatum_ratings"]) for row in sample), 43
        )

        by_id = {row["participant_id"]: row for row in sample}
        sub143 = by_id["sub-143"]
        self.assertFalse(as_bool(sub143["has_any_ultimatum_ratings"]))
        self.assertEqual(
            sub143["integrity_note"], "no Ultimatum ratings source file exists"
        )

        sub144 = by_id["sub-144"]
        self.assertTrue(as_bool(sub144["ambiguous_changed_repeat"]))
        self.assertEqual(int(sub144["pre_complete_blocks"]), 2)
        self.assertEqual(int(sub144["post_complete_blocks"]), 2)

    def test_ultimatum_ratings_sensitivity_contract(self):
        tests = read_tsv("ultimatum_ratings_within_subject_tests.tsv")
        self.assertEqual(
            {row["policy"] for row in tests},
            {
                "last_complete_block",
                "first_complete_block",
                "exclude_ambiguous_sessions",
            },
        )
        key = [
            row
            for row in tests
            if row["test_family"] == "partner_contrasts"
            and row["comparison"] == "human_mean_minus_computer"
            and row["dimension"] in {"fairness", "likeability"}
        ]
        self.assertEqual(len(key), 12)
        self.assertTrue(all(float(row["mean_difference"]) > 0 for row in key))
        self.assertTrue(all(float(row["p_fdr_bh_within_family"]) < 0.05 for row in key))

    def test_fairness_sensitivity_rating_association_contract(self):
        associations = read_tsv("ultimatum_ratings_fairness_sensitivity.tsv")
        self.assertEqual(len(associations), 24)
        counts = {}
        for row in associations:
            counts[row["policy"]] = counts.get(row["policy"], 0) + 1
        self.assertTrue(all(count == 8 for count in counts.values()))

        primary = {
            (row["rating_dimension"], row["rating_timepoint"]): row
            for row in associations
            if row["policy"] == "last_complete_block"
        }
        self.assertGreater(float(primary[("fairness", "pre")]["pearson_p"]), 0.5)
        self.assertGreater(float(primary[("fairness", "post")]["pearson_p"]), 0.5)
        self.assertLess(float(primary[("likeability", "pre")]["pearson_r"]), 0)
        self.assertGreater(float(primary[("anger", "post")]["pearson_r"]), 0)

    def test_rating_choice_moderation_contract(self):
        focal = read_tsv("rating_choice_moderation_focal_tests.tsv")
        diagnostics = read_tsv("rating_choice_moderation_model_diagnostics.tsv")

        self.assertEqual(len(diagnostics), 24)
        self.assertFalse(any(as_bool(row["singular"]) for row in diagnostics))
        self.assertTrue(
            all(row["convergence_message"] in {"", "n/a"} for row in diagnostics)
        )
        self.assertEqual(
            {row["focal_effect"] for row in focal},
            {
                "common_rating_moderation",
                "age_difference_in_rating_moderation",
            },
        )

        primary = [
            row
            for row in focal
            if row["policy"] == "last_complete_block"
            and row["random_effects"] == "random_offer_slope"
        ]
        common = {
            row["rating_dimension"]: row
            for row in primary
            if row["focal_effect"] == "common_rating_moderation"
            and row["specification"] == "separate_rating_model"
        }
        self.assertEqual(set(common), {"fairness", "likeability"})
        self.assertTrue(all(float(row["p_value"]) > 0.8 for row in common.values()))

        age_difference = {
            row["rating_dimension"]: row
            for row in primary
            if row["focal_effect"] == "age_difference_in_rating_moderation"
        }
        self.assertEqual(set(age_difference), {"fairness", "likeability"})
        self.assertTrue(
            all(float(row["p_value"]) > 0.2 for row in age_difference.values())
        )


if __name__ == "__main__":
    unittest.main()
