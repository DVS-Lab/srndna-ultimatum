"""Independent structural, timing, and real-FSL tests of partner post epochs."""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tests"))
from audit_l1_designs import read_vest_matrix
from audit_l1_estimability import diagnostics
from decision_postresponse_model import (
    CONTRAST, PAIR, SINGLE, public_template, rebuild_fsf, render_l1, settings,
    split_epochs,
)
from make_ultimatum_3col import PARTNERS, TRIAL_RE, read_events
from prepare_sub144_imaging_repair import fsf_value
from prepare_ultimatum_l3_repair import parse_inputs
from run_full_rt_correction import MODELS
from test_decision_postresponse_model import source_fixture, template


FSL_FEAT = shutil.which("feat_model")
if not FSL_FEAT and os.environ.get("FSLDIR"):
    configured_feat = Path(os.environ["FSLDIR"]) / "bin/feat_model"
    FSL_FEAT = str(configured_feat) if configured_feat.is_file() else None


def semantic_contrasts(family, nuisance_columns=0):
    """Build scientific contrasts directly, independently of the remapping code."""
    n = 10 if family == "act" else 30
    columns = (1, 2, 3, 4, 5, 6) if family == "act" else (12, 13, 14, 15, 16, 17)
    matrix = np.zeros((10 if family == "act" else 11, n + nuisance_columns))
    computer, computer_offer, similar, similar_offer, dissimilar, dissimilar_offer = columns
    specifications = [
        {computer: 1}, {computer_offer: 1}, {similar: 1}, {similar_offer: 1},
        {dissimilar: 1}, {dissimilar_offer: 1},
        {similar_offer: 1, dissimilar_offer: -1},
        {computer_offer: -2, similar_offer: 1, dissimilar_offer: 1},
        {similar: 1, dissimilar: -1},
        {computer: -2, similar: 1, dissimilar: 1},
    ]
    if family == "nppi":
        specifications.append({11: 1})
    for row, values in enumerate(specifications):
        for column, weight in values.items():
            matrix[row, column - 1] = weight
    return matrix


def event_file(subject, run="01"):
    return ROOT / f"source_data/bids/{subject}/func/{subject}_task-ultimatum_run-{run}_events.tsv"


class PartnerPostresponseModelTests(unittest.TestCase):
    def test_complete_ev_arrays_and_semantic_contrasts(self):
        for family, n in (("act", 10), ("nppi", 30)):
            with self.subTest(family=family):
                original = template(family)
                text = public_template(original, post_model="partner")
                old, new = settings(original), settings(text)
                self.assertEqual(new["evs_orig"], str(n))
                self.assertEqual(new["evs_real"], str(n))
                expected = semantic_contrasts(family)
                self.assertEqual(new["ncon_real"], str(len(expected)))
                self.assertEqual(new["ncon_orig"], str(len(expected)))
                for prefix in ("evtitle", "shape", "convolve", "convolve_phase",
                               "tempfilt_yn", "deriv_yn"):
                    indices = {int(m[1]) for key in new
                               if (m := re.fullmatch(prefix + r"(\d+)", key))}
                    self.assertEqual(indices, set(range(1, n + 1)))
                for key in new:
                    if (match := SINGLE.fullmatch(key)):
                        self.assertTrue(1 <= int(match[2]) <= n)
                    if (match := PAIR.fullmatch(key)):
                        self.assertTrue(1 <= int(match[2]) <= n)
                        self.assertTrue(0 <= int(match[3]) <= n)
                    if (match := CONTRAST.fullmatch(key)):
                        self.assertTrue(1 <= int(match[2]) <= len(expected))
                        self.assertTrue(1 <= int(match[3]) <= n)
                for mode in ("orig", "real"):
                    actual = np.array([[float(new[f"con_{mode}{c}.{ev}"])
                                        for ev in range(1, n + 1)]
                                       for c in range(1, len(expected) + 1)])
                    np.testing.assert_array_equal(actual, expected)
                    keys = {key for key in new if key.startswith(f"con_{mode}")}
                    self.assertEqual(len(keys), n * len(expected))
                for ev in range(1, n + 1):
                    for other in range(n + 1):
                        self.assertEqual(new[f"ortho{ev}.{other}"],
                                         str(int(ev in (2, 4, 6) and other in (0, ev - 1))))
                    self.assertEqual(new[f"deriv_yn{ev}"], "0")
                    self.assertEqual(new[f"convolve{ev}"], "3" if ev <= 10 else "0")
                    self.assertNotIn(fsf_value(text, f"evtitle{ev}"), ("rt", "rt_p"))
                for ev, partner in zip((8, 9, 10), PARTNERS):
                    self.assertEqual(new[f"shape{ev}"], "3")
                    self.assertIn(partner, fsf_value(text, f"evtitle{ev}"))
                    self.assertIn(partner, fsf_value(text, f"custom{ev}"))
                self.assertEqual(new["custom7"], old["custom7"])
                # Every setting outside the rebuilt arrays is invariant except counts.
                for key, value in old.items():
                    if SINGLE.fullmatch(key) or PAIR.fullmatch(key) or CONTRAST.fullmatch(key):
                        continue
                    if key not in ("evs_orig", "evs_real"):
                        self.assertEqual(new[key], value, key)
                if family == "nppi":
                    self.assertEqual(new["custom11"], old["custom10"])
                    for old_ev in range(20, 29):
                        self.assertEqual(new[f"custom{old_ev + 2}"], old[f"custom{old_ev}"])

    def test_all_ten_interactions_have_only_the_intended_parents_and_centering(self):
        config = settings(public_template(template("nppi"), post_model="partner"))
        interaction_evs = {int(m[1]) for key in config
                           if (m := re.fullmatch(r"interactions(\d+)\.\d+", key))}
        self.assertEqual(interaction_evs, set(range(12, 22)))
        for ev in range(12, 22):
            psych = ev - 11
            self.assertEqual(config[f"shape{ev}"], "4")
            self.assertNotIn(f"custom{ev}", config)
            for parent in range(1, ev):
                self.assertEqual(config[f"interactions{ev}.{parent}"],
                                 str(int(parent in (psych, 11))))
                mode = 2 if parent == 11 else 1 if parent == psych and psych in (2, 4, 6) else 0
                self.assertEqual(config[f"interactionsd{ev}.{parent}"], str(mode))
            for prefix in ("interactions", "interactionsd"):
                parents = {int(m[1]) for key in config
                           if (m := re.fullmatch(prefix + str(ev) + r"\.(\d+)", key))}
                self.assertEqual(parents, set(range(1, ev)))

    def test_pooled_templates_remain_byte_identical_and_new_templates_are_reproducible(self):
        historical_hashes = {
            "act": "528b9ed197629f8f9f05e34e5108565342a68756f54072567980c29eaa269bb6",
            "nppi": "649b2f928b076cf035ae7f34e1b7e2d17bf16298e7db834c3396e70b4ba24601",
        }
        for family in ("act", "nppi"):
            with self.subTest(family=family):
                pooled = ROOT / f"templates/revision/L1_task-ultimatum_model-decision-postresponse_type-{family}.fsf"
                self.assertEqual(hashlib.sha256(pooled.read_bytes()).hexdigest(), historical_hashes[family])
                self.assertEqual(public_template(template(family)), pooled.read_text())
                self.assertEqual(public_template(template(family), post_model="pooled"), pooled.read_text())
                partner = ROOT / f"templates/revision/L1_task-ultimatum_model-decision-postresponse-partner_type-{family}.fsf"
                self.assertEqual(public_template(template(family), post_model="partner"), partner.read_text())

    def test_render_preserves_all_sources_amplitudes_miss_confounds_and_networks(self):
        events = read_events(event_file("sub-105"))
        with tempfile.TemporaryDirectory() as temporary:
            for family, n in (("act", 10), ("nppi", 30)):
                root = Path(temporary) / family
                source, standard, source_path, confounds = source_fixture(root / "source", family, events)
                original = {path: path.read_bytes() for path in (root / "source").iterdir()}
                rendered, required, stats = render_l1(
                    source, root / "output", root / "evs", events, standard,
                    source_fsf=source_path, post_model="partner")
                self.assertEqual(stats["responded_trials"], 71)
                self.assertEqual(stats["n_evs"], n)
                self.assertEqual([stats[f"post_rows_{p}"] for p in PARTNERS], [15, 24, 32])
                self.assertEqual(Path(fsf_value(rendered, "custom7")).resolve(),
                                 Path(fsf_value(source, "custom7")).resolve())
                self.assertEqual(fsf_value(rendered, "confoundev_files(1)"), str(confounds.resolve()))
                self.assertTrue(all(path.is_file() for path in required))
                self.assertEqual(len(list((root / "evs").iterdir())), 9)
                for ev in range(1, 7):
                    before = np.loadtxt(fsf_value(source, f"custom{ev}"), ndmin=2)
                    after = np.loadtxt(fsf_value(rendered, f"custom{ev}"), ndmin=2)
                    np.testing.assert_array_equal(before[:, [0, 2]], after[:, [0, 2]])
                    self.assertTrue(np.all(after[:, 1] < before[:, 1]))
                for ev, partner in zip((8, 9, 10), PARTNERS):
                    post = np.loadtxt(fsf_value(rendered, f"custom{ev}"), ndmin=2)
                    np.testing.assert_allclose(post, split_epochs(events)[1][partner], atol=1e-9, rtol=0)
                    self.assertTrue(np.all(post[:, 2] == 1))
                self.assertTrue(all(path.read_bytes() == data for path, data in original.items()))
                with self.assertRaises(FileExistsError):
                    render_l1(source, root / "output", root / "evs", events, standard,
                              source_fsf=source_path, post_model="partner")

    def test_all_94_runs_partition_partner_phases_and_exclude_misses(self):
        members = parse_inputs((ROOT / "templates/revision" / MODELS["act"]).read_text().splitlines())
        runs = responded_count = miss_count = 0
        with tempfile.TemporaryDirectory() as temporary:
            for _, subject, _ in members:
                for run in ("01", "02"):
                    events = read_events(event_file(subject, run))
                    decisions, post = split_epochs(events)
                    misses = [row for row in events if row["trial_type"] == "missed_trial"]
                    self.assertEqual(len(decisions) + len(misses), 72)
                    root = Path(temporary) / f"{subject}-{run}"
                    source, standard, source_path, _ = source_fixture(root / "source", "act", events)
                    text, _, stats = render_l1(source, root / "out", root / "evs", events, standard,
                                              source_fsf=source_path, post_model="partner")
                    missed_onsets = {float(row["onset"]) for row in misses}
                    for offset, partner in enumerate(PARTNERS):
                        main = np.loadtxt(fsf_value(text, f"custom{2 * offset + 1}"), ndmin=2)
                        offer = np.loadtxt(fsf_value(text, f"custom{2 * offset + 2}"), ndmin=2)
                        tail = np.loadtxt(fsf_value(text, f"custom{8 + offset}"), ndmin=2)
                        subset = [row for row in decisions
                                  if TRIAL_RE.fullmatch(row["trial_type"]).group("partner") == partner]
                        self.assertEqual(len(main), len(subset))
                        self.assertEqual(len(tail), len(subset))
                        self.assertEqual(stats[f"post_rows_{partner}"], len(subset))
                        np.testing.assert_array_equal(main[:, :2], offer[:, :2])
                        np.testing.assert_array_equal(main[:, 2], np.ones(len(main)))
                        np.testing.assert_array_equal(offer[:, 2], [float(row["Offer"]) for row in subset])
                        np.testing.assert_allclose(main[:, 1], [float(row["response_time"]) for row in subset], rtol=0, atol=1e-9)
                        np.testing.assert_allclose(tail, post[partner], rtol=0, atol=1e-9)
                        self.assertFalse(missed_onsets.intersection(main[:, 0]))
                        for index, row in enumerate(subset):
                            onset, duration = float(row["onset"]), float(row["duration"])
                            self.assertAlmostEqual(main[index, 0] + main[index, 1], tail[index, 0], places=8)
                            self.assertAlmostEqual(tail[index, 0] + tail[index, 1], onset + duration, places=8)
                        # No positive-duration event overlap with missed trials.
                        for miss in misses:
                            left = float(miss["onset"])
                            right = left + float(miss["duration"])
                            for segment in (main, offer, tail):
                                overlap = np.minimum(right, segment[:, 0] + segment[:, 1]) - np.maximum(left, segment[:, 0])
                                self.assertFalse(np.any(overlap > 1e-8))
                    runs += 1
                    responded_count += len(decisions)
                    miss_count += len(misses)
        self.assertEqual((runs, responded_count, miss_count), (94, 6654, 114))

    def test_invalid_mode_mapping_source_and_epochs_fail_without_creating_outputs(self):
        with self.assertRaises(ValueError):
            public_template(template("act"), post_model="invalid")
        files = {ev: Path(f"/example/ev{ev}.txt") for ev in (*range(1, 7), 8, 9, 10)}
        for bad in ({key: value for key, value in files.items() if key != 10},
                    files | {11: Path("/example/ev11.txt")}):
            with self.assertRaises(ValueError):
                rebuild_fsf(template("act"), bad, template=True, post_model="partner")
        bad_sources = [
            template("nppi").replace("set fmri(deriv_yn1) 0", "set fmri(deriv_yn1) 1"),
            template("nppi").replace("set fmri(interactions11.10) 1", "set fmri(interactions11.10) 0"),
            template("nppi").replace("set fmri(con_real7.14) 1", "set fmri(con_real7.14) 2"),
        ]
        for source in bad_sources:
            with self.assertRaises(ValueError):
                public_template(source, post_model="partner")
        events = read_events(event_file("sub-105"))
        responded = next(row for row in events if TRIAL_RE.fullmatch(row["trial_type"]))
        bad_events = [events + [responded],
                      [dict(row, response_time=row["duration"]) if row is responded else row for row in events],
                      [row for row in events if not row["trial_type"].endswith("ingroup")]]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, standard, source_path, _ = source_fixture(root / "source", "act", events)
            for index, bad in enumerate(bad_events):
                destination = root / f"bad-{index}"
                with self.assertRaises(ValueError):
                    render_l1(source, root / "out", destination, bad, standard,
                              source_fsf=source_path, post_model="partner")
                self.assertFalse(destination.exists())

    def test_duplicate_overlapping_nonfinite_or_nonpositive_misses_are_rejected(self):
        events = read_events(event_file("sub-105"))
        missed = next(row for row in events if row["trial_type"] == "missed_trial")
        responded = next(row for row in events if TRIAL_RE.fullmatch(row["trial_type"]))
        invalid = [
            events + [missed],
            events + [dict(missed, onset=responded["onset"])],
            events + [dict(missed, onset=str(float(responded["onset"]) + .5))],
            [dict(row, duration="0") if row is missed else row for row in events],
            [dict(row, onset="nan") if row is missed else row for row in events],
        ]
        for index, rows in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                split_epochs(rows)
        # Touching boundaries have no positive overlap; companion BIDS rows
        # represent the same trial and are not additional modeled trials.
        adjacent = [dict(onset=str(3.5 * index), duration="3.5", response_time="1.2",
                         trial_type=f"event_accept_{partner}", Offer="7")
                    for index, partner in enumerate(PARTNERS)]
        adjacent.append(dict(onset="10.5", duration="3.5", response_time="n/a",
                             trial_type="missed_trial", Offer="7"))
        adjacent.append(dict(adjacent[0], trial_type="event_computer"))
        rows, post = split_epochs(adjacent)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(len(values) for values in post.values()), 3)

    def test_retained_missed_ev_shape_or_numeric_content_mismatch_is_rejected(self):
        events = read_events(event_file("sub-105"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, standard, source_path, _ = source_fixture(root / "source", "nppi", events)
            missed_path = Path(fsf_value(source, "custom7"))
            original = missed_path.read_bytes()
            bad_shape = source.replace("set fmri(shape7) 3", "set fmri(shape7) 10")
            with self.assertRaisesRegex(ValueError, "missed-trial EV shape"):
                render_l1(bad_shape, root / "out", root / "bad-shape", events, standard,
                          source_fsf=source_path, post_model="partner")
            self.assertFalse((root / "bad-shape").exists())
            for index, content in enumerate(("245.226\t3.539990\t1.0\n",
                                             "245.126\t3.539990\t2.0\n",
                                             "245.126\t3.539990\tnan\n",
                                             original.decode() * 2)):
                with self.subTest(case=index):
                    missed_path.write_text(content)
                    target = root / f"bad-miss-{index}"
                    with self.assertRaisesRegex(ValueError, "missed-trial EV differs"):
                        render_l1(source, root / "out", target, events, standard,
                                  source_fsf=source_path, post_model="partner")
                    self.assertFalse(target.exists())
            missed_path.write_bytes(original)
            rounded = np.loadtxt(missed_path, ndmin=2)
            rounded[0, 0] += .0004
            np.savetxt(missed_path, rounded, delimiter="\t")
            # Submillisecond historical serialization differences are allowed.
            _, _, stats = render_l1(source, root / "out", root / "rounded", events, standard,
                                    source_fsf=source_path, post_model="partner")
            self.assertEqual(stats["responded_trials"], 71)

    @unittest.skipUnless(FSL_FEAT, "requires real FSL feat_model")
    def test_real_fsl_compilation_with_and_without_a_real_missed_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            for family, n in (("act", 10), ("nppi", 30)):
                for subject, missed in (("sub-104", False), ("sub-105", True)):
                    with self.subTest(family=family, subject=subject):
                        events = read_events(event_file(subject))
                        root = Path(temporary) / f"{family}-{subject}"
                        source, standard, source_path, confounds = source_fixture(root / "source", family, events)
                        text, _, stats = render_l1(source, root / "out", root / "evs", events, standard,
                                                  source_fsf=source_path, post_model="partner")
                        fsf = root / "partner.fsf"
                        fsf.write_text(text)
                        result = subprocess.run([FSL_FEAT, str(fsf.with_suffix("")), str(confounds)],
                                                capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        matrix = read_vest_matrix(fsf.with_suffix(".mat"))
                        contrasts = read_vest_matrix(fsf.with_suffix(".con"))
                        self.assertEqual(stats["n_evs"], n)
                        self.assertEqual(matrix.shape, (250, n + 3))
                        np.testing.assert_array_equal(contrasts, semantic_contrasts(family, nuisance_columns=3))
                        metrics, checks = diagnostics(matrix, contrasts)
                        self.assertEqual(metrics["remaining_deficiency_after_zero_removal"], 0)
                        self.assertEqual(metrics["zero_columns"], [] if missed else [7] if family == "act" else [7, 18])
                        self.assertTrue(all(row["estimable"] and row["sensitivity_estimable"] for row in checks))
                        self.assertTrue(all(not row["weights_on_zero_columns"] for row in checks))


if __name__ == "__main__":
    unittest.main()
