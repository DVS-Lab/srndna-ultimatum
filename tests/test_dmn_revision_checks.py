import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from audit_corrected_dmn_influence import align_rows, fit_diagnostics
from prepare_dmn_revision_checks import render_group, group_design, CONTRASTS, safe_empty, REFERENCE_NAMES, digest
from prepare_dmn_condition_bar_models import SOURCE_FSF_RELATIVE, parse_inputs
from prepare_dmn_rt_sensitivity import decisions, replace_durations, render_l1
from prepare_activation_fairness_main_pipeline import contrast_vectors
from summarize_dmn_participant_bars import weighted_summary
from collect_dmn_revision_checks import collect
from run_dmn_revision_stage import verify_inputs, verify_stage_outputs


class RevisionTests(unittest.TestCase):
    def test_reference_templates_match_executable_renderer(self):
        source = (ROOT / SOURCE_FSF_RELATIVE).read_text()
        for robust,name in REFERENCE_NAMES.items():
            self.assertEqual((ROOT/'templates/revision'/name).read_text(),
                             render_group(source,Path('OUTPUTDIR'),Path('STANDARD_IMAGE'),robust))

    def test_modified_prepared_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            source=root/'event.txt'
            source.write_text('0 1 1\n')
            (root/'input_provenance.json').write_text(json.dumps({str(source):{'sha256':digest(source)}}))
            verify_inputs(root/'revision_jobs.tsv')
            source.write_text('0 1 2\n')
            with self.assertRaises(ValueError):
                verify_inputs(root/'revision_jobs.tsv')

    def test_missing_cluster_table_is_not_a_null_result(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            fsf=root/'x.fsf'
            fsf.write_text('fixture')
            manifest=root/'revision_jobs.tsv'
            with manifest.open('w') as stream:
                w=csv.DictWriter(stream,fieldnames=['stage','run','output','fsf','fsf_sha256'],delimiter='\t')
                w.writeheader()
                for name in ('simple-effects','outlier-deweighted'):
                    w.writerow(dict(stage='l3',run=name,output=str(root/name),fsf=str(fsf),fsf_sha256=digest(fsf)))
            with self.assertRaises(FileNotFoundError):
                collect(manifest,root/'collected')
            self.assertFalse((root/'collected').exists())

    def test_missing_intermediate_cope_or_variance_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'feat/stats').mkdir(parents=True)
            for i in (1,3):
                for stem in ('cope','varcope','zstat'):
                    (root/f'feat/stats/{stem}{i}.nii.gz').write_bytes(b'fixture')
            manifest=root/'revision_jobs.tsv'
            manifest.write_text(f'stage\toutput\texpected_copes\nl1\t{root/"feat"}\t3\n')
            with self.assertRaises(FileNotFoundError):
                verify_stage_outputs(manifest,'l1')

    def test_weighted_sem_reduces_to_ordinary_sem_when_weights_equal(self):
        y = np.array([1.,2.,4.,7.])
        mean, sem, neff = weighted_summary(y, np.ones(4))
        self.assertAlmostEqual(mean, y.mean())
        self.assertAlmostEqual(sem, y.std(ddof=1)/2)
        self.assertAlmostEqual(neff, 4)
        self.assertEqual(weighted_summary(y, np.ones(4)), weighted_summary(y, 10*np.ones(4)))
        with self.assertRaises(ValueError):
            weighted_summary(y, [1,1,0,1])

    def test_group_design_and_original_contrasts_preserved(self):
        text = (ROOT / SOURCE_FSF_RELATIVE).read_text()
        rendered = render_group(text, Path('/scratch/new'), Path('/fsl/standard.nii.gz'), robust=1)
        np.testing.assert_array_equal(group_design(text), group_design(rendered))
        self.assertEqual(parse_inputs(text.splitlines()), parse_inputs(rendered.splitlines()))
        original = contrast_vectors(text.splitlines(), 'real')
        new = contrast_vectors(rendered.splitlines(), 'real')
        self.assertEqual({i: new[i] for i in range(1, 5)}, original)
        self.assertEqual(new[5], tuple(CONTRASTS[4][1]))
        self.assertEqual(new[6], tuple(CONTRASTS[5][1]))
        self.assertIn('set fmri(robust_yn) 1', rendered)
        self.assertIn('set fmri(poststats_yn) 1', rendered)
        self.assertEqual(rendered.count('set fmri(conmask6_5)'), 1)

    def test_wrong_group_design_rejected(self):
        text = (ROOT / SOURCE_FSF_RELATIVE).read_text().replace('"RT"', '"oldRT"')
        with self.assertRaises(ValueError):
            render_group(text, Path('/tmp/a'), Path('/tmp/std'))

    def test_protected_or_existing_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError):
                safe_empty(root/'output', [root])
            (root/'data').write_text('keep')
            with self.assertRaises(FileExistsError):
                safe_empty(root, [])

    def test_participant_alignment_not_row_order(self):
        inputs = [(1,'sub-1','x'), (2,'sub-2','y')]
        rows = [{'participant':'sub-2','design_index':'2'}, {'participant':'sub-1','design_index':'1'}]
        self.assertEqual(align_rows(rows, inputs)[0]['participant'], 'sub-1')
        with self.assertRaises(ValueError):
            align_rows([rows[0], rows[0]], inputs)

    def test_influence_matches_explicit_leave_one_out(self):
        rng = np.random.default_rng(42)
        x = np.column_stack([np.r_[np.ones(20),np.zeros(20)], np.r_[np.zeros(20),np.ones(20)], rng.normal(size=40)])
        y = x @ np.array([2., -1., .3]) + rng.normal(size=40)
        beta, residual, h, student, cooks, full, loo = fit_diagnostics(x, y)
        self.assertAlmostEqual(h.sum(), 3)
        for i in range(40):
            keep = np.arange(40) != i
            b = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
            self.assertAlmostEqual(loo[i], b[0]-b[1])
        self.assertTrue(np.all(cooks >= 0))

    def test_response_duration_is_raw_rt_not_plus_one(self):
        rows = [dict(trial_type='event_accept_ingroup', onset='2', duration='3.51', response_time='1.65', Offer='8')]
        checked = decisions(rows)
        changed = replace_durations(np.array([[2.,3.51,8.]]), checked, 'ingroup')
        np.testing.assert_array_equal(changed, [[2.,1.65,8.]])
        with self.assertRaises(ValueError):
            decisions([*rows, *rows])
        with self.assertRaises(ValueError):
            decisions([{**rows[0], 'response_time':'999'}])
        with self.assertRaises(ValueError):
            replace_durations(np.array([[3.,3.51,8.]]), checked, 'ingroup')

    def test_fixed_rt_keeps_task_input_and_replaces_both_rt_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'source.txt'
            source.write_text('2 3.51 1\n')
            standard = root/'std.nii.gz'
            standard.write_bytes(b'fixture')
            lines = ['set fmri(evs_orig) 28', 'set fmri(evs_real) 28', 'set fmri(outputdir) "old"',
                     'set fmri(evtitle8) "rt"', 'set fmri(evtitle9) "rt_p"']
            lines += [f'set {k} "{source}"' for k in ('feat_files(1)', 'confoundev_files(1)')]
            lines += [f'set fmri(regstandard) "{standard}"']
            for i in range(1,29):
                lines += [f'set fmri(shape{i}) '+('3' if i in (1,8,9) else '10'), f'set fmri(custom{i}) "{source}"']
            events = [dict(trial_type='event_accept_ingroup', onset='2', duration='3.51', response_time='1.65', Offer='8')]
            rendered, _, n = render_l1('\n'.join(lines), root/'out', root/'ev', events, 'all-trial-rt', standard)
            self.assertIn(f'set fmri(custom1) "{source.resolve()}"', rendered)
            self.assertEqual(n, 1)
            self.assertEqual((root/'ev/event_RT_pmod.txt').read_text(), '2\t0\t1.65\n')


if __name__ == '__main__':
    unittest.main()
