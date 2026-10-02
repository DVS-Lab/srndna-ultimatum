import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from audit_l1_designs import read_vest_matrix
from audit_l1_estimability import diagnostics
from decision_postresponse_model import (expected_contrasts, public_template, rebuild_fsf,
                                         render_l1, settings, split_epochs, validate_source)
from make_ultimatum_3col import PARTNERS, ev_rows, read_events, three_column
from prepare_sub144_imaging_repair import fsf_value, replace_fsf_value, replace_fsf_number


def template(family):
    return (ROOT/f'templates/L1_task-ultimatum_model-02_type-{family}.fsf').read_text()


def source_fixture(root, family, events):
    """Rendered legacy FSF with real task timing and reproducible physiological inputs."""
    root.mkdir(parents=True,exist_ok=True)
    s = template(family)
    npoints = 250
    s = replace_fsf_number(s,'npts',npoints)
    s = s.replace('EV_SHAPE','3' if ev_rows(events,'substantive')['missed_trial'] else '10')
    s = s.replace('BBR','6')
    s = replace_fsf_value(s,'outputdir',str(root/'original'),fmri=True)
    standard = root/'standard.nii.gz'
    standard.write_bytes(b'path fixture; feat_model does not read image voxels')
    s = replace_fsf_value(s,'regstandard',str(standard),fmri=True)
    s = replace_fsf_value(s,'feat_files(1)',str(standard),fmri=False)
    rng = np.random.default_rng(2026)
    confounds = root/'confounds.txt'
    np.savetxt(confounds,rng.normal(size=(npoints,3)))
    s = replace_fsf_value(s,'confoundev_files(1)',str(confounds),fmri=False)
    all_rows = ev_rows(events,'substantive')
    names = [f'event_{p}{suf}' for p in PARTNERS for suf in ('','_pmod')]
    names += ['missed_trial','event_RT','event_RT_pmod']
    for i,name in enumerate(names,1):
        path = root/f'{name}.txt'
        path.write_text(three_column(all_rows[name]))
        s = replace_fsf_value(s,f'custom{i}',str(path),fmri=True)
    if family=='nppi':
        for i in (10,*range(20,29)):
            path = root/f'network{i}.txt'
            np.savetxt(path,rng.normal(size=npoints))
            s = replace_fsf_value(s,f'custom{i}',str(path),fmri=True)
    source = root/'design.fsf'
    source.write_text(s)
    return s,standard,source,confounds


class PhaseModelTests(unittest.TestCase):
    def test_all_94_event_files_partition_every_response(self):
        from run_full_rt_correction import MODELS
        from prepare_ultimatum_l3_repair import parse_inputs
        members = parse_inputs((ROOT/'templates/revision'/MODELS['act']).read_text().splitlines())
        count = 0
        for _,subject,_ in members:
            for run in ('01','02'):
                events = read_events(ROOT/f'source_data/bids/{subject}/func/{subject}_task-ultimatum_run-{run}_events.tsv')
                rows,post = split_epochs(events)
                self.assertEqual(sum(map(len,post.values())),len(rows))
                for row in rows:
                    onset,rt,duration = (float(row[k]) for k in ('onset','response_time','duration'))
                    actual = [p for values in post.values() for p in values if abs(p[0]-(onset+rt))<1e-8]
                    self.assertEqual(len(actual),1)
                    self.assertAlmostEqual(rt+actual[0][1],duration)
                    self.assertAlmostEqual(actual[0][0]+actual[0][1],onset+duration)
                count += len(rows)
        self.assertEqual(count,6654)

    def test_duplicate_or_impossible_response_is_rejected(self):
        rows = [dict(onset=str(i*5),duration='3.5',response_time='1.2',
                     trial_type=f'event_accept_{p}',Offer='6') for i,p in enumerate(PARTNERS)]
        for bad in (rows+[rows[0]], [dict(r,response_time='3.51') for r in rows],
                    [dict(r,response_time='3.5') for r in rows],
                    [dict(r,response_time='0') for r in rows]):
            with self.assertRaises(ValueError):
                split_epochs(bad)

    def test_complete_mapping_preserves_contrasts_and_nuisance(self):
        for family,n in [('act',8),('nppi',26)]:
            source = template(family)
            rendered = public_template(source)
            old,new = settings(source),settings(rendered)
            self.assertEqual(new['evs_orig'],str(n))
            self.assertEqual(new['evs_real'],str(n))
            expected = np.zeros((11 if family=='nppi' else 10,n))
            old_to_new = {i:i for i in range(1,8)}
            if family=='nppi':
                old_to_new.update({10:9,**{i:i-1 for i in range(11,18)},**{i:i-2 for i in range(20,29)}})
                for a,b in [(10,9),*[(i,i-2) for i in range(20,29)]]:
                    self.assertEqual(new[f'custom{b}'],old[f'custom{a}'])
            for a,b in old_to_new.items():
                expected[:,b-1] = expected_contrasts(family=='nppi')[:,a-1]
            for mode in ('real','orig'):
                actual = np.array([[float(new[f'con_{mode}{c}.{i}']) for i in range(1,n+1)]
                                   for c in range(1,len(expected)+1)])
                np.testing.assert_array_equal(actual,expected)
            self.assertFalse(any('rt' in fsf_value(rendered,f'evtitle{i}') for i in range(1,n+1)))
            for key in ('smooth','prewhiten_yn','temphp_yn','paradigm_hp','tr','motionevs'):
                self.assertEqual(old[key],new[key])
            if family=='nppi':
                for ev in range(10,18):
                    parents = [j for j in range(1,ev) if new[f'interactions{ev}.{j}']=='1']
                    self.assertEqual(parents,[ev-9,9])
                    self.assertEqual(new[f'interactionsd{ev}.9'],'2')
                    self.assertEqual(new[f'interactionsd{ev}.{ev-9}'],'1' if ev in (11,13,15) else '0')

    def test_reject_unexpected_source_settings(self):
        source = template('nppi')
        for bad in (source.replace('set fmri(deriv_yn1) 0','set fmri(deriv_yn1) 1'),
                    source.replace('set fmri(con_real7.14) 1','set fmri(con_real7.14) 2'),
                    source.replace('set fmri(interactions11.10) 1','set fmri(interactions11.10) 0'),
                    source+'\nset fmri(new_unknown_ev1) 5\n',
                    source+'\nset fmri(evs_orig) 28\n'):
            with self.assertRaises(ValueError):
                public_template(bad)

    def test_render_preserves_source_amplitudes_and_files(self):
        events = read_events(ROOT/'source_data/bids/sub-143/func/sub-143_task-ultimatum_run-01_events.tsv')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source,standard,source_path,_ = source_fixture(root/'source','nppi',events)
            original = {p:p.read_bytes() for p in (root/'source').iterdir()}
            text,required,stats = render_l1(source,root/'out',root/'evs',events,standard,source_fsf=source_path)
            self.assertEqual(stats['responded_trials'],72)
            self.assertEqual(sum(stats[f'post_rows_{p}'] for p in PARTNERS),72)
            self.assertEqual(stats['n_evs'],26)
            pooled = np.loadtxt(fsf_value(text,'custom8'))
            self.assertEqual(len(pooled),72)
            self.assertTrue(np.all(pooled[:,2]==1))
            self.assertTrue(all(p.is_file() for p in required))
            for i in range(1,7):
                before = np.loadtxt(fsf_value(source,f'custom{i}'))
                after = np.loadtxt(fsf_value(text,f'custom{i}'))
                np.testing.assert_array_equal(before[:,[0,2]],after[:,[0,2]])
                self.assertTrue(np.all(after[:,1]<before[:,1]))
            self.assertTrue(all(p.read_bytes()==b for p,b in original.items()))
            with self.assertRaises(FileExistsError):
                render_l1(source,root/'out',root/'evs',events,standard,source_fsf=source_path)

    def test_published_templates_equal_transform(self):
        for family in ('act','nppi'):
            path = ROOT/f'templates/revision/L1_task-ultimatum_model-decision-postresponse_type-{family}.fsf'
            self.assertEqual(path.read_text(),public_template(template(family)))

    @unittest.skipUnless(shutil.which('feat_model'),'requires FSL feat_model')
    def test_fsl_compile_with_and_without_missed_ev(self):
        events = read_events(ROOT/'source_data/bids/sub-104/func/sub-104_task-ultimatum_run-01_events.tsv')
        with tempfile.TemporaryDirectory() as d:
            for family in ('act','nppi'):
                for missed in (False,True):
                    rows = list(events)
                    if missed:
                        rows.append(dict(trial_type='missed_trial',onset='460',duration='3.5',response_time='n/a',Offer='n/a'))
                    root = Path(d)/f'{family}-{missed}'
                    source,standard,source_path,confounds = source_fixture(root/'source',family,rows)
                    text,_,stats = render_l1(source,root/'out',root/'evs',rows,standard,source_fsf=source_path)
                    fsf = root/'new.fsf'
                    fsf.write_text(text)
                    result = subprocess.run(['feat_model',str(fsf.with_suffix('')),str(confounds)],capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                    matrix,con = read_vest_matrix(fsf.with_suffix('.mat')),read_vest_matrix(fsf.with_suffix('.con'))
                    self.assertEqual(matrix.shape,(250,stats['n_evs']+3))
                    self.assertEqual(con.shape,(10 if family=='act' else 11,stats['n_evs']+3))
                    expected = np.array([[float(fsf_value(text,f'con_real{i}.{j}')) for j in range(1,stats['n_evs']+1)]
                                         for i in range(1,len(con)+1)])
                    np.testing.assert_allclose(con[:,:stats['n_evs']],expected)
                    self.assertTrue(np.all(con[:,stats['n_evs']:]==0))
                    metrics,checks = diagnostics(matrix,con)
                    self.assertEqual(metrics['remaining_deficiency_after_zero_removal'],0)
                    self.assertTrue(all(r['estimable'] and r['sensitivity_estimable'] for r in checks))
                    self.assertEqual(7 in metrics['zero_columns'],not missed)
                    if family=='nppi':
                        self.assertEqual(16 in metrics['zero_columns'],not missed)


if __name__=='__main__':
    unittest.main()
