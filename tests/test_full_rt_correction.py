import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from audit_retained_rt_hashes import audit, match_candidates
from make_ultimatum_3col import parse_args
from prepare_dmn_revision_checks import REFERENCE_NAMES
from prepare_dmn_condition_bar_models import SOURCE_FSF_RELATIVE
from prepare_sub144_imaging_repair import fsf_value
from prepare_ultimatum_l3_repair import parse_evs, parse_inputs
from run_full_rt_correction import MODELS, prepare, group_replacement


class FullRTTests(unittest.TestCase):
    def test_default_is_all_responded(self):
        self.assertEqual(parse_args(['--events','events.tsv','--output-prefix','ev']).rt_source,'substantive')

    def test_unknown_hash_is_not_an_omission(self):
        rows=[dict(onset='1',duration='3.5',trial_type='event_accept_ingroup',response_time='1.5'),
              dict(onset='1',duration='33.5',trial_type='block_ingroup_fair',response_time='n/a')]
        self.assertEqual(match_candidates(rows,9,'0'*64)[2],[])
        target=hashlib.sha256(b'1\t0\t1.5\n').hexdigest()
        self.assertEqual(match_candidates(rows,9,target)[2][0]['block_first_onsets'],[1.0])

    def test_real_retained_rt_evidence(self):
        _,_,summary=audit(ROOT)
        self.assertEqual(summary['exact_matches'],188)
        self.assertEqual(summary['unknown_files'],0)
        self.assertEqual(summary['block_first_excluded'],786)
        self.assertEqual(summary['inclusion_by_subject'],{'sub-143':18})

    def test_group_replacement_keeps_covariates_and_contrasts(self):
        for family,name in MODELS.items():
            text=(ROOT/'templates/revision'/name).read_text()
            paths={s:Path('/scratch')/family/s/'L2.gfeat' for _,s,_ in parse_inputs(text.splitlines())}
            rendered=group_replacement(text,Path('/scratch/out'),Path('/standard.nii.gz'),paths)
            self.assertEqual(parse_evs(rendered.splitlines()),parse_evs(text.splitlines()))
            self.assertEqual([x for x in text.splitlines() if 'fmri(con' in x],
                             [x for x in rendered.splitlines() if 'fmri(con' in x])
            self.assertEqual(len(parse_inputs(rendered.splitlines())),47)
            with self.assertRaises(ValueError):
                group_replacement(text,Path('/scratch/out'),Path('/std'),{})

    def test_full_batch_prepares_all_families_without_mutating_sources(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);repo=base/'repo';production=base/'production';repaired=base/'repaired'
            work=base/'work';standard=base/'standard.nii.gz';standard.write_bytes(b'fixture')
            for relative in [SOURCE_FSF_RELATIVE,*[Path('templates/revision')/n for n in
                                                    list(MODELS.values())+list(REFERENCE_NAMES.values())]]:
                dest=repo/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,dest)
            subjects=[s for _,s,_ in parse_inputs((ROOT/SOURCE_FSF_RELATIVE).read_text().splitlines())]
            inp=base/'input.txt';inp.write_text('1 3.5 1\n')
            source_hashes={}
            for subject in subjects:
                for run in ('01','02'):
                    rel=Path(f'source_data/bids/{subject}/func/{subject}_task-ultimatum_run-{run}_events.tsv')
                    dest=repo/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
                for family in MODELS:
                    parent=(repaired if subject=='sub-144' else production)/subject
                    l2=parent/f'L2_task-ultimatum_model-02_type-{family}_sm-6.gfeat/design.fsf'
                    l2.parent.mkdir(parents=True)
                    l2.write_text(f'set fmri(outputdir) "old"\nset fmri(regstandard) "{standard}"\nset feat_files(1) "old1"\nset feat_files(2) "old2"\n')
                    for run in ('01','02'):
                        path=parent/f'L1_task-ultimatum_model-02_type-{family}_run-{run}_sm-6.feat/design.fsf'
                        path.parent.mkdir(parents=True)
                        n=9 if family=='act' else 28
                        text=(f'set fmri(evs_orig) {n}\nset fmri(evs_real) {n}\nset fmri(ncon_real) {10 if n==9 else 11}\n'
                              'set fmri(outputdir) "old"\nset fmri(evtitle8) "rt"\nset fmri(evtitle9) "rt_p"\n'
                              f'set feat_files(1) "{standard}"\nset confoundev_files(1) "{inp}"\nset fmri(regstandard) "{standard}"\n')
                        for i in range(1,n+1):
                            text+=f'set fmri(shape{i}) 3\nset fmri(custom{i}) "{inp}"\n'
                        path.write_text(text);path.with_suffix('.mat').write_text('retained fixture')
                        source_hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
            prepare(repo,production,repaired,work,standard,())
            with (work/'revision_jobs.tsv').open() as f:jobs=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual([sum(j['stage']==s for j in jobs) for s in ('l1','l2','l3')],[282,141,5])
            self.assertEqual(sum(int(j['expected_zstats']) for j in jobs if j['stage']=='l3'),32)
            self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in source_hashes.items()))
            for family in MODELS:
                fsf=work/f'fsf/{family}_sub-104_run-01.fsf'
                text=fsf.read_text()
                self.assertEqual(fsf_value(text,'custom1'),str(inp.resolve()))
                rt=Path(fsf_value(text,'custom9')).read_text().splitlines()
                self.assertEqual(len(rt),72)
                self.assertEqual(rt[0],'4.00822\t0\t1.29731')
            with self.assertRaises(FileExistsError):
                prepare(repo,production,repaired,work,standard,())


if __name__=='__main__':
    unittest.main()
