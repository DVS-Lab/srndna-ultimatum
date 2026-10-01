import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from locate_dmn_rt_text_inputs import locate, normalized_name


class TextInputSearchTests(unittest.TestCase):
    def item(self, name='run-01_event_computer.txt', role='custom1'):
        return dict(subject='sub-104',run='01',role=role,recorded='/old/sub-104/'+name,status='missing')

    def test_normalized_runs_do_not_change_condition(self):
        self.assertEqual(normalized_name('run-01_event_computer.txt'), 'run-1_event_computer.txt')
        self.assertNotEqual(normalized_name('run-01_event_computer_pmod.txt'), 'run-1_event_computer.txt')

    def test_subject_and_model_files_are_not_blindly_selected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for subject in ('sub-104','sub-1040','sub-105'):
                folder=root/subject/'ultimatum-pmod'
                folder.mkdir(parents=True)
                (folder/'run-1_event_computer.txt').write_text('1 2 1\n')
            required,candidates,summary=locate([self.item()],[root])
            self.assertEqual(len(candidates),1)
            self.assertEqual(summary['references_with_readable_candidates'],1)
            self.assertEqual(required[0]['status'],'candidate_found')

    def test_conflicting_copies_remain_ambiguous(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for label,content in (('old','1 2 1\n'),('new','1 2 5\n')):
                folder=root/label/'sub-104'
                folder.mkdir(parents=True)
                (folder/'run-01_event_computer.txt').write_text(content)
            required,candidates,summary=locate([self.item()],[root])
            self.assertEqual(required[0]['status'],'different_copies')
            self.assertEqual(summary['references_with_different_copies'],1)

    def test_absent_directories_and_feat_outputs_are_not_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            folder=root/'sub-104'/'old.feat'
            folder.mkdir(parents=True)
            (folder/'run-01_event_computer.txt').write_text('1 2 1\n')
            required,candidates,summary=locate([self.item()],[root,root/'absent'])
            self.assertEqual(candidates,[])
            self.assertEqual(summary['references_without_readable_candidates'],1)
            self.assertFalse(summary['search_roots'][1]['exists'])

    def test_only_missing_text_inputs_are_searched(self):
        rows=[{**self.item(),'status':'found'}, self.item('bold.nii.gz','feat_files(1)'), self.item()]
        required,_,summary=locate(rows,[])
        self.assertEqual(len(required),1)
        self.assertEqual(summary['required_missing_text_references'],1)

    def test_root_search_is_rejected(self):
        with self.assertRaises(ValueError):
            locate([self.item()],[Path('/')])


if __name__ == '__main__':
    unittest.main()
