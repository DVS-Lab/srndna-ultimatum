import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import prepare_palm_resume as pr


class PalmResumeTests(unittest.TestCase):
    def make_repo(self, root):
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        (root / 'palm_help.m').write_text('unchanged source\n')
        pr.git(root, 'add', 'palm_help.m')
        pr.git(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
               'commit', '-qm', 'fixture')
        return pr.git(root, 'rev-parse', 'HEAD')

    def test_detached_head_fixed_without_source_changes_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); commit = self.make_repo(root)
            pr.git(root, 'checkout', '--detach', commit)
            before = pr.git(root, 'rev-parse', 'HEAD^{tree}')
            result = pr.ensure_readable_head(root, commit)
            self.assertTrue(result['changed'])
            self.assertEqual(pr.git(root, 'rev-parse', 'HEAD^{tree}'), before)
            self.assertFalse(pr.ensure_readable_head(root, commit)['changed'])
            self.assertEqual(pr.git(root, 'status', '--porcelain'), '')

    def test_packed_branch_gets_readable_ref_without_moving_old_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); commit = self.make_repo(root)
            first = pr.ensure_readable_head(root, commit)['branch']
            pr.git(root, 'pack-refs', '--all', '--prune')
            result = pr.ensure_readable_head(root, commit)
            self.assertTrue(result['changed'])
            self.assertEqual(pr.git(root, 'rev-parse', first), commit)
            ref = root / '.git/refs/heads' / result['branch']
            self.assertEqual(ref.read_text().strip(), commit)

    def test_dirty_or_wrong_commit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); commit = self.make_repo(root)
            with self.assertRaises(ValueError): pr.ensure_readable_head(root, '0' * 40)
            (root / 'palm_help.m').write_text('local edits\n')
            with self.assertRaises(ValueError): pr.ensure_readable_head(root, commit)

    def test_incomplete_smoke_archived_with_symlinks_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); report = work / 'report'; smoke = work / 'smoke'
            smoke.mkdir(); (smoke / 'run.txt').write_text('original failure\n')
            source = work / 'input.nii.gz'; source.write_bytes(b'input')
            (smoke / 'copes.nii.gz').symlink_to(source)
            jobs = work / 'jobs'; jobs.mkdir(); (jobs / 'complete.json').write_text('{}')
            archived = Path(pr.preserve_smoke(work, report))
            self.assertFalse(smoke.exists())
            self.assertEqual((archived / 'run.txt').read_text(), 'original failure\n')
            self.assertTrue((archived / 'copes.nii.gz').is_symlink())
            self.assertEqual((jobs / 'complete.json').read_text(), '{}')
            self.assertEqual(len(list((report / 'smoke-history').glob('*.txt'))), 1)

    def test_completed_smoke_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); smoke = work / 'smoke'; smoke.mkdir()
            (smoke / 'complete.json').write_text(json.dumps({'passed': True}))
            self.assertIsNone(pr.preserve_smoke(work, work / 'report'))
            self.assertTrue((smoke / 'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
