import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import run_palm_age_main as batch


class PalmAgeMainTests(unittest.TestCase):
    def setUp(self):
        capture = contextlib.redirect_stdout(io.StringIO()); capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def fixture(self):
        rng = np.random.default_rng(18); subjects = [f'sub-{104+i:03}' for i in range(47)]
        old = np.r_[np.zeros(25), np.ones(22)]
        values = rng.normal(size=(47, 4))
        sample = {s: dict(older=old[i], ismale=values[i, 0], tsnr=values[i, 1],
            fd_mean=values[i, 2], RT=99999) for i, s in enumerate(subjects)}
        nuisance = {key: values[:, i].tolist() for i, key in enumerate(batch.COLUMNS[2:])}
        model = dict(name='full-rt__act', subjects=subjects, nuisance=nuisance,
            primary_cope=7, family='act', timing='full-rt')
        return model, sample

    def save(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(data, float), np.eye(4)), path)

    def new_model(self):
        model, sample = self.fixture()
        x, c, vg, changes = batch.build_design(model, sample, model)
        return dict(model, x=x.tolist(), c=c.tolist(), labels=list(batch.LABELS),
            variance_groups=vg.tolist(), covariate_changes=changes)

    def prepared_fixture(self, root):
        model = self.new_model(); inputs = root/'preparation'/model['name']; inputs.mkdir(parents=True)
        mask = np.ones((3, 3, 3)); self.save(inputs/'mask.nii.gz', mask)
        self.save(inputs/'copes.nii.gz', np.ones((3, 3, 3, 47)))
        batch.pa.vest(inputs/'design.mat', np.array(model['x']))
        batch.pa.vest(inputs/'design.con', np.array(model['c']), contrast=True)
        np.savetxt(inputs/'variance_groups.csv', model['variance_groups'], fmt='%d')
        return dict(root=str(inputs.parent), models=[model]), model

    def test_locked_scope_four_test_signs_and_sample_weighted_mean(self):
        self.assertEqual(len(batch.source_specs()), 12)
        self.assertEqual(len({r[0] for r in batch.source_specs()}), 12)
        self.assertEqual(batch.METHODS, ('cluster-z3.1', 'cluster-z2.6'))
        model, sample = self.fixture()
        x, c, vg, changes = batch.build_design(model, sample, model)
        self.assertEqual(x.shape, (47, 6)); self.assertEqual(np.linalg.matrix_rank(x), 6)
        np.testing.assert_allclose(x[:, 1:].mean(axis=0), 0, atol=1e-14)
        self.assertEqual(list(np.bincount(vg)[1:]), [25, 22])
        self.assertEqual(len(changes), 47*4)
        y = 8-3*np.array([float(sample[s]['older']) for s in model['subjects']])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        np.testing.assert_allclose(c@beta, [8-3*22/47, -8+3*22/47, 3, -3], atol=1e-12)
        self.assertNotAlmostEqual(beta[0], 6.5)
        # The deliberately invalid historical table RT is never used.
        np.testing.assert_allclose(x[:, 5], np.array(model['nuisance']['RT'])-np.mean(model['nuisance']['RT']))

    def test_covariates_align_by_id_and_age_test_matches_two_group_parameterization(self):
        model, sample = self.fixture()
        altered = dict(model, nuisance={k: list(v) for k, v in model['nuisance'].items()})
        altered['nuisance']['tsnr'][3] += 4; altered['nuisance']['fd_mean'][4] -= 8
        x, c, _, changes = batch.build_design(altered, sample, model)
        self.assertEqual(sum(abs(r['centered_change']) > 1e-10 for r in changes), 94)
        reorder = np.random.default_rng(31).permutation(47)
        shuffled = dict(model, subjects=[model['subjects'][i] for i in reorder],
            nuisance={k: [v[i] for i in reorder] for k, v in model['nuisance'].items()})
        sx, sc, _, _ = batch.build_design(shuffled, sample, model)
        np.testing.assert_allclose(sx, x[reorder], atol=1e-12)
        old = np.array([sample[s]['older'] for s in model['subjects']])
        original = np.c_[1-old, old, x[:, 2:]]
        y = np.random.default_rng(31).normal(size=47)
        b0 = np.linalg.lstsq(original, y, rcond=None)[0]
        b1 = np.linalg.lstsq(x, y, rcond=None)[0]
        np.testing.assert_allclose(original@b0, x@b1, atol=1e-12)
        self.assertAlmostEqual(b0[0]-b0[1], (c@b1)[2])
        np.testing.assert_array_equal(sc, c)

    def test_covariate_disagreement_and_rank_loss_fail_instead_of_guessing(self):
        model, sample = self.fixture()
        wrong = dict(model, nuisance={k: list(v) for k, v in model['nuisance'].items()})
        wrong['nuisance']['RT'][0] += 1
        with self.assertRaisesRegex(ValueError, 'corrected RT differs'):
            batch.build_design(wrong, sample, model)
        changed = {s: dict(r) for s, r in sample.items()}; changed[model['subjects'][0]]['tsnr'] += 1
        with self.assertRaisesRegex(ValueError, 'participant table disagree'):
            batch.build_design(model, changed, model)
        for r in sample.values(): r['older'] = 0
        with self.assertRaisesRegex(ValueError, 'age-group'):
            batch.build_design(model, sample, model)

    def test_real_compact_designs_support_the_final_controls(self):
        repo = Path(__file__).resolve().parents[1]
        original = json.loads((repo/'results/permutation_audit/v1/config.json').read_text())
        dm = json.loads((repo/'results/dmn_covariate_standardized/v1/randomise/config.json').read_text())
        current = {m['name']: m for m in original['models']}
        current.update({m['name']: m for m in dm['models'] if m['name'].endswith('__dmn-age')})
        sample = {r['subjID']: r for r in batch.pa.read_rows(repo/'behavioral_analyses/data/participant_L3_47.csv', ',')}
        reference = current['full-rt__dmn-age']
        matrices = []
        for _, source, _, _, _ in batch.source_specs():
            x, c, vg, _ = batch.build_design(current[source], sample, reference)
            matrices.append(x)
            self.assertEqual(np.linalg.matrix_rank(x), 6)
            np.testing.assert_array_equal(c[2], [0, -1, 0, 0, 0, 0])
            self.assertEqual(vg.sum(), 69)
        for x in matrices[1:]: np.testing.assert_allclose(x, matrices[0], atol=1e-12)

    def test_all_run_and_l2_coverage_no_subject_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = self.new_model(); shape = (3, 3, 3)
            mask = root/'historical-mask.nii.gz'; self.save(mask, np.ones(shape)); model['mask'] = str(mask)
            model['images'] = []; expected = {}
            for i, subject in enumerate(model['subjects']):
                base = root/'source'/subject; l2 = base/'L2.gfeat'
                run_paths = [base/f'L1_run-{run:02}.feat' for run in (1, 2)]
                for run, folder in enumerate(run_paths, 1):
                    self.save(folder/'mask.nii.gz', np.ones(shape))
                    self.save(folder/'stats/cope7.nii.gz', np.full(shape, i+1))
                    v = np.ones(shape)
                    if i == 0 and run == 1: v.ravel()[0] = 0
                    self.save(folder/'stats/varcope7.nii.gz', v)
                    matrix = np.zeros((7, 2)); matrix[:, 0] = 1
                    batch.pa.vest(folder/'design.con', matrix, contrast=True)
                    (folder/'design.fsf').write_text('set fmri(evs_real) 2\nset fmri(con_real7.1) 1\nset fmri(con_real7.2) 0\n')
                l2.mkdir()
                (l2/'design.fsf').write_text(''.join(f'set feat_files({j}) "{p}"\n' for j, p in enumerate(run_paths, 1))+
                    ''.join(f'set fmri(copeinput.{j}) 1\n' for j in range(1, 8)))
                path = l2/'cope7.feat/stats/cope1.nii.gz'
                self.save(path, np.full(shape, i+1)); self.save(path.parent.parent/'mask.nii.gz', np.ones(shape))
                v = np.ones(shape)
                if i == 1: v.ravel()[1] = 0
                self.save(path.with_name('varcope1.nii.gz'), v)
                expected[str(path)] = batch.pa.digest(path); model['images'].append(str(path))
            _, retained, valid = batch.prepare_inputs(model, root/'new', expected, {})
            self.assertEqual(retained.sum(), 27); self.assertEqual(valid.sum(), 25)
            self.assertFalse(valid.ravel()[0]); self.assertFalse(valid.ravel()[1])
            self.assertEqual(len(batch.pa.read_rows(root/'new/run_coverage.tsv')), 94)
            p = Path(model['images'][0]); p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'completed analysis provenance'):
                batch.prepare_inputs(model, root/'changed', expected, {})

    def test_family_common_masks_and_receipt_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = root/'work'; work.mkdir(); source = root/'source'; source.write_text('locked')
            models = []
            for name, original, family, cope, timing in batch.source_specs():
                models.append(dict(self.new_model(), name=name, source_model=original, family=family, primary_cope=cope, timing=timing))
            def fake(model, folder, expected, sources):
                folder.mkdir(parents=True); mask = np.ones((3, 3, 3), bool); valid = mask.copy()
                i = batch.TIMINGS.index(model['timing']) if model['timing'] in batch.TIMINGS else 5
                valid.ravel()[i] = False
                self.save(folder/'copes.nii.gz', np.ones((3, 3, 3, 47)))
                ref = nib.Nifti1Image(mask.astype('uint8'), np.eye(4))
                return ref, mask, valid
            with patch.object(batch, 'load_catalog', return_value=(models, {str(source): batch.pa.digest(source)}, {})), \
                 patch.object(batch, 'prepare_inputs', side_effect=fake):
                prepared = batch.prepare(root, work)
            for model in models:
                mask = nib.load(Path(prepared['root'])/model['name']/'mask.nii.gz').get_fdata()
                self.assertEqual(mask.sum(), 24 if model['family'] in batch.FAMILIES else 26)
            with patch.object(batch, 'load_catalog', side_effect=AssertionError('completed preparation must skip')):
                self.assertEqual(batch.prepare(root, work), prepared)
            design = Path(prepared['root'])/models[0]['name']/'design.mat'
            design.write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'changed'): batch.prepare(root, work)

    def fake_launch(self, engine, palm, args, attempt, **kwargs):
        self.assertIn('-ise', args); self.assertIn('-vg', args); self.assertIn('-zstat', args)
        self.assertNotIn('-T', args); self.assertNotIn('-D', args)
        self.assertEqual(args[args.index('-rmethod')+1], 'Freedman-Lane')
        mask = nib.load(args[args.index('-m')+1]); data = mask.get_fdata() > 0
        count = batch.cov.read_vest_matrix(Path(args[args.index('-t')+1])).shape[0]
        self.assertEqual(count, 4)
        (attempt/'run.txt').write_text('synthetic PALM run')
        batch.pa.write_json(attempt/'command.json', args)
        for c in range(1, 5):
            values = data*(.97 if c == 1 else .2)
            self.save(attempt/f'palm_clustere_zstat_fwep_c{c}.nii.gz', values)

    def test_job_retry_resume_export_and_incomplete_is_not_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, model = self.prepared_fixture(root)
            work = root/'work'; work.mkdir(); report = root/'report'; report.mkdir()
            rt = dict(engine='matlab', palm_root='/palm', mode='inherited')
            with patch.object(batch.headless, 'launch', side_effect=RuntimeError('engine failed')):
                with self.assertRaisesRegex(RuntimeError, 'engine failed'):
                    batch.run_job(model, batch.METHODS[0], prepared, work, report, 'frozen', rt)
            self.assertFalse((work/'jobs'/model['name']/batch.METHODS[0]/'complete.json').exists())
            with patch.object(batch.headless, 'launch', side_effect=self.fake_launch):
                rows = batch.run_job(model, batch.METHODS[0], prepared, work, report, 'frozen', rt)
            self.assertEqual(len(rows), 4); self.assertEqual(rows[0]['significant_voxels'], 27)
            self.assertAlmostEqual(rows[0]['minimum_spatial_fwe_p'], .03)
            self.assertEqual(batch.collect(prepared, work, report, 'frozen'), (1, 2))
            self.assertEqual(len(batch.pa.read_rows(report/'summary.tsv')), 4)
            status = json.loads((report/'status.json').read_text()); self.assertFalse(status['complete'])
            with patch.object(batch.headless, 'launch', side_effect=AssertionError('must not rerun')):
                batch.run_job(model, batch.METHODS[0], prepared, work, report, 'frozen', rt)
            self.assertEqual(len(list((work/'jobs'/model['name']/batch.METHODS[0]).glob('attempt-*'))), 2)
            saved = json.loads((work/'jobs'/model['name']/batch.METHODS[0]/'complete.json').read_text())
            (Path(saved['attempt'])/'palm_clustere_zstat_fwep_c1.nii.gz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'changed'):
                batch.run_job(model, batch.METHODS[0], prepared, work, report, 'frozen', rt)

    def test_smoke_both_thresholds_and_preserved_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, _ = self.prepared_fixture(root)
            work = root/'work'; work.mkdir(); report = root/'report'; report.mkdir()
            rt = dict(engine='matlab', palm_root='/palm', mode='inherited')
            with patch.object(batch.headless, 'launch', side_effect=self.fake_launch) as launch:
                batch.smoke(prepared, work, report, 'frozen', rt)
                self.assertEqual(launch.call_count, 2)
                self.assertEqual({call.args[2][call.args[2].index('-C')+1] for call in launch.call_args_list}, {'2.6', '3.1'})
                self.assertTrue(all(call.args[2][call.args[2].index('-n')+1] == '20' for call in launch.call_args_list))
            with patch.object(batch.headless, 'launch', side_effect=AssertionError('must skip smoke')):
                batch.smoke(prepared, work, report, 'frozen', rt)
            self.assertTrue(json.loads((work/'smoke_complete.json').read_text())['not_scientific_results'])
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                batch.smoke(prepared, work, report, 'changed', rt)

    def test_scheduler_stops_new_jobs_after_failure_and_drains(self):
        import threading
        barrier = threading.Barrier(2); seen = []; after = []
        tasks = [(dict(name=str(i)), 'cluster-z3.1') for i in range(8)]
        def worker(item):
            name = item[0]['name']; seen.append(name); barrier.wait(timeout=5)
            if name == '0': raise RuntimeError('first failure')
        # Force both running tasks to be delivered before scheduling more.
        with patch.object(batch, 'wait', side_effect=lambda active, **kwargs: __import__('concurrent.futures').futures.wait(active)):
            failures, pending = batch.run_pending(tasks, worker, 2, lambda: after.append('snapshot'))
        self.assertEqual(set(seen), {'0', '1'}); self.assertEqual(len(failures), 1)
        self.assertEqual(len(pending), 6); self.assertTrue(after)

    def test_orphan_detection_is_work_specific_and_never_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = Path('/scratch/new-palm')
            for pid, cmd, state in [
                (1, b'/usr/bin/matlab\0-batch\0palm output /scratch/new-palm/jobs/a/palm', 'S'),
                (2, b'/usr/bin/matlab\0-batch\0palm output /scratch/original-palm/jobs/b/palm', 'S'),
                (3, b'python3\0runner.py\0--work-root\0/scratch/new-palm/', 'S'),
                (4, b'/usr/bin/matlab\0-batch\0palm output /scratch/new-palm/jobs/z/palm', 'Z')]:
                folder = root/str(pid); folder.mkdir()
                (folder/'cmdline').write_bytes(cmd); (folder/'stat').write_text(f'{pid} (name) {state} rest')
            self.assertEqual(batch.active_children(work, root), [1])

    def test_inventory_tracks_outputs_and_excludes_live_launch_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'status.json').write_text('{}')
            launch = root/'launch-today'; launch.mkdir(); (launch/'run.txt').write_text('live')
            batch.inventory(root)
            rows = batch.pa.read_rows(root/'inventory.tsv')
            self.assertEqual([r['path'] for r in rows], ['status.json'])
            expected = batch.pa.digest(root/'inventory.tsv'); (launch/'run.txt').write_text('more progress')
            batch.inventory(root); self.assertEqual(expected, batch.pa.digest(root/'inventory.tsv'))


if __name__ == '__main__':
    unittest.main()
