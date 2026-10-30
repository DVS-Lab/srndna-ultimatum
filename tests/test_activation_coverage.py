import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import run_activation_coverage as check


class ActivationCoverageTests(unittest.TestCase):
    def setUp(self):
        capture = contextlib.redirect_stdout(io.StringIO()); capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def save(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=float), np.eye(4)), path)

    def fixture(self, root):
        shape = (4, 4, 4); subjects = [f'sub-{i:03d}' for i in range(47)]
        expected = {}; images = {}; sources = {}; rng = np.random.default_rng(2)
        for family, primary in [('human', 10), ('parietal', 7)]:
            images[family] = []
            contrasts = (10,) if family == 'human' else (2, 4, 6, 7)
            for i, subject in enumerate(subjects):
                parent = root/'source'/family/subject
                paths = [parent/'L1_run-01.feat', parent/'L1_run-02.feat']
                for folder in paths:
                    folder.mkdir(parents=True)
                    self.save(folder/'mask.nii.gz', np.ones(shape))
                    con = np.zeros((10, 4))
                    for k in range(10): con[k, k % 4] = 1
                    check.pa.vest(folder/'design.con', con, contrast=True)
                    text = 'set fmri(evs_real) 4\n'
                    for k in contrasts:
                        text += ''.join(f'set fmri(con_real{k}.{j}) {con[k-1,j-1]}\n' for j in range(1, 5))
                        self.save(folder/'stats'/f'cope{k}.nii.gz', np.full(shape, i+1.+k/10))
                        variance = np.ones(shape)
                        if i == 0 and folder == paths[0] and k == contrasts[0]:
                            variance.ravel()[1 if family == 'human' else 2] = 0
                        self.save(folder/'stats'/f'varcope{k}.nii.gz', variance)
                    (folder/'design.fsf').write_text(text)
                l2 = parent/'L2.gfeat'; l2.mkdir()
                (l2/'design.fsf').write_text(''.join(f'set feat_files({k}) "{p}"\n' for k,p in enumerate(paths, 1))+
                    ''.join(f'set fmri(copeinput.{k}) 1\n' for k in range(1, primary+1)))
                for k in contrasts:
                    path = l2/f'cope{k}.feat/stats/cope1.nii.gz'
                    self.save(path, np.full(shape, i+1.+k/10))
                    variance = np.ones(shape)
                    if i == 1 and k == primary: variance.ravel()[0 if family == 'human' else 3] = 0
                    self.save(path.with_name('varcope1.nii.gz'), variance)
                    self.save(path.parent.parent/'mask.nii.gz', np.ones(shape))
                    expected[str(path)] = check.pa.digest(path)
                    if k == primary: images[family].append(str(path))
        old = np.r_[np.zeros(25), np.ones(22)]
        behavior = rng.normal(size=47); behavior -= behavior.mean()
        human_x = np.c_[np.ones(47), old-old.mean(), rng.normal(size=(47, 4))]
        human_c = np.zeros((4, 6)); human_c[0, 0] = 1; human_c[1, 0] = -1; human_c[2, 1] = 1; human_c[3, 1] = -1
        parietal_x = np.c_[rng.normal(size=(47, 4)), 1-old, old, behavior*(1-old), behavior*old]
        parietal_c = np.zeros((8, 8)); parietal_c[0, 6:8] = [-1, 1]; parietal_c[1] = -parietal_c[0]
        parietal_c[2, 4:6] = [1, -1]; parietal_c[3] = -parietal_c[2]
        parietal_c[4, 6] = 1; parietal_c[5, 6] = -1; parietal_c[6, 7] = 1; parietal_c[7, 7] = -1
        models = []; seeds = []
        for name in check.NAMES:
            human = name == check.NAMES[0]; family = 'human' if human else 'parietal'
            x, c = (human_x, human_c) if human else (parietal_x, parietal_c)
            report = root/'historical'/name; design = report/'design'; design.mkdir(parents=True)
            fsf = design/'design.fsf'; fsf.write_text(''.join(f'set feat_files({j}) "{p}"\n' for j,p in enumerate(images[family], 1)))
            check.pa.vest(design/'design.mat', x); check.pa.vest(design/'design.con', c, contrast=True)
            mask = report/'mask.nii.gz'; self.save(mask, np.ones(shape))
            cx, cc = check.pa.canonical_design(x, c)
            model = dict(name=name, subjects=subjects, images=images[family], x=cx.tolist(), c=cc.tolist(),
                labels=[f'contrast{i}' for i in range(1, len(c)+1)], mask=str(mask), block=name, group=name,
                source_hashes={str(fsf):check.pa.digest(fsf)}, scope='synthetic', robust=False)
            models.append(model)
            for i in range(1, len(c)+1):
                for method in check.pa.METHODS:
                    seeds.append(dict(model=name, contrast=i, method=method, seed=100+i, permutations_recorded=10000))
                z = np.full(shape, 4.); significant = np.zeros(shape); significant.ravel()[5] = 1
                self.save(report/f'contrast{i}/zstat{i}.nii.gz', z)
                self.save(report/f'contrast{i}/cluster_mask_zstat{i}.nii.gz', significant)
        reference = root/'original-randomise'; reference.mkdir()
        check.pa.write_rows(reference/'model_contrast_summary.tsv', seeds)
        check.pa.write_rows(root/'historical/inventory.tsv', [dict(destination=str(p.relative_to(root/'historical')), sha256=check.pa.digest(p))
            for p in (root/'historical').rglob('*.nii.gz')])
        roi = np.zeros(64); roi[:8] = 1; roi = nib.Nifti1Image(roi.reshape(shape), np.eye(4))
        work = root/'work'; work.mkdir()
        return models, sources, expected, reference, roi, work

    def prepare(self, root):
        models, sources, expected, original, roi, work = self.fixture(root)
        with patch.object(check, 'source_catalog', return_value=(models, sources, expected, original)), \
             patch.object(check.figures, 'supplemental_rois', return_value=[{}, dict(roi=roi)]), \
             patch.object(check.figures, 'parietal_roi', return_value=roi):
            prepared = check.prepare(root, work)
        return prepared, work

    def test_whole_brain_preparation_native_conditions_seed_resume_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, work = self.prepare(root)
            rows = check.pa.read_rows(Path(prepared['root'])/'coverage_summary.tsv')
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(int(r['participants']) == 47 and int(r['runs']) == 94 for r in rows))
            self.assertTrue(all(int(r['tested_voxels']) == 62 and int(r['valid_selected_voxels']) == 6 for r in rows))
            self.assertEqual(len(prepared['mapping']), 28); self.assertEqual(len(prepared['tests']), 28)
            self.assertEqual({t['seed'] for t in prepared['tests']}, set(range(101,109)))
            masks = [nib.load(Path(prepared['root'])/name/'mask.nii.gz').get_fdata() for name in check.NAMES[1:]]
            self.assertTrue(all(np.array_equal(masks[0], m) for m in masks))
            self.assertEqual(masks[0].ravel()[2], 0)  # native COPE2, not just focal COPE7
            with patch.object(check, 'source_catalog', side_effect=AssertionError('do not repeat preparation')):
                self.assertEqual(check.prepare(root, work), prepared)
            source = Path(prepared['models'][0]['images'][0]); source.write_bytes(b'changed')
            with self.assertRaises(ValueError): check.prepare(root, work)

    def test_l2_declarations_and_compiled_contrast_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); models, sources, _, _, _, _ = self.fixture(root)
            image = Path(models[0]['images'][0])
            runs = check.l1_inputs(image, models[0]['subjects'][0], 10, sources)
            self.assertEqual([r for r,_ in runs], [1,2])
            check.validate_contrast(runs[0][1], 10, sources)
            path = runs[0][1]/'design.con'
            c = check.read_vest_matrix(path); c[9,0] += 1
            check.pa.vest(path, c, contrast=True)
            with self.assertRaisesRegex(ValueError, 'compiled L1 contrast'):
                check.validate_contrast(runs[0][1], 10, {})
            fsf = image.parents[2]/'design.fsf'; fsf.write_text(fsf.read_text().replace('copeinput.3) 1', 'copeinput.3) 0'))
            with self.assertRaisesRegex(ValueError, 'COPE numbering'):
                check.l1_inputs(image, models[0]['subjects'][0], 10, {})

    def test_smoothness_volume_and_invalid_values_rejected(self):
        self.assertEqual(check.smoothness('DLH 0.001\nVOLUME 62\nRESELS 2\n',62)['VOLUME'],62)
        with self.assertRaisesRegex(ValueError,'volume differs'):check.smoothness('DLH .001\nVOLUME 64\nRESELS 2\n',62)
        with self.assertRaisesRegex(ValueError,'nonpositive'):check.smoothness('DLH nan\nVOLUME 62\nRESELS 2\n',62)

    def test_historical_map_inventories_accept_both_schemas_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models,_,_,_,_,_=self.fixture(root)
            self.assertEqual(len(check.historical_maps(models)),56)
            path=root/'historical/inventory.tsv';rows=check.pa.read_rows(path)
            check.pa.write_rows(path,[dict(artifact=r['destination'],sha256=r['sha256']) for r in rows])
            self.assertEqual(len(check.historical_maps(models)),56)
            (root/'historical'/models[0]['name']/'contrast1/zstat1.nii.gz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'retained FLAME map'):
                check.historical_maps(models)

    def test_frozen_operational_design_tampering_stops_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared,work=self.prepare(root)
            path=work/'groups'/prepared['models'][0]['group']/'design.mat'
            path.write_text('changed derived design')
            with self.assertRaises(ValueError):check.prepare(root,work)

    def fake_run(self, cmd, stdout=None, **kwargs):
        if cmd[0] == 'smoothest':
            count = int((nib.load(cmd[cmd.index('-m')+1]).get_fdata()>0).sum())
            stdout.write(f'DLH 0.001\nVOLUME {count}\nRESELS 2\n')
        elif cmd[0] == 'fsl-cluster':
            opts = dict(a[2:].split('=',1) for a in cmd[1:] if a.startswith('--') and '=' in a)
            ref = nib.load(opts['in']); a = np.zeros(ref.shape); a.ravel()[5] = 1
            self.save(Path(opts['oindex']), a); self.save(Path(opts['othresh']), a*4)
            Path(opts['olmax']).write_text('local maxima\n')
            stdout.write('Cluster Index\tVoxels\tP\tZ-MAX\n1\t1\t0.03\t4\n')
        elif cmd[0] == 'randomise':
            root = Path(cmd[cmd.index('-o')+1]).parent
            mask = nib.load(cmd[cmd.index('-m')+1]).get_fdata() > 0
            self.save(root/'randomise_tstat1.nii.gz', mask*4)
            self.save(root/'randomise_clustere_corrp_tstat1.nii.gz', mask*.96)
            np.savetxt(root/'randomise_clustere_corrp_tstat1.txt', np.ones(10000))
            self.assertEqual(cmd[cmd.index('-n')+1], '10000'); self.assertIn('-R',cmd); self.assertIn('-N',cmd)
            stdout.write('synthetic randomise\n')
        else:
            raise AssertionError(cmd)
        return SimpleNamespace(returncode=0)

    def test_completed_jobs_summary_and_resume_without_rerunning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared, work = self.prepare(root); report = root/'report'; report.mkdir()
            binaries = {name:dict(path=name) for name in ['smoothest','fsl-cluster','randomise']}
            model = prepared['models'][0]; test = prepared['tests'][0]
            with patch.object(check.subprocess,'run',side_effect=self.fake_run):
                for mode in ['original-mask','coverage-mask']:
                    check.run_flame(model,1,mode,prepared,work,report,'locked',binaries)
                check.run_randomise(test,'cluster-t3.1',prepared,work,report,'locked',binaries)
            complete, total = check.summarize(prepared,work,report,'locked')
            self.assertEqual(complete,3); self.assertEqual(total,112)
            with patch.object(check.subprocess,'run',side_effect=AssertionError('must skip')):
                check.run_flame(model,1,'coverage-mask',prepared,work,report,'locked',binaries)
                check.run_randomise(test,'cluster-t3.1',prepared,work,report,'locked',binaries)
            marker=work/'jobs'/test['test']/'cluster-t3.1/complete.json'
            value=json.loads(marker.read_text());value.pop('coverage_contract');check.pa.write_json(marker,value)
            self.assertEqual(check.summarize(prepared,work,report,'locked')[0],2)
            with patch.object(check.subprocess,'run',side_effect=AssertionError('do not rerun a completed, unannotated export')):
                check.run_randomise(test,'cluster-t3.1',prepared,work,report,'locked',binaries)
            self.assertEqual(check.summarize(prepared,work,report,'locked')[0],3)
            artifact=Path(json.loads(marker.read_text())['attempt'])/'randomise_tstat1.nii.gz';artifact.write_bytes(b'tampered')
            with self.assertRaises(ValueError): check.run_randomise(test,'cluster-t3.1',prepared,work,report,'locked',binaries)

    def test_descriptive_figures_use_fixed_valid_rois_and_are_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared,work=self.prepare(root);report=root/'report';report.mkdir()
            check.export_figures(prepared,report)
            self.assertEqual(len(list((report/'figures').glob('*.png'))),4)
            rows=check.pa.read_rows(report/'figures/parietal_slope_comparison.tsv')
            self.assertEqual(len(rows),9);self.assertTrue(all(r['voxels']=='6' for r in rows))
            self.assertEqual(len(check.pa.read_rows(report/'figures/parietal_deletions.tsv')),3*(47+1081))
            before={p:check.pa.digest(p) for p in (report/'figures').iterdir() if p.is_file()}
            check.export_figures(prepared,report)
            for p,h in before.items():self.assertEqual(check.pa.digest(p),h)

    def test_failed_smoothness_preserves_attempt_and_retry_uses_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared,work=self.prepare(root);report=root/'report';report.mkdir()
            model=prepared['models'][0];binaries={name:dict(path=name) for name in ['smoothest','fsl-cluster','randomise']}
            with patch.object(check.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
                with self.assertRaisesRegex(RuntimeError,'smoothest failed'):
                    check.run_flame(model,1,'coverage-mask',prepared,work,report,'locked',binaries)
            parent=work/'flame'/model['name']/'contrast1/coverage-mask'
            self.assertFalse((parent/'complete.json').exists())
            old=list(parent.glob('attempt-*'));self.assertEqual(len(old),1)
            self.assertTrue((old[0]/'commands.json').is_file())
            with patch.object(check.subprocess,'run',side_effect=self.fake_run):
                check.run_flame(model,1,'coverage-mask',prepared,work,report,'locked',binaries)
            self.assertEqual(len(list(parent.glob('attempt-*'))),2)
            self.assertTrue((old[0]/'commands.json').is_file())


if __name__ == '__main__':
    unittest.main()
