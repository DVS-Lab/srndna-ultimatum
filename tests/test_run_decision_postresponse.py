import argparse
import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

import run_decision_postresponse as runner
from make_ultimatum_3col import generate
from prepare_dmn_condition_bar_models import SOURCE_FSF_RELATIVE
from prepare_dmn_revision_checks import REFERENCE_NAMES, digest
from prepare_sub144_imaging_repair import (CUSTOM_NAMES, fsf_value,
                                           replace_fsf_number, replace_fsf_value)
from prepare_ultimatum_l3_repair import parse_evs, parse_inputs


def read_table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


class DecisionPostresponseRunnerTests(unittest.TestCase):
    def test_reuse_exact_input_configuration_and_reject_conflicts(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            config = base / 'preparation_config.json'
            values = dict(production=str(base/'production'), repaired=str(base/'repaired'),
                          standard=str(base/'standard.nii.gz'),
                          input_maps=[['/data/projects/old', str(base/'relocated')]])
            config.write_text(json.dumps(values))
            args = argparse.Namespace(reuse_input_config=config, production_fsl_root=None,
                                      repaired_fsl_root=None, standard_image=None, input_map=[])
            production, repaired, standard, maps = runner.resolve_config(args)
            self.assertEqual((production, repaired, standard),
                             tuple(Path(values[k]).resolve() for k in ('production', 'repaired', 'standard')))
            self.assertEqual(maps, ((Path('/data/projects/old'), base/'relocated'),))
            args.production_fsl_root = base/'different'
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                runner.resolve_config(args)
            args.production_fsl_root = None
            args.input_map = ['/old=/new']
            with self.assertRaisesRegex(ValueError, 'combine'):
                runner.resolve_config(args)

    def test_execute_requires_explicit_acceptance_and_existing_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            flags = ['--work-root', str(base/'work'), '--output-root', str(base/'reports')]
            with patch.object(runner, 'prepare') as prepare, patch.object(runner, 'run_jobs') as run_jobs:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    runner.main(flags + ['--execute'])
                with self.assertRaisesRegex(ValueError, 'dry-run'):
                    runner.main(flags + ['--execute', '--accept-design-diagnostics'])
                prepare.assert_not_called()
                run_jobs.assert_not_called()

    def test_dry_run_then_execute_routes_reports_and_caps_higher_level_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            work, output = base/'work', base/'reports'
            def prepare(*args):
                work.mkdir()
                (work/'revision_jobs.tsv').write_text('stage\tl1\n')
            flags = ['--work-root', str(work), '--output-root', str(output), '--jobs', '40']
            with patch.object(runner, 'prepare', side_effect=prepare) as prep, \
                    patch.object(runner, 'compile_designs') as compile_designs, \
                    patch.object(runner, 'audit_compiled') as audit, \
                    patch.object(runner, 'run_jobs', return_value=0) as jobs, \
                    patch.object(runner, 'verify_stage_outputs') as verify, \
                    patch.object(runner, 'collect') as collect, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(flags), 0)
                prep.assert_called_once()
                compile_designs.assert_called_once_with(work/'revision_jobs.tsv', 'l1')
                audit.assert_called_once_with(work/'revision_jobs.tsv', output/'design_audit')
                jobs.assert_called_once_with(work/'revision_jobs.tsv', 'l1', 40,
                                             resume=True, dry_run=True)
                verify.assert_not_called()
                collect.assert_not_called()
                jobs.reset_mock()
                self.assertEqual(runner.main(flags + ['--execute', '--accept-design-diagnostics']), 0)
                self.assertEqual([(call.args[1], call.args[2]) for call in jobs.call_args_list],
                                 [('l1', 40), ('l2', 2), ('l3', 2)])
                self.assertTrue(all(call.kwargs == dict(resume=True, dry_run=False)
                                    for call in jobs.call_args_list))
                self.assertEqual(verify.call_count, 3)
                collect.assert_called_once_with(work/'revision_jobs.tsv', output/'imaging', expected_jobs=5)
                self.assertEqual(prep.call_count, 1)

    def test_resume_rejects_configuration_drift_and_output_work_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            work = base/'work'
            work.mkdir()
            (work/'revision_jobs.tsv').write_text('stage\tl1\n')
            (work/'preparation_config.json').write_text('{}')
            with patch.object(runner, 'compile_designs') as compile_designs:
                with self.assertRaisesRegex(ValueError, 'code/configuration changed'):
                    runner.main(['--work-root', str(work), '--output-root', str(base/'reports')])
                with self.assertRaisesRegex(ValueError, 'separate'):
                    runner.main(['--work-root', str(work), '--output-root', str(work/'reports')])
                compile_designs.assert_not_called()

    def test_audit_is_labeled_conditional_basis_and_preserves_pair_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            manifest = base/'revision_jobs.tsv'
            manifest.write_text('stage\tmodel\tsubject\trun\tfsf\n'
                                f'l1\tnppi-dmn\tsub-104\t01\t{base}/input.fsf\n')
            for name in ('phase_construction.tsv', 'baseline_preflight.tsv', 'l1_preflight.tsv', 'analysis_plan.json'):
                (base/name).write_text('test fixture\n')
            matrix = np.random.default_rng(42).normal(size=(100, 26))
            contrasts = np.zeros((11, 26))
            for i in range(6):
                contrasts[i, 9+i] = 1
            contrasts[6, [12, 14]] = [1, -1]
            contrasts[7:, 8] = 1
            def read(path):
                return matrix if path.suffix == '.mat' else contrasts
            with patch.object(runner, 'read_vest_matrix', side_effect=read), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.audit_compiled(manifest, base/'reports')
            rows = read_table(base/'reports/contrast_diagnostics.tsv')
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(row['estimable'] == '1' for row in rows))
            self.assertIn('compiled_basis_variance_ratio', rows[0])
            self.assertNotIn('contrast_vif', rows[0])
            pairs = read_table(base/'reports/phase_correlations.tsv')
            self.assertEqual({(r['decision_ev'], r['post_ev']) for r in pairs},
                             {('1','8'), ('3','8'), ('5','8'), ('10','17'), ('12','17'), ('14','17')})
            scope = json.loads((base/'reports/diagnostic_scope.json').read_text())
            self.assertIn('not original-stimulus cVIF', scope['interpretation'])

    def test_real_templates_prepare_all_jobs_and_leave_source_tree_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            repo, production, repaired, work = (base/name for name in ('repo', 'production', 'repaired', 'work'))
            standard = base/'standard.nii.gz'
            standard.write_bytes(b'input fixture: no imaging fits are run in this test')
            phys = base/'phys.txt'
            phys.write_text('0\n1\n-1\n')
            for relative in {SOURCE_FSF_RELATIVE, *[Path('templates/revision')/name for name in
                             list(runner.MODELS.values()) + list(REFERENCE_NAMES.values())]}:
                target = repo/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/relative, target)
            subjects = [s for _,s,_ in parse_inputs((ROOT/SOURCE_FSF_RELATIVE).read_text().splitlines())]
            templates = {family: (ROOT/f'templates/L1_task-ultimatum_model-02_type-{"act" if family=="act" else "nppi"}.fsf').read_text()
                         for family in runner.MODELS}
            sources = []
            for subject in subjects:
                for run in ('01', '02'):
                    relative = Path(f'source_data/bids/{subject}/func/{subject}_task-ultimatum_run-{run}_events.tsv')
                    events = repo/relative
                    events.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT/relative, events)
                    prefix = base/'source_evs'/subject/f'run-{run}'
                    counts = generate(events, prefix, 'substantive')
                    for family, template in templates.items():
                        parent = (repaired if subject=='sub-144' else production)/subject
                        l1 = parent/f'L1_task-ultimatum_model-02_type-{family}_run-{run}_sm-6.feat/design.fsf'
                        l1.parent.mkdir(parents=True)
                        text = template
                        for role, fmri in [('feat_files(1)', False), ('confoundev_files(1)', False), ('regstandard', True)]:
                            text = replace_fsf_value(text, role, str(standard if role!='confoundev_files(1)' else phys), fmri=fmri)
                        text = replace_fsf_number(text, 'shape7', 3 if counts['missed_trial'] else 10)
                        for i, name in CUSTOM_NAMES.items():
                            text = replace_fsf_value(text, f'custom{i}', f'{prefix}_{name}.txt', fmri=True)
                        if family != 'act':
                            for i in (10, *range(20,29)):
                                text = replace_fsf_value(text, f'custom{i}', str(phys), fmri=True)
                        l1.write_text(text)
                        l1.with_suffix('.mat').write_text('retained design fixture; baseline compilation tested separately\n')
                        sources += [l1, l1.with_suffix('.mat')]
                        if run == '01':
                            l2 = parent/f'L2_task-ultimatum_model-02_type-{family}_sm-6.gfeat/design.fsf'
                            l2.parent.mkdir(parents=True)
                            l2.write_text(f'set fmri(outputdir) "old"\nset fmri(regstandard) "{standard}"\n'
                                          'set feat_files(1) "old1"\nset feat_files(2) "old2"\n')
                            sources.append(l2)
            sources += [p for p in (base/'source_evs').rglob('*') if p.is_file()]
            sources += [p for p in repo.rglob('*') if p.is_file()]
            before = {p:digest(p) for p in sources}
            runner.prepare(repo, production, repaired, work, standard, ())
            self.assertTrue(all(digest(p)==value for p,value in before.items()))
            jobs = read_table(work/'revision_jobs.tsv')
            self.assertEqual([sum(r['stage']==stage for r in jobs) for stage in ('l1', 'l2', 'l3')], [282,141,5])
            self.assertEqual(sum(int(r['expected_zstats']) for r in jobs if r['stage']=='l3'), 32)
            for row in jobs:
                text = Path(row['fsf']).read_text()
                if row['stage']=='l1':
                    self.assertEqual(int(row['n_evs']), 8 if row['model']=='act' else 26)
                    self.assertIn('post', fsf_value(text, 'evtitle8'))
                    self.assertTrue(Path(row['baseline_fsf']).is_file())
                    self.assertIn(str(repaired if row['subject']=='sub-144' else production), row['source_fsf'])
                elif row['stage']=='l3':
                    self.assertEqual(row['group_contract'], 'locked-source')
                    source = Path(row['source_fsf']).read_text()
                    self.assertEqual(parse_evs(text.splitlines()), parse_evs(source.splitlines()))
                    self.assertEqual(len(parse_inputs(text.splitlines())), 47)
            construction = read_table(work/'phase_construction.tsv')
            self.assertEqual(len(construction), 282)
            for row in construction:
                self.assertEqual(int(row['responded_trials']), sum(int(row[f'post_rows_{p}'])
                                                                 for p in ('computer','ingroup','outgroup')))
            with self.assertRaises(FileExistsError):
                runner.prepare(repo, production, repaired, work, standard, ())


if __name__ == '__main__':
    unittest.main()
