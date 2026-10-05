import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import run_gu_choice_extensions as ext
import run_gu_hierarchical as base
from export_gu_stan_diagnostics import variable_label, parameter_table
import pandas as pd


class ChoiceExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = ext.load_trials(ROOT)[0]

    def test_baseline_unchanged_and_identical_data(self):
        self.assertEqual(len(base.MODELS), 4)
        for name in ext.MODELS:
            for stage in ('full', 'run1'):
                a = ext.build_data(self.units, name, stage)
                b = base.build_data(self.units, 'rw_free', stage)
                for key in ('offers','choice','use_choice','observed','run','subject','partner','contrast'):
                    self.assertEqual(a[key], b[key])
                self.assertEqual((a['S'],a['U'],a['T'],a['K']), (47,141,48,4))
        with self.assertRaises(ValueError):
            ext.build_data(self.units, 'rw_positive', age_mode='group')

    def test_equations_and_bias_on_logodds(self):
        offers = np.array([1.,8.,2.,7.])
        theta = np.array([.7,.4,12.,.2])
        z = base.logits(theta, offers, 'rw_free')
        np.testing.assert_allclose(ext.logits(theta, offers, 'rw_positive'), z)
        np.testing.assert_allclose(ext.logits(np.r_[theta, -1.2], offers, 'rw_positive_bias'), z-1.2)
        np.testing.assert_allclose(ext.logits(np.array([np.r_[theta,0.]]*3), offers,
                                           'rw_positive_bias'), [z]*3)
        # Removal of the $1 offer ceiling; not a change of dollar units.
        self.assertGreater(expit(ext.logits(np.array([1.,4.,.1,.01]), [1.], 'rw_positive')[0]), .98)

    def test_support_shared_bias_prior_and_order(self):
        for name in ext.MODELS:
            units = list(reversed(self.units))
            data = ext.build_data(units, name)
            a = ext.draw_prior(data, name, np.random.default_rng(42))
            b = ext.draw_prior(data, name, np.random.default_rng(42))
            np.testing.assert_array_equal(a,b)
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue((a[:,:2]>0).all())
            self.assertTrue((a[:,:2]>1).any())
            self.assertTrue(((a[:,2]>0)&(a[:,2]<20)).all())
            self.assertTrue(((a[:,3]>0)&(a[:,3]<1)).all())
            if name.endswith('bias'):
                for s in range(1,48):
                    bias = a[np.array(data['subject'])==s,4]
                    np.testing.assert_array_equal(bias, [bias[0]]*3)
                for v in ext.difference_draws(a,data).values():
                    np.testing.assert_allclose(v[:,4],0,atol=1e-15)

    def test_causal_offer_history(self):
        theta=np.array([1.5,2.,12.,.2,.3])
        z=ext.logits(theta,[2.,8.,4.,3.],'rw_positive_bias')
        alt=ext.logits(theta,[2.,8.,10.,10.],'rw_positive_bias')
        np.testing.assert_array_equal(z[:2],alt[:2])
        self.assertNotEqual(z[2],ext.logits(theta,[2.,4.,3.],'rw_positive_bias')[1])

    def test_new_bias_terms_enter_diagnostics(self):
        config={'parameters':list(ext.MODELS['rw_positive_bias'][1]),'subjects':['sub-104','sub-105']}
        data={'subject':[2,1],'partner':[3,1]}
        self.assertEqual(variable_label('bias_z[2,1]',config,data)['subject'],'sub-105')
        self.assertEqual(variable_label('theta[1,5]',config,data)['partner'],'all')
        frame=pd.DataFrame({'R_hat':[1.02]*3,'ESS_bulk':[300]*3,'ESS_tail':[500]*3},
                           index=['bias_mu[1]','bias_sigma[1]','bias_z[2,1]'])
        table=parameter_table(frame,config,data)
        self.assertEqual(table.flagged.sum(),3)
        self.assertEqual(set(table.parameter),{'acceptance_bias'})

    def test_config_and_completion_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'job'; config={'model':'rw_positive'}
            ext.config_guard(p,config)
            ext.config_guard(p,config)
            with self.assertRaises(ValueError):
                ext.config_guard(p,{'model':'rw_positive_bias'})
            output=p/'out.tsv'; output.write_text('done')
            record={'config_hash':ext.digest(config),'files':{'out.tsv':ext.sha(output)}}
            ext.atomic_json(p/'completed.json',record)
            self.assertEqual(ext.completed(p,ext.digest(config)),record)
            output.write_text('changed')
            with self.assertRaises(ValueError): ext.completed(p,ext.digest(config))

    def test_resume_skips_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); cfg={'model':'rw_positive'}
            record={'config_hash':ext.digest(cfg),'files':{},'diagnostics':{'passed':False}}
            ext.atomic_json(p/'completed.json',record)
            with patch.dict(sys.modules,{'cmdstanpy':object()}):
                self.assertEqual(ext.fit_job((p,None,cfg,None),None,None,None),record)

    def test_summary_exports_shared_bias_once_and_predictions(self):
        units=self.units[:6]
        data=ext.build_data(units,'rw_positive_bias','run1')
        rng=np.random.default_rng(7)
        theta=np.stack([ext.draw_prior(data,'rw_positive_bias',rng) for _ in range(40)])
        z=ext.logits(theta,np.array(data['offers']),'rw_positive_bias')
        ll=np.array(data['choice'])*z-np.logaddexp(0,z)
        heldout=np.array(data['observed'],bool)&~np.array(data['use_choice'],bool)
        variables={'theta':theta,'replicated_accept':np.zeros((40,3,2,11)),
                   'heldout_log_lik':np.where(heldout,ll,0).sum(2)}
        class FakeFit:
            def summary(self):
                return pd.DataFrame({'R_hat':[1.]*3,'ESS_bulk':[900]*3,'ESS_tail':[900]*3},
                                    index=['mu[1]','bias_sigma[1]','theta[1,5]'])
            def method_variables(self):
                return {'energy__':rng.normal(size=(40,4)),
                        'divergent__':np.zeros((40,4)),'treedepth__':np.ones((40,4))}
            def stan_variable(self,name): return variables[name]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            ext.summarize_fit(FakeFit(),data,'rw_positive_bias',units,directory,12)
            pars=pd.read_csv(directory/'parameters.tsv',sep='\t')
            bias=pars[pars.parameter=='acceptance_bias']
            self.assertEqual(len(bias),2)
            self.assertEqual(set(bias.partner),{'all'})
            contrasts=pd.read_csv(directory/'partner_contrasts.tsv',sep='\t')
            self.assertNotIn('acceptance_bias',set(contrasts.parameter))
            self.assertEqual(len(pd.read_csv(directory/'posterior_parameter_correlations.tsv',sep='\t')),60)
            ppc=pd.read_csv(directory/'participant_predictive.tsv',sep='\t')
            self.assertEqual(set(ppc.offer_bin),{'all','offer_1','offer_1_to_3'})
            row=ppc[(ppc.subject==units[0]['subject'])&(ppc.partner==units[0]['partner'])&
                    (ppc.run==2)&(ppc.offer_bin=='all')].iloc[0]
            mask=heldout[0]
            self.assertAlmostEqual(row['mean'],expit(z[:,0,mask]).mean())
            scores=pd.read_csv(directory/'heldout_prediction.tsv',sep='\t')
            self.assertEqual(scores.n.sum(),heldout.sum())

    def test_wrapper_syntax_and_rejects_typo(self):
        script=ROOT/'code/run_gu_choice_extensions.sh'
        subprocess.run(['bash','-n',str(script)],check=True)
        result=subprocess.run(['bash',str(script),'fit','--exceute'],capture_output=True)
        self.assertNotEqual(result.returncode,0)


@unittest.skipUnless(os.environ.get('SRNDNA_GU_EXTENSION_EXE'),'compiled extension executable not supplied')
class CompiledTests(unittest.TestCase):
    def test_actual_stan_target(self):
        import cmdstanpy
        model=cmdstanpy.CmdStanModel(exe_file=os.environ['SRNDNA_GU_EXTENSION_EXE'])
        with tempfile.TemporaryDirectory() as tmp:
            ext.validate_compiled(model,ext.load_trials(ROOT)[0],Path(tmp))


if __name__ == '__main__':
    unittest.main()
