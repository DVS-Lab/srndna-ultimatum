#!/usr/bin/env python3
"""Build a dated scientific report from retained results; perform no new fits.

The complete standardized PALM inventory is authoritative for current direct
age/main tests. Historical estimators and behavioral interaction tests are
reported separately, with their different designs and unresolved checks.
"""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np

from plot_current_result_review import Review, ROOT, AUDIT, digest, rows

DATE = '2026-11-02'
COLLECTION = Path('results/dmn_followup/v1/palm/collections/20261030T131328Z-zjcxasmt')


def write_table(path, records):
    if not records:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(records)


def pformat(value):
    if value in ('', None):
        return 'No clusters'
    x = float(value)
    return f'{x:.2g}' if x < .001 else f'{x:.4f}'


class Report(Review):
    def __init__(self, output, standard):
        super().__init__(ROOT, output, standard)
        self.source(ROOT/'code/build_scientific_results_report.py')
        self.sections = []
        self.tables = {}
        self.export = output/'tables'; self.export.mkdir(exist_ok=True)
        self.commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        for key, rel in {
            'behavior': 'results/reviewer/tables/primary_acceptance_models.tsv',
            'rt': 'results/reviewer/tables/response_time_model.tsv',
            'misses': 'results/reviewer/tables/missed_trials_age_test.tsv',
            'sensitivity': 'results/reviewer/tables/fairness_sensitivity_age_test.tsv',
            'ratings': 'results/vmpfc_ratings/v1/ratings/retained_paired_tests.tsv',
            'rating_associations': 'results/reviewer/ultimatum_ratings/ultimatum_ratings_fairness_sensitivity.tsv',
            'rating_choice': 'results/reviewer/ultimatum_ratings/rating_choice_moderation_focal_tests.tsv',
            'crossed': 'results/crossed_behavior_l3/v1/imaging/summary.tsv',
            'randomise': 'results/permutation_audit/v1/model_contrast_summary.tsv',
            'dmn_flame': 'results/dmn_covariate_standardized/v1/imaging/summary.tsv',
            'dmn_randomise': 'results/dmn_covariate_standardized/v1/randomise/model_contrast_summary.tsv',
            'stable': 'results/norm_learning/stan-bias-stable-v1/baseline_stable_comparison.tsv',
            'recovery': 'results/norm_learning/stan-bias-validation-v1/recovery_summary_by_target.tsv',
            'deletion': str(AUDIT/'influence/deletion_summary.tsv'),
            'robust': str(AUDIT/'influence/slope_comparison.tsv'),
            'historical_palm': str(COLLECTION/'interim_results.tsv'),
        }.items():
            path = self.source(ROOT/rel); self.tables[key] = rows(path)
            shutil.copyfile(path, self.export/(key+'.tsv'))
        write_table(self.export/'palm_age_main_all_96.tsv', self.summary)
        self.contract = json.loads(self.source(self.palm/'contract.json').read_text())
        self.old_status = json.loads(self.source(ROOT/COLLECTION/'collection_status.json').read_text())
        self.norm_status = json.loads(self.source(ROOT/'results/norm_learning/stan-bias-validation-v1/workflow_status.json').read_text())
        self.source(ROOT/'templates/revision/L1_task-ultimatum_model-02_type-act_fairness-main.fsf')
        self.source(ROOT/'templates/revision/L1_task-ultimatum_model-decision-postresponse-partner_type-act.fsf')
        self.source(ROOT/'code/make_ultimatum_3col.py')
        self.source(ROOT/'code/run_full_rt_correction.py')
        self.source(ROOT/'code/stan/gu_choice_extensions.stan')
        self.source(ROOT/'code/analyze_reviewer_behavior.R')
        if len(self.summary) != 96 or self.contract['shuffles'] != 10000:
            raise ValueError('PALM report scope changed')
        if self.norm_status['parameter_recovery_validated'] or self.norm_status['imaging_covariates_released']:
            raise ValueError('norm-model eligibility changed: update report')

    def section(self, title, paragraphs=(), table=None, widths=None, image=None, caption=None):
        self.sections.append(dict(title=title, paragraphs=list(paragraphs), table=table,
                                  widths=widths, image=image, caption=caption))

    def figure_section(self, name):
        page = next(x for x in self.pages if x['name'] == name)
        self.section(page['title'], image=page['image'], caption=page['caption'])

    def retained_figure(self, folder, name, title, caption):
        manifest = json.loads(self.source(ROOT/folder/'output_inventory.json').read_text())
        image = ROOT/folder/(name+'.png')
        if digest(self.source(image)) != manifest[image.name]:
            raise ValueError(f'retained image differs: {image}')
        dest = self.output/(name+'.png'); shutil.copyfile(image, dest)
        self.pages.append(dict(name=name, image=dest.name, title=title, caption=caption))

    def cell(self, model, method, contrast):
        r = next(x for x in self.summary if x['model'] == model and x['method'] == method and int(x['contrast']) == contrast)
        return f"{pformat(r['minimum_spatial_fwe_p'])} / {r['significant_voxels']}"

    def overlay_page(self, specs, name, title, subtitle, caption):
        # Leave separate vertical space for the colorbar label and orientation
        # note; the older compact two-row layout crowds those annotations.
        count=len(specs)
        fig=plt.figure(figsize=(13, 2.65*count+2.2))
        grid=fig.add_gridspec(count,4,width_ratios=[1.25,1,1,1],left=.025,right=.965,
                             top=.83,bottom=.22,hspace=.30,wspace=.05)
        for row,spec in enumerate(specs):
            payload=self.payload(*spec);record=payload[0]
            ax=fig.add_subplot(grid[row,0]);ax.axis('off')
            ax.text(.02,.82,record['title'],fontsize=12,weight='bold',va='top')
            ax.text(.02,.34,f"pFWE = {record['minimum_spatial_fwe_p']:.4f}\n{record['native_voxels']} native voxels\n{record['method'].replace('cluster-z','Z > ')}",fontsize=10,va='top')
            self.planes([fig.add_subplot(grid[row,col])for col in range(1,4)],payload)
        fig.suptitle(title,fontsize=18,y=.98)
        fig.text(.5,.915,subtitle,ha='center',fontsize=10.5)
        cbax=fig.add_axes([.34,.11,.38,.018])
        cb=fig.colorbar(ScalarMappable(norm=Normalize(-6,6),cmap='coolwarm'),cax=cbax,orientation='horizontal')
        cb.set_label('Signed equivalent Z relative to the original contrast',fontsize=9,labelpad=4)
        fig.text(.025,.015,'Native-map inference; nearest-neighbor display resampling only. Neurological L/R. All 47 participants retained.',fontsize=8)
        self.save(fig,name,title,caption)

    def model_schematic(self):
        offers = np.arange(1, 11)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True)
        curves = [(offers-5.5, 'Current: linear offer slope', 'Negative slope: smaller > larger offers'),
                  (np.where(offers <= 4, 1, np.where(offers >= 7, 0, .5)),
                   'Different question: category contrast', 'Unfair ($1-4) versus fair ($7-10)'),
                  (np.where(offers <= 4, 0, np.where(offers >= 7, 0, 1)),
                   'A response the linear slope can miss', 'Illustrative peak at intermediate offers')]
        for ax, (values, title, subtitle) in zip(axes, curves):
            ax.plot(offers, values, 'o-', color='#397e9e'); ax.set_title(title, fontsize=11)
            ax.set_xlabel('Offer to participant ($)'); ax.set_xticks([1, 4, 5, 6, 7, 10])
            ax.set_yticks([]); ax.text(.5, -.27, subtitle, ha='center', transform=ax.transAxes, fontsize=9)
        axes[0].plot(offers,-curves[0][0],'o--',color='#ba6c43',label='Negative slope')
        axes[0].lines[0].set_label('Positive slope');axes[0].legend(frameon=False,fontsize=8)
        axes[0].set_ylabel('Illustrative response (arbitrary units)')
        fig.suptitle('Offer slope and unfair > fair are related, but not identical', fontsize=16)
        fig.subplots_adjust(left=.06, right=.99, bottom=.25, top=.78, wspace=.22)
        self.save(fig, '00_model_questions', 'Model clarification | What we have and have not tested',
            'Schematic curves, not participant data. The existing offer modulator tests a linear response to amount. '
            'Its negative contrast tests a linear increase for smaller offers. A categorical unfair-minus-fair test '
            'would require separately modeled categories (or an appropriate richer basis); it cannot be obtained '
            'by relabeling the negative linear slope. The third curve illustrates one possible nonlinear response, '
            'not evidence that such a response exists in this dataset. No categorical refit is included in this report.')

    def figures(self):
        self.model_schematic(); self.behavior(); self.ratings(); self.dmn_bars()
        self.overlay_page([
            ('taskwide-offer', 'cluster-z3.1', 1, 'Larger offers\nAll partners / all ages', 1),
            ('human-computer-response', 'cluster-z3.1', 1, 'Human > computer\nMean task response', 1),
            ('human-computer-response', 'cluster-z3.1', 2, 'Computer > human\nMean task response', -1)],
            '03_foundation', 'Whole-sample results: offer slopes and social-context responses',
            'PALM | Z > 3.1 cluster formation; spatial FWE p < .05 | 47 participants',
            'The first row is parametric; the other rows are unmodulated partner-response contrasts from models '
            'that also contain offer modulators. They test different quantities. Positive offer-related activation '
            'is posterior; it is not a convergent anterior-insula/dACC unfairness finding. All displayed PALM voxels '
            'lie within earlier significant FLAME and randomise maps, whose covariates/masks are not fully matched.')
        self.overlay_page([
            ('human-computer-response', 'cluster-z3.1', 3, 'Younger > older\nHuman - computer', 1),
            ('human-computer-response', 'cluster-z2.6', 3, 'Same age contrast\nLower cluster threshold', 1)],
            '04_social_context_age', 'Age difference in the human-computer response',
            'Offer/display-period activation | Z > 3.1 and Z > 2.6 shown separately',
            'Posterior midline human-minus-computer response difference: PALM p = .0163, 25 voxels at Z > 3.1; '
            'p = .0253, 118 voxels at Z > 2.6. This is an age-by-partner-context effect, not an age difference '
            'in the offer slope. It includes the modeled display period and does not isolate a pure decision '
            'operation or mentalizing. Participant bars for this exact 25-voxel map have not been extracted; '
            'previous bars from a larger selected region are not relabeled here.')
        self.overlay_page([
            ('taskwide-offer', 'cluster-z2.6', 3, 'Younger > older\nAverage offer slope', 1),
            ('full-rt__nppi-dmn', 'cluster-z2.6', 3, 'Younger > older\nSimilar - dissimilar\nDMN offer coupling', 1)],
            '05_fairness_age', 'Offer-related age differences at Z > 2.6',
            'Spatial FWE p < .05 | At Z > 3.1: offer-age p = .1017; DMN-age p = .2526 (both 0 significant voxels)',
            'Task-wide activation: p = .0352, 149 voxels. Full-display DMN PPI: p = .0468, 49 voxels. '
            'Both are corrected whole-brain findings at the lower cluster-forming threshold, with the higher-threshold '
            'results disclosed. They remain threshold-sensitive evidence, not findings rescued by choosing the '
            'threshold after inspecting results. The activation effect measures an average offer slope; the PPI '
            'effect measures a between-partner difference in offer-dependent network coupling.')
        self.overlay_page([
            ('pooled-post__nppi-dmn', 'cluster-z3.1', 1, 'DMN similar > dissimilar\nPooled post-response', 1),
            ('pooled-post__nppi-ecn', 'cluster-z3.1', 2, 'ECN dissimilar > similar\nPooled post-response', -1),
            ('partner-post__nppi-ecn', 'cluster-z3.1', 2, 'ECN dissimilar > similar\nPartner post-response', -1)],
            '07_network_means', 'Offer-modulated coupling: small full-sample partner effects',
            'PALM | Z > 3.1; spatial FWE p < .05 | Main effects, not age differences',
            'DMN: p = .0424, 10 voxels. ECN: p = .0298, 10 voxels (pooled post); p = .0121, 13 voxels '
            '(partner post). The ECN regions overlap in 8 voxels. These effects do not survive Z > 3.1 in '
            'the full-display model; they remain exploratory and timing dependent. Precise tissue localization '
            'and influence checks are required before substantive regional interpretation.')
        self.retained_figure(Path('results/figure_review/2026-11-02-parametric'),
            '01_across_partner_offer_activation', 'Across-partner offer activation | Anatomical comparison',
            'Corrected positive offer-related activation, averaged across all three partners. The retained FLAME '
            'right-insular cluster contains 82 voxels, p = .000129, Zmax = 4.49, peak [40.3, 1.29, 5.22] mm. '
            'It reflects larger offers, not greater activation for unfair offers. Neither permutation analysis '
            'confirms that insular cluster at either displayed cluster threshold. No corrected negative '
            'offer-slope cluster is established. This is whole-brain inference; no new ROI analysis is included.')
        source = self.source(ROOT/AUDIT/'influence/parietal_huber_scatter.png')
        shutil.copyfile(source, self.output/'09_parietal_diagnostics.png')
        self.pages.append(dict(name='09_parietal_diagnostics', image='09_parietal_diagnostics.png',
            title='Exploratory brain-behavior interaction | Influence diagnostics',
            caption='Descriptive selected-cluster scatterplots, not independent ROI tests. Outcome: similar-minus-dissimilar '
            'activation offer slope; predictor: the corresponding acceptance-intercept difference, not the fairness '
            'choice slope and not a recovered Gu norm parameter. Robust fits and single/pair deletions preserve '
            'the interaction sign, but region selection is circular and a coverage-corrected audit remains uncollected. '
            'The 125-voxel extraction region is distinct from the 61-voxel FLAME/randomise overlap region.'))
        self.complete_grid()

    def comparisons(self):
        out = []
        foundations = [('taskwide-offer', ROOT/'results/reviewer/activation_fairness_main'),
                       ('human-computer-offer', ROOT/'results/reviewer/activation_social_computer/social_offer_slope'),
                       ('human-computer-response', ROOT/'results/reviewer/activation_social_computer/social_task_response')]
        for model, folder in foundations:
            for c in range(1, 5):
                if model == 'taskwide-offer':
                    sub = {1:'focal_contrast1', 2:'contrast2_adjusted-mean-negative',
                           3:'contrast3_older-younger', 4:'contrast4_younger-older'}[c]
                else:
                    sub = f'contrast{c}'
                cluster_rows = rows(self.source(folder/sub/f'cluster_zstat{c}_std.txt'))
                # Older retained foundation contrasts 3/4 have reverse age order
                # from PALM; means 1/2 have the same order.
                target_c = {1:1, 2:2, 3:4, 4:3}[c]
                out.append(dict(model=model, direction=self.contract['labels'][target_c-1],
                    engine='FLAME', threshold='Z>3.1', p=min((float(r['P']) for r in cluster_rows), default=''),
                    voxels=sum(int(r['Voxels']) for r in cluster_rows), design='earlier corrected covariates/mask'))
                for method in ['cluster-t3.1', 'cluster-t2.6']:
                    r = next(r for r in self.tables['randomise'] if r['model'] == model and int(r['contrast']) == c and r['method'] == method)
                    out.append(dict(model=model, direction=self.contract['labels'][target_c-1], engine='randomise',
                        threshold=method, p=float(r['minimum_spatial_fwe_p']), voxels=int(r['significant_voxels']),
                        design='earlier corrected covariates/mask'))
        for timing in ['full-rt', 'pooled-post', 'partner-post']:
            model = timing+'__dmn-age'
            for c in (3, 4):
                r = next(r for r in self.tables['dmn_flame'] if r['model'] == model and int(r['contrast']) == c)
                out.append(dict(model=timing+'__nppi-dmn', direction=self.contract['labels'][c-1],
                    engine='FLAME', threshold='Z>3.1', p=r['minimum_cluster_p'], voxels=int(r['voxels']),
                    design='standardized DMN; earlier mask'))
                for method in ['cluster-t3.1','cluster-t2.6']:
                    r = next(r for r in self.tables['dmn_randomise'] if r['model'] == model and int(r['contrast']) == c and r['method'] == method)
                    out.append(dict(model=timing+'__nppi-dmn', direction=self.contract['labels'][c-1],
                        engine='randomise', threshold=method, p=float(r['minimum_spatial_fwe_p']),
                        voxels=int(r['significant_voxels']), design='standardized DMN; earlier mask'))
        for r in self.summary:
            out.append(dict(model=r['model'], direction=r['contrast_name'], engine='PALM', threshold=r['method'],
                p=float(r['minimum_spatial_fwe_p']), voxels=int(r['significant_voxels']),
                design='standardized covariates; strict coverage; age variance groups'))
        write_table(self.export/'method_comparison.tsv', out)
        self.comparison = out
        overlap=[]
        for model,c,folder,oldc in [
                ('taskwide-offer',1,'results/reviewer/activation_fairness_main/focal_contrast1',1),
                ('human-computer-response',1,'results/reviewer/activation_social_computer/social_task_response/contrast1',1),
                ('human-computer-response',2,'results/reviewer/activation_social_computer/social_task_response/contrast2',2),
                ('human-computer-response',3,'results/reviewer/activation_social_computer/social_task_response/contrast4',4)]:
            a=nib.load(self.source(self.palm/f'maps/{model}/cluster-z3.1/palm_clustere_zvstat_fwep_c{c}.nii'))
            b=nib.load(self.source(ROOT/folder/f'cluster_mask_zstat{oldc}.nii.gz'))
            r=next(r for r in self.tables['randomise']if r['model']==model and int(r['contrast'])==oldc and r['method']=='cluster-t3.1')
            d=nib.load(self.source(ROOT/f"results/permutation_audit/v1/maps/{r['test']}/cluster-t3.1/randomise_clustere_corrp_tstat1.nii.gz"))
            if a.shape!=b.shape or a.shape!=d.shape or not np.allclose(a.affine,b.affine)or not np.allclose(a.affine,d.affine):
                raise ValueError('method comparison grids differ')
            sig=a.get_fdata()>.95
            row=dict(model=model,palm_contrast=c,palm_voxels=int(sig.sum()),
                     flame_overlap=int(np.sum(sig&(b.get_fdata()>0))),
                     randomise_overlap=int(np.sum(sig&(d.get_fdata()>.95))))
            if row['palm_voxels']!=row['flame_overlap']or row['palm_voxels']!=row['randomise_overlap']:
                raise ValueError('method-convergence claim changed')
            overlap.append(row)
        write_table(self.export/'method_spatial_overlap.tsv',overlap)

    def narratives(self):
        self.section('SRNDNA Ultimatum | Updated results report', [
            '<b>Scientific review for the author team | 2 November 2026.</b> N = 47 (25 younger, 22 older). '
            'This report consolidates the corrected results and supersedes the earlier figure packets. It does not '
            'constitute a revised manuscript or a completed response to reviewers.',
            '<b>Behavior:</b> acceptance increases strongly with offer amount. The primary mixed model does not '
            'detect age moderation of the offer slope or its similar-dissimilar difference. Partner ratings '
            'distinguish humans from the computer; they do not establish an age-specific behavioral mechanism.',
            '<b>Imaging:</b> the full-sample positive offer slope converges in posterior cortex. An age difference '
            'in the unmodulated human-computer response survives PALM at both cluster thresholds. Offer-related '
            'age findings are weaker: task-wide activation and full-display DMN PPI survive at Z &gt; 2.6, '
            'but not Z &gt; 3.1. No original ECN age claim is restored.',
            '<b>Scope:</b> whole-brain cluster inference at Z &gt; 2.6 and Z &gt; 3.1 is shown side by side. '
            'No TFCE result, vmPFC ROI test, or new targeted ROI inference is used to support the current narrative. '
            'Selected-cluster plots are descriptive decompositions of effects, not additional independent tests.',
            '<b>Computational behavior:</b> Gu-style hierarchical models and bias extensions were fitted. '
            'Recovery and diagnostic gates remain uncleared, so individual model parameters are not released '
            'as imaging covariates or presented as established age mechanisms.',
            '<b>Completed current batch:</b> 24/24 standardized PALM jobs, 96 signed summaries, 10,000 shuffles '
            'per test, all 47 participants retained. Earlier broad PALM and activation-coverage work have separate '
            'uncollected or unfinished status; they are not silently treated as complete.'
        ])
        self.section('What the current imaging model tests', [
            '<b>Original and corrected full-display model:</b> three partner-specific display regressors '
            '(computer, similar, dissimilar), three matching offer-amount modulators, missed trials, a response-related '
            'constant term and an RT-height term, plus confounds. The correction restores dedicated RT coverage '
            'for block-first responses; it does not replace the original task with an accept/reject model. '
            'The corrected RT EVs are zero-duration events at offer onset, with heights 1 and raw RT. '
            'In nPPI, network main effects and task-by-network interactions are added; the offer-PPI '
            'contrast concerns those interactions, not the corresponding activation coefficient.',
            '<b>Offer slope:</b> the modulator asks whether activity changes approximately linearly with the '
            'amount offered. Positive means more activity for larger offers; negative means more activity for '
            'smaller offers. Amount also changes the participant\'s own payoff, so the task cannot isolate fairness '
            'from monetary value. PPI tests analogous offer-dependent network coupling, not activation itself.',
            '<b>Unmodulated partner response:</b> the human-computer contrast compares mean offer/display '
            'responses while retaining the offer modulators. The templates orthogonalize each offer modulator '
            'to its matching constant partner EV. This is a context-response contrast; it is not a purified '
            'fairness effect or proof of mentalizing.',
            '<b>Were we doing unfair &gt; fair as well?</b> No separate categorical unfair-minus-fair model '
            'appears in the current analysis. The negative offer-slope contrast addresses a related linear '
            'prediction. Under a linear response, the two should be directionally consistent; they are not '
            'identical tests when responses are nonlinear. No categorical refit was launched for this report.'
        ], table=[['Contrast','Quantity tested'],
            ['COPE7: similar - dissimilar offer slopes','Partner difference in linear offer sensitivity (or its PPI analogue)'],
            ['COPE8: -2 computer_p + similar_p + dissimilar_p','Human-computer offer-slope difference; twice the human-average contrast'],
            ['COPE10: -2 computer + similar + dissimilar','Unmodulated human-computer response; twice the human-average contrast'],
            ['COPE11: (computer_p + similar_p + dissimilar_p)/3','Average offer slope across all three partners']], widths=[320,450])
        self.figure_section('00_model_questions')
        self.section('Corrections, timing specifications and group inference', [
            '<b>Data identity:</b> sub-144 Ultimatum events were repaired from retained source records. '
            'Sub-143 had substantive RT information despite lacking redundant event_RT rows; it did not require '
            'the same identity repair. The dedicated historical RT EVs omitted block-first responses. The '
            'all-responded correction propagated through activation and DMN/ECN L1/L2 analyses.',
            '<b>Group model:</b> all-ones mean, centered categorical age, sex, verified tSNR, verified mean '
            'framewise displacement, and corrected mean RT. The new direct age/main PALM batch uses six '
            'full-rank columns consistently. Its mean is sample-weighted across 25 younger and 22 older '
            'participants, rather than an equal-weight mean of the two age groups.',
            '<b>RT in timing alternatives:</b> both the constant partner term and its offer modulator '
            'use onset-to-button-press duration in the decision-epoch models. These replace the dedicated '
            'RT-height terms with one or three post-response nuisance EVs. Imaging RT is measured from '
            'offer onset; the auxiliary behavioral RT analysis subtracts the fixed one-second delay '
            'before response options appear.',
            '<b>Coverage:</b> the strict mask contains 59,297 native voxels with valid estimates and positive '
            'VARCOPE support across all 94 runs and 47 participant-level estimates. No participants were excluded. '
            'FILM threshold --thr=1000 was verified; low-signal zero estimates inside broader masks were not '
            'interpreted as reliable observations. Original signed continuous network maps were retained.',
            '<b>Inference:</b> FLAME is the submitted mixed-effects estimator and uses within-participant '
            'variance information. randomise provides an unweighted permutation comparison. Current PALM uses '
            'age variance groups, symmetric-error sign flips and Freedman-Lane nuisance handling; it does not '
            'reproduce FLAME VARCOPE weighting. Unequal-variance handling need not make every test more conservative.',
            '<b>Thresholds:</b> PALM uses Z &gt; 3.1 and Z &gt; 2.6 cluster formation with spatial FWE p &lt; .05. '
            'randomise uses t &gt; 3.1 and t &gt; 2.6, which are not identical Z thresholds. Report both '
            'prespecified audit thresholds rather than choosing whichever is significant. Spatial FWE applies '
            'within each signed test, not jointly across all models, directions and thresholds.'
        ], table=[['Timing label','Task epochs and RT treatment','Interpretation'],
            ['Full RT','Full ~3.5-second display + corrected response/RT-height EVs','Closest to the submitted model; includes visual and post-response time'],
            ['Pooled post','Offer onset to response; one pooled post-response EV','Separates residual display time with a shared nuisance response'],
            ['Partner post','Offer onset to response; three partner-specific post-response EVs','More flexible nuisance model; high decision/post-response correlations']], widths=[100,335,335])
        b = [r for r in self.tables['behavior'] if r['model_id']=='primary_maximal']
        bnames = {'(Intercept)':'Intercept (reference cell)', 'offer_c':'Offer slope (younger / dissimilar reference)',
            'age_groupolder':'Older - younger at centered offer / dissimilar', 'similaritysimilar':'Similar - dissimilar (younger)',
            'offer_c:age_groupolder':'Age difference in offer slope (dissimilar)',
            'offer_c:similaritysimilar':'Partner difference in offer slope (younger)',
            'age_groupolder:similaritysimilar':'Age x partner at centered offer',
            'offer_c:age_groupolder:similaritysimilar':'Age x offer x partner'}
        self.section('Corrected behavior | Choices, response time and misses', [
            '<b>Primary social-choice GLMM:</b> 47 participants, 4,438 responded human-partner trials. '
            'Participant-specific intercepts and offer slopes; converged, nonsingular, with agreement across '
            'optimizers. The strong offer coefficient is conditional on the reference age/partner; it is not '
            'a marginal all-condition average. Null interactions do not establish equivalence.',
            '<b>Response-selection RT:</b> time measured after the response options appeared (one second '
            'after offer onset). Older-minus-younger coefficient = -0.1187 seconds, p = .0382; partner '
            'p = .8367, age x partner p = .5872. These are retained auxiliary tests, not a mechanism explaining '
            'the imaging age contrast.',
            '<b>Misses:</b> younger mean 1.48 (37 total), older mean 3.50 (77 total); Welch p = .0959. '
            'The similar-minus-dissimilar behavioral offer-sensitivity score has younger-minus-older '
            'difference -0.0979, p = .5837. The simpler random-intercept choice model also does not detect '
            'the three-way interaction (p = .4547).'
        ], table=[['Primary term','Estimate','SE','p']] + [[bnames[r['term']],f"{float(r['estimate']):.4f}",f"{float(r['std_error']):.4f}",pformat(r['p_value'])] for r in b], widths=[470,100,100,100])
        self.figure_section('01_choice_behavior'); self.figure_section('02_partner_ratings')
        rt = [r for r in self.tables['ratings'] if r['dimension'] in ['fairness','likeability'] and r['comparison'] in ['human_mean_minus_computer','similar_minus_dissimilar']]
        self.section('Ratings | Social context, fairness sensitivity and limits', [
            '<b>Sample and provenance:</b> 42 pre-task and 42 post-task ratings; 41 paired. The final complete '
            'attempt is the version of record. Sub-143 has no ratings source; sub-144 ratings were checked '
            'for participant provenance and are not duplicates of another participant\'s ratings.',
            '<b>Choice-slope associations:</b> exploratory similar-minus-dissimilar pre-task likeability '
            'correlates negatively with the corresponding choice-slope difference (Spearman rho = -.4055, '
            'p = .00772, BH q = .03662). Post-task anger correlates positively (rho = .3974, p = .00916, '
            'q = .03662). Corresponding Pearson tests have q = .09767. Fairness-rating associations do '
            'not survive correction; age interactions are not established.',
            '<b>Direct trial-level moderation:</b> pre-task fairness p = .8682 and likeability p = .8342 '
            '(42 participants, 3,969 trials); their age interactions p = .2032/.6220. These findings '
            'provide social-context description, not evidence that ratings explain an age-related fairness mechanism.'
        ], table=[['Time','Rating','Contrast','n','Mean difference','BH q']] + [[r['timepoint'],r['dimension'],r['comparison'].replace('_',' '),r['n'],f"{float(r['mean_difference']):.3f}",pformat(r['p_fdr_bh_within_family'])] for r in rt], widths=[60,90,270,45,150,155])
        self.figure_section('03_foundation')
        main_table = [['Effect','Z > 3.1: p / voxels','Z > 2.6: p / voxels']]
        for model,c,label in [('taskwide-offer',1,'Positive average offer slope'),('taskwide-offer',2,'Negative average offer slope'),
                ('human-computer-offer',1,'Human > computer offer slope'),('human-computer-offer',2,'Computer > human offer slope'),
                ('human-computer-response',1,'Human > computer response'),('human-computer-response',2,'Computer > human response')]:
            main_table.append([label,self.cell(model,'cluster-z3.1',c),self.cell(model,'cluster-z2.6',c)])
        self.section('Full-sample activation | Both directions and thresholds', [
            '<b>Foundational result:</b> larger offers have a corrected positive activation effect across '
            'all partners and ages. Current permutation convergence is posterior (lingual/occipital regions), '
            'rather than an established anterior-insula or dACC unfairness result.',
            '<b>Social context:</b> human-computer response contrasts identify different sets of regions '
            'in the positive and negative directions. Human-computer offer-slope differences are nonsignificant '
            'in the standardized PALM batch. Mean task responses must not be described as offer sensitivity.',
            '<b>Insula:</b> earlier corrected-data FLAME detects a positive right-insular cluster; its '
            'anatomical label is supported by the atlas audit. This finding is not confirmed by the '
            'permutation maps at either threshold. No corrected negative offer-slope or dACC cluster '
            'is established. A null whole-brain result is not proof of no response.'
        ], table=main_table, widths=[370,200,200])
        self.figure_section('01_across_partner_offer_activation')
        self.figure_section('04_social_context_age')
        self.figure_section('05_fairness_age'); self.figure_section('06_dmn_partner_bars')
        self.section('Age effects | What holds, what changes with the model', [
            '<b>Direct age evidence:</b> the human-computer mean response is the only direct age contrast '
            'in the current standardized PALM batch that survives Z &gt; 3.1. It concerns posterior midline '
            'activation during the offer/display period, not a network PPI or an offer slope.',
            '<b>Offer-related age evidence:</b> full-display DMN younger &gt; older survives Z &gt; 2.6 '
            '(49 voxels) but not Z &gt; 3.1. The task-wide offer-slope age contrast has the same '
            'threshold dependence. Human-computer offer-slope age tests and all older &gt; younger '
            'tests are nonsignificant at both thresholds. Network timing alternatives do not establish '
            'a Z &gt; 3.1 offer-related age effect.',
            '<b>DMN decomposition:</b> within the selected 49-voxel region, the similar-dissimilar age gap '
            'is 20.1 without group covariates, 25.2 with sex/tSNR/RT, and 36.9 after also including group FD. '
            'Direction is present before group motion adjustment; magnitude increases with adjustment. '
            'This is a descriptive sensitivity analysis, not three independent whole-brain tests.',
            '<b>Partner-age interpretation:</b> similar and dissimilar reverse proposer age across '
            'participant groups. The observed descriptive crossover aligns with positive offer coupling '
            'for younger proposers and negative coupling for older proposers in both groups. It should '
            'not be presented as simple own-age favoritism.',
            '<b>Original claims:</b> the original DMN-ACC and ECN fairness-sensitivity narrative cannot '
            'be carried forward with its original numbers. The corrected evidence must replace it; '
            'within-age significance alone does not establish an age interaction.'
        ])
        self.figure_section('07_network_means')
        self.method_page()
        self.section('Decision-period social context | Literature and claim boundaries', [
            '<b>Closest analytic precedent:</b> Crockett et al. (2013, Journal of Neuroscience) modeled '
            'human/computer offer onsets plus amount and fairness modulators, pooling choices in their primary '
            'model. They also tested the human-computer response contrast. They could not separately estimate '
            'accepted and rejected computer offers because many participants never rejected them. '
            '<link href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3593678/" color="#28617d">Full paper</link>.',
            '<b>Aging precedent:</b> Harle and Sanfey (2012, Neuropsychologia) modeled decision epochs '
            'by partner and offer amount, pooling acceptance/rejection. Amount was categorical, not omitted. '
            'Their behavioral age difference was concentrated at a moderately unfair offer, illustrating '
            'why a categorical and a linear question can differ. '
            '<link href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3358414/" color="#28617d">Full paper</link>.',
            '<b>Broader rationale:</b> Rilling et al. (2004, NeuroImage) reported human/computer '
            'differences during economic interaction. This supports studying partner context as a meaningful '
            'question, but does not establish that our observed posterior cluster implements mentalizing. '
            '<link href="https://doi.org/10.1016/j.neuroimage.2004.04.015" color="#28617d">Article</link>.',
            '<b>Interpretation limit:</b> Abu-Akel et al. (2020) dissociated perceived intentionality '
            'from human/computer identity. Our contrast does not make that dissociation and includes different '
            'partner imagery. Use social-context response as the descriptive construct, with mentalizing '
            'as a possible interpretation rather than a demonstrated mechanism. '
            '<link href="https://doi.org/10.1098/rspb.2020.0244" color="#28617d">Article</link>.',
            '<b>Potential result wording:</b> Younger adults showed a larger human-versus-computer '
            'offer-period response difference in posterior midline cortex. This difference survived '
            'whole-brain cluster correction in the standardized unequal-variance PALM analysis '
            '(Z &gt; 3.1; p = .0163). The contrast measures partner-context responses across offers '
            'and should be distinguished from age differences in linear offer sensitivity.'
        ])
        self.brain_behavior()
        self.figure_section('09_parietal_diagnostics')
        self.computational()
        self.section('Current figure inventory and remaining decisions', [
            '<b>Candidate main figures:</b> (1) corrected task schematic and timing; (2) social-choice '
            'acceptance curves with participant SEM; (3) full-sample offer activation and clearly separated '
            'human/computer context effects; (4) posterior social-context age effect if retained by the authors. '
            'Final numbering depends on how much the revision can change the original narrative.',
            '<b>Supplementary candidates:</b> three-partner ratings; paired Z &gt; 2.6 / Z &gt; 3.1 '
            'fairness-age results; the DMN three-partner/motion decomposition; complete signed-test inventory. '
            'Timing-dependent network means and the parietal interaction require the outstanding checks '
            'before promotion to central findings. No vmPFC ROI figure is proposed.',
            '<b>Required before final claims:</b> collect the activation-coverage batch and current broader '
            'PALM snapshot; resolve calibration mismatches in that batch; verify matched masks/covariates '
            'before attributing method discrepancies to FLAME/randomise/PALM; finalize a timing specification '
            'on design grounds. Extract participant bars from the exact new posterior age region if desired.',
            '<b>Optional, not already run:</b> a separate categorical unfair &gt; fair model could test '
            'nonlinear fairness responses. The linear negative offer contrast already addresses the basic '
            'smaller-offer prediction. A new categorical model should be a named exploratory addition, '
            'not assumed necessary because an expected region failed to survive.',
            '<b>Writing:</b> the manuscript has not yet been updated. Replace superseded imaging claims, '
            'describe the event/RT correction and new robustness analyses, distinguish original and '
            'revision-added questions, and then draft the analytical reviewer responses from this result set. '
            'Keep original outputs as provenance rather than mixing them with corrected results.',
            '<b>Data release:</b> OpenNeuro ds003745 version 2.2.0 is published. '
            '<link href="https://openneuro.org/datasets/ds003745/versions/2.2.0" color="#28617d">Dataset release</link>. '
            'The paper repository of record is DVS-Lab/srndna-ultimatum.'
        ])
        self.figure_section('08_complete_result_grid')
        self.appendices()

    def method_page(self):
        records = [['Effect / direction','FLAME Z > 3.1','randomise t > 3.1','PALM Z > 3.1','PALM Z > 2.6']]
        cases = [('taskwide-offer','full-sample-positive','Mean positive offer slope'),
                 ('human-computer-response','full-sample-positive','Mean human > computer'),
                 ('human-computer-response','full-sample-negative','Mean computer > human'),
                 ('human-computer-response','younger-minus-older','Age: human-computer response'),
                 ('taskwide-offer','younger-minus-older','Age: average offer slope'),
                 ('human-computer-offer','younger-minus-older','Age: human-computer offer slope'),
                 ('full-rt__nppi-dmn','younger-minus-older','Age: DMN full display'),
                 ('pooled-post__nppi-dmn','younger-minus-older','Age: DMN pooled post'),
                 ('partner-post__nppi-dmn','younger-minus-older','Age: DMN partner post')]
        for model,direction,label in cases:
            values=[]
            for engine,method in [('FLAME','Z>3.1'),('randomise','cluster-t3.1'),('PALM','cluster-z3.1'),('PALM','cluster-z2.6')]:
                r=next(r for r in self.comparison if r['model']==model and r['direction']==direction and r['engine']==engine and r['threshold']==method)
                values.append(f"{pformat(r['p'])} / {r['voxels']}")
            records.append([label]+values)
        self.section('FLAME, randomise and PALM | Side-by-side evidence', [
            'Cells show minimum spatially corrected p / total significant voxels. FLAME without a surviving '
            'cluster has no cluster p to report. These are earlier corrected FLAME/randomise models versus '
            'the newest standardized PALM models; masks/covariates differ in places. The table therefore '
            'shows current evidence, not a controlled comparison of statistical software.',
            '<b>Convergence:</b> the positive offer mean, both human/computer response means, and '
            'the human-computer age contrast have overlapping significant regions across all three '
            'approaches. All 45 / 270 / 155 / 25 significant current PALM voxels, respectively, lie '
            'within the earlier FLAME and randomise maps. This is method robustness on the same data, '
            'not independent replication.',
            '<b>Discrepancies:</b> task-wide offer-age and DMN effects depend on method, mask, '
            'timing and/or threshold. No method should be selected because it yields a desired finding. '
            'Unequal-variance PALM is valuable given age-related variance concerns; it does not turn '
            'a nonsignificant fairness result into a confirmed absence.'
        ],table=records,widths=[230,135,135,135,135])

    def brain_behavior(self):
        focal=[]
        for timing in ['full-rt','pooled-post','partner-post']:
            model=timing+'__activation-norm'
            r=next(r for r in self.tables['randomise']if r['model']==model and r['contrast']=='1' and r['method']=='cluster-t3.1')
            ols=next(r for r in self.tables['robust']if r['timing']==timing and r['method']=='OLS')
            huber=next(r for r in self.tables['robust']if r['timing']==timing and r['method']=='Huber')
            focal.append([timing,pformat(r['minimum_spatial_fwe_p']),r['significant_voxels'],f"{float(ols['older_minus_younger']):.3f}",f"{float(huber['older_minus_younger']):.3f}"])
        self.section('Brain-behavior results | Important distinction between two predictors',[
            '<b>Choice sensitivity to fairness:</b> the participant-specific offer slope of acceptance, '
            'and its similar-minus-dissimilar difference. This is not the same as the acceptance '
            'intercept, nor a Gu norm-sensitivity parameter. The crossed FLAME analysis includes '
            'both predictor types separately across activation/DMN/ECN and three timing models '
            '(18 models, 144 signed contrasts; complete table in the appendix).',
            '<b>Parietal interaction:</b> the most consistent older retained activation result relates '
            'the similar-minus-dissimilar neural offer slope to an acceptance-intercept difference. '
            'The older-minus-younger behavioral-association contrast survives earlier randomise '
            't &gt; 3.1 across all three timings. It is an age difference in a brain-behavior association, '
            'not a direct age difference in activation and not demonstrated norm learning.',
            '<b>Influence:</b> within the selected region, all 47 single deletions and all 1,081 '
            'pair deletions per timing preserve the interaction sign. Huber fits reduce its magnitude '
            'without reversing it. These diagnostics do not remove selection circularity or establish '
            'independent significance.',
            '<b>Unfinished coverage check:</b> an unsupported voxel for sub-117 was detected in the '
            '125-voxel extraction region; the coverage-corrected activation batch is not available '
            'in the current pushed result set. The last user-reported live count was 104/112, '
            'with two calibration mismatches. Do not present this as a finalized, fully audited effect.'
        ],table=[['Timing','randomise p','Significant voxels','OLS interaction','Huber interaction']]+focal,widths=[150,140,160,160,160])

    def computational(self):
        stable=[r for r in self.tables['stable']if r['batch']=='stable']
        rec=[r for r in self.tables['recovery']if r['selection']=='all_completed_descriptive' and r['target']=='partner_difference']
        self.section('Computational behavior | What the norm-model work established',[
            '<b>Model family:</b> Gu-style offer/norm utility with a learned expectation, choice consistency '
            'and an initial norm; fixed/free-prior and hierarchical variants were examined. The latest '
            'extension constrains norm sensitivity and choice consistency positive and adds a shared '
            'participant acceptance bias. Bias can be negative or positive, allowing reject-prone and '
            'accept-prone behavior. It is shared across partners and the extension is age-blind.',
            '<b>Current diagnostic status:</b> longer stable empirical fits improved R-hat/ESS but '
            'retain divergences. Eight conditional recovery fits completed; six require diagnostic '
            'review. The exported workflow explicitly marks parameter recovery as unvalidated and '
            'individual imaging covariates as unreleased. This is a scientific gate, not an unfinished '
            'upload or missing commit.',
            '<b>What the conditional screen suggests:</b> recovery of partner differences is weaker '
            'for norm sensitivity and initial norm than for choice consistency. The table below '
            'summarizes rank recovery across eight completed simulations, including flagged fits; '
            'these descriptive values do not validate the model or supply new empirical age effects.',
            '<b>Implication for this paper:</b> retain the conventional choice and ratings analyses '
            'as the behavioral evidence. Do not use alpha, learning rate, initial norm, consistency '
            'or acceptance bias as established brain correlates while the gates remain uncleared. '
            'No age-related norm mechanism is established by these latest age-blind fits.'
        ],table=[['Stable fit','Retained draws','Divergences','Maximum R-hat','Min. bulk ESS']]+[[r['stage'],r['retained_draws'],r['divergences'],r['max_rhat'],f"{float(r['min_bulk_ess']):.0f}"]for r in stable]+[['Conditional recovery target','Parameter','Replicates','Median rank r','']]+[[r['partner_or_contrast'].replace('_',' '),r['parameter'],r['completed_replicates'],f"{float(r['median_spearman_r']):.3f}",'']for r in rec], widths=[265,135,120,125,125])

    def appendices(self):
        models=list(dict.fromkeys(r['model']for r in self.summary))
        labels={'full-rt':'Full display','pooled-post':'Pooled post','partner-post':'Partner post'}
        for method,title in [('cluster-z3.1','Z > 3.1'),('cluster-z2.6','Z > 2.6')]:
            data=[['Model','Mean positive','Mean negative','Younger > older','Older > younger']]
            for model in models:
                label=model.replace('__',' | ').replace('nppi-','')
                data.append([label]+[self.cell(model,method,c)for c in range(1,5)])
            self.section('Complete standardized PALM results | '+title,[
                'All 12 models and four signed directions; minimum spatial FWE p / significant voxels. '
                'All-ones means average the tested participant contrast across ages. A network row concerns '
                'offer-dependent coupling, not regional activation. No correction across this model/contrast '
                'family is implied. Non-significant cells are retained, not filtered out.'
            ],table=data,widths=[250,130,130,130,130])
        crossed=self.tables['crossed']; models=sorted({r['model']for r in crossed})
        for offset in (0,9):
            data=[['Crossed FLAME model','O-Y assoc.','Y-O assoc.','Y-O at mean','O-Y at mean','Y slope +','Y slope -','O slope +','O slope -']]
            for model in models[offset:offset+9]:
                values=[]
                for c in range(1,9):
                    r=next(r for r in crossed if r['model']==model and int(r['contrast'])==c)
                    values.append((pformat(r['minimum_cluster_p'])+' / '+r['voxels'])if int(r['voxels'])else 'None')
                data.append([model.replace('_',' ').replace('nppi-','')]+values)
            self.section('Earlier crossed behavioral FLAME results | '+('1/2'if offset==0 else '2/2'),[
                'Exploratory Z > 3.1 cluster results: minimum cluster p / total voxels. These models include '
                'age-specific behavioral covariates and differ from the standardized six-column direct-age '
                'PALM batch. Assoc. = age difference in the brain-behavior slope; at mean = age contrast '
                'at centered behavior. Within-age slopes are not age interactions. None means no corrected '
                'clusters, not missing data. Coverage/permutation validation is not complete for this family.'
            ],table=data,widths=[210]+[70]*8)
        self.section('Provenance and reproducibility',[
            '<b>Data cutoff:</b> report built from repository source commit '+self.commit[:12]+'. '
            'The standardized PALM batch is complete. The last retained broad PALM collection '
            '(30 October 2026, 13:13 UTC) contains 26/45 complete jobs and 19 unfinished; that snapshot '
            'cannot establish live process status. Its interim table is supplied separately and is not '
            'merged into the standardized result table.',
            '<b>Evidence files:</b> accompanying tables contain all 96 current PALM cells, all 144 '
            'earlier crossed FLAME cells, retained behavioral/ratings/norm diagnostics, and method '
            'comparisons. source_inventory.json records SHA-256 hashes of consumed inputs; '
            'output_inventory.json records final report artifacts. The figures were generated or '
            'copied only after their retained sources were checked.',
            '<b>Scope:</b> no FSL, PALM, Stan or behavioral-model fit was launched in producing this '
            'report; no data were excluded or altered. No new categorical unfair/fair or targeted '
            'ROI analysis was performed. Old outputs and earlier figure packets are preserved.',
            '<b>Rebuild:</b> run code/build_scientific_results_report.py with --standard pointing '
            'to MNI152_T1_2mm_brain.nii.gz and --output pointing to a new report directory. '
            'The report uses the retained results present in this checkout; the existing verified '
            'anatomical figure is reused rather than silently substituting a different atlas.'
        ])

    def export_report(self):
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        from reportlab.pdfgen import canvas
        from PIL import Image as PILImage
        folder=self.output/'output/pdf';folder.mkdir(parents=True,exist_ok=True)
        pdf=folder/'SRNDNA_Ultimatum_results_report_2026-11-02.pdf'
        width,height=landscape(A4)
        styles=getSampleStyleSheet()
        styles.add(ParagraphStyle(name='BodyScientific',fontName='Helvetica',fontSize=10.8,leading=14.6,spaceAfter=10))
        styles.add(ParagraphStyle(name='TableScientific',fontName='Helvetica',fontSize=9,leading=11.5))
        styles.add(ParagraphStyle(name='CaptionScientific',fontSize=9.5,leading=12.5,spaceAfter=7))
        doc=SimpleDocTemplate(str(pdf),pagesize=(width,height),leftMargin=35,rightMargin=35,topMargin=32,bottomMargin=34,
            title='SRNDNA Ultimatum: corrected results and draft figures',author='DVS-Lab')
        story=[];html=[]
        for i,section in enumerate(self.sections):
            story.append(Paragraph(escape(section['title']),styles['Heading1']))
            html.append('<section><h2>'+escape(section['title'])+'</h2>')
            for paragraph in section['paragraphs']:
                story.append(Paragraph(paragraph,styles['BodyScientific']))
                html.append('<p>'+paragraph.replace('<link href=','<a href=').replace('</link>','</a>')+'</p>')
            if section['image']:
                path=self.output/section['image']
                with PILImage.open(path)as image:iw,ih=image.size
                scale=min((width-70)/iw,370/ih)
                story.append(Image(str(path),width=iw*scale,height=ih*scale));story.append(Spacer(1,7))
                story.append(Paragraph(escape(section['caption']),styles['CaptionScientific']))
                html+=['<img src="'+section['image']+'"><p>'+escape(section['caption'])+'</p>']
            if section['table']:
                data=section['table']; style=styles['TableScientific']
                table=Table([[Paragraph(escape(str(c)),style)for c in r]for r in data],
                            colWidths=section['widths'],repeatRows=1)
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#deebf1')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f8')]),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                    ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),
                    ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
                story.append(table)
                html.append('<table>'+''.join('<tr>'+''.join(('<th>'if j==0 else '<td>')+escape(str(c))+('</th>'if j==0 else '</td>')for c in r)+'</tr>'for j,r in enumerate(data))+'</table>')
            html.append('</section>')
            if i<len(self.sections)-1:story.append(PageBreak())
        def footer(c,d):
            c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#536675'))
            c.drawString(35,16,'SRNDNA Ultimatum | Scientific results review | 2 November 2026')
            c.drawRightString(width-35,16,str(d.page))
        def stable_canvas(*a,**kw):kw['invariant']=1;return canvas.Canvas(*a,**kw)
        doc.build(story,onFirstPage=footer,onLaterPages=footer,canvasmaker=stable_canvas)
        css='body{font:17px system-ui;max-width:1150px;margin:35px auto;padding:0 24px;color:#23333d}p{line-height:1.5}section{margin:45px 0 65px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #ddd}th{background:#deebf1}a{color:#28617d}'
        (self.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SRNDNA scientific results</title><style>'+css+'</style><h1>SRNDNA Ultimatum: updated scientific results</h1>'+''.join(html))
        (self.output/'report_sections.json').write_text(json.dumps(self.sections,indent=2)+'\n')
        (self.output/'source_inventory.json').write_text(json.dumps(self.sources,indent=2,sort_keys=True)+'\n')
        (self.output/'figure_manifest.json').write_text(json.dumps(self.pages,indent=2)+'\n')
        (self.output/'map_provenance.json').write_text(json.dumps(self.map_records,indent=2)+'\n')
        # Verify that every input remained stable throughout report generation.
        for path,sha in self.sources.items():
            if digest(path)!=sha:raise ValueError(f'input changed while rendering: {path}')
        outputs={str(p.relative_to(self.output)):digest(p)for p in sorted(self.output.rglob('*'))if p.is_file()and p.name!='output_inventory.json'}
        (self.output/'output_inventory.json').write_text(json.dumps(outputs,indent=2)+'\n')
        print(f'PASS: {len(self.sections)} report sections; {len(self.summary)} current PALM cells; {len(self.pages)} figures; {len(self.sources)} stable sources')
        print(pdf)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standard',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/scientific_report/2026-11-02')
    args=parser.parse_args()
    report=Report(args.output.resolve(),args.standard.resolve())
    report.figures();report.comparisons();report.narratives();report.export_report()


if __name__=='__main__':main()
