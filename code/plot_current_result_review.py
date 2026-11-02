#!/usr/bin/env python3
"""Draft current-result figures from verified retained maps and source tables.

No fitting, new ROI inference, smoothing, exclusions or changes to source data.
PDF export is optional and requires reportlab. Historical figures stay intact.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from xml.sax.saxutils import escape

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage

from plot_permutation_followup import slice_plane
from build_event_corrected_trials import event_trials

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-11-02'
AUDIT = Path('results/age_figure_audit/v1/attempt-20261030T161854Z-2NzpRx/outputs')
BLUE, ORANGE, TEAL, GRAY = '#397e9e', '#ba6c43', '#297f87', '#888d92'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(p):
    with Path(p).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


class Review:
    def __init__(self, repo, output, standard):
        self.repo, self.output = repo, output
        output.mkdir(parents=True, exist_ok=True)
        self.sources = {}; self.pages = []; self.map_records = []
        self.source(repo/'code/plot_current_result_review.py')
        self.source(repo/'code/plot_permutation_followup.py')
        self.source(repo/'code/build_event_corrected_trials.py')
        for path in [repo/'results/norm_learning/stan-bias-validation-v1/workflow_status.json',
                     repo/'results/norm_learning/stan-bias-stable-v1/baseline_stable_comparison.tsv',
                     repo/'results/reviewer/ultimatum_ratings/ultimatum_ratings_fairness_sensitivity.tsv',
                     repo/'results/reviewer/ultimatum_ratings/rating_choice_moderation_focal_tests.tsv',
                     repo/'results/reviewer/ultimatum_ratings/ultimatum_ratings_age_tests.tsv']:
            self.source(path)
        self.palm = repo/'results/palm_age_main/v1'
        status = json.loads(self.source(self.palm/'status.json').read_text())
        if not status['complete'] or status['completed_jobs'] != 24 or status['summary_rows'] != 96:
            raise ValueError('all 24 PALM jobs and 96 summaries are required')
        for r in rows(self.palm/'inventory.tsv'):
            p = self.palm/r['path']
            if not p.is_file() or p.stat().st_size != int(r['bytes']) or digest(p) != r['sha256']:
                raise ValueError(f'PALM inventory differs: {p}')
        self.source(self.palm/'inventory.tsv')
        self.summary = rows(self.source(self.palm/'summary.tsv'))
        self.bg = nib.as_closest_canonical(nib.load(self.source(standard)))
        self.anatomy = self.bg.get_fdata()
        self.vmax = np.percentile(self.anatomy[self.anatomy > 0], 99.5)
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'savefig.facecolor': 'white'})

    def source(self, p):
        p = Path(p); self.sources[str(p)] = digest(p); return p

    def save(self, fig, name, title, caption):
        path = self.output/(name+'.png'); fig.savefig(path, dpi=170); plt.close(fig)
        self.pages.append(dict(name=name, title=title, caption=caption, image=path.name))
        return path

    def payload(self, model, method, contrast, title, sign=1):
        item = next(r for r in self.summary if r['model'] == model and r['method'] == method and int(r['contrast']) == contrast)
        folder = self.palm/'maps'/model/method
        stat = nib.load(self.source(folder/f'palm_vox_zvstat_c{contrast}.nii'))
        prob = nib.load(self.source(folder/f'palm_clustere_zvstat_fwep_c{contrast}.nii'))
        mask = nib.load(self.source(self.palm/'preparation'/model/'mask.nii.gz'))
        if stat.shape != prob.shape or stat.shape != mask.shape or not np.allclose(stat.affine, prob.affine) or not np.allclose(stat.affine, mask.affine):
            raise ValueError('native statistical grids differ')
        z, p = stat.get_fdata(), prob.get_fdata(); keep = mask.get_fdata() > 0
        sig = keep & (p > .95)
        if int(sig.sum()) != int(item['significant_voxels']) or abs(1-p[keep].max()-float(item['minimum_spatial_fwe_p'])) > 1e-9:
            raise ValueError('map/summary discrepancy')
        if not sig.any() or np.any(z[sig] <= 0):
            raise ValueError('requested map has no positive significant signed-test statistics')
        labels, _ = ndimage.label(sig, structure=np.ones((3, 3, 3)))
        counts = np.bincount(labels.ravel()); counts[0] = 0
        locations = np.argwhere(labels == counts.argmax())
        peak = locations[np.argmax(z[tuple(locations.T)])]
        cuts = nib.affines.apply_affine(stat.affine, peak)
        overlay = resample_from_to(nib.Nifti1Image(np.where(sig, sign*z, 0).astype('float32'), stat.affine), self.bg, order=0).get_fdata()
        record = dict(model=model, method=method, contrast=contrast, title=title,
            native_voxels=int(sig.sum()), minimum_spatial_fwe_p=float(item['minimum_spatial_fwe_p']),
            cut_rule='maximum signed statistic in largest 26-connected significant component',
            native_cut_x=float(cuts[0]), native_cut_y=float(cuts[1]), native_cut_z=float(cuts[2]),
            statistic_map=str(folder/f'palm_vox_zvstat_c{contrast}.nii'), sign_for_display=sign)
        self.map_records.append(record)
        return record, overlay, cuts

    def planes(self, axes, payload):
        record, overlay, cuts = payload
        for i, ax in enumerate(axes):
            base, extent, actual = slice_plane(self.anatomy, self.bg.affine, i, cuts[i])
            value, _, _ = slice_plane(overlay, self.bg.affine, i, cuts[i])
            ax.imshow(base, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=self.vmax)
            ax.imshow(np.ma.masked_equal(value, 0), cmap='coolwarm', origin='lower', extent=extent,
                      vmin=-6, vmax=6, interpolation='nearest')
            ax.set_title(f'{["Sagittal", "Coronal", "Axial"][i]} | {"xyz"[i]} = {actual:.0f} mm', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for pos, letter in [(.025, 'P' if i == 0 else 'L'), (.94, 'A' if i == 0 else 'R')]:
                ax.text(pos, .025, letter, color='white', transform=ax.transAxes, fontsize=10)
            ax.spines[['left', 'bottom']].set_visible(False)

    def overlay_page(self, specs, name, title, subtitle, caption):
        single = len(specs) == 1
        fig = plt.figure(figsize=(13, 6.5 if single else 2.8*len(specs)+1.75))
        grid = fig.add_gridspec(len(specs), 4, width_ratios=[1.25, 1, 1, 1],
            left=.025, right=.965, top=.73 if single else .85, bottom=.25 if single else .13,
            hspace=.30, wspace=.05)
        for row, spec in enumerate(specs):
            item = self.payload(*spec); rec = item[0]
            ax = fig.add_subplot(grid[row, 0]); ax.axis('off')
            ax.text(.02, .82, rec['title'], fontsize=12, weight='bold', va='top', wrap=True)
            ax.text(.02, .36, f"pFWE = {rec['minimum_spatial_fwe_p']:.4f}\n{rec['native_voxels']} native voxels\n{rec['method'].replace('cluster-z', 'Z = ')}", fontsize=10, va='top')
            self.planes([fig.add_subplot(grid[row, col]) for col in range(1, 4)], item)
        fig.suptitle(title, fontsize=18, y=.975)
        fig.text(.5, .88 if single else .91, subtitle, ha='center', fontsize=10.5)
        cbax = fig.add_axes([.34, .14 if single else .072, .38, .018]); cb = fig.colorbar(ScalarMappable(norm=Normalize(-6, 6), cmap='coolwarm'), cax=cbax, orientation='horizontal')
        cb.set_label('Signed equivalent Z relative to the original contrast', fontsize=9)
        fig.text(.025, .025 if single else .012, 'Native-map inference; nearest-neighbor display resampling only. Neurological L/R. All 47 participants retained.', fontsize=8)
        self.save(fig, name, title, caption)

    def behavior(self):
        observed = rows(self.source(self.repo/'results/reviewer/tables/acceptance_observed_source_data.tsv'))
        self.source(self.repo/'results/reviewer/tables/primary_acceptance_models.tsv')
        sample_path = self.source(self.repo/'behavioral_analyses/data/participant_L3_47.csv')
        with sample_path.open(encoding='utf-8-sig') as stream:
            people = {r['subjID']: r for r in csv.DictReader(stream)}
        buckets = defaultdict(list); totals = defaultdict(list)
        for subject, person in people.items():
            age = 'older' if int(float(person['older'])) else 'younger'
            for run in (1, 2):
                path = self.source(self.repo/'source_data/bids'/subject/'func'/f'{subject}_task-ultimatum_run-{run:02}_events.tsv')
                for r in event_trials(path, 1.0):
                    if r['human'] != '1' or r['missed'] != '0': continue
                    partner = 'similar' if r['ingroup'] == '1' else 'dissimilar'
                    key = (age, partner, int(float(r['offer'])))
                    value = int(r['accept']); buckets[key+(subject,)].append(value); totals[key].append(value)
        if len(people) != 47 or sum(len(v) for v in totals.values()) != 4438:
            raise ValueError('descriptive choice sample differs from corrected model sample')
        # Require exact agreement with the retained model's observed source
        # table before summarizing participants rather than pooling trials.
        for row in observed:
            key = (row['age_group'], row['similarity'], int(float(row['offer'])))
            values = totals[key]
            if len(values) != int(row['n']) or sum(values) != int(row['accepted']):
                raise ValueError(f'corrected choice counts differ: {key}')
        means = defaultdict(list)
        individual = []
        for (age, partner, offer, subject), values in sorted(buckets.items()):
            mean = float(np.mean(values)); means[age, partner, offer].append(mean)
            individual.append(dict(subject=subject, age_group=age, partner=partner, offer=offer,
                trials=len(values), accepted=sum(values), acceptance_proportion=mean))
        with (self.output/'choice_participant_source.tsv').open('w') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(individual[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader();writer.writerows(individual)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        for ax, group, n in zip(axes, ['younger', 'older'], [25, 22]):
            for partner, color, offset in [('similar', TEAL, -.06), ('dissimilar', ORANGE, .06)]:
                keys = sorted(k for k in means if k[:2] == (group, partner))
                m = [np.mean(means[k]) for k in keys]
                sem = [np.std(means[k], ddof=1)/np.sqrt(len(means[k])) for k in keys]
                ax.errorbar([k[2]+offset for k in keys], m, yerr=sem, fmt='o-', color=color,
                    lw=1.8, capsize=3, markersize=4, label=partner.title())
            ax.set_title(f'{group.title()} adults | n = {n}', fontsize=13)
            ax.set_xlabel('Offer to participant ($)'); ax.set_ylim(-.03, 1.03); ax.set_xlim(.6, 10.4)
            ax.set_xticks([1, 2, 4, 6, 8, 10]); ax.set_yticks([0, .25, .5, .75, 1]); ax.grid(alpha=.15)
        axes[0].set_ylabel('Probability of accepting'); axes[1].legend(frameon=False)
        fig.suptitle('Choice follows offer amount; age moderation was not detected', fontsize=16, y=.98)
        fig.subplots_adjust(left=.075, right=.98, bottom=.22, top=.83, wspace=.12)
        fig.text(.075, .075, '47 participants; 4,438 responded human-partner trials. Offer coefficient = 1.801 (SE 0.237), p < .001.\nAge x offer p = .616; age x offer x partner p = .230. Points/lines: observed participant means +/- SEM.\nEach offer/partner cell includes participants with responded trials in that cell. Model tests are retained; no new fits.', fontsize=9.5)
        self.save(fig, '01_choice_behavior', 'Draft Figure 1 | Corrected social-choice behavior',
            'Acceptance increased with offer size. The corrected random-intercept/random-offer-slope model converged without singularity. '
            'No reliable age or age-by-partner moderation was detected; this does not establish equivalence. '
            'Points show observed participant means and SEM, rather than a conditional logistic curve at zero random effects. '
            'Similar/dissimilar refer to proposer age relative to the participant. Offer amount also changes the participant\'s monetary payoff.')

    def ratings(self):
        base = self.repo/'results/vmpfc_ratings/v1/ratings'
        table = rows(self.source(base/'bar_source.tsv')); tests = rows(self.source(base/'retained_paired_tests.tsv'))
        self.source(base/'contract.json')
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.1), sharey=True)
        for i, time in enumerate(['pre', 'post']):
            for j, dimension in enumerate(['fairness', 'likeability']):
                ax = axes[i, j]
                rs = [next(r for r in table if r['age_group'] == 'all' and r['timepoint'] == time and r['rating_dimension'] == dimension and r['partner'] == partner)
                      for partner in ['computer', 'similar', 'dissimilar']]
                ax.bar(range(3), [float(r['mean']) for r in rs], yerr=[float(r['sem']) for r in rs],
                       color=[GRAY, TEAL, ORANGE], width=.66, capsize=4)
                ax.set_xticks(range(3), ['Computer', 'Similar', 'Dissimilar']); ax.set_ylim(0, 10)
                ax.set_title(f'{time.title()}-task {dimension} | n = 42', fontsize=12)
                if j == 0: ax.set_ylabel('Rating (0-10)')
                test = next(r for r in tests if r['dimension'] == dimension and r['timepoint'] == time and r['comparison'] == 'human_mean_minus_computer')
                ax.text(.5, .95, f"Human - computer: BH q = {float(test['p_fdr_bh_within_family']):.4g}", ha='center', va='top', transform=ax.transAxes, fontsize=9)
        fig.suptitle('Partner ratings distinguish humans from the computer', fontsize=17, y=.99)
        fig.subplots_adjust(left=.07, right=.98, bottom=.16, top=.90, hspace=.30, wspace=.15)
        fig.text(.07, .035, 'Means +/- participant SEM. Last complete attempt retained. Pre/post samples overlap (41 paired participants).\nSimilar versus dissimilar fairness/likeability contrasts do not survive BH adjustment. These plots do not test changes caused by the task.', fontsize=9.5)
        self.save(fig, '02_partner_ratings', 'Draft Figure 2 | Partner ratings',
            'Humans were rated fairer and more likeable than the computer before and after the task. '
            'The corresponding similar-minus-dissimilar comparisons are nonsignificant after BH adjustment. '
            'These are ratings in a subsample, not evidence that partner ratings mediate an age effect.')

    def dmn_bars(self):
        root = self.repo/AUDIT/'dmn'; table = rows(self.source(root/'bar_summary.tsv'))
        self.source(root/'age_contrast_diagnostics.tsv')
        old = self.repo/'results/dmn_followup/v1/coverage_check/v1/maps/palm/cluster-z2.6/palm_clustere_zvstat_fwep_c3.nii'
        new = self.palm/'maps/full-rt__nppi-dmn/cluster-z2.6/palm_clustere_zvstat_fwep_c3.nii'
        a, b = nib.load(self.source(old)), nib.load(self.source(new))
        if a.shape != b.shape or not np.allclose(a.affine, b.affine) or not np.array_equal(a.get_fdata() > .95, b.get_fdata() > .95):
            raise ValueError('old DMN extraction region no longer matches the new 49-voxel map')
        fig, axes = plt.subplots(1, 3, figsize=(14, 6.8), sharey=True)
        for ax, policy, title in zip(axes, ['age-only', 'no-group-FD', 'full'],
            ['Age groups only', 'Sex + tSNR + RT\nNo group motion covariate', 'Sex + tSNR + RT + motion\nFull model']):
            for age, color, offset in [('younger', BLUE, -.18), ('older', ORANGE, .18)]:
                data = [next(r for r in table if r['representation'] == 'matched' and r['policy'] == policy and r['age_group'] == age and r['metric'] == partner)
                        for partner in ['computer', 'similar', 'dissimilar']]
                ax.bar(np.arange(3)+offset, [float(r['adjusted_mean']) for r in data],
                       yerr=[float(r['conditional_participant_sem']) for r in data], color=color, width=.34, capsize=3, label=age.title())
            ax.axhline(0, color='.5', lw=.8); ax.set_xticks(range(3), ['Computer', 'Similar', 'Dissimilar']); ax.set_title(title, fontsize=12)
        axes[0].set_ylabel('Offer-modulated DMN coupling (model units)'); axes[2].legend(frameon=False)
        fig.suptitle('DMN age sensitivity: partner decomposition and motion adjustment', fontsize=17, y=.98)
        fig.subplots_adjust(left=.07, right=.98, bottom=.25, top=.80, wspace=.14)
        fig.text(.07, .07, 'Same 49-voxel region as the new PALM Z = 2.6 age effect. Means +/- conditional participant SEM.\nDescriptive selected-region summaries using common COPE7 run weights; not independent ROI tests or PALM group weights.\nYounger-minus-older contrast: 20.1 age-only; 25.2 without group FD; 36.9 with all covariates.\nSimilar/dissimilar reverses proposer age between groups. Removing group FD does not remove L1 motion adjustment.', fontsize=9.5)
        self.save(fig, '06_dmn_partner_bars', 'Sensitivity Figure | DMN three-partner bars',
            'The age-gap direction is present before group motion adjustment, but its magnitude increases substantially after adjustment. '
            'The apparent crossover corresponds to positive offer-related coupling for younger proposers and negative coupling for older proposers in both participant groups. '
            'The computer condition supplies context and is not part of the similar-minus-dissimilar contrast. No new ROI significance test is performed.')

    def complete_grid(self):
        models = list(dict.fromkeys(r['model'] for r in self.summary))
        labels = [x.replace('full-rt__', 'Full RT | ').replace('pooled-post__', 'Pooled post | ').replace('partner-post__', 'Partner post | ') for x in models]
        fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), sharey=True)
        for ax, method, title in zip(axes, ['cluster-z3.1', 'cluster-z2.6'], ['Primary: cluster Z = 3.1', 'Sensitivity: cluster Z = 2.6']):
            p = np.array([[float(next(r for r in self.summary if r['model'] == m and r['method'] == method and int(r['contrast']) == c)['minimum_spatial_fwe_p']) for c in range(1, 5)] for m in models])
            ax.imshow((p < .05).astype(int), cmap=matplotlib.colors.ListedColormap(['#f2f3f5', '#d0e5ef']), vmin=0, vmax=1, aspect='auto')
            for i in range(len(models)):
                for j in range(4):
                    ax.text(j, i, f'{p[i,j]:.4f}' if p[i,j] < .05 else 'NS', ha='center', va='center', fontsize=10, weight='bold' if p[i,j] < .05 else 'normal')
            ax.set_xticks(range(4), ['Mean +', 'Mean -', 'Younger > older', 'Older > younger'], rotation=25, ha='right')
            ax.set_yticks(range(len(labels)), labels); ax.set_title(title, fontsize=13)
            ax.tick_params(axis='both', length=0)
        fig.suptitle('Complete PALM age/main-effect inventory: all 96 result cells', fontsize=17, y=.98)
        fig.subplots_adjust(left=.22, right=.98, top=.88, bottom=.17, wspace=.18)
        fig.text(.22, .035, 'Cell values are minimum spatial-FWE p across the map. NS means no corrected p < .05.\nNo correction across the model/direction/threshold search. No TFCE. Main effects are not age differences.\nAll cells use the same six group-level columns and all 47 participants; network/task contrasts differ.', fontsize=9.5)
        self.save(fig, '08_complete_result_grid', 'Audit Figure | Complete PALM inventory',
            'All planned results are shown, including null findings. The cluster-Z = 3.1 human-computer task-response age contrast is the only primary PALM age finding. '
            'The DMN age effect and task-wide offer age effect occur only at the lower cluster threshold. '
            'Full-sample network effects depend on timing specification.')

    def finish(self):
        (self.output/'map_provenance.json').write_text(json.dumps(self.map_records, indent=2)+'\n')
        (self.output/'source_inventory.json').write_text(json.dumps(self.sources, indent=2, sort_keys=True)+'\n')
        (self.output/'figure_manifest.json').write_text(json.dumps(self.pages, indent=2)+'\n')
        css = 'body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#26313b}img{width:100%}section{margin:55px 0}p{line-height:1.55}.note{background:#f2f5f7;padding:18px}a{color:#28617d}'
        body = '<h1>SRNDNA Ultimatum: draft figure review</h1><p>2 November 2026 | N = 47 | PALM batch complete: 24 jobs, 96 result cells.</p>'
        body += '<p class="note">Current strongest imaging candidates: full-sample offer-size and human/computer response effects, plus the human/computer task-response age difference. DMN age and small network main effects remain sensitivity findings. No TFCE or vmPFC ROI figures are promoted here.</p>'
        for page in self.pages:
            body += '<section><h2>'+escape(page['title'])+'</h2><img src="'+page['image']+'" alt="'+escape(page['title'])+'"><p>'+escape(page['caption'])+'</p></section>'
        (self.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SRNDNA figure review</title><style>'+css+'</style>'+body)


def pdf_packet(review):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.pdfgen import canvas
    from PIL import Image as PillowImage
    out = review.output/'output/pdf'; out.mkdir(parents=True, exist_ok=True)
    path = out/'srndna_current_results_2026-11-02.pdf'
    width, height = landscape(A4)
    doc = SimpleDocTemplate(str(path), pagesize=(width, height), rightMargin=35, leftMargin=35, topMargin=30, bottomMargin=30,
        title='SRNDNA Ultimatum: current results and draft figures', author='DVS-Lab')
    styles = getSampleStyleSheet(); styles.add(ParagraphStyle(name='BodyReview', fontSize=10.5, leading=14, spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallReview', fontSize=9, leading=12, spaceAfter=7))
    p = lambda s: Paragraph(s, styles['BodyReview'])
    story = [Paragraph('SRNDNA Ultimatum: current results', styles['Title']), p('Draft scientific review | 2 November 2026 | Repository: DVS-Lab/srndna-ultimatum'),
        p('<b>Complete:</b> 24 PALM jobs, 96 signed result summaries, 47 participants (25 younger, 22 older), 10,000 sign-flips per test. No failures, exclusions or L1/L2 refits. All tracked PALM inventory hashes were verified before rendering.'),
        p('<b>Main message:</b> choice depends strongly on offer amount; the clearest current imaging results concern full-sample task responses and an age difference in the human-computer response contrast. The original DMN age narrative is not stable at the paper-matched cluster threshold.')]
    cells = [['Result', 'Current evidence', 'Figure status'],
        ['Social-choice offer effect', 'beta 1.801; SE .237; p < .001', 'Main behavioral figure'],
        ['Age x offer x similarity', 'p = .230; no detected moderation', 'Report alongside behavior'],
        ['Positive full-sample offer slope', 'PALM Z3.1 p = .0052; 45 voxels', 'Foundational task effect'],
        ['Human > computer task response', 'PALM Z3.1 p = .0002; 270 voxels', 'Foundational partner effect'],
        ['Computer > human task response', 'PALM Z3.1 p = .0012; 155 voxels', 'Separate regional effect'],
        ['Younger > older: human-computer', 'PALM Z3.1 p = .0163; 25 voxels', 'Current age candidate'],
        ['Younger > older: DMN full-RT', 'Z3.1 p = .2526; Z2.6 p = .0468', 'Threshold-sensitive supplement'],
        ['Younger > older: task-wide offer', 'Z3.1 p = .1017; Z2.6 p = .0352', 'Threshold-sensitive supplement'],
        ['Partner ratings', 'Human/computer differences; no reliable age effect', 'Behavioral context / supplement'],
        ['Gu-style individual parameters', 'Recovery gates not cleared', 'No imaging covariates released']]
    t = Table([[Paragraph(escape(str(c)), styles['SmallReview']) for c in row] for row in cells], colWidths=[228, 275, 260])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#75818a')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f8f9')]),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story += [t, Spacer(1, 10), Paragraph('Spatial FWE is per signed contrast, not across the full model search. PALM has separate age variance groups and does not reproduce FLAME VARCOPE weighting. Offer amount confounds fairness and own monetary payoff.', styles['SmallReview']), PageBreak()]
    story += [Paragraph('Interpretation and remaining work', styles['Title']),
        p('<b>Paper consistency:</b> retain categorical younger/older groups and the four reported controls: sex, tSNR, mean framewise displacement and corrected mean task RT. The new PALM batch uses verified motion/tSNR values. Full-sample means use an all-ones column with centered age and nuisance columns. FLAME is the submitted estimator; PALM is an unequal-variance robustness analysis, not an automatically adopted replacement.'),
        p('<b>Historical comparison:</b> at threshold 3.1, PALM significant regions for positive offer slope, human > computer, computer > human and the human-computer age contrast are completely contained in significant FLAME and randomise regions. Historical models use some different covariates/masks; this is spatial convergence, not a fully matched comparison of estimators.'),
        p('<b>Timing-dependent network means:</b> pooled-post DMN similar > dissimilar offer coupling: p = .0424, 10 voxels. ECN dissimilar > similar: pooled-post p = .0298, 10 voxels; partner-post p = .0121, 13 voxels (8 overlap). These are full-sample effects, not restored age findings. Precise anatomical labels require atlas review.'),
        p('<b>DMN sensitivity:</b> the lower-threshold full-RT age region contains the same 49 voxels as the earlier coverage-corrected extraction. The descriptive age gap increases from 25.2 without group motion to 36.9 with group motion; selected-region bars are not independent inference.'),
        p('<b>Other work preserved:</b> the parietal brain-behavior interaction survived earlier robust/deletion diagnostics, but is a different estimand from a direct age difference. Its coverage-corrected batch has not been pushed. The original broader PALM collection remains at its last pushed snapshot. vmPFC ROI results are deliberately omitted from this paper figure set.'),
        p('<b>Computational behavior:</b> empirical bias-model fits still have divergences; six of eight conditional recovery fits need diagnostic review. Norm sensitivity, learning rate, initial norm and choice consistency are not currently cleared as individual-difference imaging covariates.'),
        p('<b>Before submission:</b> settle the primary timing specification; finish matched-design comparisons if reporting estimator robustness; inspect anatomical/influence diagnostics for new primary maps; finalize figure extraction/captions; revise the manuscript and response. The manuscript has not yet been updated. OpenNeuro ds003745 version 2.2.0 is published.'), PageBreak()]
    story += [Paragraph('What was repaired and what remains exploratory', styles['Title']),
        p('<b>Event provenance:</b> the sub-144 Ultimatum/Shared Reward event-file problem was repaired from the retained source records. Sub-143 did not require synthetic event_RT duplicate rows to recover its behavioral RT values. The dataset and the paper analysis sample are distinct: 50 recruited, 47 retained for this paper.'),
        p('<b>Dedicated RT regressors:</b> the historical RT EVs omitted block-first responses. The corrected full-display model includes all responded trials in its dedicated constant and RT-height terms. Full-run activation and network models were regenerated. Fixed display, decision plus pooled post-response, and decision plus partner post-response specifications remain separate; they are not interchangeable estimates of decision time.'),
        p('<b>Coverage and group inputs:</b> the strict whole-brain mask requires supported finite estimates and positive VARCOPEs in all 94 L1 runs and all 47 L2 estimates. It contains 59,297 native voxels, versus 64,444 in the historical mask. All participants are retained. Motion/tSNR columns were reconciled against independently verified measurements; corrected mean RT was retained. Continuous signed network maps were preserved.'),
        p('<b>Ratings and choices:</b> 42 participants have pre-task ratings, 42 post-task ratings and 41 paired records, using the last complete attempt. Humans were rated fairer and more likeable than the computer. Similar-minus-dissimilar choice-slope associations with pre-task likeability (Spearman rho = -.405) and post-task anger (rho = .397) survive the retained BH adjustment (q = .0366 each), but corresponding Pearson associations do not. Trial-level rating moderation is nonsignificant for pre-task fairness (p = .868) and likeability (p = .834). These are exploratory associations, not an age mechanism.'),
        p('<b>Norm-model diagnostics:</b> the retained stable bias extension has 2 divergences in the full fit and 7 in the run-1 fit. Improved convergence does not establish parameter recovery; the conditional screen still flags six of eight recovery fits. Individual parameters are not currently used to support imaging claims.'),
        p('<b>Claims to avoid:</b> no proof of behavioral equivalence; no isolated neural fairness signal distinct from own payoff; no restored original ECN age effect; no robust DMN age claim based on selecting Z = 2.6; no causal or directional interpretation of PPI; and no relabeling older selected-region bars as estimates from the new 25-voxel age region.'), PageBreak()]
    for i, page in enumerate(review.pages):
        story.append(Paragraph(escape(page['title']), styles['Heading1']))
        path_image = review.output/page['image']
        with PillowImage.open(path_image) as im: iw, ih = im.size
        scale = min((width-70)/iw, 385/ih)
        story.append(Image(str(path_image), width=iw*scale, height=ih*scale))
        story.append(Spacer(1, 7)); story.append(Paragraph(escape(page['caption']), styles['SmallReview']))
        if i != len(review.pages)-1: story.append(PageBreak())
    def footer(c, d):
        c.setFont('Helvetica', 8); c.setFillColor(colors.HexColor('#596874'))
        c.drawString(35, 16, 'Draft figure review | 2 November 2026 | No new inferential tests')
        c.drawRightString(width-35, 16, str(d.page))
    def stable_canvas(*a, **kw):
        kw['invariant'] = 1; return canvas.Canvas(*a, **kw)
    doc.build(story, onFirstPage=footer, onLaterPages=footer, canvasmaker=stable_canvas)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standard', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'results/figure_review/2026-11-02')
    parser.add_argument('--pdf', action='store_true')
    a = parser.parse_args(); review = Review(ROOT, a.output.resolve(), a.standard.resolve())
    review.behavior(); review.ratings()
    review.overlay_page([
        ('taskwide-offer', 'cluster-z3.1', 1, 'Larger offers\nFull-sample activation', 1),
        ('human-computer-response', 'cluster-z3.1', 1, 'Human > computer\nFull-sample response', 1),
        ('human-computer-response', 'cluster-z3.1', 2, 'Computer > human\nFull-sample response', -1)],
        '03_foundational_activation', 'Foundational task effects across the full sample',
        'Primary PALM: cluster Z = 3.1; spatial FWE p < .05',
        'All three effects survive PALM and overlap the historical FLAME/randomise significant regions. '
        'Positive offer-related activity is largely posterior, not an established vmPFC/ventral-striatal fairness effect. '
        'Human/computer response effects concern the full task display and do not isolate offer sensitivity.')
    review.overlay_page([('human-computer-response', 'cluster-z3.1', 3,
        'Younger > older\nHuman - computer\ntask response', 1)],
        '04_human_computer_age', 'Current age candidate: human-computer task response',
        'Primary PALM: p = .0163; all 25 voxels also significant in historical FLAME and randomise',
        'A posterior midline age difference in the human-minus-computer task-response contrast. '
        'This is not a fairness-slope age effect. Historical methods use some different covariates/masks. '
        'Participant-level bars for this exact new 25-voxel region require a fresh Linux extraction; older-region bars are not relabeled here.')
    review.overlay_page([('full-rt__nppi-dmn', 'cluster-z2.6', 3,
        'Younger > older\nSimilar - dissimilar\noffer-related DMN PPI', 1)],
        '05_dmn_age_sensitivity', 'DMN age result: lower-threshold sensitivity finding',
        'Full display + RT height | Z = 2.6: p = .0468, 49 voxels | Z = 3.1: p = .2526, no significant voxels',
        'The 49-voxel map exactly matches the previous coverage-corrected PALM region. '
        'The result does not survive the paper-matched Z = 3.1 threshold under any timing specification and should not be described as a robust primary age effect.')
    review.dmn_bars()
    review.overlay_page([
        ('pooled-post__nppi-dmn', 'cluster-z3.1', 1, 'DMN similar > dissimilar\nPooled post-response', 1),
        ('pooled-post__nppi-ecn', 'cluster-z3.1', 2, 'ECN dissimilar > similar\nPooled post-response', -1),
        ('partner-post__nppi-ecn', 'cluster-z3.1', 2, 'ECN dissimilar > similar\nPartner post-response', -1)],
        '07_network_main_effects', 'Small full-sample network effects: exploratory candidates',
        'Primary PALM: cluster Z = 3.1; spatial FWE p < .05 | These are main effects, not age differences',
        'Offer-modulated partner differences in network coupling. The two ECN maps overlap in 8 voxels. '
        'These effects are absent at Z = 3.1 in the full-RT model. Precise tissue/atlas localization and influence checks remain before interpretation.')
    review.complete_grid(); review.finish()
    if a.pdf: print(pdf_packet(review))
    outputs = {str(p.relative_to(review.output)): digest(p) for p in sorted(review.output.rglob('*')) if p.is_file() and p.name != 'output_inventory.json'}
    (review.output/'output_inventory.json').write_text(json.dumps(outputs, indent=2)+'\n')
    print(f'PASS: {len(review.pages)} draft figures; {len(review.map_records)} map panels; sources verified; output={review.output}')


if __name__ == '__main__':
    main()
