#!/usr/bin/env python3
"""Offer-parametric review only; preserve the broader historical figure packet.

Across-partner mean offer slope, partner differences in slopes, and offer PPI.
No new fits, ROI tests, small-volume correction, or threshold searching.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'srndna-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import map_coordinates

from plot_current_result_review import Review, ROOT, digest, rows, slice_plane


def table(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(records)


class ParametricReview(Review):
    def __init__(self, repo, output, standard, atlas_root):
        super().__init__(repo, output, standard)
        self.source(repo/'code/plot_parametric_result_review.py')
        self.source(repo/'masks/README.md')
        self.summary = [r for r in self.summary if r['model'] != 'human-computer-response']
        if len(self.summary) != 88 or len({r['model'] for r in self.summary}) != 11:
            raise ValueError('expected 11 parametric models / 88 signed result cells')
        self.atlas_path = self.source(atlas_root/'HarvardOxford/HarvardOxford-cort-prob-2mm.nii.gz')
        self.atlas_xml = self.source(atlas_root/'HarvardOxford-Cortical.xml')
        self.atlas = nib.load(self.atlas_path); self.atlas_values = self.atlas.get_fdata()
        self.labels = {int(n.attrib['index']): n.text for n in ET.parse(self.atlas_xml).findall('.//label')}
        if self.atlas.shape[-1] != len(self.labels) or self.atlas_values.min() < 0 or self.atlas_values.max() > 100:
            raise ValueError('atlas volume/dictionary/range mismatch')
        self.historical = rows(self.source(repo/'results/permutation_audit/v1/model_contrast_summary.tsv'))
        self.flame_root = repo/'results/reviewer/activation_fairness_main'
        self.source(self.flame_root/'job_provenance.tsv')
        self.flame_clusters = rows(self.source(self.flame_root/'focal_contrast1/cluster_zstat1_std.txt'))
        negative = rows(self.source(self.flame_root/'contrast2_adjusted-mean-negative/cluster_zstat2_std.txt'))
        if negative:
            raise ValueError('negative FLAME offer result changed; review narrative')
        self.insula_cluster = next(r for r in self.flame_clusters if r['Cluster Index'] == '4')

    def atlas_probabilities(self, image, keep):
        xyz = nib.affines.apply_affine(image.affine, np.argwhere(keep))
        ijk = nib.affines.apply_affine(np.linalg.inv(self.atlas.affine), xyz).T
        values = np.array([map_coordinates(self.atlas_values[..., i], ijk, order=1, mode='constant')
                           for i in range(len(self.labels))])
        return values.mean(axis=1)

    def fairness_audit(self):
        records = []; region_rows = []
        for n, direction in [(1, 'larger-offer-positive'), (2, 'smaller-offer-negative')]:
            folder = self.flame_root/('focal_contrast1' if n == 1 else 'contrast2_adjusted-mean-negative')
            stats = nib.load(self.source(folder/f'zstat{n}.nii.gz'))
            labels = nib.load(self.source(folder/f'cluster_mask_zstat{n}.nii.gz')).get_fdata()
            records.append(dict(engine='FLAME', direction=direction, threshold='Z=3.1',
                minimum_spatial_fwe_p=min((float(r['P']) for r in self.flame_clusters), default=1.) if n == 1 else '',
                significant_voxels=int((labels > 0).sum()), covariates='historical', mask='historical whole brain'))
            if n == 1:
                for row in self.flame_clusters:
                    keep = labels == int(row['Cluster Index']); probs = self.atlas_probabilities(stats, keep)
                    peak_xyz = np.array([float(row[f'Z-MAX {axis} (mm)']) for axis in 'XYZ'])
                    peak_ijk = nib.affines.apply_affine(np.linalg.inv(self.atlas.affine), peak_xyz)[:, None]
                    for i in np.argsort(probs)[::-1][:3]:
                        region_rows.append(dict(engine='FLAME', cluster=row['Cluster Index'], voxels=int(keep.sum()),
                            cluster_p=float(row['P']), peak_x=row['Z-MAX X (mm)'], peak_y=row['Z-MAX Y (mm)'], peak_z=row['Z-MAX Z (mm)'],
                            atlas_label=self.labels[i], mean_probability_percent=float(probs[i]),
                            peak_probability_percent=float(map_coordinates(self.atlas_values[..., i], peak_ijk, order=1, mode='constant')[0])))
            for method in ['cluster-t3.1', 'cluster-t2.6']:
                r = next(r for r in self.historical if r['model'] == 'taskwide-offer' and r['method'] == method and int(r['contrast']) == n)
                records.append(dict(engine='randomise', direction=direction, threshold=method,
                    minimum_spatial_fwe_p=float(r['minimum_spatial_fwe_p']), significant_voxels=int(r['significant_voxels']),
                    covariates='historical', mask='historical whole brain'))
            for method in ['cluster-z3.1', 'cluster-z2.6']:
                r = next(r for r in self.summary if r['model'] == 'taskwide-offer' and r['method'] == method and int(r['contrast']) == n)
                records.append(dict(engine='PALM', direction=direction, threshold=method,
                    minimum_spatial_fwe_p=float(r['minimum_spatial_fwe_p']), significant_voxels=int(r['significant_voxels']),
                    covariates='verified standardized controls', mask='all 94 L1 / 47 L2 valid coverage'))
        # Existing aINS/dACC spheres are checked descriptively, not used to
        # change the whole-brain family or perform a new ROI inference.
        checks = []
        for roi_name in ['roi-aINS', 'roi-dACC']:
            roi = nib.load(self.source(self.repo/'masks'/f'{roi_name}.nii.gz'))
            for engine in ['FLAME', 'randomise', 'PALM']:
                for n, direction in [(1, 'larger-offer-positive'), (2, 'smaller-offer-negative')]:
                    if engine == 'FLAME':
                        folder = self.flame_root/('focal_contrast1' if n == 1 else 'contrast2_adjusted-mean-negative')
                        z = nib.load(self.source(folder/f'zstat{n}.nii.gz'))
                        sig = nib.load(self.source(folder/f'cluster_mask_zstat{n}.nii.gz')).get_fdata() > 0
                    elif engine == 'PALM':
                        folder = self.palm/'maps/taskwide-offer/cluster-z3.1'
                        z = nib.load(self.source(folder/f'palm_vox_zvstat_c{n}.nii'))
                        sig = nib.load(self.source(folder/f'palm_clustere_zvstat_fwep_c{n}.nii')).get_fdata() > .95
                    else:
                        row = next(r for r in self.historical if r['model'] == 'taskwide-offer' and r['method'] == 'cluster-t3.1' and int(r['contrast']) == n)
                        folder = self.repo/'results/permutation_audit/v1/maps'/row['test']/'cluster-t3.1'
                        z = nib.load(self.source(folder/'randomise_tstat1.nii.gz'))
                        sig = nib.load(self.source(folder/'randomise_clustere_corrp_tstat1.nii.gz')).get_fdata() > .95
                    keep = resample_from_to(roi, z, order=0).get_fdata() > 0
                    checks.append(dict(roi=roi_name, engine=engine, direction=direction,
                        roi_native_voxels=int(keep.sum()), whole_brain_corrected_overlap_voxels=int((keep & sig).sum()),
                        maximum_descriptive_signed_statistic=float(z.get_fdata()[keep].max()),
                        interpretation='descriptive inspection of an existing whole-brain map; not an ROI test'))
        table(self.output/'across_partner_activation_results.tsv', records)
        table(self.output/'flame_positive_cluster_atlas.tsv', region_rows)
        table(self.output/'predefined_region_map_check.tsv', checks)
        self.audit = records; self.roi_checks = checks
        insula_mask = nib.load(self.flame_root/'focal_contrast1/cluster_mask_zstat1.nii.gz').get_fdata() == 4
        overlap = []
        for method in ['cluster-t3.1', 'cluster-t2.6']:
            r = next(r for r in self.historical if r['model']=='taskwide-offer' and r['method']==method and r['contrast']=='1')
            image = nib.load(self.source(self.repo/'results/permutation_audit/v1/maps'/r['test']/method/'randomise_clustere_corrp_tstat1.nii.gz'))
            overlap.append(dict(engine='randomise', method=method, corrected_overlap_voxels=int(np.sum(insula_mask & (image.get_fdata()>.95)))))
        for method in ['cluster-z3.1', 'cluster-z2.6']:
            image=nib.load(self.source(self.palm/'maps/taskwide-offer'/method/'palm_clustere_zvstat_fwep_c1.nii'))
            overlap.append(dict(engine='PALM', method=method, corrected_overlap_voxels=int(np.sum(insula_mask & (image.get_fdata()>.95)))))
        table(self.output/'insula_cluster_permutation_overlap.tsv',overlap)
        if any(r['corrected_overlap_voxels'] for r in overlap) or any(r['whole_brain_corrected_overlap_voxels'] for r in checks):
            raise ValueError('insula/dACC overlap changed; review narrative before rendering')
        # Anatomical localization verifies that the FLAME-only cluster really
        # includes insula, rather than inferring anatomy from an expected effect.
        insula = [r for r in region_rows if r['cluster'] == '4' and r['atlas_label'] == 'Insular Cortex']
        if len(insula) != 1 or insula[0]['mean_probability_percent'] < 10:
            raise ValueError('right-insula localization no longer supported')

    def collapsed_overlays(self):
        payloads = []
        payloads.append(self.payload('taskwide-offer', 'cluster-z3.1', 1, 'PALM: larger offers\nAcross all partners', 1))
        r = next(r for r in self.historical if r['model'] == 'taskwide-offer' and r['method'] == 'cluster-t3.1' and r['contrast'] == '1')
        folder = self.repo/'results/permutation_audit/v1/maps'/r['test']/'cluster-t3.1'
        stats = nib.load(self.source(folder/'randomise_tstat1.nii.gz'))
        sig = nib.load(self.source(folder/'randomise_clustere_corrp_tstat1.nii.gz')).get_fdata() > .95
        payloads.append(self.custom_payload(stats, sig, 'randomise: larger offers\nAcross all partners', float(r['minimum_spatial_fwe_p']), 't = 3.1'))
        folder = self.flame_root/'focal_contrast1'; stats = nib.load(self.source(folder/'zstat1.nii.gz'))
        sig = nib.load(self.source(folder/'cluster_mask_zstat1.nii.gz')).get_fdata() == 4
        if int(sig.sum()) != int(self.insula_cluster['Voxels']): raise ValueError('FLAME insula cluster count changed')
        payloads.append(self.custom_payload(stats, sig, 'FLAME: right insula\nPositive offer slope', float(self.insula_cluster['P']), 'Z = 3.1'))
        fig = plt.figure(figsize=(13, 10)); grid = fig.add_gridspec(3, 4, width_ratios=[1.25, 1, 1, 1],
            left=.025, right=.965, top=.84, bottom=.13, hspace=.28, wspace=.05)
        for i, item in enumerate(payloads):
            rec = item[0]; ax = fig.add_subplot(grid[i, 0]); ax.axis('off')
            ax.text(.02, .82, rec['title'], va='top', fontsize=12, weight='bold')
            ax.text(.02, .36, f"pFWE = {rec['minimum_spatial_fwe_p']:.4g}\n{rec['native_voxels']} native voxels\n{rec.get('threshold', 'Z = 3.1')}", va='top', fontsize=10)
            self.planes([fig.add_subplot(grid[i, col]) for col in range(1, 4)], item)
        fig.suptitle('Offer-related activation across all three partners and age groups', fontsize=17, y=.975)
        fig.text(.5, .91, 'Convergent posterior effect; additional right-insula effect under historical FLAME only', ha='center', fontsize=11)
        cbax = fig.add_axes([.35, .072, .36, .018]); cb = fig.colorbar(ScalarMappable(norm=Normalize(-6, 6), cmap='coolwarm'), cax=cbax, orientation='horizontal')
        cb.set_label('FLAME/PALM: equivalent Z; randomise: t', fontsize=9)
        fig.text(.025, .012, 'Historical covariates/masks differ from new PALM. Native inference, nearest-neighbor display only. Insula is a POSITIVE offer slope.', fontsize=8)
        self.save(fig, '01_across_partner_offer_activation', 'Offer activation | Collapsed across partners and age groups',
            'Participant COPE11 averages computer, similar-partner and dissimilar-partner offer slopes with equal one-third weights. '
            'The group intercept averages all 47 participants with age and nuisance covariates centered. '
            'PALM detects 45 posterior voxels; randomise detects 95. Historical FLAME additionally detects an 82-voxel right-insula cluster '
            '(p = .000129; peak [40.3, 1.29, 5.22]). That cluster increases with larger offers and has no corrected randomise/PALM overlap. '
            'This is not evidence of the classic smaller-offer/unfairness response. The historical FLAME map has five positive clusters, 348 voxels in total.')

    def custom_payload(self, image, sig, title, p, threshold):
        z = image.get_fdata(); locations = np.argwhere(sig)
        peak = locations[np.argmax(z[tuple(locations.T)])]; cuts = nib.affines.apply_affine(image.affine, peak)
        overlay = resample_from_to(nib.Nifti1Image(np.where(sig, z, 0).astype('float32'), image.affine), self.bg, order=0).get_fdata()
        rec = dict(title=title, minimum_spatial_fwe_p=p, native_voxels=int(sig.sum()), threshold=threshold,
            native_cut_x=float(cuts[0]), native_cut_y=float(cuts[1]), native_cut_z=float(cuts[2]),
            cut_rule='largest statistic within displayed corrected cluster(s)')
        self.map_records.append(rec); return rec, overlay, cuts

    def region_diagnostic(self):
        folder = self.palm/'maps/taskwide-offer/cluster-z3.1'
        z = nib.load(self.source(folder/'palm_vox_zvstat_c1.nii')); field = resample_from_to(z, self.bg, order=1).get_fdata()
        fig = plt.figure(figsize=(13, 8.2)); grid = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1], left=.025, right=.965, top=.81, bottom=.19, hspace=.30, wspace=.05)
        for row, (name, title, cuts) in enumerate([
            ('roi-aINS', 'Existing anterior-insula\nspheres', [38, 20, 0]),
            ('roi-dACC', 'Existing dACC\nsphere', [-4, 16, 48])]):
            roi = nib.load(self.source(self.repo/'masks'/f'{name}.nii.gz'))
            contour = resample_from_to(roi, self.bg, order=0).get_fdata() > 0
            ax = fig.add_subplot(grid[row, 0]); ax.axis('off'); ax.text(.02, .82, title, weight='bold', va='top', fontsize=12)
            ax.text(.02, .40, 'No overlap with corrected\nwhole-brain clusters\nin FLAME, randomise\nor PALM (both signs).', fontsize=10, va='top')
            for col in range(3):
                ax = fig.add_subplot(grid[row, col+1]); base, extent, actual = slice_plane(self.anatomy, self.bg.affine, col, cuts[col])
                values, _, _ = slice_plane(field, self.bg.affine, col, cuts[col]); outline, _, _ = slice_plane(contour, self.bg.affine, col, cuts[col])
                ax.imshow(base, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=self.vmax)
                ax.imshow(np.ma.masked_where(base <= 0, values), cmap='coolwarm', vmin=-4, vmax=4, alpha=.65, origin='lower', extent=extent)
                if outline.any():
                    xs = np.linspace(extent[0], extent[1], outline.shape[1]); ys = np.linspace(extent[2], extent[3], outline.shape[0])
                    ax.contour(xs, ys, outline, levels=[.5], colors=['#ffed52'], linewidths=1)
                ax.set_title(f'{"xyz"[col]} = {actual:.0f} mm', fontsize=10); ax.set_xticks([]); ax.set_yticks([])
                ax.spines[['left','bottom']].set_visible(False)
        fig.suptitle('Anterior insula and dACC: descriptive inspection, not a new ROI test', fontsize=16, y=.975)
        fig.text(.5, .91, 'UNTHRESHOLDED PALM signed Z field: positive = larger offers, negative = smaller offers', ha='center', fontsize=11)
        cbax = fig.add_axes([.35, .105, .36, .022]); cb = fig.colorbar(ScalarMappable(norm=Normalize(-4, 4), cmap='coolwarm'), cax=cbax, orientation='horizontal'); cb.set_label('Unthresholded signed equivalent Z', fontsize=9)
        fig.text(.025, .018, 'Yellow outlines: existing Feng-based sphere masks, not the entire anatomical insula/dACC. Color shading uses linear display interpolation only.\nNo small-volume correction, ROI inference, model refit, or threshold relaxation. Absence of corrected clusters does not establish absence of a response.', fontsize=8.5)
        self.save(fig, '02_insula_dacc_descriptive', 'Insula/dACC | Existing masks and unthresholded field',
            'These images explain the sign and anatomical position of the existing maps; colored voxels are NOT significance claims. '
            'Neither the predefined anterior-insula spheres nor the dACC sphere contains a corrected whole-brain significant voxel in either direction. '
            'The positive historical FLAME insula cluster is more posterior than the anterior-insula spheres. No new ROI statistical test was run.')

    def parametric_grid(self):
        models = list(dict.fromkeys(r['model'] for r in self.summary))
        labels = [m.replace('full-rt__','Full RT | ').replace('pooled-post__','Pooled post | ').replace('partner-post__','Partner post | ') for m in models]
        fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
        for ax, method, title in zip(axes, ['cluster-z3.1','cluster-z2.6'], ['Paper-matched PALM: Z = 3.1', 'Sensitivity PALM: Z = 2.6']):
            p = np.array([[float(next(r for r in self.summary if r['model']==m and r['method']==method and int(r['contrast'])==c)['minimum_spatial_fwe_p']) for c in range(1,5)] for m in models])
            ax.imshow(p<.05, cmap=ListedColormap(['#f2f3f5','#d0e5ef']), vmin=0, vmax=1, aspect='auto')
            for i in range(len(models)):
                for j in range(4):ax.text(j,i,f'{p[i,j]:.4f}' if p[i,j]<.05 else 'NS',ha='center',va='center',fontsize=10,weight='bold' if p[i,j]<.05 else 'normal')
            ax.set_xticks(range(4), ['Mean +','Mean -','Younger > older','Older > younger'], rotation=25,ha='right');ax.set_yticks(range(len(labels)),labels);ax.set_title(title,fontsize=12);ax.tick_params(length=0)
        fig.suptitle('Parametric effects only: all 88 planned result cells',fontsize=17,y=.98)
        fig.subplots_adjust(left=.22,right=.98,top=.88,bottom=.18,wspace=.18)
        fig.text(.22,.035,'All rows concern offer slopes or offer-modulated network coupling. The unmodulated task-response model is excluded.\nNo direct age contrast in this batch survives PALM at Z = 3.1. Minimum spatial-FWE p per map; no correction across models/directions/thresholds.',fontsize=9.5)
        self.save(fig,'08_parametric_inventory','Parametric inventory | All means and age directions',
            'The unmodulated human-computer task-response model is removed. All 11 parametric models and both signed mean/age directions are shown. '
            'No direct age contrast in this batch survives at Z = 3.1. Lower-threshold DMN and task-wide offer age findings remain sensitivity results. Age-by-behavior interactions are separate analyses.')

    def finish(self):
        for name,value in [('map_provenance.json',self.map_records),('source_inventory.json',self.sources),('figure_manifest.json',self.pages)]:
            (self.output/name).write_text(json.dumps(value,indent=2)+'\n')
        body='<h1>SRNDNA Ultimatum: parametric effects</h1><p>Revised 2 November 2026. Imaging scope: offer slopes and offer-modulated PPI only.</p>'
        body+='<p>Across all three partners and age groups, positive offer-related activation converges posteriorly. Historical FLAME also detects a positive right-insula cluster; randomise/PALM do not confirm it. No corrected negative offer-slope cluster or dACC result is established. No direct age contrast in this parametric PALM batch survives Z = 3.1.</p>'
        for page in self.pages:body+='<section><h2>'+escape(page['title'])+'</h2><img src="'+page['image']+'"><p>'+escape(page['caption'])+'</p></section>'
        (self.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SRNDNA parametric review</title><style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#26313b}img{width:100%}section{margin:50px 0}p{line-height:1.5}</style>'+body)


def pdf(review):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.pdfgen import canvas
    from PIL import Image as PILImage
    folder=review.output/'output/pdf';folder.mkdir(parents=True,exist_ok=True);path=folder/'srndna_parametric_results_2026-11-02.pdf'
    w,h=landscape(A4);doc=SimpleDocTemplate(str(path),pagesize=(w,h),leftMargin=35,rightMargin=35,topMargin=30,bottomMargin=30,title='SRNDNA Ultimatum: parametric offer effects',author='DVS-Lab')
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='ReviewBody',fontSize=10.5,leading=14,spaceAfter=9));styles.add(ParagraphStyle(name='ReviewSmall',fontSize=9,leading=12,spaceAfter=6))
    p=lambda s:Paragraph(s,styles['ReviewBody'])
    story=[Paragraph('SRNDNA Ultimatum: parametric offer effects',styles['Title']),p('Revised scientific figure packet | 2 November 2026 | N = 47'),
        p('<b>Scope corrected:</b> imaging figures now concern offer-amount parametric modulators and offer-modulated network coupling. Unmodulated human-computer task responses and their age effect are removed from the result narrative and figure set.'),
        p('<b>Across-partner, full-sample test:</b> L1 COPE11 is (computer offer slope + similar-partner offer slope + dissimilar-partner offer slope)/3. The L3 all-ones coefficient gives the adjusted mean across 25 younger and 22 older participants, with centered age and the reported controls. It is already the requested across-partner, across-age activation analysis.')]
    data=[['Offer-related activation','FLAME Z3.1','randomise t3.1','PALM Z3.1'],
        ['Positive slope: larger offers','p = .00000256; 348 vox','p = .0364; 95 vox','p = .0052; 45 vox'],
        ['Negative slope: smaller offers','No corrected clusters','p = .2814; 0 vox','p = .7646; 0 vox'],
        ['Right-insula positive cluster','82 vox; p = .000129','0 corrected overlap','0 corrected overlap'],
        ['Existing aINS / dACC spheres','0 corrected overlap','0 corrected overlap','0 corrected overlap']]
    t=Table([[Paragraph(escape(c),styles['ReviewSmall'])for c in r]for r in data],colWidths=[225,180,180,180]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f8')])]))
    story += [t,Spacer(1,10),p('<b>What this supports:</b> positive offer-related activation converges in posterior cortex. Retained FLAME additionally supports a right-insular positive slope, confirmed anatomically with the Harvard-Oxford atlas. It is not a convergent permutation finding and does not show greater activation for unfair/smaller offers.'),
        Paragraph('FLAME here uses corrected events and all-responded-trial RT regressors, not the unrepaired submitted analysis. Its group covariates/mask predate the latest PALM harmonization.',styles['ReviewSmall']),
        p('<b>What this does not establish:</b> no corrected negative offer-slope cluster and no corrected dACC effect are present. The predefined anterior-insula spheres have no corrected overlap, but they do not cover all anatomical insula. These are whole-brain findings, not new targeted ROI tests.'),PageBreak()]
    story += [Paragraph('Interpretation, age effects and remaining work',styles['Title']),
        p('<b>Sign matters:</b> positive activation slopes mean greater activity for larger offers; negative slopes mean greater activity for smaller offers. A contrast between partners or age groups is a difference between slopes and does not require either slope itself to be positive. Offer size changes both fairness and own monetary payoff in this design.'),
        p('<b>Activation is not PPI:</b> offer-modulated PPI measures how network-region coupling depends on offers. The small network main effects remain in this packet because they are parametric: pooled-post DMN similar > dissimilar p = .0424; pooled-post and partner-post ECN dissimilar > similar p = .0298/.0121. They remain timing-dependent and require anatomical/influence review.'),
        p('<b>Age findings within scope:</b> no direct age contrast in this parametric PALM batch survives Z = 3.1. Full-RT DMN younger > older has p = .2526 at Z3.1 and p = .0468 at Z2.6. Task-wide offer younger > older has p = .1017 at Z3.1 and p = .0352 at Z2.6. The unmodulated human-computer age contrast tests a different process. Age-by-behavior interactions, including the separate parietal finding, are not part of this batch.'),
        p('<b>Method comparison:</b> historical FLAME/randomise use some different motion/tSNR values and masks. PALM uses verified covariates, strict all-run coverage, age variance groups, 10,000 sign-flips and unweighted group inference. A fully matched estimator comparison remains necessary before attributing differences to software alone. FLAME is the submitted estimator; PALM is a robustness analysis.'),
        p('<b>Behavioral context retained:</b> acceptance rises with offer size, without reliable age-by-offer-by-similarity moderation (p = .230). Ratings distinguish humans from the computer, not reliably similar from dissimilar partners. Individual norm-model parameters have not cleared recovery gates for imaging use.'),
        p('<b>Still to finish:</b> resolve matched-design comparisons; choose the primary timing specification on design grounds; review tissue localization/influence for the parametric candidates; finalize figures and revise the manuscript/responses. No new ROI tests, small-volume correction or model changes were made to seek insula/dACC significance.'),
        p('<b>Anatomical audit:</b> the historical right-insula cluster peaks near [40.3, 1.29, 5.22] mm and is more posterior than the existing anterior-insula sphere. Atlas percentages are descriptive template probabilities, not statistical significance or participant-specific tissue fractions. Source hashes and descriptive checks accompany the figures.'),PageBreak()]
    for i,page in enumerate(review.pages):
        story.append(Paragraph(escape(page['title']),styles['Heading1']))
        im=review.output/page['image']
        with PILImage.open(im)as image:iw,ih=image.size
        scale=min((w-70)/iw,375/ih);story.append(Image(str(im),width=iw*scale,height=ih*scale));story.append(Spacer(1,7));story.append(Paragraph(escape(page['caption']),styles['ReviewSmall']))
        if i<len(review.pages)-1:story.append(PageBreak())
    def footer(c,d):
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#596874'));c.drawString(35,16,'Parametric review | 2 November 2026 | No new inferential tests');c.drawRightString(w-35,16,str(d.page))
    def stable(*a,**kw):kw['invariant']=1;return canvas.Canvas(*a,**kw)
    doc.build(story,onFirstPage=footer,onLaterPages=footer,canvasmaker=stable);return path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--standard',type=Path,required=True);parser.add_argument('--atlas-root',type=Path,required=True);parser.add_argument('--output',type=Path,default=ROOT/'results/figure_review/2026-11-02-parametric');parser.add_argument('--pdf',action='store_true');a=parser.parse_args()
    review=ParametricReview(ROOT,a.output.resolve(),a.standard.resolve(),a.atlas_root.resolve());review.fairness_audit();review.collapsed_overlays();review.region_diagnostic()
    review.overlay_page([('pooled-post__nppi-dmn','cluster-z3.1',1,'DMN similar > dissimilar\nPooled post-response',1),('pooled-post__nppi-ecn','cluster-z3.1',2,'ECN dissimilar > similar\nPooled post-response',-1),('partner-post__nppi-ecn','cluster-z3.1',2,'ECN dissimilar > similar\nPartner post-response',-1)],
        '03_offer_network_means','Offer-modulated network main effects','Cluster Z = 3.1; spatial FWE p < .05 | Coupling slopes, not overall task responses',
        'Small full-sample partner differences in offer-modulated coupling. They are not age effects and are absent at this threshold in the full-RT model. Tissue localization and influence checks remain.')
    review.overlay_page([('full-rt__nppi-dmn','cluster-z2.6',3,'Younger > older\nSimilar - dissimilar\noffer-modulated DMN PPI',1)],
        '04_dmn_offer_age_sensitivity','DMN offer-related age difference: sensitivity result','Full RT | Z2.6 p = .0468 (49 voxels) | Z3.1 p = .2526 (0 significant voxels)',
        'The lower-threshold map matches the retained 49-voxel descriptive extraction region. No direct age contrast in this parametric PALM batch survives the paper-matched Z3.1 threshold.')
    review.dmn_bars();review.behavior();review.ratings();review.parametric_grid();review.finish()
    if a.pdf:print(pdf(review))
    (review.output/'output_inventory.json').write_text(json.dumps({str(p.relative_to(review.output)):digest(p)for p in sorted(review.output.rglob('*'))if p.is_file()and p.name!='output_inventory.json'},indent=2)+'\n')
    print(f'PASS: parametric-only review; {len(review.pages)} figures, 88 result cells; no new inference')


if __name__=='__main__':main()
