# Corrected manuscript result set

This directory is the source of record for the revised manuscript figures and
their compact source data. The scientific endpoint is the fully corrected
analysis: corrected sub-144 trial identity, regenerated sub-144 L1/L2 images,
event-corrected task-wide mean RT, and corrected behavioral covariates where
the group model uses them.

## Figure set

1. `figure1_task_schematic.png` is the unchanged task schematic recovered from
   the manuscript working repository.
2. `figure2_corrected_acceptance.png` is generated from the converged,
   event-corrected behavioral model and its tracked source tables.
3. `figure3_corrected_dmn.png` shows the corrected DMN result and the four
   descriptive age-group-by-partner FLAME estimates from its significant
   cluster.

The submitted ECN figure is deliberately absent. Its focal cluster does not
survive the fully corrected model. The image-only rerun is retained solely as
a forensic provenance check and is not a reportable intermediate analysis.

The corrected activation/norm-proxy result also remains in the audit record.
It was not a headline result or figure in the submitted manuscript and is not
promoted into the revision as a new post hoc claim.

## Reproduction

On Linux, after the corrected group model has completed:

```bash
python3 code/export_corrected_dmn_figure_data.py \
  --repair-root /ZPOOL/data/scratch/srndna-ultimatum-l3-repair-v2
```

This exports only the compact 29-voxel corrected cluster, its thresholded
Z-stat values, and the 47-row descriptive source table. It validates that the
participant order matches the rendered FSF and extracts the exact L2 cope 7
images entered into the corrected group model. Although first-level cope 7 is
defined as cope 4 minus cope 6, separately combined L2 cope estimates need not
retain that numerical identity because FSL combines them using their own
run-level variance estimates. The figure therefore uses the actual L2 cope 7
group inputs rather than subtracting independently combined L2 copes 4 and 6.

Then build all three figures with a Python environment containing NumPy,
pandas, matplotlib, nibabel, and nilearn:

```bash
python3 -m pip install -r requirements-figures.txt
python3 code/plot_manuscript_figures.py
```

The bars in Figure 3 are descriptive because the displayed cluster was
selected by the group analysis. Inferential reporting must use the whole-brain
similar-minus-dissimilar FLAME 1+2 result, not a test on the displayed bars.

### Four-bar FLAME decomposition

The final panel contains four bars (younger/older by similar/dissimilar) from
two condition-specific FLAME 1+2 models. These models reuse the exact corrected
six-column group design and substitute L2 cope 4 or cope 6 for the cope 7
inputs. They retain the repaired sub-144 images. The legacy 94-row
stacked-condition template is not used because it does not model the two
observations per participant.

Prepare and run the two small Linux models with:

```bash
DMN_BAR_ROOT=/ZPOOL/data/scratch/srndna-ultimatum-dmn-bars-v1
python3 code/prepare_dmn_condition_bar_models.py --work-root "$DMN_BAR_ROOT"
python3 code/run_ultimatum_repair_jobs.py \
  --manifest "$DMN_BAR_ROOT/dmn_condition_bar_jobs.tsv" \
  --stage l3 --jobs 2
python3 code/export_dmn_condition_bar_data.py --work-root "$DMN_BAR_ROOT"
```

The export records the 94 input-level cluster-average COPE and VARCOPE values
for diagnostics. Bar heights come from the corresponding condition-specific
FLAME group COPE maps, not a hand-computed inverse-varcope average. Display
error bars use the cluster mean of the voxelwise FLAME standard-error map and
are descriptive rather than an independent cluster-average inferential test.
