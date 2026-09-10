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
3. `figure3_corrected_dmn.png` shows the corrected DMN result and descriptive
   participant values extracted from its significant cluster.

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
participant order matches the rendered FSF and that cope 7 equals cope 4 minus
cope 6 within the cluster.

Then build all three figures with a Python environment containing NumPy,
pandas, matplotlib, nibabel, and nilearn:

```bash
python3 -m pip install -r requirements-figures.txt
python3 code/plot_manuscript_figures.py
```

Participant values in Figure 3 are descriptive because the displayed cluster
was selected by the group analysis. Inferential reporting must use the
whole-brain FLAME 1+2 result, not a test on the extracted values.
