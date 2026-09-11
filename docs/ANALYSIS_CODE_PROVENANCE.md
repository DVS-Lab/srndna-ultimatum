# Analysis-code provenance for the Ultimatum Game paper

`DVS-Lab/srndna-ultimatum` is the repository of record for this paper. The
other SRNDNA repositories are evidence about how analyses evolved; they are
not interchangeable analysis roots.

## Verified repository roles

| Repository | Verified role | What belongs here |
| --- | --- | --- |
| `DVS-Lab/srndna-ultimatum` | Original 2021 Ultimatum production workflow, retained FEAT outputs, and current reviewer-repair code | Authoritative rendered production FSFs and matrices; corrected source events; tested repair/audit entry points |
| `DVS-Lab/srndna-ug` | Jen's later 2024–2025 paper-analysis working repository | Paper-specific L3/SANS templates and execution history, used as provenance candidates |
| `DVS-Lab/srndna_ug_public` | Public behavioral-analysis snapshot | Behavioral notebooks/tables only; no authoritative imaging pipeline was found |
| `DVS-Lab/srndna-datapaper` | Data-paper/OpenNeuro maintenance workflow | Reusable modern utilities, including the tested fMRIPrep-to-FSL confound converter |
| `DVS-Lab/srndna` | Earlier task conversion and model-01 lineage | Source-history evidence only; not the submitted paper's model-02 workflow |

## First- and second-level model provenance

The four model-02 template files in `srndna-ultimatum` and `srndna-ug` are
byte-identical. Their SHA-256 checksums are:

| Template | SHA-256 |
| --- | --- |
| `L1_task-ultimatum_model-02_type-act.fsf` | `24d0bb69fe853148d652ecf42a5e15af3c3bdd58deb902a255a9a4308b9410f7` |
| `L1_task-ultimatum_model-02_type-nppi.fsf` | `2453e5e203f2311932dc2c1c4370ea78c5eb39f0063a1ea1a14d90598eaeec86` |
| `L2_task-ultimatum_model-02_type-act.fsf` | `27f2a157796fa93cff5dc6fecf3205ae715be5617043551001c666b8c10af5f4` |
| `L2_task-ultimatum_model-02_type-ppi.fsf` | `044d95385458e6323a0815d8dd8aeffbdd1864d09c4d6daf16f34223cc05b442` |

The files entered `srndna-ultimatum` in commit `1e3f476` (2021-08-08); its
later `8914e36` change altered permissions, not contents. `code/L2stats.sh` is
also byte-identical across the two repositories. For the repair, retained
rendered `design.fsf` files and `design.mat` files take precedence over either
generic template or shell-script defaults.

The historical `MakeConfounds.py` implementations select, in order, all
`cosine*` columns, all `non_steady_state*` columns, six motion parameters, the
first six aCompCor components, and framewise displacement, replacing missing
values with zero and omitting the header. `code/make_fsl_confounds.py` ports
that behavior from the tested data-paper repair at commit `cec9d0a` without
its pandas/numpy runtime dependency. `code/audit_prepared_ultimatum_designs.py`
verifies the port against the nuisance-regressor columns of the retained
designs before an image fit is allowed.

The L1 `DATA` path in the original repository points outside the analysis
repository to `srndna-data/derivatives/fmriprep` and requests
`MNI152NLin2009cAsym`. The OpenNeuro/data-paper derivative metadata records
fMRIPrep 21.0.2, and its 21.0.2 wrapper requests that space. Jen later changed
the wrapper in `srndna-ug` to fMRIPrep 23.2.1 with only
`MNI152NLin6Asym`, but did not change the L1 filename to that space. Thus the
later wrapper is not evidence that the retained L1 models used 23.2.1. A hash
comparison on Linux established that both affected sub-144 BOLD files are
byte-identical between the downloaded OpenNeuro derivative and Jen's surviving
`srndna-ug` tree (run-1 SHA-256 begins `2055f273`; run-2 begins `73139354`).
There is no preprocessing rerun in the minimal sub-144 repair.

## Third-level paper-analysis provenance

The later group-analysis history lives primarily in `srndna-ug`: its
paper-specific L3/SANS scripts and templates were edited from 2024-08 through
2025-08, including the final age-by-sensitivity templates in commit `2a60e18`
(2025-08-12). The 15 current top-level L3 templates from that working tree are
already preserved under `legacy/jen_working_tree/templates/` for comparison.

The three submitted focal masks are binary derivatives rather than untouched
FSL `cluster_mask_zstat` files: their values are 0/1, while their copied image
headers retain FSL's Z-score intent and the `2203.12` build description. Git
history in `srndna-ug` adds the DMN, ECN, and activation masks on 2025-04-10,
2025-04-11, and 2025-04-12, respectively. Consequently, a compressed-file
checksum mismatch is expected if a labeled cluster was selected and binarized.
Production provenance requires an exact labeled-voxel-support match, not merely
a similar location or visual appearance.

The Linux support audit supplied that match for all three focal masks. DMN is
cluster label 1 of zstat 3 in the original 47-participant cope-7 DMN group
model; the same cluster and identical design matrix, contrast, and group files
survive in both repositories. ECN is cluster label 1 of zstat 1 in the
47-participant cope-7 group-specific sensitivity model. Activation is cluster
label 1 of zstat 1 in the corresponding `norm2_logit` model. Each rendered
design specifies FLAME 1+2, no automatic outlier deweighting, Z > 3.1, and
cluster-corrected p = .05.

The ECN output directory retains the earlier `sensitivity2` name. This is not a
different estimand: commit `924c1a4` renamed the production-era source FSF to
`sensitivity2_logit` with 100% content identity. The production design matrix,
contrast, and group-file SHA-256 values exactly match the companions committed
with that historical template, and its group-specific sensitivity EV values
match `in_out_sensitivity_indiv_logit.csv` to rounding error. The activation
production design files likewise exactly match the committed `norm2_logit`
companions. The recovered standard-space tables report DMN: 26 voxels,
corrected p = .0233, Zmax = 4.15; ECN: 23 voxels, corrected p = .0287,
Zmax = 4.25; and activation: 161 voxels, corrected p = 1.79e-7, Zmax = 4.33.

The legacy SANS shell scripts are not safe to copy wholesale. They combine
hard-coded output naming, unconditional removal of partial results, optional
`randomise` execution, and—in one version—a broken `sed` redirection. Their
scientific content is in the templates and rendered production designs. The
new guarded L3 preparer renders scratch-only jobs directly from the recovered
production FSFs after the sub-144 L1/L2 outputs pass; the runner refuses to
overwrite any existing output and verifies every expected group Z statistic.
No legacy script is treated as authoritative merely because it is newer.

## Porting rule

Port only a paper-relevant function after matching it to retained production
artifacts. Keep the source repository and commit in this document or the
script docstring, add a test or an explicit design comparison, write all new
outputs to scratch, and never replace retained FEAT/GFEAT directories during
the reviewer audit.
