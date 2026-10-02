#!/usr/bin/env python3
"""Unwhitened, contrast-specific diagnostics for compiled FEAT design matrices.

Reference implementation inspected: jmumford/vif_contrasts,
src/vif_for_contrasts/vif_contrasts.py (accessed 2026-10-01):
https://github.com/jmumford/vif_contrasts/blob/main/src/vif_for_contrasts/vif_contrasts.py

The implemented variance ratio is actual contrast variance divided by the
variance after setting off-diagonal Gram entries to zero. It need not exceed
one for multi-column contrasts. Unlike the upstream implementation's unchanged
contrast after column normalization, the main ratio here transforms the contrast
as well, preserving the original FEAT COPE's estimand and units. An additional
explicitly labeled value reproduces the upstream unit-norm contrast definition.

Both calculations center varying columns and remove constants, equivalent to
conditioning on an intercept. Constants cannot be tested by this routine. Empty
nuisance columns are harmless; nonzero contrast weights on empty columns are not.
Nonzero rank deficiencies and nonestimable contrasts are reported explicitly.

IMPORTANT: the reference cVIF requires unorthogonalized, condition-vs-baseline
coding (continuous modulators are permitted). An orthogonalized design's ratio
is only a descriptive ratio in that compiled basis, not original-stimulus cVIF.
This module neither unorthogonalizes EVs nor models FILM prewhitening. These
design diagnostics do not establish physiological validity or post-fit precision.
"""
from __future__ import annotations

import numpy as np


def pad_contrasts(contrasts, ncolumns):
    """Append zero weights for nuisance columns; reject excess/nonfinite weights."""
    array = np.asarray(contrasts, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("contrasts must be a nonempty vector or matrix")
    if array.shape[1] > ncolumns:
        raise ValueError("more contrast weights than design columns")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite contrast weights")
    return np.pad(array, ((0, 0), (0, ncolumns - array.shape[1])))


def diagnose_design(matrix, contrasts, *, orthogonalized=False, rcond=1e-10,
                    estimability_tolerance=1e-7):
    """Return JSON-safe design and contrast diagnostics, without fitting images.

    Parameters
    ----------
    matrix : array (observations, compiled EVs including appended confounds)
        FEAT design.mat values, already convolved/filtered as appropriate.
    contrasts : array (contrasts, <= compiled EVs), or one contrast vector
        Original compiled-column weights. Omitted trailing nuisance weights are 0.
    orthogonalized : bool
        True if any scientific EVs were orthogonalized before compilation. Then
        ``contrast_vif`` is deliberately null; ``actual_basis_variance_ratio`` is
        still supplied as a basis-dependent supplemental diagnostic.
    rcond : float
        Relative singular-value rank cutoff for column-normalized centered X.

    Each finite variance is c (X_centered' X_centered)^+ c' per unit IID residual
    variance, after checking estimability, not a fitted or prewhitened variance.
    ``iid_efficiency`` is its reciprocal and cannot be compared across contrasts
    with different scaling/units. No arbitrary VIF threshold or design approval
    is imposed. Column numbers in output are one-based.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("design must have at least two rows and one column")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite design values")
    if not 0 < rcond < 1 or not 0 < estimability_tolerance < 1:
        raise ValueError("rank cutoff and estimability tolerance must lie in (0, 1)")
    input_contrasts = np.atleast_2d(np.asarray(contrasts, dtype=float))
    c = pad_contrasts(contrasts, x.shape[1])
    constant = np.ptp(x, axis=0) == 0
    exact_zero = np.all(x == 0, axis=0)
    active = ~constant
    centered = x[:, active] - x[:, active].mean(axis=0)
    scales = np.linalg.norm(centered, axis=0)
    if active.any():
        normalized = centered / scales
        _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
        cutoff = rcond * singular[0]
        rank = int(np.count_nonzero(singular > cutoff))
        basis = vt[:rank]
        gram = normalized.T @ normalized
        correlations = np.abs(gram - np.diag(np.diag(gram)))
        max_correlation = float(correlations.max())
    else:
        singular = np.array([])
        basis = np.empty((0, 0))
        cutoff, rank, max_correlation = 0.0, 0, 0.0

    def evaluate(weights):
        size = np.linalg.norm(weights)
        error = np.linalg.norm(weights - (weights @ basis.T) @ basis)
        error = float(error / size) if size else 0.0
        if error > estimability_tolerance:
            return error, None
        coordinates = weights @ basis.T
        variance = float(np.sum((coordinates / singular[:rank]) ** 2))
        return error, variance

    checks = []
    for index, row in enumerate(c, 1):
        result = {
            "contrast": index,
            "status": "estimable",
            "estimable": False,
            "relative_projection_error": None,
            "weights_on_zero_columns": (np.flatnonzero((row != 0) & exact_zero) + 1).tolist(),
            "weights_on_constant_nonzero_columns": (
                np.flatnonzero((row != 0) & constant & ~exact_zero) + 1).tolist(),
            "iid_contrast_variance": None,
            "iid_orthogonal_reference_variance": None,
            "iid_efficiency": None,
            "actual_basis_variance_ratio": None,
            "contrast_vif": None,
            "upstream_unit_norm_contrast_vif": None,
        }
        if not np.any(row):
            result["status"] = "zero_contrast"
        elif result["weights_on_zero_columns"]:
            result["status"] = "nonestimable_empty_column"
        elif result["weights_on_constant_nonzero_columns"]:
            result["status"] = "unsupported_constant_contrast_after_centering"
        else:
            # X = Z D; gamma = D beta; c beta = (c D^-1) gamma.
            weights = row[active] / scales
            error, variance = evaluate(weights)
            result["relative_projection_error"] = error
            if variance is None:
                result["status"] = "nonestimable_rank_deficiency"
            else:
                reference_variance = float(weights @ weights)
                ratio = variance / reference_variance
                result.update(
                    estimable=True,
                    iid_contrast_variance=variance,
                    iid_orthogonal_reference_variance=reference_variance,
                    iid_efficiency=1.0 / variance,
                    actual_basis_variance_ratio=ratio,
                    contrast_vif=None if orthogonalized else ratio,
                )
                # Direct comparison with upstream is available only where its
                # full inverse exists, and still has a different estimand.
                if rank == int(active.sum()) and not orthogonalized:
                    upstream_weights = row[active]
                    _, upstream_variance = evaluate(upstream_weights)
                    result["upstream_unit_norm_contrast_vif"] = (
                        upstream_variance / float(upstream_weights @ upstream_weights))
        checks.append(result)

    full_rank = rank == int(active.sum())
    return {
        "design": {
            "npoints": int(x.shape[0]),
            "ncolumns": int(x.shape[1]),
            "contrast_padding_nuisance_zeros": int(x.shape[1] - input_contrasts.shape[1]),
            "centering": "varying_columns_centered_intercept_conditioned_out",
            "noise_assumption": "IID_design_diagnostic_not_FILM_prewhitened",
            "orthogonalized": bool(orthogonalized),
            "vif_interpretation": (
                "actual_compiled_basis_only_not_original_stimulus_cvif" if orthogonalized
                else "original_contrast_with_orthogonal_reference_same_column_norms"),
            "zero_columns": (np.flatnonzero(exact_zero) + 1).tolist(),
            "constant_nonzero_columns": (np.flatnonzero(constant & ~exact_zero) + 1).tolist(),
            "varying_columns": int(active.sum()),
            "scaled_centered_rank": rank,
            "remaining_rank_deficiency": int(active.sum()) - rank,
            "full_rank_after_constant_removal": full_rank,
            "rcond": float(rcond),
            "svd_cutoff": float(cutoff),
            "scaled_condition_number": (
                float(singular[0] / singular[-1]) if full_rank and len(singular) else None),
            "max_abs_centered_column_correlation": max_correlation,
        },
        "contrasts": checks,
    }
