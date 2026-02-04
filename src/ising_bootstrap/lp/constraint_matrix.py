"""
Constraint matrix assembly for the conformal bootstrap LP.

Builds the matrix A and identity vector f_id such that the LP feasibility
problem is:
    Find λ ∈ R^66 such that:
        λ^T f_id = 1          (normalization)
        λ^T f_i  ≥ 0          for each operator i  (positivity)

where:
    f_id  = F_id^{m,n}(Δσ)            (identity contribution)
    f_i   = F^{m,n}_{Δi,li}(Δσ)       (crossing derivatives for operator i)

The constraint matrix A has shape (N_operators, 66) where N_operators is the
number of discretized operators in the spectrum.

Reference: arXiv:1203.6064, Appendix D
"""

import warnings
import numpy as np
from typing import List, Tuple, Optional
from mpmath import mpf

from ..config import (
    N_MAX, MAX_DERIV_ORDER, FULL_DISCRETIZATION, REDUCED_DISCRETIZATION,
    DiscretizationTable,
)
from ..spectrum.index_set import generate_index_set
from ..spectrum.discretization import (
    SpectrumPoint,
    build_spectrum_with_gaps,
    generate_full_spectrum,
)
from .crossing import (
    compute_prefactor_table,
    compute_identity_vector,
    compute_extended_h_array,
    compute_crossing_vector_fast,
    build_comb_cache,
)


def build_constraint_matrix(
    spectrum: List[SpectrumPoint],
    delta_sigma: float,
    n_max: int = N_MAX,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the constraint matrix A and identity vector f_id.

    For each operator in the spectrum, computes the 66-component crossing
    derivative vector F^{m,n}_{Δ,l}(Δσ).

    Parameters
    ----------
    spectrum : list of SpectrumPoint
        Discretized operator spectrum (from build_spectrum_with_gaps).
    delta_sigma : float
        External scalar dimension Δσ.
    n_max : int
        Truncation parameter.
    verbose : bool
        If True, print progress during computation.

    Returns
    -------
    A : np.ndarray
        Constraint matrix of shape (N_operators, 66). Row i corresponds to
        operator i in the spectrum.
    f_id : np.ndarray
        Identity contribution vector of shape (66,).
    """
    index_set = generate_index_set(n_max)
    n_components = len(index_set)
    n_operators = len(spectrum)
    max_order = 2 * n_max + 1

    # Precompute Δσ-dependent prefactor table (shared across all operators)
    U = compute_prefactor_table(delta_sigma, n_max)

    # Precompute identity vector
    f_id = compute_identity_vector(delta_sigma, n_max)

    # Precompute binomial coefficient cache
    comb_cache = build_comb_cache(n_max)

    # Build constraint matrix row by row
    A = np.zeros((n_operators, n_components), dtype=np.float64)

    n_skipped = 0
    for i, point in enumerate(spectrum):
        if verbose and (i % 1000 == 0 or i == n_operators - 1):
            print(f"  [{i+1}/{n_operators}] (Δ={point.delta:.4f}, l={point.spin})")

        try:
            # Compute extended block derivatives for this operator
            H = compute_extended_h_array(point.delta, point.spin, n_max)

            # Compute crossing derivatives via Leibniz rule
            A[i, :] = compute_crossing_vector_fast(
                H, U, index_set, comb_cache, max_order
            )
        except (ZeroDivisionError, ValueError) as e:
            # Skip operators that hit poles in the block computation
            # (e.g., spin-0 at exact unitarity bound Δ=0.5 in D=3).
            # Row remains zeros → trivially satisfied constraint.
            n_skipped += 1
            if verbose:
                print(f"  Warning: skipping (Δ={point.delta}, l={point.spin}): {e}")

    if n_skipped > 0:
        warnings.warn(
            f"Skipped {n_skipped}/{n_operators} operators due to block computation errors "
            f"(e.g., 3F2 pole at unitarity bound)."
        )

    return A, f_id


def build_constraint_matrix_from_cache(
    spectrum: List[SpectrumPoint],
    delta_sigma: float,
    h_cache: dict,
    n_max: int = N_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build constraint matrix using precomputed extended block derivatives.

    Parameters
    ----------
    spectrum : list of SpectrumPoint
        Discretized operator spectrum.
    delta_sigma : float
        External scalar dimension Δσ.
    h_cache : dict
        Dictionary mapping (delta, spin) -> np.ndarray (extended H array).
    n_max : int
        Truncation parameter.

    Returns
    -------
    A : np.ndarray
        Constraint matrix of shape (N_operators, 66).
    f_id : np.ndarray
        Identity vector of shape (66,).
    """
    index_set = generate_index_set(n_max)
    n_components = len(index_set)
    n_operators = len(spectrum)
    max_order = 2 * n_max + 1

    U = compute_prefactor_table(delta_sigma, n_max)
    f_id = compute_identity_vector(delta_sigma, n_max)
    comb_cache = build_comb_cache(n_max)

    A = np.zeros((n_operators, n_components), dtype=np.float64)

    for i, point in enumerate(spectrum):
        key = (round(point.delta, 8), point.spin)
        if key in h_cache:
            H = h_cache[key]
        else:
            try:
                H = compute_extended_h_array(point.delta, point.spin, n_max)
            except (ZeroDivisionError, ValueError):
                # Skip operators that hit poles (row stays zero)
                continue
            h_cache[key] = H

        A[i, :] = compute_crossing_vector_fast(
            H, U, index_set, comb_cache, max_order
        )

    return A, f_id


def precompute_extended_blocks(
    spectrum: List[SpectrumPoint],
    n_max: int = N_MAX,
    verbose: bool = False,
) -> dict:
    """
    Precompute extended block derivatives for all operators in the spectrum.

    Returns a cache dictionary mapping (delta, spin) -> H array.

    Parameters
    ----------
    spectrum : list of SpectrumPoint
        Discretized operator spectrum.
    n_max : int
        Truncation parameter.
    verbose : bool
        If True, print progress.

    Returns
    -------
    dict
        Mapping (delta, spin) -> np.ndarray of shape (max_order+1, max_k+1).
    """
    cache = {}
    unique_ops = set()
    for p in spectrum:
        unique_ops.add((round(p.delta, 8), p.spin))

    total = len(unique_ops)
    for i, (delta, spin) in enumerate(sorted(unique_ops)):
        if verbose and (i % 100 == 0 or i == total - 1):
            print(f"  [{i+1}/{total}] (Δ={delta:.4f}, l={spin})")
        try:
            cache[(delta, spin)] = compute_extended_h_array(delta, spin, n_max)
        except (ZeroDivisionError, ValueError):
            if verbose:
                print(f"  Warning: skipping (Δ={delta}, l={spin}): block computation error")

    return cache
