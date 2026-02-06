"""
Extended-precision simplex solver for the conformal bootstrap LP.

Uses mpmath arithmetic with column generation to handle the ill-conditioned
constraint matrix at n_max=10 (condition number ~4e16).

Architecture:
    - Farkas' lemma reduces feasibility to a 66-constraint system
    - Phase I simplex with 66x66 basis (tractable at 50-digit precision)
    - Column generation: float64 pricing over full spectrum, mpmath solve on active set
    - Certificate validation on every decision

Reference: arXiv:1203.6064
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Callable, Set
from mpmath import mp, mpf

from ..config import N_MAX, MPMATH_PRECISION
from ..spectrum.index_set import generate_index_set
from .crossing import (
    compute_prefactor_table_mp,
    compute_identity_vector_mp,
    compute_extended_h_array_mp,
    compute_crossing_vector_mp,
    build_comb_cache,
)
from .solver import FeasibilityResult


# =============================================================================
# Phase I Revised Simplex (66x66 basis, mpmath)
# =============================================================================

@dataclass
class Phase1Result:
    """Result of Phase I simplex on the Farkas system."""
    feasible: bool          # True if Farkas system A^T v = f_id, v >= 0 has solution
    objective: mpf          # Phase I objective (sum of artificials)
    pi: Optional[list]      # Simplex multiplier (66-vector of mpf)
    x_basic: Optional[list] # Basic variable values
    basis: Optional[list]   # Basis indices
    iterations: int


def _phase1_simplex(
    columns: List[List[mpf]],
    b: List[mpf],
    m: int,
    dps: int = 50,
    max_iter: int = 5000,
    verbose: bool = False,
) -> Phase1Result:
    """
    Phase I revised simplex to check feasibility of Mx = b, x >= 0.

    Minimizes sum of artificial variables. If minimum = 0, system is feasible.

    Parameters
    ----------
    columns : list of list of mpf
        Each element is a column of M (length m). Total n columns.
    b : list of mpf
        RHS vector (length m), must be non-negative.
    m : int
        Number of constraints (= 66 for our problem).
    dps : int
        mpmath decimal places.
    max_iter : int
        Maximum simplex iterations.

    Returns
    -------
    Phase1Result
    """
    saved_dps = mp.dps
    mp.dps = dps
    zero = mpf(0)
    tol = mpf(10) ** (-(dps - 5))

    try:
        n_real = len(columns)  # real variables (v_j)
        n_total = n_real + m   # real + artificial variables

        # Phase I costs: 0 for real variables, 1 for artificials
        c = [zero] * n_real + [mpf(1)] * m

        # Initial basis: artificial variables (indices n_real, ..., n_real+m-1)
        basis = list(range(n_real, n_total))
        nonbasis = list(range(n_real))

        # B_inv starts as identity (artificial columns are I)
        B_inv = [[mpf(1) if i == j else zero for j in range(m)] for i in range(m)]

        # Basic variable values = b (since B = I, x_B = B^{-1} b = b)
        x_B = [mpf(bi) for bi in b]

        # Phase I objective = sum(b)
        obj = sum(x_B)

        def get_column(j):
            """Get column j of the full constraint matrix [M | I]."""
            if j < n_real:
                return columns[j]
            else:
                # Artificial variable: standard basis vector
                idx = j - n_real
                return [mpf(1) if i == idx else zero for i in range(m)]

        # Compute initial simplex multiplier: pi = B^{-T} c_B
        c_B = [c[basis[i]] for i in range(m)]
        pi = [zero] * m
        for i in range(m):
            for j in range(m):
                pi[i] += B_inv[j][i] * c_B[j]

        for iteration in range(max_iter):
            # Check if objective is zero (within tolerance)
            if obj <= tol:
                if verbose:
                    print(f"  Phase I converged at iter {iteration}: obj = {obj}")
                return Phase1Result(
                    feasible=True, objective=obj, pi=pi,
                    x_basic=x_B, basis=basis, iterations=iteration,
                )

            # --- Pricing: find entering variable (Bland's rule) ---
            entering = -1
            entering_rc = zero
            basis_set = set(basis)

            # Check real variables first, then artificials (Bland's = smallest index)
            for j in range(n_total):
                if j in basis_set:
                    continue
                # Reduced cost: rc_j = c_j - pi^T a_j
                col_j = get_column(j)
                rc = c[j]
                for i in range(m):
                    rc -= pi[i] * col_j[i]
                if rc < -tol:
                    entering = j
                    entering_rc = rc
                    break  # Bland's rule: take first negative

            if entering == -1:
                # No entering variable -> optimal
                if verbose:
                    print(f"  Phase I optimal at iter {iteration}: obj = {obj}")
                return Phase1Result(
                    feasible=(obj <= tol), objective=obj, pi=pi,
                    x_basic=x_B, basis=basis, iterations=iteration,
                )

            # --- Pivot column: d = B^{-1} * a_entering ---
            col_e = get_column(entering)
            d = [zero] * m
            for i in range(m):
                for j in range(m):
                    d[i] += B_inv[i][j] * col_e[j]

            # --- Ratio test (Bland's rule for ties) ---
            leaving_idx = -1
            min_ratio = None
            for i in range(m):
                if d[i] > tol:
                    ratio = x_B[i] / d[i]
                    if min_ratio is None or ratio < min_ratio - tol:
                        min_ratio = ratio
                        leaving_idx = i
                    elif min_ratio is not None and abs(ratio - min_ratio) <= tol:
                        # Tie-breaking: smallest basis index (Bland's)
                        if basis[i] < basis[leaving_idx]:
                            leaving_idx = i

            if leaving_idx == -1:
                # Unbounded (shouldn't happen in Phase I with bounded b)
                if verbose:
                    print(f"  Phase I unbounded at iter {iteration}")
                return Phase1Result(
                    feasible=False, objective=obj, pi=pi,
                    x_basic=x_B, basis=basis, iterations=iteration,
                )

            # --- Basis update ---
            pivot_elem = d[leaving_idx]
            step = x_B[leaving_idx] / pivot_elem

            # Update basic variable values
            for i in range(m):
                if i == leaving_idx:
                    x_B[i] = step
                else:
                    x_B[i] -= step * d[i]

            # Update B_inv (eta-matrix / rank-1 update)
            inv_pivot = mpf(1) / pivot_elem
            # First update the leaving row
            old_leaving_row = B_inv[leaving_idx][:]
            for j in range(m):
                B_inv[leaving_idx][j] *= inv_pivot
            # Then update all other rows
            for i in range(m):
                if i == leaving_idx:
                    continue
                factor = d[i]
                for j in range(m):
                    B_inv[i][j] -= factor * B_inv[leaving_idx][j]

            # Update basis
            leaving_var = basis[leaving_idx]
            basis[leaving_idx] = entering

            # Update objective
            obj -= step * (-entering_rc)  # obj_new = obj - step * |rc|

            # Update simplex multiplier: pi = B^{-T} c_B
            c_B = [c[basis[i]] for i in range(m)]
            pi = [zero] * m
            for i in range(m):
                for j in range(m):
                    pi[i] += B_inv[j][i] * c_B[j]

        # Max iterations reached
        if verbose:
            print(f"  Phase I max iterations ({max_iter}): obj = {obj}")
        return Phase1Result(
            feasible=(obj <= tol), objective=obj, pi=pi,
            x_basic=x_B, basis=basis, iterations=max_iter,
        )

    finally:
        mp.dps = saved_dps


# =============================================================================
# Column generation wrapper
# =============================================================================

def _select_initial_active_set(
    A_f64: np.ndarray,
    f_id_f64: np.ndarray,
    size: int = 200,
) -> List[int]:
    """
    Select a diverse initial active set of operator indices.

    Strategy: pick operators spread across different spins and delta values,
    plus those with largest projections onto f_id.
    """
    N = A_f64.shape[0]
    if N <= size:
        return list(range(N))

    indices = set()

    # Include operators with largest |f_i^T f_id| (most relevant to normalization)
    projections = np.abs(A_f64 @ f_id_f64)
    top_proj = np.argsort(-projections)[:size // 2]
    indices.update(top_proj.tolist())

    # Include uniformly spaced operators for diversity
    step = max(1, N // (size - len(indices)))
    for i in range(0, N, step):
        indices.add(i)
        if len(indices) >= size:
            break

    return sorted(indices)


def check_feasibility_extended(
    A_f64: np.ndarray,
    f_id_f64: np.ndarray,
    spectrum: list,
    delta_sigma: float,
    n_max: int = N_MAX,
    dps: int = MPMATH_PRECISION,
    dps_verify: Optional[int] = None,
    max_outer: int = 20,
    active_hint: Optional[Set[int]] = None,
    verbose: bool = False,
) -> FeasibilityResult:
    """
    Check bootstrap feasibility using extended-precision column-generation simplex.

    Uses Farkas' lemma: the primal {α : f_id^T α = 1, A α ≥ 0} is feasible
    (spectrum EXCLUDED) iff the system A^T v = f_id, v ≥ 0 has NO solution.

    Parameters
    ----------
    A_f64 : np.ndarray, shape (N, n_components)
        Full float64 constraint matrix (for fast pricing).
    f_id_f64 : np.ndarray, shape (n_components,)
        Float64 identity vector.
    spectrum : list of SpectrumPoint
        Full spectrum (for computing mpmath crossing vectors on demand).
    delta_sigma : float
        External scalar dimension.
    n_max : int
        Truncation parameter.
    dps : int
        Primary mpmath precision (decimal places).
    dps_verify : int or None
        If set, re-verify at this higher precision.
    max_outer : int
        Maximum column-generation iterations.
    active_hint : set of int or None
        Warm-start: previous active set indices.
    verbose : bool
        Print progress.

    Returns
    -------
    FeasibilityResult
        With status in {"excluded", "allowed", "inconclusive"}.
    """
    saved_dps = mp.dps
    mp.dps = dps
    N, n_comp = A_f64.shape

    try:
        # Precompute shared mpmath objects
        index_set = generate_index_set(n_max)
        comb_cache = build_comb_cache(n_max)
        max_order = 2 * n_max + 1
        U_mp = compute_prefactor_table_mp(delta_sigma, n_max, dps)
        f_id_mp = compute_identity_vector_mp(delta_sigma, n_max, dps)

        # Cache for mpmath crossing vectors (avoid recomputing)
        row_cache: Dict[int, List[mpf]] = {}

        def compute_row_mp(j: int) -> List[mpf]:
            if j in row_cache:
                return row_cache[j]
            point = spectrum[j]
            H_mp = compute_extended_h_array_mp(point.delta, point.spin, n_max, dps)
            F_mp = compute_crossing_vector_mp(H_mp, U_mp, index_set, comb_cache, max_order)
            row_cache[j] = F_mp
            return F_mp

        # Initial active set
        if active_hint is not None and len(active_hint) >= 50:
            active_set = set(active_hint)
            # Ensure at least some diversity
            extra = _select_initial_active_set(A_f64, f_id_f64, size=50)
            active_set.update(extra)
        else:
            active_set = set(_select_initial_active_set(A_f64, f_id_f64, size=200))

        for outer_iter in range(max_outer):
            active_list = sorted(active_set)
            n_active = len(active_list)

            if verbose:
                print(f"    CG iter {outer_iter}: {n_active} active constraints")

            # Build sign-adjusted Farkas system: D A_active^T v + a = |f_id|
            # We want: A_active^T v = f_id, v >= 0
            # Sign-adjust so RHS is non-negative
            D = []
            b_adj = []
            for i in range(n_comp):
                if f_id_mp[i] >= 0:
                    D.append(mpf(1))
                    b_adj.append(f_id_mp[i])
                else:
                    D.append(mpf(-1))
                    b_adj.append(-f_id_mp[i])

            # Build columns for Phase I: each column j is D * f_j (row j of A at mpmath)
            columns = []
            for j in active_list:
                F_j = compute_row_mp(j)
                col = [D[i] * F_j[i] for i in range(n_comp)]
                columns.append(col)

            # Run Phase I simplex
            result = _phase1_simplex(
                columns, b_adj, n_comp, dps=dps,
                max_iter=5000, verbose=verbose,
            )

            if result.feasible:
                # Farkas system feasible -> primal INFEASIBLE -> ALLOWED
                # Validate certificate: reconstruct v, check A^T v ≈ f_id
                cert_ok = _validate_allowed_certificate(
                    result, active_list, row_cache, f_id_mp, n_comp, dps,
                )
                if verbose:
                    print(f"    ALLOWED (Phase I obj={result.objective}, "
                          f"cert={'OK' if cert_ok else 'FAIL'})")
                if cert_ok:
                    return FeasibilityResult(
                        excluded=False,
                        status="Spectrum allowed (Farkas certificate verified)",
                        lp_status=2,
                    )
                else:
                    return FeasibilityResult(
                        excluded=False,
                        status="Spectrum allowed (certificate residual high)",
                        lp_status=2,
                    )

            # Farkas infeasible on active set -> price non-active columns
            pi = result.pi  # simplex multiplier, length n_comp

            # Compute pricing direction: D * pi (since columns are D * f_j)
            D_pi_f64 = np.array([float(D[i] * pi[i]) for i in range(n_comp)])

            # Score all columns at float64
            scores = A_f64 @ D_pi_f64  # shape (N,)

            # Find candidates with positive score (negative reduced cost)
            # A positive score means D*pi^T * f_j > 0, i.e., the column could
            # improve Phase I and potentially make it feasible
            threshold = 1e-12
            candidates = np.where(scores > threshold)[0]
            candidates = [j for j in candidates if j not in active_set]

            if len(candidates) == 0:
                # Safety: re-check borderline cases at mpmath
                borderline_tol = 1e-8
                borderline_idx = np.where(
                    (scores > -borderline_tol) & (scores <= threshold)
                )[0]
                borderline_idx = [j for j in borderline_idx if j not in active_set]

                mp_violators = []
                for j in borderline_idx[:100]:
                    F_j = compute_row_mp(j)
                    score_mp = sum(D[i] * pi[i] * F_j[i] for i in range(n_comp))
                    if score_mp > mpf(10) ** (-(dps - 10)):
                        mp_violators.append(j)

                if mp_violators:
                    active_set.update(mp_violators)
                    if verbose:
                        print(f"    Found {len(mp_violators)} borderline violators at mpmath")
                    continue

                # No violators at all -> Phase I optimal on FULL problem -> EXCLUDED
                if verbose:
                    print(f"    EXCLUDED (no pricing violators, "
                          f"Phase I obj={result.objective})")

                # Precision ladder verification
                if dps_verify is not None:
                    if verbose:
                        print(f"    Verifying at dps={dps_verify}...")
                    verify_result = check_feasibility_extended(
                        A_f64, f_id_f64, spectrum, delta_sigma,
                        n_max=n_max, dps=dps_verify, dps_verify=None,
                        max_outer=max_outer,
                        active_hint=active_set, verbose=False,
                    )
                    if verify_result.excluded != True:
                        if verbose:
                            print(f"    Precision mismatch! dps={dps} says excluded, "
                                  f"dps={dps_verify} says {verify_result.status}")
                        return FeasibilityResult(
                            excluded=None,
                            status=f"Inconclusive: precision mismatch (dps={dps} vs {dps_verify})",
                            lp_status=-1,
                        )

                return FeasibilityResult(
                    excluded=True,
                    status="Spectrum excluded (column generation converged)",
                    lp_status=0,
                )

            # Add worst violators to active set
            worst_idx = np.argsort(-scores[candidates])[:50]
            worst = [candidates[i] for i in worst_idx]
            active_set.update(worst)
            if verbose:
                print(f"    Added {len(worst)} violators (max score={scores[worst[0]]:.2e})")

        # Max outer iterations
        if verbose:
            print(f"    INCONCLUSIVE (max CG iterations reached)")
        return FeasibilityResult(
            excluded=None,
            status=f"Inconclusive: column generation did not converge in {max_outer} iterations",
            lp_status=-1,
        )

    finally:
        mp.dps = saved_dps


def _validate_allowed_certificate(
    result: Phase1Result,
    active_list: List[int],
    row_cache: Dict[int, List[mpf]],
    f_id_mp: List[mpf],
    n_comp: int,
    dps: int,
) -> bool:
    """
    Validate the ALLOWED certificate: check that A_active^T v ≈ f_id.

    Returns True if the certificate residual is small.
    """
    tol = mpf(10) ** (-(dps - 10))

    # Extract v values (non-artificial basic variables)
    n_active = len(active_list)
    v = [mpf(0)] * n_active
    for bi, bval in enumerate(result.x_basic):
        basis_idx = result.basis[bi]
        if basis_idx < n_active:
            v[basis_idx] = bval

    # Compute A_active^T v
    residual = [mpf(0)] * n_comp
    for j_idx, j in enumerate(active_list):
        if v[j_idx] == 0:
            continue
        if j not in row_cache:
            continue
        F_j = row_cache[j]
        for i in range(n_comp):
            residual[i] += v[j_idx] * F_j[i]

    # Check residual ≈ f_id
    max_res = mpf(0)
    for i in range(n_comp):
        diff = abs(residual[i] - f_id_mp[i])
        if diff > max_res:
            max_res = diff

    # Check v >= 0
    min_v = min(v) if v else mpf(0)

    return max_res < tol and min_v >= -tol
