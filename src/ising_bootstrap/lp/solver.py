"""
LP feasibility solver for the conformal bootstrap.

Wraps scipy.optimize.linprog (HiGHS backend) to solve the bootstrap
feasibility problem:

    Find λ ∈ R^66 such that:
        λ^T f_id = 1          (normalization)
        λ^T f_i  ≥ 0          for each operator i  (positivity)

If such λ exists, the hypothesized spectrum is EXCLUDED (inconsistent with
crossing symmetry). If no such λ exists, the spectrum is ALLOWED.

Row and column scaling is critical since the constraint matrix entries
span many orders of magnitude.

Reference: arXiv:1203.6064, Appendix D
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from scipy.optimize import linprog

from ..config import LP_TOLERANCE, N_MAX
from ..spectrum.index_set import generate_index_set
from ..spectrum.discretization import SpectrumPoint


@dataclass
class FeasibilityResult:
    """
    Result of a bootstrap feasibility test.

    Attributes
    ----------
    excluded : bool
        True if the spectrum is excluded (LP feasible, functional α found).
        False if the spectrum is allowed (LP infeasible, no functional found).
    status : str
        Human-readable status message.
    lp_status : int
        Raw status code from scipy.optimize.linprog:
        0 = optimal (LP feasible → spectrum excluded)
        1 = iteration limit
        2 = infeasible (LP infeasible → spectrum allowed)
        3 = unbounded
        4 = other
    alpha : Optional[np.ndarray]
        The linear functional coefficients λ_{m,n} if excluded, else None.
    fun : Optional[float]
        Objective function value (0 for feasibility).
    """
    excluded: bool
    status: str
    lp_status: int
    alpha: Optional[np.ndarray] = None
    fun: Optional[float] = None


def scale_constraints(
    A: np.ndarray,
    f_id: np.ndarray,
    n_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply geometric mean row/column scaling to the constraint system.

    Iteratively scales rows and columns so that the maximum absolute value
    in each row/column approaches 1. This greatly improves LP solver
    conditioning.

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix of shape (N, 66).
    f_id : np.ndarray
        Identity vector of shape (66,).
    n_iterations : int
        Number of scaling iterations.

    Returns
    -------
    A_scaled : np.ndarray
        Scaled constraint matrix.
    f_id_scaled : np.ndarray
        Scaled identity vector.
    row_scale : np.ndarray
        Row scaling factors (shape (N,)).
    col_scale : np.ndarray
        Column scaling factors (shape (66,)).
    """
    A_s = A.copy()
    f_id_s = f_id.copy()
    n_rows, n_cols = A_s.shape

    row_scale = np.ones(n_rows)
    col_scale = np.ones(n_cols)

    for _ in range(n_iterations):
        # Row scaling: scale each row by 1/max|row|
        for i in range(n_rows):
            row_max = np.max(np.abs(A_s[i, :]))
            if row_max > 0:
                factor = 1.0 / row_max
                A_s[i, :] *= factor
                row_scale[i] *= factor

        # Column scaling: scale each column by 1/max|col|
        # Include f_id in the column scaling
        for j in range(n_cols):
            col_max = max(np.max(np.abs(A_s[:, j])), abs(f_id_s[j]))
            if col_max > 0:
                factor = 1.0 / col_max
                A_s[:, j] *= factor
                f_id_s[j] *= factor
                col_scale[j] *= factor

    return A_s, f_id_s, row_scale, col_scale


def check_feasibility(
    A: np.ndarray,
    f_id: np.ndarray,
    tolerance: float = LP_TOLERANCE,
    scale: bool = True,
    method: str = "highs",
) -> FeasibilityResult:
    """
    Check bootstrap feasibility using linear programming.

    Solves the LP: minimize 0 subject to
        A_eq @ x = b_eq     (normalization: x^T f_id = 1)
        A_ub @ x ≤ b_ub     (positivity: x^T f_i ≥ 0, i.e. -f_i^T x ≤ 0)

    where x = λ are the linear functional coefficients (unbounded).

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix of shape (N_operators, 66).
        Row i is the crossing derivative vector for operator i.
    f_id : np.ndarray
        Identity vector of shape (66,).
    tolerance : float
        LP solver tolerance.
    scale : bool
        If True, apply row/column scaling before solving.
    method : str
        LP solver method (default "highs").

    Returns
    -------
    FeasibilityResult
        Result indicating whether the spectrum is excluded.
    """
    n_operators, n_vars = A.shape

    if scale:
        A_s, f_id_s, row_scale, col_scale = scale_constraints(A, f_id)
    else:
        A_s, f_id_s = A.copy(), f_id.copy()
        col_scale = np.ones(n_vars)

    # LP formulation:
    # Variables: x = λ (66 coefficients of the linear functional)
    # Objective: minimize 0 (feasibility only)
    c = np.zeros(n_vars)

    # Equality constraint: x^T f_id = 1  →  f_id^T @ x = 1
    A_eq = f_id_s.reshape(1, -1)
    b_eq = np.array([1.0])

    # Inequality constraints: x^T f_i ≥ 0  →  -f_i^T @ x ≤ 0
    A_ub = -A_s
    b_ub = np.zeros(n_operators)

    # Bounds: λ is unbounded
    bounds = [(None, None)] * n_vars

    # Solve
    options = {"presolve": True, "dual_feasibility_tolerance": tolerance,
               "primal_feasibility_tolerance": tolerance}

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method=method, options=options,
    )

    if res.status == 0:
        # LP feasible → spectrum EXCLUDED
        # Unscale the solution
        alpha = res.x / col_scale if scale else res.x
        return FeasibilityResult(
            excluded=True,
            status="Spectrum excluded (functional found)",
            lp_status=0,
            alpha=alpha,
            fun=res.fun,
        )
    elif res.status == 2:
        # LP infeasible → spectrum ALLOWED
        return FeasibilityResult(
            excluded=False,
            status="Spectrum allowed (no functional exists)",
            lp_status=2,
            alpha=None,
            fun=None,
        )
    else:
        # Other status (iteration limit, unbounded, etc.)
        return FeasibilityResult(
            excluded=False,
            status=f"LP solver returned status {res.status}: {res.message}",
            lp_status=res.status,
            alpha=None,
            fun=None,
        )


def solve_bootstrap(
    delta_sigma: float,
    delta_epsilon: Optional[float] = None,
    delta_epsilon_prime: Optional[float] = None,
    tables: Optional[List] = None,
    n_max: int = N_MAX,
    tolerance: float = LP_TOLERANCE,
    scale: bool = True,
    verbose: bool = False,
    h_cache: Optional[dict] = None,
) -> FeasibilityResult:
    """
    End-to-end bootstrap feasibility test.

    Builds the spectrum with gap constraints, constructs the constraint matrix,
    and solves the LP feasibility problem.

    Parameters
    ----------
    delta_sigma : float
        External scalar dimension Δσ.
    delta_epsilon : float, optional
        Gap below ε (Stage A): no scalars with Δ < delta_epsilon.
    delta_epsilon_prime : float, optional
        Gap between ε and ε' (Stage B): no scalars with
        delta_epsilon < Δ < delta_epsilon_prime.
    tables : list of DiscretizationTable, optional
        Which discretization tables to use. Default is FULL_DISCRETIZATION.
    n_max : int
        Truncation parameter.
    tolerance : float
        LP solver tolerance.
    scale : bool
        Whether to apply constraint scaling.
    verbose : bool
        If True, print progress.
    h_cache : dict, optional
        Precomputed extended block derivatives cache.

    Returns
    -------
    FeasibilityResult
        Result indicating whether the spectrum is excluded.
    """
    from .constraint_matrix import (
        build_constraint_matrix,
        build_constraint_matrix_from_cache,
    )
    from ..spectrum.discretization import build_spectrum_with_gaps

    # Build spectrum with gap constraints
    spectrum = build_spectrum_with_gaps(
        delta_epsilon=delta_epsilon,
        delta_epsilon_prime=delta_epsilon_prime,
        tables=tables,
    )

    if verbose:
        print(f"Spectrum: {len(spectrum)} operators")
        print(f"Δσ = {delta_sigma}, gap_ε = {delta_epsilon}, gap_ε' = {delta_epsilon_prime}")

    # Build constraint matrix
    if h_cache is not None:
        A, f_id = build_constraint_matrix_from_cache(
            spectrum, delta_sigma, h_cache, n_max
        )
    else:
        A, f_id = build_constraint_matrix(
            spectrum, delta_sigma, n_max, verbose=verbose
        )

    if verbose:
        print(f"Constraint matrix: {A.shape}")
        print(f"Identity norm: {np.linalg.norm(f_id):.6e}")

    # Solve LP
    result = check_feasibility(A, f_id, tolerance=tolerance, scale=scale)

    if verbose:
        print(f"Result: {result.status}")

    return result


def check_feasibility_extended(
    A: np.ndarray,
    f_id: np.ndarray,
    spectrum: list,
    delta_sigma: float,
    n_max: int = N_MAX,
    dps: int = 50,
    dps_verify: Optional[int] = 80,
    active_hint: Optional[set] = None,
    verbose: bool = False,
) -> FeasibilityResult:
    """
    Extended-precision feasibility check using mpmath column-generation simplex.

    Delegates to simplex.check_feasibility_extended(). See that function for
    full documentation.

    Parameters
    ----------
    A : np.ndarray
        Float64 constraint matrix (N, n_components) for fast pricing.
    f_id : np.ndarray
        Float64 identity vector.
    spectrum : list of SpectrumPoint
        Full spectrum for mpmath row computation.
    delta_sigma : float
        External scalar dimension.
    n_max : int
        Truncation parameter.
    dps : int
        Primary mpmath precision.
    dps_verify : int or None
        Verification precision (set None to skip).
    active_hint : set or None
        Warm-start active set from previous solve.
    verbose : bool
        Print progress.

    Returns
    -------
    FeasibilityResult
        With tri-state status: excluded=True/False/None.
    """
    from .simplex import check_feasibility_extended as _cfe
    return _cfe(
        A, f_id, spectrum, delta_sigma,
        n_max=n_max, dps=dps, dps_verify=dps_verify,
        active_hint=active_hint, verbose=verbose,
    )
