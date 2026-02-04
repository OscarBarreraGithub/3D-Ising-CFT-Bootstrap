"""
Crossing function derivatives for the conformal bootstrap LP.

Computes F^{Δσ}_{Δ,l}(m,n) = ∂_a^m ∂_b^n [v^{Δσ} G(u,v) - u^{Δσ} G(v,u)]
evaluated at the crossing-symmetric point (a=1, b=0), i.e. u=v=1/4.

Key formula (for m odd):
    F^{m,n}_{Δ,l} = 2 Σ_{j,k} C(m,j) C(n,k) (-1)^j U^{j,k}(Δσ) h_{m-j,n-k}(Δ,l)

where:
    U^{j,k}(Δσ) = ∂_a^j ∂_b^k u^{Δσ} at (a=1, b=0)
    h_{p,q}(Δ,l)  = ∂_a^p ∂_b^q G_{Δ,l}  at (a=1, b=0)

The symmetry V^{j,k} = (-1)^j U^{j,k} (since v(a,b) = u(2-a, b)) simplifies
the general Leibniz rule to depend only on U^{j,k}.

Identity contribution: F_id^{m,n} = -2 U^{m,n}(Δσ)  for m odd.

Reference: arXiv:1203.6064, Sections 5-6
"""

import numpy as np
from math import comb, factorial
from mpmath import mp, mpf
from typing import Dict, Tuple, List, Optional, Union

from ..config import N_MAX, MAX_DERIV_ORDER, MPMATH_PRECISION
from ..spectrum.index_set import generate_index_set


def generate_extended_pairs(n_max: int = N_MAX) -> List[Tuple[int, int]]:
    """
    Generate all (p, q) pairs with p + 2q ≤ 2*n_max + 1, p ≥ 0, q ≥ 0.

    This is the "extended index set" including both odd and even p.
    For n_max=10: 132 pairs (66 odd-p from the standard index set + 66 even-p).

    Parameters
    ----------
    n_max : int
        Truncation parameter.

    Returns
    -------
    list of (int, int)
        All (p, q) pairs sorted lexicographically.
    """
    max_order = 2 * n_max + 1
    pairs = []
    for p in range(max_order + 1):
        max_q = (max_order - p) // 2
        for q in range(max_q + 1):
            pairs.append((p, q))
    return pairs


def extended_pair_count(n_max: int = N_MAX) -> int:
    """Number of (p, q) pairs in the extended index set."""
    max_order = 2 * n_max + 1
    count = 0
    for p in range(max_order + 1):
        count += (max_order - p) // 2 + 1
    return count


def compute_prefactor_table(delta_sigma: float, n_max: int = N_MAX) -> np.ndarray:
    """
    Compute U^{j,k}(Δσ) = ∂_a^j ∂_b^k u^{Δσ} at (a=1, b=0).

    Uses a stable recursion on the Taylor coefficients of w^α where
    w(ε, δ) = (1+ε)² - δ  and  u = w/4, so u^α = (w/4)^α = 4^{-α} w^α.

    The recursion derives from w · ∂_δ(w^α) = -α · w^α, which gives:
        T[j, k+1] = (k - α)/(k+1) T[j, k] - 2 T[j-1, k+1] - T[j-2, k+1]

    Parameters
    ----------
    delta_sigma : float
        External scalar dimension Δσ (= α in the formula).
    n_max : int
        Truncation parameter. max_order = 2*n_max + 1.

    Returns
    -------
    np.ndarray
        2D array of shape (max_order+1, max_k+1) where
        result[j, k] = ∂_a^j ∂_b^k u^{Δσ}  at (a=1, b=0).
        Entries with j + 2k > max_order are zero.
    """
    mp.dps = MPMATH_PRECISION
    alpha = mpf(delta_sigma)
    max_order = 2 * n_max + 1  # = 21 for n_max=10
    max_k = max_order // 2      # = 10

    # T[j][k] = Taylor coefficient of w^α at ε^j δ^k
    # where w = (1+ε)² - δ = 1 + 2ε + ε² - δ
    T = [[mpf(0)] * (max_k + 1) for _ in range(max_order + 1)]

    # ----- Column k = 0 -----
    # From w · ∂_ε(w^α) = α (∂_ε w) · w^α with ∂_ε w = 2(1+ε):
    #   T[j+1, 0] = [(2α - 2j) T[j, 0] + (2α - j + 1) T[j-1, 0]] / (j+1)
    T[0][0] = mpf(1)
    for j in range(max_order):
        t_j = T[j][0]
        t_jm1 = T[j - 1][0] if j >= 1 else mpf(0)
        T[j + 1][0] = ((2 * alpha - 2 * j) * t_j
                        + (2 * alpha - j + 1) * t_jm1) / (j + 1)

    # ----- Columns k = 1, 2, ..., max_k -----
    # From w · ∂_δ(w^α) = -α · w^α:
    #   T[j, k+1] = (k - α)/(k+1) T[j, k] - 2 T[j-1, k+1] - T[j-2, k+1]
    for k in range(max_k):
        max_j = max_order - 2 * (k + 1)
        for j in range(max_j + 1):
            t_jk = T[j][k]
            t_jm1 = T[j - 1][k + 1] if j >= 1 else mpf(0)
            t_jm2 = T[j - 2][k + 1] if j >= 2 else mpf(0)
            T[j][k + 1] = (k - alpha) / (k + 1) * t_jk - 2 * t_jm1 - t_jm2

    # Convert to actual derivatives: U^{j,k} = j! k! 4^{-α} T[j][k]
    prefactor_4 = mpf('0.25') ** alpha
    result = np.zeros((max_order + 1, max_k + 1), dtype=np.float64)
    for j in range(max_order + 1):
        for k in range(max_k + 1):
            if j + 2 * k <= max_order:
                result[j, k] = float(
                    prefactor_4 * factorial(j) * factorial(k) * T[j][k]
                )

    return result


def compute_identity_vector(delta_sigma: float, n_max: int = N_MAX) -> np.ndarray:
    """
    Compute F_id^{m,n}(Δσ) for the identity contribution.

    F_id(u,v) = v^{Δσ} - u^{Δσ}
    F_id^{m,n} = V^{m,n} - U^{m,n}

    Since V^{j,k} = (-1)^j U^{j,k} and m is odd:
        F_id^{m,n} = (-1)^m U^{m,n} - U^{m,n} = -2 U^{m,n}

    Parameters
    ----------
    delta_sigma : float
        External scalar dimension Δσ.
    n_max : int
        Truncation parameter.

    Returns
    -------
    np.ndarray
        1D array of length 66 (for n_max=10) with F_id^{m,n}
        in standard index set order.
    """
    U = compute_prefactor_table(delta_sigma, n_max)
    index_set = generate_index_set(n_max)

    f_id = np.zeros(len(index_set), dtype=np.float64)
    for idx, (m, n) in enumerate(index_set):
        f_id[idx] = -2.0 * U[m, n]

    return f_id


def compute_extended_h_array(
    delta: Union[float, mpf],
    spin: int,
    n_max: int = N_MAX
) -> np.ndarray:
    """
    Compute block derivatives h_{p,q} for ALL (p,q) with p+2q ≤ max_order.

    This includes both odd and even p (the standard index set only has odd p).
    The even-p values are needed for the Leibniz rule in crossing derivatives.

    Parameters
    ----------
    delta : float or mpf
        Operator scaling dimension Δ.
    spin : int
        Operator spin l.
    n_max : int
        Truncation parameter.

    Returns
    -------
    np.ndarray
        2D array H[p, q] = h_{p,q}  of shape (max_order+1, max_k+1).
        Entries with p+2q > max_order are zero.
    """
    from ..blocks.coordinate_transform import compute_h_m0_from_block_derivs
    from ..blocks.transverse_derivs import compute_all_h_mn

    max_order = 2 * n_max + 1
    max_k = max_order // 2

    # Compute h_{m,0} for m = 0, ..., max_order + 4 (buffer for Casimir recursion)
    h_m0 = compute_h_m0_from_block_derivs(mpf(delta), spin, max_order + 4)

    # Compute full h_{m,n} dictionary via Casimir recursion
    h_all = compute_all_h_mn(delta, spin, h_m0, n_max)

    # Extract into 2D array
    H = np.zeros((max_order + 1, max_k + 1), dtype=np.float64)
    for p in range(max_order + 1):
        for q in range(max_k + 1):
            if p + 2 * q <= max_order and (p, q) in h_all:
                H[p, q] = float(h_all[(p, q)])

    return H


def compute_crossing_vector(
    h_extended: np.ndarray,
    prefactor_table: np.ndarray,
    n_max: int = N_MAX
) -> np.ndarray:
    """
    Compute crossing function derivatives F^{m,n}_{Δ,l} for one operator.

    F^{m,n} = 2 Σ_{j=0}^{m} Σ_{k=0}^{n} C(m,j) C(n,k) (-1)^j U^{j,k} h_{m-j,n-k}

    Parameters
    ----------
    h_extended : np.ndarray
        2D array H[p, q] of block derivatives (from compute_extended_h_array).
    prefactor_table : np.ndarray
        2D array U[j, k] of prefactor derivatives (from compute_prefactor_table).
    n_max : int
        Truncation parameter.

    Returns
    -------
    np.ndarray
        1D array of length 66 (for n_max=10) with F^{m,n} in index set order.
    """
    index_set = generate_index_set(n_max)
    max_order = 2 * n_max + 1
    H = h_extended
    U = prefactor_table

    result = np.zeros(len(index_set), dtype=np.float64)

    for idx, (m, n) in enumerate(index_set):
        val = 0.0
        sign_j = 1.0  # (-1)^j, starting at j=0
        for j in range(m + 1):
            p = m - j
            if p < 0:
                break
            for k in range(n + 1):
                q = n - k
                if p + 2 * q > max_order:
                    continue
                h_pq = H[p, q]
                if h_pq == 0.0 and p + q > 0:
                    continue
                val += comb(m, j) * comb(n, k) * sign_j * U[j, k] * h_pq
            sign_j = -sign_j  # alternate sign with j

        result[idx] = 2.0 * val

    return result


def compute_crossing_vector_fast(
    h_extended: np.ndarray,
    prefactor_table: np.ndarray,
    index_set: List[Tuple[int, int]],
    comb_cache: Dict[Tuple[int, int], int],
    max_order: int
) -> np.ndarray:
    """
    Optimized version of compute_crossing_vector using precomputed comb cache.

    Parameters
    ----------
    h_extended : np.ndarray
        2D block derivative array H[p, q].
    prefactor_table : np.ndarray
        2D prefactor array U[j, k].
    index_set : list of (int, int)
        The 66 (m, n) pairs.
    comb_cache : dict
        Precomputed binomial coefficients.
    max_order : int
        Maximum derivative order (21).

    Returns
    -------
    np.ndarray
        1D array of crossing derivatives F^{m,n}.
    """
    H = h_extended
    U = prefactor_table
    result = np.zeros(len(index_set), dtype=np.float64)

    for idx, (m, n) in enumerate(index_set):
        val = 0.0
        sign_j = 1.0
        for j in range(m + 1):
            p = m - j
            c_mj = comb_cache[(m, j)]
            for k in range(n + 1):
                q = n - k
                if p + 2 * q > max_order:
                    continue
                h_pq = H[p, q]
                if h_pq == 0.0 and p + q > 0:
                    continue
                val += c_mj * comb_cache[(n, k)] * sign_j * U[j, k] * h_pq
            sign_j = -sign_j

        result[idx] = 2.0 * val

    return result


def build_comb_cache(n_max: int = N_MAX) -> Dict[Tuple[int, int], int]:
    """Precompute binomial coefficients C(m, j) for all needed values."""
    max_order = 2 * n_max + 1
    max_k = max_order // 2
    cache = {}
    for m in range(max_order + 1):
        for j in range(m + 1):
            cache[(m, j)] = comb(m, j)
    for n in range(max_k + 1):
        for k in range(n + 1):
            cache[(n, k)] = comb(n, k)
    return cache
