"""
Transverse (b) derivatives of conformal blocks via Casimir recursion.

The Casimir differential equation implies a recursion relation for
h_{m,n} = ∂_a^m ∂_b^n G|_{a=1,b=0}

This recursion (Eq. C.1 from arXiv:1203.6064) expresses h_{m,n} with n > 0
in terms of h values with lower n, using h_{m,0} as initial conditions.

The conformal blocks are eigenfunctions of the quadratic Casimir:
    D G_{Δ,l}(z,z̄) = (1/2) C_{Δ,l} G_{Δ,l}(z,z̄)

where C_{Δ,l} = Δ(Δ - D) + l(l + D - 2).

Reference: arXiv:1203.6064, Appendix C, Eq. C.1
"""

from mpmath import mp, mpf
from typing import Union, Dict, List, Tuple, Optional

from ..config import D, MPMATH_PRECISION, N_MAX, MAX_DERIV_ORDER
from ..spectrum.index_set import generate_index_set, is_valid_index_pair

# Set precision
mp.dps = MPMATH_PRECISION


def casimir_eigenvalue(delta: Union[float, mpf], spin: int, d: int = D) -> mpf:
    """
    Compute the quadratic Casimir eigenvalue C_{Δ,l}.

    C_{Δ,l} = Δ(Δ - D) + l(l + D - 2)

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    d : int
        The spacetime dimension D.

    Returns
    -------
    mpf
        The Casimir eigenvalue.
    """
    delta = mpf(delta)
    return delta * (delta - d) + spin * (spin + d - 2)


def casimir_recursion_coefficient_m_term(m: int, n: int, d: int = D) -> Tuple[mpf, mpf, mpf]:
    """
    Compute coefficients for h_{m-1,n}, h_{m-2,n}, h_{m-3,n} terms.

    From Eq. C.1, the first line of the RHS is:
    2m(D + 2n - 3)[-h_{m-1,n} + (m-1)h_{m-2,n} + (m-1)(m-2)h_{m-3,n}]

    Returns
    -------
    tuple of (mpf, mpf, mpf)
        Coefficients for (h_{m-1,n}, h_{m-2,n}, h_{m-3,n}).
    """
    prefactor = 2 * m * (d + 2*n - 3)
    coeff_m1 = -prefactor
    coeff_m2 = prefactor * (m - 1)
    coeff_m3 = prefactor * (m - 1) * (m - 2)
    return (mpf(coeff_m1), mpf(coeff_m2), mpf(coeff_m3))


def casimir_recursion_coefficient_n1_terms(m: int, n: int, delta: mpf, spin: int, d: int = D) -> Tuple[mpf, mpf, mpf, mpf]:
    """
    Compute coefficients for n-1 level terms.

    From Eq. C.1:
    - h_{m+2,n-1} + (D - m - 4n + 4)h_{m+1,n-1}
    + [2C + 2D(m+n-1) + m² + 8mn - 9m + 4n² - 6n + 2] h_{m,n-1}
    + m[D(m-2n+1) + m² + 12mn - 15m + 12n² - 30n + 20] h_{m-1,n-1}

    Returns
    -------
    tuple of (mpf, mpf, mpf, mpf)
        Coefficients for (h_{m+2,n-1}, h_{m+1,n-1}, h_{m,n-1}, h_{m-1,n-1}).
    """
    C = casimir_eigenvalue(delta, spin, d)

    coeff_mp2 = mpf(-1)
    coeff_mp1 = mpf(d - m - 4*n + 4)
    coeff_m0 = 2*C + 2*d*(m + n - 1) + m**2 + 8*m*n - 9*m + 4*n**2 - 6*n + 2
    coeff_mm1 = m * (d*(m - 2*n + 1) + m**2 + 12*m*n - 15*m + 12*n**2 - 30*n + 20)

    return (mpf(coeff_mp2), mpf(coeff_mp1), mpf(coeff_m0), mpf(coeff_mm1))


def casimir_recursion_coefficient_n2_terms(m: int, n: int, d: int = D) -> Tuple[mpf, mpf]:
    """
    Compute coefficients for n-2 level terms.

    From Eq. C.1:
    (n-1)[h_{m+2,n-2} - (D - 3m - 4n + 4)h_{m+1,n-2}]

    Returns
    -------
    tuple of (mpf, mpf)
        Coefficients for (h_{m+2,n-2}, h_{m+1,n-2}).
    """
    prefactor = n - 1
    coeff_mp2 = mpf(prefactor)
    coeff_mp1 = mpf(-prefactor * (d - 3*m - 4*n + 4))
    return (coeff_mp2, coeff_mp1)


def compute_h_mn_from_recursion(h_values: Dict[Tuple[int, int], mpf],
                                 m: int, n: int,
                                 delta: mpf, spin: int,
                                 d: int = D) -> mpf:
    """
    Compute h_{m,n} using the Casimir recursion (Eq. C.1).

    Assumes all required h values with lower n are already computed.

    The recursion (Eq. C.1):
    2(D + 2n - 3) h_{m,n} = [RHS terms]

    Parameters
    ----------
    h_values : dict
        Dictionary mapping (m, n) -> h_{m,n} for previously computed values.
    m : int
        The a-derivative order.
    n : int
        The b-derivative order (must be >= 1).
    delta : mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    d : int
        The spacetime dimension.

    Returns
    -------
    mpf
        The value h_{m,n}.
    """
    if n == 0:
        raise ValueError("Use diagonal derivatives (h_{m,0}) for n=0, not recursion")

    if n < 0 or m < 0:
        raise ValueError(f"Invalid indices m={m}, n={n}")

    # LHS coefficient
    lhs_coeff = 2 * (d + 2*n - 3)
    if abs(lhs_coeff) < mpf('1e-100'):
        raise ValueError(f"LHS coefficient is zero for n={n}, D={d}")

    rhs = mpf('0')

    # Helper to safely get h value (returns 0 if index invalid or not computed)
    def get_h(mm, nn):
        if mm < 0 or nn < 0:
            return mpf('0')
        if (mm, nn) in h_values:
            return h_values[(mm, nn)]
        # This shouldn't happen if we compute in correct order
        return mpf('0')

    # First line: m-dependent terms at level n (these contribute only for n >= 1)
    # BUT these involve h at the SAME level n, which we're computing!
    # Looking at the structure more carefully...

    # Actually, re-reading Eq. C.1, the terms h_{m-k,n} on the RHS are at the SAME
    # level n as the LHS. This means the recursion relates h_{m,n} to h_{m',n}
    # with m' < m, as well as h values at lower n.
    #
    # The first term being proportional to m means h_{0,n} is determined directly
    # from lower n values. Then h_{1,n}, h_{2,n}, etc. are built up.

    # Coefficients for same-n terms (m-dependent)
    coeff_m1, coeff_m2, coeff_m3 = casimir_recursion_coefficient_m_term(m, n, d)

    # These involve h_{m-1,n}, h_{m-2,n}, h_{m-3,n}
    # Since we compute in order of increasing m for each n, these should be available
    rhs += coeff_m1 * get_h(m - 1, n)
    rhs += coeff_m2 * get_h(m - 2, n)
    rhs += coeff_m3 * get_h(m - 3, n)

    # n-1 level terms
    coeff_mp2_n1, coeff_mp1_n1, coeff_m0_n1, coeff_mm1_n1 = \
        casimir_recursion_coefficient_n1_terms(m, n, delta, spin, d)

    rhs += coeff_mp2_n1 * get_h(m + 2, n - 1)
    rhs += coeff_mp1_n1 * get_h(m + 1, n - 1)
    rhs += coeff_m0_n1 * get_h(m, n - 1)
    rhs += coeff_mm1_n1 * get_h(m - 1, n - 1)

    # n-2 level terms (only if n >= 2)
    if n >= 2:
        coeff_mp2_n2, coeff_mp1_n2 = casimir_recursion_coefficient_n2_terms(m, n, d)
        rhs += coeff_mp2_n2 * get_h(m + 2, n - 2)
        rhs += coeff_mp1_n2 * get_h(m + 1, n - 2)

    return rhs / lhs_coeff


def compute_all_h_mn(delta: Union[float, mpf], spin: int,
                      h_m0: List[mpf],
                      n_max: int = N_MAX) -> Dict[Tuple[int, int], mpf]:
    """
    Compute all h_{m,n} for the index set using Casimir recursion.

    The index set consists of (m, n) pairs where:
    - m is odd (m = 1, 3, 5, ...)
    - n >= 0
    - m + 2n <= 2*n_max + 1

    However, the recursion requires computing h_{m,n} for ALL m (odd and even)
    because the recursion involves h_{m+2,n-1} and h_{m+1,n-1} which may have
    m values outside the final index set.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    h_m0 : list of mpf
        The diagonal derivatives [h_{0,0}, h_{1,0}, h_{2,0}, ...].
        Must have length >= max_deriv_order + 1 where max_deriv_order = 2*n_max + 1.
    n_max : int
        The truncation parameter (default 10 gives 66 index pairs).

    Returns
    -------
    dict
        Dictionary mapping (m, n) -> h_{m,n} for all pairs in the index set.
    """
    delta = mpf(delta)
    max_order = 2 * n_max + 1

    # Check h_m0 has enough entries
    if len(h_m0) < max_order + 3:
        raise ValueError(
            f"h_m0 must have at least {max_order + 3} entries, got {len(h_m0)}. "
            "Need extra entries for recursion boundary terms."
        )

    h_values = {}

    # Initialize with h_{m,0} values
    for m in range(len(h_m0)):
        h_values[(m, 0)] = h_m0[m]

    # Compute h_{m,n} for n = 1, 2, 3, ...
    # For each n, we need m values such that m + 2n <= max_order
    # But we also need m+2 at level n-1, so we compute a few extra

    max_n = (max_order - 1) // 2 + 1  # Maximum n value

    for n in range(1, max_n + 1):
        # Maximum m we need at this level (plus some buffer for recursion)
        max_m_at_n = max_order - 2*n + 4  # Buffer of 4 for safety

        # Compute h_{m,n} for m = 0, 1, 2, ..., max_m_at_n
        for m in range(max_m_at_n + 1):
            h_mn = compute_h_mn_from_recursion(h_values, m, n, delta, spin)
            h_values[(m, n)] = h_mn

    return h_values


def extract_index_set_h_values(h_values: Dict[Tuple[int, int], mpf],
                                n_max: int = N_MAX) -> Dict[Tuple[int, int], mpf]:
    """
    Extract h_{m,n} values for the valid index set only.

    The index set has m odd and m + 2n <= 2*n_max + 1.

    Parameters
    ----------
    h_values : dict
        Full dictionary of h_{m,n} values.
    n_max : int
        The truncation parameter.

    Returns
    -------
    dict
        Dictionary with only valid index set entries.
    """
    index_set = generate_index_set(n_max)
    return {(m, n): h_values[(m, n)] for (m, n) in index_set if (m, n) in h_values}


def block_derivatives_full(delta: Union[float, mpf], spin: int,
                            n_max: int = N_MAX) -> Dict[Tuple[int, int], mpf]:
    """
    Compute all block derivatives h_{m,n} for the index set.

    This is the main entry point that:
    1. Computes z-derivatives of the block
    2. Converts to diagonal a-derivatives h_{m,0}
    3. Uses Casimir recursion for n > 0

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    n_max : int
        The truncation parameter.

    Returns
    -------
    dict
        Dictionary mapping (m, n) -> h_{m,n} for all pairs in the index set.

    Examples
    --------
    >>> h = block_derivatives_full(mpf('1.5'), 0, n_max=2)
    >>> len(h)  # Should be 6 for n_max=2
    6
    """
    from .coordinate_transform import compute_h_m0_from_block_derivs

    delta = mpf(delta)
    max_order = 2 * n_max + 1

    # Compute h_{m,0} with extra buffer for recursion
    # Need at least max_order + 3 entries
    h_m0 = compute_h_m0_from_block_derivs(delta, spin, max_order + 4)

    # Compute all h_{m,n} using Casimir recursion
    h_all = compute_all_h_mn(delta, spin, h_m0, n_max)

    # Extract only index set values
    h_index_set = extract_index_set_h_values(h_all, n_max)

    return h_index_set


def block_derivatives_as_vector(delta: Union[float, mpf], spin: int,
                                 n_max: int = N_MAX) -> List[mpf]:
    """
    Compute block derivatives as a flat vector in standard index order.

    The ordering follows generate_index_set(): (1,0), (1,1), ..., (3,0), (3,1), ...

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    n_max : int
        The truncation parameter.

    Returns
    -------
    list of mpf
        Flat vector of h_{m,n} values in index set order.
    """
    h_dict = block_derivatives_full(delta, spin, n_max)
    index_set = generate_index_set(n_max)

    return [h_dict[(m, n)] for (m, n) in index_set]
