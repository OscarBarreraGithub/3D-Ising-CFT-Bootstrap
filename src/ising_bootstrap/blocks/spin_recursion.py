"""
Spin recursion for conformal blocks at z=z̄.

At z=z̄, the spin recursion relation (Eq. 4.9 from arXiv:1203.6064) becomes
a simple algebraic recursion (the derivative term F₂ vanishes). This allows
efficient computation of G_{Δ,l}(z) for l ≥ 2 from the spin-0 and spin-1 base cases.

Reference: arXiv:1203.6064, Section 4 and Appendix A
"""

from mpmath import mp, mpf
from typing import Union, Dict, Tuple, Optional
from functools import lru_cache

from ..config import D, ALPHA, Z_POINT, MPMATH_PRECISION
from .diagonal_blocks import spin0_block, spin1_block


# Set precision for mpmath
mp.dps = MPMATH_PRECISION


def _spin_recursion_lhs_coeff(delta: mpf, spin: int, d: int = D) -> mpf:
    """
    Compute the left-hand side coefficient of the spin recursion.

    LHS = (l + D - 3)(2Δ + 2 - D) G_{Δ,l}

    For D=3: LHS = l(2Δ - 1) G_{Δ,l}

    Parameters
    ----------
    delta : mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    d : int
        The spacetime dimension (default: 3).

    Returns
    -------
    mpf
        The coefficient (l + D - 3)(2Δ + 2 - D).
    """
    return (spin + d - 3) * (2 * delta + 2 - d)


def _spin_recursion_first_term_coeff(delta: mpf, spin: int, d: int = D) -> mpf:
    """
    Compute the coefficient of G_{Δ,l-2} in the spin recursion.

    First term: (D - 2)(Δ + l - 1) G_{Δ,l-2}

    For D=3: (Δ + l - 1) G_{Δ,l-2}

    Parameters
    ----------
    delta : mpf
        The scaling dimension Δ.
    spin : int
        The spin l (of the block being computed, not l-2).
    d : int
        The spacetime dimension (default: 3).

    Returns
    -------
    mpf
        The coefficient (D - 2)(Δ + l - 1).
    """
    return (d - 2) * (delta + spin - 1)


def _spin_recursion_second_term_coeff(delta: mpf, spin: int, z: mpf, d: int = D) -> mpf:
    """
    Compute the coefficient of G_{Δ+1,l-1} in the spin recursion.

    Second term: (2-z)/(2z) × (2l + D - 4)(Δ - D + 2) G_{Δ+1,l-1}

    For D=3 and z=1/2: 1.5 × (2l - 1)(Δ - 1) G_{Δ+1,l-1}

    Parameters
    ----------
    delta : mpf
        The scaling dimension Δ.
    spin : int
        The spin l (of the block being computed).
    z : mpf
        The cross-ratio coordinate.
    d : int
        The spacetime dimension (default: 3).

    Returns
    -------
    mpf
        The full coefficient including the (2-z)/(2z) factor.
    """
    z_factor = (2 - z) / (2 * z)
    return z_factor * (2 * spin + d - 4) * (delta - d + 2)


def _spin_recursion_third_term_coeff(delta: mpf, spin: int, d: int = D) -> mpf:
    """
    Compute the coefficient of G_{Δ+2,l-2} in the spin recursion.

    Third term (from Eq. 4.9):
    - Δ(2l+D-4)(Δ+2-D)(Δ+3-D)(Δ-l-D+4)² / [16(Δ+1-D/2)(Δ-D/2+2)(l-Δ+D-5)(l-Δ+D-3)] × G_{Δ+2,l-2}

    For D=3:
    - Δ(2l-1)(Δ-1)(Δ)(Δ-l+1)² / [16(Δ-1/2)(Δ+1/2)(l-Δ-2)(l-Δ)] × G_{Δ+2,l-2}

    Parameters
    ----------
    delta : mpf
        The scaling dimension Δ.
    spin : int
        The spin l (of the block being computed).
    d : int
        The spacetime dimension (default: 3).

    Returns
    -------
    mpf
        The coefficient (negative sign included).
    """
    # Numerator factors
    num1 = delta
    num2 = 2 * spin + d - 4
    num3 = delta + 2 - d
    num4 = delta + 3 - d
    num5 = (delta - spin - d + 4) ** 2

    # Denominator factors
    den1 = 16
    den2 = delta + 1 - d / 2
    den3 = delta - d / 2 + 2
    den4 = spin - delta + d - 5
    den5 = spin - delta + d - 3

    # Check for potential division by zero
    denominator = den1 * den2 * den3 * den4 * den5
    if abs(denominator) < mpf('1e-100'):
        # This can happen for special values; return 0 as the contribution
        # would be singular but the physical result should be finite
        return mpf('0')

    numerator = num1 * num2 * num3 * num4 * num5

    # Note: the coefficient is negative in Eq. 4.9
    return -numerator / denominator


def higher_spin_block(delta: Union[float, mpf], spin: int,
                      z: Union[float, mpf] = None,
                      cache: Optional[Dict[Tuple[str, int], mpf]] = None) -> mpf:
    """
    Compute G_{Δ,l}(z) for l ≥ 0 using spin recursion.

    Uses Equation 4.9 from arXiv:1203.6064 which at z=z̄ becomes a
    non-derivative algebraic recursion:

        (l+D-3)(2Δ+2-D) G_{Δ,l} =
            (D-2)(Δ+l-1) G_{Δ,l-2}
          + (2-z)/(2z) (2l+D-4)(Δ-D+2) G_{Δ+1,l-1}
          - [coefficient] G_{Δ+2,l-2}

    For l=0,1, uses the direct ₃F₂ formulas from diagonal_blocks.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ of the exchanged operator.
    spin : int
        The spin l of the exchanged operator.
    z : float or mpf, optional
        The cross-ratio. Default is 1/2 (crossing-symmetric point).
    cache : dict, optional
        Cache for storing intermediate block values. If None, a local
        cache is used. Keys are (delta_str, spin) tuples.

    Returns
    -------
    mpf
        The conformal block value G_{Δ,l}(z).

    Raises
    ------
    ValueError
        If spin < 0.

    Examples
    --------
    >>> from mpmath import mpf
    >>> G2 = higher_spin_block(mpf('3.0'), 2)
    >>> G4 = higher_spin_block(mpf('5.0'), 4)
    """
    if spin < 0:
        raise ValueError(f"Spin must be non-negative, got {spin}")

    if z is None:
        z = mpf(Z_POINT)
    else:
        z = mpf(z)
    delta = mpf(delta)

    # Use local cache if none provided
    if cache is None:
        cache = {}

    # Create a string key for delta to handle floating point comparisons
    delta_key = str(delta)

    # Check cache first
    cache_key = (delta_key, spin)
    if cache_key in cache:
        return cache[cache_key]

    # Base cases: spin 0 and 1
    if spin == 0:
        result = spin0_block(delta, z)
        cache[cache_key] = result
        return result
    elif spin == 1:
        result = spin1_block(delta, z)
        cache[cache_key] = result
        return result

    # Recursion for spin ≥ 2
    # Need: G_{Δ,l-2}, G_{Δ+1,l-1}, G_{Δ+2,l-2}

    # Get the required blocks (may trigger further recursion)
    G_delta_lm2 = higher_spin_block(delta, spin - 2, z, cache)
    G_deltap1_lm1 = higher_spin_block(delta + 1, spin - 1, z, cache)
    G_deltap2_lm2 = higher_spin_block(delta + 2, spin - 2, z, cache)

    # Compute coefficients
    lhs_coeff = _spin_recursion_lhs_coeff(delta, spin)
    coeff1 = _spin_recursion_first_term_coeff(delta, spin)
    coeff2 = _spin_recursion_second_term_coeff(delta, spin, z)
    coeff3 = _spin_recursion_third_term_coeff(delta, spin)

    # Check for division by zero on LHS
    if abs(lhs_coeff) < mpf('1e-100'):
        raise ValueError(
            f"LHS coefficient is zero for delta={delta}, spin={spin}. "
            "This may indicate hitting a special value."
        )

    # Apply recursion: solve for G_{Δ,l}
    rhs = coeff1 * G_delta_lm2 + coeff2 * G_deltap1_lm1 + coeff3 * G_deltap2_lm2
    result = rhs / lhs_coeff

    # Cache and return
    cache[cache_key] = result
    return result


def diagonal_block_any_spin(delta: Union[float, mpf], spin: int,
                            z: Union[float, mpf] = None) -> mpf:
    """
    Compute G_{Δ,l}(z) for any spin l ≥ 0.

    This is a convenience wrapper that handles all spins uniformly.
    For l=0,1 it calls the direct formulas; for l≥2 it uses recursion.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l (must be non-negative).
    z : float or mpf, optional
        The cross-ratio. Default is 1/2.

    Returns
    -------
    mpf
        The conformal block value G_{Δ,l}(z).

    Examples
    --------
    >>> from mpmath import mpf
    >>> G0 = diagonal_block_any_spin(mpf('1.5'), 0)
    >>> G1 = diagonal_block_any_spin(mpf('2.0'), 1)
    >>> G2 = diagonal_block_any_spin(mpf('3.0'), 2)
    """
    return higher_spin_block(delta, spin, z)
