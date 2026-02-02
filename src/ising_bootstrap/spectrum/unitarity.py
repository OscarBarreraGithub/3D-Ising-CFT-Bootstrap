"""
Unitarity bounds for operators in D=3 CFT.

In D=3 dimensions, unitarity constrains the scaling dimensions:
- Scalars (l=0): Δ ≥ 1/2
- Spinning operators (l≥1): Δ ≥ l + 1

These bounds ensure reflection positivity of the CFT.

Reference: arXiv:1203.6064, Section 1.3
"""

from typing import Union
from ..config import D


# Unitarity bound formula: Δ_min = l + D - 2 for l ≥ 1, and (D-2)/2 for l = 0
# For D = 3: Δ_min = l + 1 for l ≥ 1, and 0.5 for l = 0


def unitarity_bound(spin: int) -> float:
    """
    Compute the unitarity bound Δ_min(l) for an operator with given spin.

    In D=3:
    - Scalars (l=0): Δ ≥ 1/2
    - Spinning (l≥1): Δ ≥ l + 1

    Parameters
    ----------
    spin : int
        The spin of the operator. Must be non-negative.

    Returns
    -------
    float
        The minimum allowed scaling dimension.

    Raises
    ------
    ValueError
        If spin is negative.

    Examples
    --------
    >>> unitarity_bound(0)
    0.5
    >>> unitarity_bound(2)
    3.0
    >>> unitarity_bound(4)
    5.0
    """
    if spin < 0:
        raise ValueError(f"Spin must be non-negative, got {spin}")

    if spin == 0:
        return (D - 2) / 2  # = 0.5 for D=3
    else:
        return float(spin + D - 2)  # = l + 1 for D=3


def satisfies_unitarity(delta: float, spin: int, strict: bool = False) -> bool:
    """
    Check if a scaling dimension satisfies the unitarity bound.

    Parameters
    ----------
    delta : float
        The scaling dimension to check.
    spin : int
        The spin of the operator.
    strict : bool, optional
        If True, require Δ > bound (strict inequality).
        If False (default), require Δ ≥ bound.

    Returns
    -------
    bool
        True if the dimension is consistent with unitarity.

    Examples
    --------
    >>> satisfies_unitarity(0.5, 0)
    True
    >>> satisfies_unitarity(0.4, 0)
    False
    >>> satisfies_unitarity(3.0, 2)
    True
    >>> satisfies_unitarity(3.0, 2, strict=True)
    False
    """
    bound = unitarity_bound(spin)

    if strict:
        return delta > bound
    else:
        return delta >= bound


def check_unitarity(delta: float, spin: int, tolerance: float = 1e-10) -> bool:
    """
    Check unitarity with a numerical tolerance.

    This is useful when Δ is computed numerically and may be slightly
    below the bound due to floating-point errors.

    Parameters
    ----------
    delta : float
        The scaling dimension to check.
    spin : int
        The spin of the operator.
    tolerance : float, optional
        Allow Δ ≥ bound - tolerance. Default is 1e-10.

    Returns
    -------
    bool
        True if the dimension is consistent with unitarity (within tolerance).

    Examples
    --------
    >>> check_unitarity(0.5 - 1e-12, 0)
    True
    >>> check_unitarity(0.4, 0)
    False
    """
    bound = unitarity_bound(spin)
    return delta >= bound - tolerance


def is_allowed_spin(spin: int) -> bool:
    """
    Check if a spin is allowed in the σ×σ OPE.

    For identical scalar correlator, only even spins appear due to Bose symmetry.

    Parameters
    ----------
    spin : int
        The spin to check.

    Returns
    -------
    bool
        True if spin is allowed (non-negative and even).

    Examples
    --------
    >>> is_allowed_spin(0)
    True
    >>> is_allowed_spin(2)
    True
    >>> is_allowed_spin(3)
    False
    >>> is_allowed_spin(-1)
    False
    """
    return spin >= 0 and spin % 2 == 0


def validate_operator(delta: float, spin: int, tolerance: float = 1e-10) -> bool:
    """
    Validate that an operator (Δ, l) is allowed in the σ×σ OPE.

    Checks both:
    1. Spin is even (Bose symmetry)
    2. Dimension satisfies unitarity bound

    Parameters
    ----------
    delta : float
        The scaling dimension.
    spin : int
        The spin.
    tolerance : float, optional
        Numerical tolerance for unitarity check. Default is 1e-10.

    Returns
    -------
    bool
        True if the operator is allowed.

    Examples
    --------
    >>> validate_operator(1.5, 0)
    True
    >>> validate_operator(1.5, 1)  # Odd spin not allowed
    False
    >>> validate_operator(0.4, 0)  # Below unitarity bound
    False
    """
    if not is_allowed_spin(spin):
        return False
    if not check_unitarity(delta, spin, tolerance):
        return False
    return True
