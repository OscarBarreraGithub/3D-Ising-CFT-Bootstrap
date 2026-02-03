"""
Diagonal conformal blocks G_{Δ,l}(z) evaluated at z=z̄.

At z=z̄ (the diagonal), the conformal block simplifies significantly.
The spin-0 and spin-1 blocks serve as base cases, and higher spins
are obtained via recursion (see spin_recursion.py).

This module implements Equations 4.10 and 4.11 from arXiv:1203.6064.

Reference: arXiv:1203.6064, Section 4.2
"""

from mpmath import mp, mpf, hyp3f2, power, log
from typing import Union

from ..config import D, ALPHA, Z_POINT, MPMATH_PRECISION


# Set precision for mpmath
mp.dps = MPMATH_PRECISION


def _hyp3f2_argument(z: mpf) -> mpf:
    """
    Compute the ₃F₂ argument: z²/(4(z-1)).

    At z=1/2: returns -1/8 = -0.125

    Parameters
    ----------
    z : mpf
        The cross-ratio coordinate.

    Returns
    -------
    mpf
        The hypergeometric argument.
    """
    return z**2 / (4 * (z - 1))


def _prefactor_base(z: mpf) -> mpf:
    """
    Compute the base prefactor z²/(1-z).

    At z=1/2: returns -1/2 = -0.5

    This is raised to power Δ/2 for spin-0 or (Δ+1)/2 for spin-1.

    Parameters
    ----------
    z : mpf
        The cross-ratio coordinate.

    Returns
    -------
    mpf
        The base prefactor.
    """
    return z**2 / (1 - z)


def spin0_block(delta: Union[float, mpf], z: Union[float, mpf] = None) -> mpf:
    """
    Compute the spin-0 conformal block G_{Δ,0}(z) at z=z̄.

    Uses Equation 4.10 from arXiv:1203.6064:

        G_{Δ,0}(z) = (z²/(1-z))^{Δ/2} × ₃F₂(Δ/2, Δ/2, Δ/2-α; (Δ+1)/2, Δ-α; x)

    where x = z²/(4(z-1)) and α = D/2 - 1 = 0.5 for D=3.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ of the exchanged operator.
    z : float or mpf, optional
        The cross-ratio. Default is 1/2 (crossing-symmetric point).

    Returns
    -------
    mpf
        The conformal block value G_{Δ,0}(z).

    Notes
    -----
    At z=1/2, the ₃F₂ argument is -1/8, which is inside the convergence
    radius |x| < 1. The prefactor (z²/(1-z))^{Δ/2} = (-1/2)^{Δ/2} requires
    careful handling of the complex branch.

    For real Δ and z=1/2, the result is real because we use the principal
    branch of the power function.

    Examples
    --------
    >>> from mpmath import mpf
    >>> G = spin0_block(mpf('1.5'))
    >>> float(G)  # doctest: +SKIP
    -0.3535...
    """
    if z is None:
        z = mpf(Z_POINT)
    else:
        z = mpf(z)
    delta = mpf(delta)
    alpha = mpf(ALPHA)

    # Compute the ₃F₂ argument
    x = _hyp3f2_argument(z)

    # Compute the prefactor: (z²/(1-z))^{Δ/2}
    base = _prefactor_base(z)
    prefactor = power(base, delta / 2)

    # ₃F₂ parameters for spin-0 (Eq. 4.10)
    # Upper parameters: (Δ/2, Δ/2, Δ/2 - α)
    # Lower parameters: ((Δ+1)/2, Δ - α)
    a1 = delta / 2
    a2 = delta / 2
    a3 = delta / 2 - alpha
    b1 = (delta + 1) / 2
    b2 = delta - alpha

    # Evaluate the hypergeometric function
    hyp_value = hyp3f2(a1, a2, a3, b1, b2, x)

    return prefactor * hyp_value


def spin1_block(delta: Union[float, mpf], z: Union[float, mpf] = None) -> mpf:
    """
    Compute the spin-1 conformal block G_{Δ,1}(z) at z=z̄.

    Uses Equation 4.11 from arXiv:1203.6064:

        G_{Δ,1}(z) = (2-z)/(2z) × (z²/(1-z))^{(Δ+1)/2}
                     × ₃F₂((Δ+1)/2, (Δ+1)/2, (Δ+1)/2-α; (Δ+2)/2, Δ+1-α; x)

    where x = z²/(4(z-1)) and α = D/2 - 1 = 0.5 for D=3.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ of the exchanged operator.
    z : float or mpf, optional
        The cross-ratio. Default is 1/2 (crossing-symmetric point).

    Returns
    -------
    mpf
        The conformal block value G_{Δ,1}(z).

    Notes
    -----
    For spin-1 operators in D=3, the unitarity bound requires Δ ≥ 2.

    Examples
    --------
    >>> from mpmath import mpf
    >>> G = spin1_block(mpf('2.0'))
    >>> abs(float(G)) > 0  # Non-zero
    True
    """
    if z is None:
        z = mpf(Z_POINT)
    else:
        z = mpf(z)
    delta = mpf(delta)
    alpha = mpf(ALPHA)

    # Compute the ₃F₂ argument
    x = _hyp3f2_argument(z)

    # Additional spin-1 prefactor: (2-z)/(2z)
    # At z=1/2: (2 - 0.5)/(2 * 0.5) = 1.5/1 = 1.5
    spin1_factor = (2 - z) / (2 * z)

    # Compute the base prefactor: (z²/(1-z))^{(Δ+1)/2}
    base = _prefactor_base(z)
    prefactor = power(base, (delta + 1) / 2)

    # ₃F₂ parameters for spin-1 (Eq. 4.11)
    # Upper parameters: ((Δ+1)/2, (Δ+1)/2, (Δ+1)/2 - α)
    # Lower parameters: ((Δ+2)/2, Δ+1 - α)
    a1 = (delta + 1) / 2
    a2 = (delta + 1) / 2
    a3 = (delta + 1) / 2 - alpha
    b1 = (delta + 2) / 2
    b2 = delta + 1 - alpha

    # Evaluate the hypergeometric function
    hyp_value = hyp3f2(a1, a2, a3, b1, b2, x)

    return spin1_factor * prefactor * hyp_value


def diagonal_block(delta: Union[float, mpf], spin: int,
                   z: Union[float, mpf] = None) -> mpf:
    """
    Compute the diagonal conformal block G_{Δ,l}(z) at z=z̄.

    This is the main entry point for computing conformal blocks.
    For spin 0 and 1, it uses the direct ₃F₂ formulas.
    For spin ≥ 2, use the spin_recursion module instead.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ of the exchanged operator.
    spin : int
        The spin l of the exchanged operator (0 or 1 for direct evaluation).
    z : float or mpf, optional
        The cross-ratio. Default is 1/2 (crossing-symmetric point).

    Returns
    -------
    mpf
        The conformal block value G_{Δ,l}(z).

    Raises
    ------
    ValueError
        If spin < 0 or spin > 1 (use spin_recursion for higher spins).

    Examples
    --------
    >>> from mpmath import mpf
    >>> G0 = diagonal_block(mpf('1.5'), 0)
    >>> G1 = diagonal_block(mpf('2.0'), 1)
    """
    if spin < 0:
        raise ValueError(f"Spin must be non-negative, got {spin}")
    if spin == 0:
        return spin0_block(delta, z)
    elif spin == 1:
        return spin1_block(delta, z)
    else:
        raise ValueError(
            f"Direct evaluation only supports spin 0 and 1, got spin={spin}. "
            "Use spin_recursion.higher_spin_block() for spin ≥ 2."
        )
