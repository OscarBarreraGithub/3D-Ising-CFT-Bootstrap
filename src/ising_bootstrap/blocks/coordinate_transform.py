"""
Coordinate transformation between (z, z̄) and (a, b) variables.

The crossing-symmetric point z = z̄ = 1/2 corresponds to a = 1, b = 0.
At the diagonal (z = z̄), the coordinate transformation simplifies:
    z = (a + √b)/2,  z̄ = (a - √b)/2
    ⟹ z = z̄ = a/2 when b = 0

This module provides:
1. Coordinate conversion functions
2. Derivative transformation: h_{m,n} = ∂_a^m ∂_b^n G|_{a=1,b=0}
3. Relation between z-derivatives and a-derivatives along the diagonal

Reference: arXiv:1203.6064, Section 4 and Eq. 4.15
"""

from mpmath import mp, mpf, sqrt
from typing import Union, List

from ..config import Z_POINT, MPMATH_PRECISION, MAX_DERIV_ORDER

# Set precision
mp.dps = MPMATH_PRECISION


def z_zbar_to_a_b(z: mpf, zbar: mpf) -> tuple:
    """
    Convert (z, z̄) coordinates to (a, b) coordinates.

    Inverse of the transformation:
        z = (a + √b)/2
        z̄ = (a - √b)/2

    Solving:
        a = z + z̄
        √b = z - z̄  ⟹  b = (z - z̄)²

    Parameters
    ----------
    z : mpf
        The z coordinate.
    zbar : mpf
        The z̄ coordinate.

    Returns
    -------
    tuple of (mpf, mpf)
        (a, b) coordinates.

    Examples
    --------
    >>> z = zbar = mpf('0.5')
    >>> a, b = z_zbar_to_a_b(z, zbar)
    >>> float(a), float(b)
    (1.0, 0.0)
    """
    a = z + zbar
    b = (z - zbar) ** 2
    return (a, b)


def a_b_to_z_zbar(a: mpf, b: mpf) -> tuple:
    """
    Convert (a, b) coordinates to (z, z̄) coordinates.

    Transformation (Eq. 4.15):
        z = (a + √b)/2
        z̄ = (a - √b)/2

    Parameters
    ----------
    a : mpf
        The a coordinate.
    b : mpf
        The b coordinate (must be non-negative).

    Returns
    -------
    tuple of (mpf, mpf)
        (z, z̄) coordinates.

    Examples
    --------
    >>> a, b = mpf('1'), mpf('0')
    >>> z, zbar = a_b_to_z_zbar(a, b)
    >>> float(z), float(zbar)
    (0.5, 0.5)
    """
    sqrt_b = sqrt(b)
    z = (a + sqrt_b) / 2
    zbar = (a - sqrt_b) / 2
    return (z, zbar)


def z_zbar_to_u_v(z: mpf, zbar: mpf) -> tuple:
    """
    Convert (z, z̄) to conformal cross-ratios (u, v).

    u = z z̄
    v = (1-z)(1-z̄)

    Parameters
    ----------
    z : mpf
        The z coordinate.
    zbar : mpf
        The z̄ coordinate.

    Returns
    -------
    tuple of (mpf, mpf)
        (u, v) cross-ratios.

    Examples
    --------
    >>> z = zbar = mpf('0.5')
    >>> u, v = z_zbar_to_u_v(z, zbar)
    >>> float(u), float(v)
    (0.25, 0.25)
    """
    u = z * zbar
    v = (1 - z) * (1 - zbar)
    return (u, v)


def diagonal_a_derivative_to_z_derivative(m: int) -> mpf:
    """
    Compute the factor relating a-derivatives to z-derivatives along the diagonal.

    At b = 0 (the diagonal z = z̄), we have:
        z = a/2
        ∂_z = 2 ∂_a
        ⟹ ∂_a = (1/2) ∂_z
        ⟹ ∂_a^m = (1/2)^m ∂_z^m

    So: h_{m,0} = ∂_a^m G|_{a=1,b=0} = (1/2)^m × ∂_z^m G|_{z=1/2}

    Parameters
    ----------
    m : int
        The derivative order.

    Returns
    -------
    mpf
        The factor (1/2)^m.
    """
    return mpf('0.5') ** m


def z_derivatives_to_h_m0(z_derivs: List[mpf]) -> List[mpf]:
    """
    Convert z-derivatives at z=1/2 to diagonal a-derivatives h_{m,0}.

    h_{m,0} = ∂_a^m G|_{a=1,b=0} = (1/2)^m × ∂_z^m G|_{z=1/2}

    Parameters
    ----------
    z_derivs : list of mpf
        [G(z), G'(z), G''(z), ...] evaluated at z = 1/2.

    Returns
    -------
    list of mpf
        [h_{0,0}, h_{1,0}, h_{2,0}, ...] = [G, (1/2)G', (1/4)G'', ...]

    Examples
    --------
    >>> z_derivs = [mpf('1'), mpf('2'), mpf('4')]
    >>> h_m0 = z_derivatives_to_h_m0(z_derivs)
    >>> [float(h) for h in h_m0]
    [1.0, 1.0, 1.0]
    """
    h_m0 = []
    for m, d_m in enumerate(z_derivs):
        factor = diagonal_a_derivative_to_z_derivative(m)
        h_m0.append(factor * d_m)
    return h_m0


def h_m0_to_z_derivatives(h_m0: List[mpf]) -> List[mpf]:
    """
    Convert diagonal a-derivatives h_{m,0} to z-derivatives at z=1/2.

    This is the inverse of z_derivatives_to_h_m0.

    ∂_z^m G|_{z=1/2} = 2^m × h_{m,0}

    Parameters
    ----------
    h_m0 : list of mpf
        [h_{0,0}, h_{1,0}, h_{2,0}, ...]

    Returns
    -------
    list of mpf
        [G(z), G'(z), G''(z), ...] evaluated at z = 1/2.
    """
    z_derivs = []
    for m, h in enumerate(h_m0):
        factor = mpf('2') ** m
        z_derivs.append(factor * h)
    return z_derivs


def crossing_point_values() -> dict:
    """
    Return the standard crossing-symmetric point values.

    At z = z̄ = 1/2:
        - a = 1, b = 0
        - u = v = 1/4

    Returns
    -------
    dict
        Dictionary with coordinate values at the crossing-symmetric point.
    """
    z = zbar = mpf(Z_POINT)
    a, b = z_zbar_to_a_b(z, zbar)
    u, v = z_zbar_to_u_v(z, zbar)

    return {
        'z': z,
        'zbar': zbar,
        'a': a,
        'b': b,
        'u': u,
        'v': v
    }


def compute_h_m0_from_block_derivs(delta: Union[float, mpf], spin: int,
                                    max_m: int = MAX_DERIV_ORDER) -> List[mpf]:
    """
    Compute h_{m,0} = ∂_a^m G_{Δ,l}|_{a=1,b=0} for m = 0, 1, ..., max_m.

    This is a convenience function that:
    1. Computes z-derivatives using block_z_derivatives
    2. Converts to a-derivatives using the (1/2)^m factor

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    max_m : int
        Maximum a-derivative order.

    Returns
    -------
    list of mpf
        [h_{0,0}, h_{1,0}, ..., h_{max_m,0}]
    """
    from .z_derivatives import block_z_derivatives

    delta = mpf(delta)

    # Get z-derivatives at z = 1/2
    z_derivs = block_z_derivatives(delta, spin, mpf(Z_POINT), max_m)

    # Convert to a-derivatives
    h_m0 = z_derivatives_to_h_m0(z_derivs)

    return h_m0
