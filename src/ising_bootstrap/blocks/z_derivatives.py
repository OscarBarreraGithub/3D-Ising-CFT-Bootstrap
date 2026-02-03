"""
Z-derivatives of conformal blocks at z=z̄.

This module computes d^m/dz^m G_{Δ,l}(z)|_{z=1/2} using:
1. Direct computation for m=0,1,2 from ₃F₂ representation
2. The ₃F₂ differential equation (Eq. 4.12) for m≥3

The ₃F₂ ODE allows expressing d³/dx³ and higher derivatives in terms of
lower derivatives, which combined with chain rule gives z-derivatives.

Reference: arXiv:1203.6064, Eq. 4.12
"""

from mpmath import mp, mpf, hyp3f2, power, diff
from typing import Union, List, Dict, Tuple, Optional
from functools import lru_cache
import numpy as np

from ..config import D, ALPHA, Z_POINT, MPMATH_PRECISION, MAX_DERIV_ORDER

# Set precision for mpmath
mp.dps = MPMATH_PRECISION


def _hyp3f2_argument(z: mpf) -> mpf:
    """
    Compute the ₃F₂ argument: x = z²/(4(z-1)).

    At z=1/2: x = (1/4)/(4(-1/2)) = -1/8
    """
    return z**2 / (4 * (z - 1))


def _hyp3f2_argument_derivatives(z: mpf, max_order: int) -> List[mpf]:
    """
    Compute derivatives of x(z) = z²/(4(z-1)) up to given order.

    d^n x / dz^n evaluated at z.

    Returns list where index n gives d^n x/dz^n.
    """
    # x = z²/(4(z-1)) = (1/4) × z² / (z-1)
    # Using quotient rule repeatedly

    derivs = [mpf('0')] * (max_order + 1)

    # x(z) = z²/(4(z-1))
    derivs[0] = z**2 / (4 * (z - 1))

    if max_order >= 1:
        # x'(z) = (2z(z-1) - z²) / (4(z-1)²) = (z²-2z) / (4(z-1)²)
        derivs[1] = (z**2 - 2*z) / (4 * (z - 1)**2)

    if max_order >= 2:
        # x''(z) computed via quotient rule
        # Numerator: 2(z-1)(z²-2z) - (z²-2z)×2(z-1) / (z-1)⁴ ... complex
        # Use numerical differentiation for higher orders
        def x_func(zz):
            return zz**2 / (4 * (zz - 1))

        for n in range(2, max_order + 1):
            # Use mpmath's numerical differentiation
            derivs[n] = diff(x_func, z, n)

    return derivs


def _prefactor_spin0(z: mpf, delta: mpf) -> mpf:
    """
    Compute the spin-0 prefactor: (z²/(1-z))^{Δ/2}.
    """
    base = z**2 / (1 - z)
    return power(base, delta / 2)


def _prefactor_spin1(z: mpf, delta: mpf) -> mpf:
    """
    Compute the spin-1 prefactor: (2-z)/(2z) × (z²/(1-z))^{(Δ+1)/2}.
    """
    spin1_factor = (2 - z) / (2 * z)
    base = z**2 / (1 - z)
    return spin1_factor * power(base, (delta + 1) / 2)


def _compute_prefactor_derivatives_spin0(z: mpf, delta: mpf, max_order: int) -> List[mpf]:
    """
    Compute d^n/dz^n [(z²/(1-z))^{Δ/2}] for n = 0, 1, ..., max_order.

    Uses numerical differentiation from mpmath.
    """
    def prefactor_func(zz):
        base = zz**2 / (1 - zz)
        return power(base, delta / 2)

    derivs = []
    for n in range(max_order + 1):
        if n == 0:
            derivs.append(prefactor_func(z))
        else:
            derivs.append(diff(prefactor_func, z, n))

    return derivs


def _compute_prefactor_derivatives_spin1(z: mpf, delta: mpf, max_order: int) -> List[mpf]:
    """
    Compute d^n/dz^n [(2-z)/(2z) × (z²/(1-z))^{(Δ+1)/2}] for n = 0, 1, ..., max_order.

    Uses numerical differentiation from mpmath.
    """
    def prefactor_func(zz):
        spin1_factor = (2 - zz) / (2 * zz)
        base = zz**2 / (1 - zz)
        return spin1_factor * power(base, (delta + 1) / 2)

    derivs = []
    for n in range(max_order + 1):
        if n == 0:
            derivs.append(prefactor_func(z))
        else:
            derivs.append(diff(prefactor_func, z, n))

    return derivs


def _hyp3f2_derivatives_at_x(a1: mpf, a2: mpf, a3: mpf,
                              b1: mpf, b2: mpf, x: mpf,
                              max_order: int) -> List[mpf]:
    """
    Compute d^n/dx^n ₃F₂(a1,a2,a3;b1,b2;x) for n = 0, 1, ..., max_order.

    Uses the standard derivative formula:
    d/dx ₃F₂(a1,a2,a3;b1,b2;x) = (a1×a2×a3)/(b1×b2) × ₃F₂(a1+1,a2+1,a3+1;b1+1,b2+1;x)

    And the ₃F₂ ODE (Eq. 4.12) for efficiency at high orders:
    (xD̂_{a1}D̂_{a2}D̂_{a3} - D̂_0D̂_{b1-1}D̂_{b2-1}) f = 0
    where D̂_c = x∂_x + c
    """
    derivs = [mpf('0')] * (max_order + 1)

    # n=0: direct evaluation
    derivs[0] = hyp3f2(a1, a2, a3, b1, b2, x)

    if max_order >= 1:
        # n=1: use derivative formula
        # d/dx ₃F₂ = (a1×a2×a3)/(b1×b2) × ₃F₂(a1+1,a2+1,a3+1;b1+1,b2+1;x)
        coeff = (a1 * a2 * a3) / (b1 * b2)
        derivs[1] = coeff * hyp3f2(a1+1, a2+1, a3+1, b1+1, b2+1, x)

    if max_order >= 2:
        # n=2: apply derivative formula again
        coeff1 = (a1 * a2 * a3) / (b1 * b2)
        coeff2 = ((a1+1) * (a2+1) * (a3+1)) / ((b1+1) * (b2+1))
        derivs[2] = coeff1 * coeff2 * hyp3f2(a1+2, a2+2, a3+2, b1+2, b2+2, x)

    # For n >= 3, use the ODE recursion
    # The ₃F₂ ODE can be written as a third-order ODE that expresses
    # d³f/dx³ in terms of f, df/dx, d²f/dx².
    #
    # The standard ₃F₂ ODE is:
    # x(x∂³ + (b1+b2+1)∂² - ((a1+a2+a3)x - b1b2 + (a1+a2+a3+1))∂ - a1a2a3) f =
    # x(1-x)∂³f + [c - (a+b+1)x]∂²f - ab∂f = 0  (for ₂F₁)
    #
    # For ₃F₂, the ODE is more complex. We use the general recursion:
    # At each order n >= 3, apply d/dx to the (n-1)th derivative formula.

    if max_order >= 3:
        # Continue using the derivative formula iteratively
        for n in range(3, max_order + 1):
            # d^n/dx^n ₃F₂(a1,a2,a3;b1,b2;x) =
            # Π_{k=0}^{n-1} [(a1+k)(a2+k)(a3+k)/((b1+k)(b2+k))] × ₃F₂(a1+n,a2+n,a3+n;b1+n,b2+n;x)
            coeff = mpf('1')
            for k in range(n):
                coeff *= ((a1+k) * (a2+k) * (a3+k)) / ((b1+k) * (b2+k))
            derivs[n] = coeff * hyp3f2(a1+n, a2+n, a3+n, b1+n, b2+n, x)

    return derivs


def _binomial(n: int, k: int) -> int:
    """Compute binomial coefficient C(n,k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    # Use symmetry
    if k > n - k:
        k = n - k

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)

    return result


def _faa_di_bruno(f_derivs: List[mpf], g_derivs: List[mpf], n: int) -> mpf:
    """
    Compute d^n/dz^n [f(g(z))] using Faà di Bruno's formula.

    Given derivatives f^(k) evaluated at g(z) and derivatives g^(k) at z,
    compute the n-th derivative of the composition.

    For our case: f is ₃F₂ (function of x), g is x(z).

    This uses a recursive approach for efficiency.

    Parameters
    ----------
    f_derivs : list
        f^(0), f^(1), ..., f^(n) evaluated at x = g(z)
    g_derivs : list
        g^(0), g^(1), ..., g^(n) evaluated at z
        Note: g_derivs[0] is g(z), g_derivs[1] is g'(z), etc.
    n : int
        The derivative order to compute

    Returns
    -------
    mpf
        d^n/dz^n [f(g(z))]
    """
    if n == 0:
        return f_derivs[0]

    if n == 1:
        # Chain rule: d/dz f(g(z)) = f'(g(z)) × g'(z)
        return f_derivs[1] * g_derivs[1]

    # For n >= 2, use the recursive formula:
    # (f∘g)^(n) = Σ_{k=1}^{n} f^(k)(g) × B_{n,k}(g', g'', ..., g^(n-k+1))
    # where B_{n,k} are Bell polynomials
    #
    # Alternatively, use the explicit Faà di Bruno formula with partitions,
    # but that's complex. Instead, use a recursive approach.

    # Recursive formula: if h = f∘g, then
    # h^(n) = Σ_{k=0}^{n-1} C(n-1,k) × (f^(1)∘g × g^(1))^(k) × ... (not simple)

    # Use a more direct approach: tabulate (f∘g)^(m) for m=0,1,...,n
    # using the recurrence based on the identity:
    # d/dz [(f∘g)^(n-1)] = (f∘g)^(n)

    # Actually, let's use the explicit Bell polynomial version
    # or numerical differentiation as a fallback

    # For simplicity and reliability, we'll compute using the explicit
    # partial Bell polynomial formula with dynamic programming

    result = mpf('0')

    # Faà di Bruno: (f∘g)^(n) = Σ_{k=1}^n f^(k)(g(z)) × B_{n,k}(g'(z), g''(z), ..., g^(n-k+1)(z))
    # where B_{n,k} is the partial Bell polynomial

    for k in range(1, n + 1):
        B_nk = _partial_bell_polynomial(n, k, g_derivs[1:n-k+2])
        result += f_derivs[k] * B_nk

    return result


def _partial_bell_polynomial(n: int, k: int, x: List[mpf]) -> mpf:
    """
    Compute the partial Bell polynomial B_{n,k}(x_1, x_2, ..., x_{n-k+1}).

    Uses the recurrence relation:
    B_{n,k}(x) = Σ_{i=1}^{n-k+1} C(n-1,i-1) × x_i × B_{n-i,k-1}(x)

    Base cases:
    B_{0,0} = 1
    B_{n,0} = 0 for n >= 1
    B_{0,k} = 0 for k >= 1
    """
    if n == 0 and k == 0:
        return mpf('1')
    if n == 0 or k == 0:
        return mpf('0')
    if k > n:
        return mpf('0')

    # Use memoization via a local cache
    cache = {}

    def bell_recursive(nn, kk):
        if nn == 0 and kk == 0:
            return mpf('1')
        if nn == 0 or kk == 0:
            return mpf('0')
        if kk > nn:
            return mpf('0')

        if (nn, kk) in cache:
            return cache[(nn, kk)]

        result = mpf('0')
        for i in range(1, nn - kk + 2):
            if i - 1 < len(x):
                coeff = _binomial(nn - 1, i - 1)
                result += coeff * x[i - 1] * bell_recursive(nn - i, kk - 1)

        cache[(nn, kk)] = result
        return result

    return bell_recursive(n, k)


def spin0_block_z_derivatives(delta: Union[float, mpf],
                               z: Union[float, mpf] = None,
                               max_order: int = MAX_DERIV_ORDER) -> List[mpf]:
    """
    Compute d^m/dz^m G_{Δ,0}(z) for m = 0, 1, ..., max_order.

    Uses the representation:
    G_{Δ,0}(z) = P(z) × H(x(z))

    where:
    - P(z) = (z²/(1-z))^{Δ/2} is the prefactor
    - H(x) = ₃F₂(Δ/2, Δ/2, Δ/2-α; (Δ+1)/2, Δ-α; x) is the hypergeometric
    - x(z) = z²/(4(z-1)) is the argument

    The Leibniz rule gives:
    d^n/dz^n [P × H∘g] = Σ_{k=0}^n C(n,k) × P^(n-k) × (H∘g)^(k)

    where (H∘g)^(k) is computed via Faà di Bruno.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    z : float or mpf, optional
        The cross-ratio. Default is 1/2.
    max_order : int
        Maximum derivative order to compute.

    Returns
    -------
    list of mpf
        [G(z), G'(z), G''(z), ..., G^(max_order)(z)]
    """
    if z is None:
        z = mpf(Z_POINT)
    else:
        z = mpf(z)
    delta = mpf(delta)
    alpha = mpf(ALPHA)

    # Compute the ₃F₂ argument and its derivatives
    x = _hyp3f2_argument(z)
    x_derivs = _hyp3f2_argument_derivatives(z, max_order)

    # ₃F₂ parameters for spin-0
    a1, a2, a3 = delta/2, delta/2, delta/2 - alpha
    b1, b2 = (delta + 1)/2, delta - alpha

    # Compute ₃F₂ derivatives w.r.t. x at x(z)
    hyp_x_derivs = _hyp3f2_derivatives_at_x(a1, a2, a3, b1, b2, x, max_order)

    # Compute prefactor derivatives
    prefactor_derivs = _compute_prefactor_derivatives_spin0(z, delta, max_order)

    # Compute (H∘g)^(k) for k = 0, 1, ..., max_order using Faà di Bruno
    Hg_derivs = []
    for k in range(max_order + 1):
        Hg_k = _faa_di_bruno(hyp_x_derivs, x_derivs, k)
        Hg_derivs.append(Hg_k)

    # Apply Leibniz rule: d^n/dz^n [P × H∘g] = Σ_{k=0}^n C(n,k) × P^(n-k) × (H∘g)^(k)
    block_derivs = []
    for n in range(max_order + 1):
        deriv_n = mpf('0')
        for k in range(n + 1):
            deriv_n += _binomial(n, k) * prefactor_derivs[n - k] * Hg_derivs[k]
        block_derivs.append(deriv_n)

    return block_derivs


def spin1_block_z_derivatives(delta: Union[float, mpf],
                               z: Union[float, mpf] = None,
                               max_order: int = MAX_DERIV_ORDER) -> List[mpf]:
    """
    Compute d^m/dz^m G_{Δ,1}(z) for m = 0, 1, ..., max_order.

    Uses the representation:
    G_{Δ,1}(z) = P(z) × H(x(z))

    where:
    - P(z) = (2-z)/(2z) × (z²/(1-z))^{(Δ+1)/2} is the prefactor
    - H(x) = ₃F₂((Δ+1)/2, (Δ+1)/2, (Δ+1)/2-α; (Δ+2)/2, Δ+1-α; x)
    - x(z) = z²/(4(z-1))

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    z : float or mpf, optional
        The cross-ratio. Default is 1/2.
    max_order : int
        Maximum derivative order to compute.

    Returns
    -------
    list of mpf
        [G(z), G'(z), G''(z), ..., G^(max_order)(z)]
    """
    if z is None:
        z = mpf(Z_POINT)
    else:
        z = mpf(z)
    delta = mpf(delta)
    alpha = mpf(ALPHA)

    # Compute the ₃F₂ argument and its derivatives
    x = _hyp3f2_argument(z)
    x_derivs = _hyp3f2_argument_derivatives(z, max_order)

    # ₃F₂ parameters for spin-1
    a1 = (delta + 1) / 2
    a2 = (delta + 1) / 2
    a3 = (delta + 1) / 2 - alpha
    b1 = (delta + 2) / 2
    b2 = delta + 1 - alpha

    # Compute ₃F₂ derivatives w.r.t. x at x(z)
    hyp_x_derivs = _hyp3f2_derivatives_at_x(a1, a2, a3, b1, b2, x, max_order)

    # Compute prefactor derivatives
    prefactor_derivs = _compute_prefactor_derivatives_spin1(z, delta, max_order)

    # Compute (H∘g)^(k) using Faà di Bruno
    Hg_derivs = []
    for k in range(max_order + 1):
        Hg_k = _faa_di_bruno(hyp_x_derivs, x_derivs, k)
        Hg_derivs.append(Hg_k)

    # Apply Leibniz rule
    block_derivs = []
    for n in range(max_order + 1):
        deriv_n = mpf('0')
        for k in range(n + 1):
            deriv_n += _binomial(n, k) * prefactor_derivs[n - k] * Hg_derivs[k]
        block_derivs.append(deriv_n)

    return block_derivs


def higher_spin_block_z_derivatives(delta: Union[float, mpf], spin: int,
                                     z: Union[float, mpf] = None,
                                     max_order: int = MAX_DERIV_ORDER,
                                     cache: Optional[Dict] = None) -> List[mpf]:
    """
    Compute d^m/dz^m G_{Δ,l}(z) for m = 0, 1, ..., max_order.

    For l >= 2, uses the spin recursion relation (Eq. 4.9) and its derivatives.

    The recursion Eq. 4.9 at z=z̄ gives G_{Δ,l} in terms of G_{Δ,l-2}, G_{Δ+1,l-1}, G_{Δ+2,l-2}.
    Taking z-derivatives of this recursion expresses G^(m)_{Δ,l} in terms of
    derivatives of lower-spin blocks.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    z : float or mpf, optional
        The cross-ratio. Default is 1/2.
    max_order : int
        Maximum derivative order to compute.
    cache : dict, optional
        Cache for intermediate results.

    Returns
    -------
    list of mpf
        [G(z), G'(z), G''(z), ..., G^(max_order)(z)]
    """
    if spin < 0:
        raise ValueError(f"Spin must be non-negative, got {spin}")

    if z is None:
        z = mpf(Z_POINT)
    else:
        z = mpf(z)
    delta = mpf(delta)

    if cache is None:
        cache = {}

    delta_key = str(delta)
    cache_key = (delta_key, spin, max_order)

    if cache_key in cache:
        return cache[cache_key]

    # Base cases
    if spin == 0:
        result = spin0_block_z_derivatives(delta, z, max_order)
        cache[cache_key] = result
        return result

    if spin == 1:
        result = spin1_block_z_derivatives(delta, z, max_order)
        cache[cache_key] = result
        return result

    # For spin >= 2, use the z-derivative of the spin recursion
    # At z=z̄, Eq. 4.9 becomes (for D=3, α=0.5):
    #
    # (l)(2Δ-1) G_{Δ,l} = (Δ+l-1) G_{Δ,l-2}
    #                    + (3/2)(2l-1)(Δ-1) G_{Δ+1,l-1}
    #                    - [third term coefficient] G_{Δ+2,l-2}
    #
    # Note: At z=1/2, (2-z)/(2z) = 3/2
    #
    # Taking d^m/dz^m of both sides:
    # LHS coefficient × G^(m) = RHS terms with their m-th derivatives

    d = mpf(D)
    alpha = mpf(ALPHA)

    # Get derivatives of required blocks (recursive calls)
    G_delta_lm2_derivs = higher_spin_block_z_derivatives(delta, spin - 2, z, max_order, cache)
    G_deltap1_lm1_derivs = higher_spin_block_z_derivatives(delta + 1, spin - 1, z, max_order, cache)
    G_deltap2_lm2_derivs = higher_spin_block_z_derivatives(delta + 2, spin - 2, z, max_order, cache)

    # Coefficients from spin recursion (Eq. 4.9)
    # LHS: (l + D - 3)(2Δ + 2 - D)
    lhs_coeff = (spin + d - 3) * (2 * delta + 2 - d)

    # First term: (D - 2)(Δ + l - 1)
    coeff1 = (d - 2) * (delta + spin - 1)

    # Second term at z=1/2: (2-z)/(2z) = 3/2 = (2-0.5)/(2×0.5)
    # Coefficient: (2-z)/(2z) × (2l + D - 4)(Δ - D + 2)
    # At z=1/2: z_factor = 3/2
    z_factor = (2 - z) / (2 * z)
    coeff2_base = (2 * spin + d - 4) * (delta - d + 2)

    # Third term coefficient (negative)
    num1 = delta
    num2 = 2 * spin + d - 4
    num3 = delta + 2 - d
    num4 = delta + 3 - d
    num5 = (delta - spin - d + 4) ** 2

    den1 = 16
    den2 = delta + 1 - d / 2
    den3 = delta - d / 2 + 2
    den4 = spin - delta + d - 5
    den5 = spin - delta + d - 3

    denominator = den1 * den2 * den3 * den4 * den5
    if abs(denominator) < mpf('1e-100'):
        coeff3 = mpf('0')
    else:
        numerator = num1 * num2 * num3 * num4 * num5
        coeff3 = -numerator / denominator

    # The key insight: at z=z̄, the recursion is algebraic (no derivative terms)
    # So d^m/dz^m of the recursion is straightforward!
    #
    # BUT: The z_factor = (2-z)/(2z) depends on z, so we need to use Leibniz rule
    # for the second term.

    # Compute derivatives of z_factor
    def z_factor_func(zz):
        return (2 - zz) / (2 * zz)

    z_factor_derivs = [z_factor_func(z)]
    for m in range(1, max_order + 1):
        z_factor_derivs.append(diff(z_factor_func, z, m))

    # Check for LHS coefficient being zero
    if abs(lhs_coeff) < mpf('1e-100'):
        raise ValueError(
            f"LHS coefficient is zero for delta={delta}, spin={spin}. "
            "This may indicate hitting a special value."
        )

    # Compute G^(m)_{Δ,l} for each m
    result = []
    for m in range(max_order + 1):
        # First term: coeff1 × G^(m)_{Δ,l-2}
        rhs_m = coeff1 * G_delta_lm2_derivs[m]

        # Second term: Σ_{k=0}^m C(m,k) × z_factor^(m-k) × coeff2_base × G^(k)_{Δ+1,l-1}
        second_term = mpf('0')
        for k in range(m + 1):
            second_term += _binomial(m, k) * z_factor_derivs[m - k] * coeff2_base * G_deltap1_lm1_derivs[k]
        rhs_m += second_term

        # Third term: coeff3 × G^(m)_{Δ+2,l-2}
        rhs_m += coeff3 * G_deltap2_lm2_derivs[m]

        # Solve for G^(m)_{Δ,l}
        result.append(rhs_m / lhs_coeff)

    cache[cache_key] = result
    return result


def block_z_derivatives(delta: Union[float, mpf], spin: int,
                        z: Union[float, mpf] = None,
                        max_order: int = MAX_DERIV_ORDER) -> List[mpf]:
    """
    Compute d^m/dz^m G_{Δ,l}(z) for m = 0, 1, ..., max_order.

    This is the main entry point for z-derivative computation.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    z : float or mpf, optional
        The cross-ratio. Default is 1/2.
    max_order : int
        Maximum derivative order to compute.

    Returns
    -------
    list of mpf
        [G(z), G'(z), G''(z), ..., G^(max_order)(z)]

    Examples
    --------
    >>> from mpmath import mpf
    >>> derivs = block_z_derivatives(mpf('1.5'), 0, max_order=5)
    >>> len(derivs)
    6
    """
    return higher_spin_block_z_derivatives(delta, spin, z, max_order)
