"""
Global configuration and physical constants for the 3D Ising bootstrap.

Reference: arXiv:1203.6064
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


# =============================================================================
# Physical Constants
# =============================================================================

D = 3
"""Spacetime dimension."""

ALPHA = D / 2 - 1  # = 0.5 for D=3
"""Parameter α = D/2 - 1 appearing in conformal blocks."""

N_MAX = 10
"""Derivative truncation parameter. Controls m + 2n ≤ 2*N_MAX + 1."""

MAX_DERIV_ORDER = 2 * N_MAX + 1  # = 21
"""Maximum value of m + 2n in the index set."""


# =============================================================================
# Crossing-Symmetric Point
# =============================================================================

Z_POINT = 0.5
"""The crossing-symmetric point z = z̄ = 1/2."""

A_POINT = 1.0
"""Value of 'a' coordinate at crossing-symmetric point (a=1, b=0)."""

B_POINT = 0.0
"""Value of 'b' coordinate at crossing-symmetric point (a=1, b=0)."""

U_POINT = Z_POINT ** 2  # = 0.25
"""Cross-ratio u = zz̄ at crossing-symmetric point."""

V_POINT = (1 - Z_POINT) ** 2  # = 0.25
"""Cross-ratio v = (1-z)(1-z̄) at crossing-symmetric point."""


# =============================================================================
# 3D Ising Model Reference Values
# =============================================================================

ISING_DELTA_SIGMA = 0.5182
"""Known dimension of the spin field σ in 3D Ising model."""

ISING_DELTA_EPSILON = 1.413
"""Known dimension of the energy field ε in 3D Ising model."""

ISING_DELTA_EPSILON_PRIME = 3.84
"""Expected dimension of ε' at n_max=10 truncation (from paper Sec. 5.2)."""


# =============================================================================
# Unitarity Bounds
# =============================================================================

def unitarity_bound(spin: int) -> float:
    """
    Return the unitarity bound Δ_min(l) for operators with given spin in D=3.

    For D=3:
    - Scalars (l=0): Δ ≥ 1/2
    - Spinning (l≥1): Δ ≥ l + 1

    Parameters
    ----------
    spin : int
        The spin of the operator (must be non-negative).

    Returns
    -------
    float
        The minimum allowed dimension.
    """
    if spin < 0:
        raise ValueError(f"Spin must be non-negative, got {spin}")
    if spin == 0:
        return 0.5
    else:
        return spin + 1


# =============================================================================
# Table 2 Discretization (from paper Appendix D)
# =============================================================================

@dataclass
class DiscretizationTable:
    """Parameters for a single discretization table from Table 2."""
    name: str
    delta: float      # Step size in Δ
    delta_max: float  # Maximum dimension
    l_max: int        # Maximum spin


# Exact values from Table 2 of arXiv:1203.6064
TABLE_1 = DiscretizationTable("T1", delta=2e-5, delta_max=3, l_max=0)
TABLE_2 = DiscretizationTable("T2", delta=5e-4, delta_max=8, l_max=6)
TABLE_3 = DiscretizationTable("T3", delta=2e-3, delta_max=22, l_max=20)
TABLE_4 = DiscretizationTable("T4", delta=0.02, delta_max=100, l_max=50)
TABLE_5 = DiscretizationTable("T5", delta=1.0, delta_max=500, l_max=100)

FULL_DISCRETIZATION = [TABLE_1, TABLE_2, TABLE_3, TABLE_4, TABLE_5]
"""All five discretization tables for full precision."""

REDUCED_DISCRETIZATION = [TABLE_1, TABLE_2]
"""Reduced discretization (T1-T2 only) for fast testing."""


# =============================================================================
# Scan Parameters
# =============================================================================

DEFAULT_SIGMA_MIN = 0.50
"""Default minimum Δσ for scan."""

DEFAULT_SIGMA_MAX = 0.60
"""Default maximum Δσ for scan."""

DEFAULT_SIGMA_STEP = 0.002
"""Default step size for Δσ scan."""

DEFAULT_EPS_TOLERANCE = 1e-4
"""Default binary search tolerance for Δε."""

DEFAULT_EPSPRIME_TOLERANCE = 1e-3
"""Default binary search tolerance for Δε'."""


# =============================================================================
# Numerical Parameters
# =============================================================================

MPMATH_PRECISION = 50
"""Number of decimal digits for mpmath extended precision."""

LP_TOLERANCE = 1e-7
"""Feasibility tolerance for LP solver."""

BINARY_SEARCH_MAX_ITER = 50
"""Maximum iterations for binary search."""


# =============================================================================
# File Paths
# =============================================================================

import os
from pathlib import Path

# Get project root directory
_THIS_DIR = Path(__file__).parent
PROJECT_ROOT = _THIS_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
"""Directory for output data files."""

CACHE_DIR = DATA_DIR / "cached_blocks"
"""Directory for cached block derivatives."""

FIGURES_DIR = PROJECT_ROOT / "figures"
"""Directory for generated figures."""

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utility Functions
# =============================================================================

def get_index_set_count(n_max: int = N_MAX) -> int:
    """
    Compute the number of (m, n) pairs in the index set.

    The index set has m odd, m ≥ 1, n ≥ 0, m + 2n ≤ 2*n_max + 1.

    For n_max=10, this should return 66.
    """
    count = 0
    max_order = 2 * n_max + 1
    for m in range(1, max_order + 1, 2):  # m = 1, 3, 5, ..., max_order
        max_n = (max_order - m) // 2
        count += max_n + 1
    return count


# Verify expected count
assert get_index_set_count(10) == 66, "Index set count should be 66 for n_max=10"
