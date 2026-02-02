"""
Spectrum discretization following Table 2 of arXiv:1203.6064.

The paper discretizes the spectrum using 5 tables (T1-T5) with different
resolutions and ranges to efficiently cover operators from low dimensions
(near unitarity bound) to asymptotically large dimensions.

For each table, operators are sampled at:
- Spins: l = 0, 2, 4, ..., L_max (even spins only)
- Dimensions: Δ from unitarity bound to Δ_max + 2(L_max - l) in steps of δ

The union of all tables provides the full constraint set for the LP.

Reference: arXiv:1203.6064, Appendix D, Table 2
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Iterator
import numpy as np

from ..config import (
    DiscretizationTable,
    TABLE_1, TABLE_2, TABLE_3, TABLE_4, TABLE_5,
    FULL_DISCRETIZATION, REDUCED_DISCRETIZATION
)
from .unitarity import unitarity_bound, is_allowed_spin


# Type alias for an operator in the spectrum
Operator = Tuple[float, int]  # (Δ, l)


@dataclass
class SpectrumPoint:
    """
    A single point in the discretized spectrum.

    Attributes
    ----------
    delta : float
        The scaling dimension Δ.
    spin : int
        The spin l.
    table : str
        Which table (T1-T5) this point came from.
    """
    delta: float
    spin: int
    table: str

    def as_tuple(self) -> Operator:
        """Return (Δ, l) tuple."""
        return (self.delta, self.spin)


def generate_table_points(
    table: DiscretizationTable,
    gap_scalar: Optional[float] = None,
    gap_eprime: Optional[float] = None
) -> List[SpectrumPoint]:
    """
    Generate all (Δ, l) points for a single discretization table.

    For a table with parameters (δ, Δ_max, L_max):
    - Spins: l = 0, 2, 4, ..., L_max
    - For each spin l, dimensions are sampled from:
        Δ_min(l) = unitarity bound
        Δ_max(l) = Δ_max + 2(L_max - l)
      in steps of δ.

    Parameters
    ----------
    table : DiscretizationTable
        The table parameters.
    gap_scalar : float, optional
        If provided, exclude scalars (l=0) with Δ < gap_scalar.
        Used for Stage A bootstrap (gap below ε).
    gap_eprime : float, optional
        If provided and gap_scalar is also provided, exclude scalars with
        gap_scalar < Δ < gap_eprime. Used for Stage B bootstrap (gap between ε and ε').

    Returns
    -------
    list of SpectrumPoint
        All spectrum points from this table.
    """
    points = []

    # Iterate over even spins from 0 to L_max
    for spin in range(0, table.l_max + 1, 2):
        # Dimension range for this spin
        delta_min = unitarity_bound(spin)
        delta_max_spin = table.delta_max + 2 * (table.l_max - spin)

        # Number of points (including endpoints)
        n_points = int(np.round((delta_max_spin - delta_min) / table.delta)) + 1

        for i in range(n_points):
            delta = delta_min + i * table.delta

            # Don't exceed the maximum
            if delta > delta_max_spin + 1e-10:
                break

            # Apply gap constraints for scalars
            if spin == 0:
                if gap_scalar is not None and delta < gap_scalar - 1e-10:
                    continue
                if gap_eprime is not None and gap_scalar is not None:
                    # Exclude scalars in the gap between ε and ε'
                    if gap_scalar - 1e-10 < delta < gap_eprime - 1e-10:
                        continue

            points.append(SpectrumPoint(delta=delta, spin=spin, table=table.name))

    return points


def generate_full_spectrum(
    tables: List[DiscretizationTable] = None,
    gap_scalar: Optional[float] = None,
    gap_eprime: Optional[float] = None,
    remove_duplicates: bool = True,
    duplicate_tolerance: float = 1e-8
) -> List[SpectrumPoint]:
    """
    Generate the full discretized spectrum by combining all tables.

    Parameters
    ----------
    tables : list of DiscretizationTable, optional
        Which tables to use. Default is FULL_DISCRETIZATION (T1-T5).
    gap_scalar : float, optional
        Exclude scalars with Δ < gap_scalar (for Stage A).
    gap_eprime : float, optional
        Exclude scalars with gap_scalar < Δ < gap_eprime (for Stage B).
    remove_duplicates : bool, optional
        If True, remove duplicate (Δ, l) points that appear in multiple tables.
        The point from the finer-resolution table is kept. Default is True.
    duplicate_tolerance : float, optional
        Two points are considered duplicates if |Δ1 - Δ2| < tolerance and l1 == l2.
        Default is 1e-8.

    Returns
    -------
    list of SpectrumPoint
        The combined spectrum sorted by (spin, delta).
    """
    if tables is None:
        tables = FULL_DISCRETIZATION

    all_points = []
    for table in tables:
        points = generate_table_points(
            table, gap_scalar=gap_scalar, gap_eprime=gap_eprime
        )
        all_points.extend(points)

    if remove_duplicates:
        all_points = _remove_duplicate_points(all_points, duplicate_tolerance)

    # Sort by (spin, delta) for consistent ordering
    all_points.sort(key=lambda p: (p.spin, p.delta))

    return all_points


def _remove_duplicate_points(
    points: List[SpectrumPoint],
    tolerance: float = 1e-8
) -> List[SpectrumPoint]:
    """
    Remove duplicate spectrum points, keeping the one from the finest table.

    Tables are ordered by resolution: T1 (finest) > T2 > T3 > T4 > T5 (coarsest).
    """
    # Table priority (lower = finer resolution = higher priority)
    table_priority = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5}

    # Group by spin
    by_spin = {}
    for p in points:
        if p.spin not in by_spin:
            by_spin[p.spin] = []
        by_spin[p.spin].append(p)

    unique_points = []
    for spin in sorted(by_spin.keys()):
        spin_points = sorted(by_spin[spin], key=lambda p: p.delta)

        if not spin_points:
            continue

        # Merge close points, keeping the one with higher priority (lower number)
        merged = [spin_points[0]]
        for p in spin_points[1:]:
            if abs(p.delta - merged[-1].delta) < tolerance:
                # Duplicate found - keep the one from finer table
                if table_priority.get(p.table, 10) < table_priority.get(merged[-1].table, 10):
                    merged[-1] = p
            else:
                merged.append(p)

        unique_points.extend(merged)

    return unique_points


def spectrum_to_array(points: List[SpectrumPoint]) -> np.ndarray:
    """
    Convert spectrum points to a numpy array of shape (N, 2).

    Parameters
    ----------
    points : list of SpectrumPoint
        The spectrum points.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) where each row is [Δ, l].
    """
    return np.array([[p.delta, p.spin] for p in points])


def count_by_spin(points: List[SpectrumPoint]) -> dict:
    """
    Count spectrum points by spin.

    Parameters
    ----------
    points : list of SpectrumPoint
        The spectrum points.

    Returns
    -------
    dict
        Dictionary mapping spin -> count.
    """
    counts = {}
    for p in points:
        counts[p.spin] = counts.get(p.spin, 0) + 1
    return counts


def count_by_table(points: List[SpectrumPoint]) -> dict:
    """
    Count spectrum points by source table.

    Parameters
    ----------
    points : list of SpectrumPoint
        The spectrum points.

    Returns
    -------
    dict
        Dictionary mapping table name -> count.
    """
    counts = {}
    for p in points:
        counts[p.table] = counts.get(p.table, 0) + 1
    return counts


def get_scalars(points: List[SpectrumPoint]) -> List[SpectrumPoint]:
    """Return only the scalar (l=0) points."""
    return [p for p in points if p.spin == 0]


def get_spinning(points: List[SpectrumPoint]) -> List[SpectrumPoint]:
    """Return only the spinning (l>0) points."""
    return [p for p in points if p.spin > 0]


def build_spectrum_with_gaps(
    delta_epsilon: Optional[float] = None,
    delta_epsilon_prime: Optional[float] = None,
    tables: List[DiscretizationTable] = None,
    remove_duplicates: bool = True
) -> List[SpectrumPoint]:
    """
    Build a discretized spectrum with gap assumptions.

    This is a convenience function for setting up Stage A and B bootstrap scans.

    Stage A (only delta_epsilon provided):
        - No scalars below delta_epsilon (gap below ε)
        - All spinning operators per unitarity

    Stage B (both provided):
        - No scalars below delta_epsilon (gap below ε)
        - No scalars between delta_epsilon and delta_epsilon_prime (gap ε to ε')
        - Scalars at and above delta_epsilon_prime are included
        - All spinning operators per unitarity

    Parameters
    ----------
    delta_epsilon : float, optional
        Lower bound on first scalar dimension. If None, include all scalars
        from unitarity bound.
    delta_epsilon_prime : float, optional
        Lower bound on second scalar dimension (Stage B).
        Only used if delta_epsilon is also provided.
    tables : list of DiscretizationTable, optional
        Which tables to use. Default is FULL_DISCRETIZATION.
    remove_duplicates : bool, optional
        Whether to remove duplicate points. Default is True.

    Returns
    -------
    list of SpectrumPoint
        The discretized spectrum with gap constraints applied.

    Examples
    --------
    >>> # Stage A: gap below ε = 1.4
    >>> spectrum_a = build_spectrum_with_gaps(delta_epsilon=1.4)

    >>> # Stage B: ε at 1.4, gap up to ε' = 3.5
    >>> spectrum_b = build_spectrum_with_gaps(delta_epsilon=1.4, delta_epsilon_prime=3.5)
    """
    return generate_full_spectrum(
        tables=tables,
        gap_scalar=delta_epsilon,
        gap_eprime=delta_epsilon_prime,
        remove_duplicates=remove_duplicates
    )


def estimate_spectrum_size(tables: List[DiscretizationTable] = None) -> int:
    """
    Estimate the total number of spectrum points (upper bound).

    This counts points before removing duplicates, so it's an upper bound
    on the actual spectrum size.

    Parameters
    ----------
    tables : list of DiscretizationTable, optional
        Which tables to count. Default is FULL_DISCRETIZATION.

    Returns
    -------
    int
        Upper bound on number of spectrum points.
    """
    if tables is None:
        tables = FULL_DISCRETIZATION

    total = 0
    for table in tables:
        for spin in range(0, table.l_max + 1, 2):
            delta_min = unitarity_bound(spin)
            delta_max_spin = table.delta_max + 2 * (table.l_max - spin)
            n_points = int(np.round((delta_max_spin - delta_min) / table.delta)) + 1
            total += n_points

    return total


def get_table_info(table: DiscretizationTable) -> dict:
    """
    Get detailed information about a discretization table.

    Parameters
    ----------
    table : DiscretizationTable
        The table to analyze.

    Returns
    -------
    dict
        Dictionary with 'name', 'delta', 'delta_max', 'l_max',
        'spins', 'n_scalars', 'n_total' keys.
    """
    points = generate_table_points(table)
    scalars = get_scalars(points)

    return {
        'name': table.name,
        'delta': table.delta,
        'delta_max': table.delta_max,
        'l_max': table.l_max,
        'spins': list(range(0, table.l_max + 1, 2)),
        'n_scalars': len(scalars),
        'n_total': len(points)
    }
