"""
Index set generation for the conformal bootstrap linear functional.

The index set consists of (m, n) pairs where:
- m is odd (m = 1, 3, 5, ...) because F^{Δσ}_{Δ,l} is antisymmetric under u↔v
- n ≥ 0
- m + 2n ≤ 2*n_max + 1 = 21 (for n_max=10)

This gives exactly 66 index pairs for n_max=10.

Reference: arXiv:1203.6064, Appendix D
"""

from typing import List, Tuple, Iterator
from ..config import N_MAX, MAX_DERIV_ORDER


IndexPair = Tuple[int, int]


def generate_index_set(n_max: int = N_MAX) -> List[IndexPair]:
    """
    Generate the (m, n) index set for derivative truncation.

    The linear functional α acts on functions via:
        α[F] = Σ_{m,n} λ_{m,n} ∂_a^m ∂_b^n F|_{a=1,b=0}

    Only odd m contribute because F^{Δσ}_{Δ,l} is antisymmetric under u↔v
    (equivalently a → -a at fixed b).

    Parameters
    ----------
    n_max : int
        The truncation parameter. Default is 10 from config.

    Returns
    -------
    list of (int, int)
        List of (m, n) pairs sorted by (m, n) lexicographically.
        For n_max=10, this contains exactly 66 pairs.

    Examples
    --------
    >>> index_set = generate_index_set(10)
    >>> len(index_set)
    66
    >>> (1, 0) in index_set
    True
    >>> (2, 0) in index_set  # m must be odd
    False
    """
    max_order = 2 * n_max + 1
    index_set = []

    # m must be odd: 1, 3, 5, ..., max_order
    for m in range(1, max_order + 1, 2):
        # For each m, n can range from 0 to floor((max_order - m) / 2)
        max_n = (max_order - m) // 2
        for n in range(max_n + 1):
            index_set.append((m, n))

    return index_set


def iter_index_set(n_max: int = N_MAX) -> Iterator[IndexPair]:
    """
    Iterate over the (m, n) index set.

    This is a memory-efficient alternative to generate_index_set()
    when you only need to iterate once.

    Parameters
    ----------
    n_max : int
        The truncation parameter.

    Yields
    ------
    tuple of (int, int)
        (m, n) pairs in lexicographic order.
    """
    max_order = 2 * n_max + 1

    for m in range(1, max_order + 1, 2):
        max_n = (max_order - m) // 2
        for n in range(max_n + 1):
            yield (m, n)


def index_set_size(n_max: int = N_MAX) -> int:
    """
    Compute the size of the index set without generating it.

    For n_max=10, this equals 66.

    The formula is: Σ_{m odd, m ≤ max_order} (floor((max_order - m) / 2) + 1)

    This simplifies to (n_max + 1)^2 for integer n_max.

    Parameters
    ----------
    n_max : int
        The truncation parameter.

    Returns
    -------
    int
        Number of (m, n) pairs in the index set.

    Examples
    --------
    >>> index_set_size(10)
    66
    >>> index_set_size(5)
    21
    """
    max_order = 2 * n_max + 1
    count = 0

    for m in range(1, max_order + 1, 2):
        max_n = (max_order - m) // 2
        count += max_n + 1

    return count


def is_valid_index_pair(m: int, n: int, n_max: int = N_MAX) -> bool:
    """
    Check if (m, n) is a valid index pair.

    Valid pairs satisfy:
    - m >= 1 and m is odd
    - n >= 0
    - m + 2n <= 2*n_max + 1

    Parameters
    ----------
    m : int
        The a-derivative order.
    n : int
        The b-derivative order.
    n_max : int
        The truncation parameter.

    Returns
    -------
    bool
        True if (m, n) is in the index set.

    Examples
    --------
    >>> is_valid_index_pair(1, 10, 10)  # 1 + 20 = 21 ≤ 21
    True
    >>> is_valid_index_pair(2, 0, 10)   # m must be odd
    False
    >>> is_valid_index_pair(1, 11, 10)  # 1 + 22 = 23 > 21
    False
    """
    max_order = 2 * n_max + 1

    if m < 1:
        return False
    if m % 2 == 0:  # m must be odd
        return False
    if n < 0:
        return False
    if m + 2 * n > max_order:
        return False

    return True


def get_index_position(m: int, n: int, n_max: int = N_MAX) -> int:
    """
    Get the linear index of (m, n) in the index set.

    This is useful for building constraint matrices where each column
    corresponds to a λ_{m,n} coefficient.

    Parameters
    ----------
    m : int
        The a-derivative order (must be odd, >= 1).
    n : int
        The b-derivative order (must be >= 0).
    n_max : int
        The truncation parameter.

    Returns
    -------
    int
        The 0-based position of (m, n) in the index set.

    Raises
    ------
    ValueError
        If (m, n) is not a valid index pair.

    Examples
    --------
    >>> get_index_position(1, 0, 10)
    0
    >>> get_index_position(1, 1, 10)
    1
    >>> get_index_position(3, 0, 10)
    11
    """
    if not is_valid_index_pair(m, n, n_max):
        raise ValueError(f"({m}, {n}) is not a valid index pair for n_max={n_max}")

    max_order = 2 * n_max + 1
    position = 0

    # Count all pairs with m' < m
    for m_prime in range(1, m, 2):
        max_n_prime = (max_order - m_prime) // 2
        position += max_n_prime + 1

    # Add n for the current m
    position += n

    return position


def get_pair_at_position(position: int, n_max: int = N_MAX) -> IndexPair:
    """
    Get the (m, n) pair at a given linear index.

    This is the inverse of get_index_position().

    Parameters
    ----------
    position : int
        The 0-based position in the index set.
    n_max : int
        The truncation parameter.

    Returns
    -------
    tuple of (int, int)
        The (m, n) pair at the given position.

    Raises
    ------
    ValueError
        If position is out of range.

    Examples
    --------
    >>> get_pair_at_position(0, 10)
    (1, 0)
    >>> get_pair_at_position(11, 10)
    (3, 0)
    >>> get_pair_at_position(65, 10)
    (21, 0)
    """
    total_size = index_set_size(n_max)
    if position < 0 or position >= total_size:
        raise ValueError(f"Position {position} out of range [0, {total_size - 1}]")

    max_order = 2 * n_max + 1
    cumulative = 0

    for m in range(1, max_order + 1, 2):
        max_n = (max_order - m) // 2
        count_for_m = max_n + 1

        if cumulative + count_for_m > position:
            # The pair is in this m block
            n = position - cumulative
            return (m, n)

        cumulative += count_for_m

    # Should never reach here
    raise RuntimeError("Internal error in get_pair_at_position")
