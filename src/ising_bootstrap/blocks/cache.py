"""
Disk caching for conformal block derivatives.

Block derivatives h_{m,n}(Δ,l) depend only on (Δ,l,D), NOT on Δσ.
This means we can precompute and cache them for all (Δ,l) in the
Table 2 discretization, then load them quickly for each Δσ scan point.

Cache format: NPZ files at data/cached_blocks/d{delta:.6f}_l{spin}.npz

Reference: arXiv:1203.6064, Appendix D
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from mpmath import mpf

from ..config import CACHE_DIR, N_MAX, MPMATH_PRECISION
from ..spectrum.index_set import generate_index_set, index_set_size


def get_cache_filename(delta: float, spin: int) -> Path:
    """
    Generate the cache filename for a given (Δ, l) pair.

    Format: d{delta:.6f}_l{spin}.npz

    Parameters
    ----------
    delta : float
        The scaling dimension Δ.
    spin : int
        The spin l.

    Returns
    -------
    Path
        Full path to the cache file.

    Examples
    --------
    >>> p = get_cache_filename(1.5, 2)
    >>> p.name
    'd1.500000_l2.npz'
    """
    filename = f"d{delta:.6f}_l{spin}.npz"
    return CACHE_DIR / filename


def cache_exists(delta: float, spin: int) -> bool:
    """
    Check if cache file exists for the given (Δ, l) pair.

    Parameters
    ----------
    delta : float
        The scaling dimension.
    spin : int
        The spin.

    Returns
    -------
    bool
        True if cache file exists.
    """
    return get_cache_filename(delta, spin).exists()


def save_block_derivatives(delta: float, spin: int,
                           h_values: Dict[Tuple[int, int], mpf],
                           n_max: int = N_MAX,
                           overwrite: bool = False) -> Path:
    """
    Save block derivatives to cache file.

    Converts mpf values to float64 for storage (sufficient for LP solver).

    Parameters
    ----------
    delta : float
        The scaling dimension Δ.
    spin : int
        The spin l.
    h_values : dict
        Dictionary mapping (m, n) -> h_{m,n}.
    n_max : int
        The truncation parameter.
    overwrite : bool
        If True, overwrite existing cache file.

    Returns
    -------
    Path
        Path to the saved cache file.

    Raises
    ------
    FileExistsError
        If cache file exists and overwrite=False.
    """
    cache_path = get_cache_filename(delta, spin)

    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"Cache file already exists: {cache_path}")

    # Get index set and convert to arrays
    index_set = generate_index_set(n_max)
    n_indices = len(index_set)

    # Store as structured arrays
    m_indices = np.array([m for m, n in index_set], dtype=np.int32)
    n_indices_arr = np.array([n for m, n in index_set], dtype=np.int32)
    h_array = np.array([float(h_values[(m, n)]) for m, n in index_set], dtype=np.float64)

    # Save metadata alongside values
    np.savez_compressed(
        cache_path,
        delta=np.float64(delta),
        spin=np.int32(spin),
        n_max=np.int32(n_max),
        m_indices=m_indices,
        n_indices=n_indices_arr,
        h_values=h_array
    )

    return cache_path


def load_block_derivatives(delta: float, spin: int,
                           n_max: int = N_MAX) -> Dict[Tuple[int, int], float]:
    """
    Load block derivatives from cache file.

    Parameters
    ----------
    delta : float
        The scaling dimension Δ.
    spin : int
        The spin l.
    n_max : int
        Expected truncation parameter.

    Returns
    -------
    dict
        Dictionary mapping (m, n) -> h_{m,n} as float64 values.

    Raises
    ------
    FileNotFoundError
        If cache file doesn't exist.
    ValueError
        If cached n_max doesn't match expected.
    """
    cache_path = get_cache_filename(delta, spin)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    data = np.load(cache_path)

    # Verify metadata
    cached_n_max = int(data['n_max'])
    if cached_n_max != n_max:
        raise ValueError(
            f"Cached n_max={cached_n_max} doesn't match expected n_max={n_max}"
        )

    # Reconstruct dictionary
    m_indices = data['m_indices']
    n_indices = data['n_indices']
    h_array = data['h_values']

    h_dict = {}
    for i in range(len(m_indices)):
        m = int(m_indices[i])
        n = int(n_indices[i])
        h_dict[(m, n)] = float(h_array[i])

    return h_dict


def load_block_derivatives_as_vector(delta: float, spin: int,
                                      n_max: int = N_MAX) -> np.ndarray:
    """
    Load block derivatives as a flat numpy array in index set order.

    This is the preferred format for constructing LP constraint matrices.

    Parameters
    ----------
    delta : float
        The scaling dimension Δ.
    spin : int
        The spin l.
    n_max : int
        The truncation parameter.

    Returns
    -------
    np.ndarray
        1D array of h_{m,n} values in index set order.
    """
    cache_path = get_cache_filename(delta, spin)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    data = np.load(cache_path)

    # Verify n_max
    cached_n_max = int(data['n_max'])
    if cached_n_max != n_max:
        raise ValueError(
            f"Cached n_max={cached_n_max} doesn't match expected n_max={n_max}"
        )

    return data['h_values'].copy()


def compute_and_cache(delta: Union[float, mpf], spin: int,
                      n_max: int = N_MAX,
                      overwrite: bool = False) -> Dict[Tuple[int, int], mpf]:
    """
    Compute block derivatives and save to cache.

    If cache exists and overwrite=False, loads from cache instead.

    Parameters
    ----------
    delta : float or mpf
        The scaling dimension Δ.
    spin : int
        The spin l.
    n_max : int
        The truncation parameter.
    overwrite : bool
        If True, recompute even if cache exists.

    Returns
    -------
    dict
        Dictionary mapping (m, n) -> h_{m,n}.
    """
    delta_float = float(delta)

    # Check cache first
    if not overwrite and cache_exists(delta_float, spin):
        # Load and convert back to mpf
        h_dict_float = load_block_derivatives(delta_float, spin, n_max)
        return {k: mpf(v) for k, v in h_dict_float.items()}

    # Compute block derivatives
    from .transverse_derivs import block_derivatives_full

    h_values = block_derivatives_full(delta, spin, n_max)

    # Save to cache
    save_block_derivatives(delta_float, spin, h_values, n_max, overwrite=True)

    return h_values


def precompute_spectrum_blocks(spectrum_points: List[Tuple[float, int]],
                               n_max: int = N_MAX,
                               skip_existing: bool = True,
                               verbose: bool = True) -> int:
    """
    Precompute block derivatives for a list of (Δ, l) points.

    This is used to cache all blocks for the Table 2 discretization
    before running the bootstrap scan.

    Parameters
    ----------
    spectrum_points : list of (float, int)
        List of (Δ, l) pairs to compute.
    n_max : int
        The truncation parameter.
    skip_existing : bool
        If True, skip points that are already cached.
    verbose : bool
        If True, print progress.

    Returns
    -------
    int
        Number of blocks computed (excluding skipped).
    """
    from .transverse_derivs import block_derivatives_full

    computed = 0
    total = len(spectrum_points)

    for i, (delta, spin) in enumerate(spectrum_points):
        if verbose:
            print(f"[{i+1}/{total}] Computing (Δ={delta:.4f}, l={spin})...", end=" ")

        if skip_existing and cache_exists(delta, spin):
            if verbose:
                print("cached, skipping")
            continue

        try:
            h_values = block_derivatives_full(mpf(delta), spin, n_max)
            save_block_derivatives(delta, spin, h_values, n_max, overwrite=True)
            computed += 1
            if verbose:
                print("done")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    return computed


def clear_cache(confirm: bool = False) -> int:
    """
    Delete all cache files.

    Parameters
    ----------
    confirm : bool
        Must be True to actually delete files.

    Returns
    -------
    int
        Number of files deleted.
    """
    if not confirm:
        raise ValueError("Set confirm=True to delete cache files")

    count = 0
    for cache_file in CACHE_DIR.glob("*.npz"):
        cache_file.unlink()
        count += 1

    return count


def cache_stats() -> Dict:
    """
    Get statistics about the cache.

    Returns
    -------
    dict
        Dictionary with cache statistics.
    """
    cache_files = list(CACHE_DIR.glob("*.npz"))

    total_size = sum(f.stat().st_size for f in cache_files)

    # Parse delta and spin from filenames
    deltas = []
    spins = []
    for f in cache_files:
        try:
            parts = f.stem.split("_")
            delta = float(parts[0][1:])  # Remove 'd' prefix
            spin = int(parts[1][1:])  # Remove 'l' prefix
            deltas.append(delta)
            spins.append(spin)
        except (IndexError, ValueError):
            pass

    return {
        'num_files': len(cache_files),
        'total_size_mb': total_size / (1024 * 1024),
        'delta_range': (min(deltas), max(deltas)) if deltas else (None, None),
        'spin_range': (min(spins), max(spins)) if spins else (None, None),
        'cache_dir': str(CACHE_DIR)
    }


def get_or_compute(delta: Union[float, mpf], spin: int,
                   n_max: int = N_MAX) -> np.ndarray:
    """
    Get block derivatives as numpy array, computing if not cached.

    This is the primary interface for the LP constraint builder.

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
    np.ndarray
        1D array of h_{m,n} values in index set order.
    """
    delta_float = float(delta)

    if cache_exists(delta_float, spin):
        return load_block_derivatives_as_vector(delta_float, spin, n_max)

    # Compute and cache
    compute_and_cache(delta, spin, n_max)

    return load_block_derivatives_as_vector(delta_float, spin, n_max)
