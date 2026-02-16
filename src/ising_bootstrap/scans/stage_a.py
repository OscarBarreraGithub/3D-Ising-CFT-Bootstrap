"""
Stage A scan: compute upper bound on Δε as a function of Δσ.

For each Δσ in the scan grid, binary search over Δε to find the
largest gap (no scalars below Δε) that is still consistent with
crossing symmetry. This yields the curve Δε_max(Δσ), as in Fig. 3
of arXiv:1203.6064.

Binary search logic:
    - Imposing a gap removes scalars with Δ < Δε from the positivity
      constraints, making the LP easier to satisfy (fewer constraints).
    - LP feasible (excluded): the gap is too large → lower hi
    - LP infeasible (allowed): the gap is consistent → raise lo
    - At convergence: lo ≈ Δε_max

Usage:
    python -m ising_bootstrap.scans.stage_a \\
        --sigma-min 0.50 --sigma-max 0.60 --sigma-step 0.002 \\
        --tolerance 1e-4 --output data/eps_bound.csv --verbose

Reference: arXiv:1203.6064, Sections 5-6 and Appendix D
"""

import argparse
import csv
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Callable

from ..config import (
    N_MAX, LP_TOLERANCE, BINARY_SEARCH_MAX_ITER,
    DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_MAX, DEFAULT_SIGMA_STEP,
    DEFAULT_EPS_TOLERANCE,
    FULL_DISCRETIZATION, REDUCED_DISCRETIZATION,
    DATA_DIR, CACHE_DIR,
    DiscretizationTable,
)
from ..spectrum.discretization import (
    SpectrumPoint, generate_full_spectrum,
)
from ..lp.crossing import (
    compute_prefactor_table,
    compute_identity_vector,
    compute_crossing_vector_fast,
    build_comb_cache,
)
from ..lp.constraint_matrix import (
    build_constraint_matrix_from_cache,
    precompute_extended_blocks,
)
from ..lp.solver import check_feasibility, FeasibilityResult
from ..blocks.cache import (
    extended_cache_exists,
    list_extended_cache_filenames,
    load_extended_h_array,
    precompute_extended_spectrum_blocks,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScanConfig:
    """Configuration for a Stage A scan."""
    sigma_min: float = DEFAULT_SIGMA_MIN
    sigma_max: float = DEFAULT_SIGMA_MAX
    sigma_step: float = DEFAULT_SIGMA_STEP
    tolerance: float = DEFAULT_EPS_TOLERANCE
    max_iter: int = BINARY_SEARCH_MAX_ITER
    eps_lo: float = 0.5       # unitarity bound for scalars
    eps_hi: float = 2.5       # generous upper bound
    n_max: int = N_MAX
    tables: Optional[List[DiscretizationTable]] = None
    reduced: bool = False
    cache_dir: Optional[Path] = None
    output: Path = field(default_factory=lambda: DATA_DIR / "eps_bound.csv")
    verbose: bool = False
    precompute_only: bool = False
    scale: bool = True
    backend: str = "scipy"
    sdpb_image: Optional[Path] = None
    sdpb_precision: int = 1024
    sdpb_cores: int = 4
    sdpb_timeout: int = 600
    validate_bracket: bool = True
    workers: int = 1
    shard_id: Optional[int] = None
    num_shards: Optional[int] = None

    def get_tables(self) -> List[DiscretizationTable]:
        """Return the discretization tables to use."""
        if self.tables is not None:
            return self.tables
        return REDUCED_DISCRETIZATION if self.reduced else FULL_DISCRETIZATION

    def get_sigma_grid(self) -> np.ndarray:
        """Return the Δσ scan grid."""
        return np.arange(
            self.sigma_min,
            self.sigma_max + self.sigma_step / 2,
            self.sigma_step,
        )


# =============================================================================
# Binary search
# =============================================================================

def binary_search_eps(
    is_excluded_fn: Callable[[float], bool],
    lo: float,
    hi: float,
    tol: float,
    max_iter: int,
) -> Tuple[float, int]:
    """
    Generic binary search for the exclusion boundary.

    Finds the largest gap value where is_excluded_fn returns False
    (i.e., the spectrum is still allowed).

    Parameters
    ----------
    is_excluded_fn : callable
        Function that takes a gap value and returns True if the
        spectrum with that gap is excluded (LP feasible).
    lo : float
        Lower bound (should be allowed).
    hi : float
        Upper bound (should be excluded).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    boundary : float
        The estimated Δε_max (largest allowed gap).
    n_iter : int
        Number of iterations performed.
    """
    for iteration in range(max_iter):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2.0
        if is_excluded_fn(mid):
            # LP feasible → gap inconsistent → too high
            hi = mid
        else:
            # LP infeasible → gap consistent → can go higher
            lo = mid
    return lo, iteration + 1


# =============================================================================
# Constraint matrix with metadata
# =============================================================================

def build_full_constraint_matrix(
    spectrum: List[SpectrumPoint],
    delta_sigma: float,
    h_cache: dict,
    n_max: int = N_MAX,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full constraint matrix for the ungapped spectrum, plus metadata
    for fast row-subsetting during binary search.

    Parameters
    ----------
    spectrum : list of SpectrumPoint
        Full (ungapped) spectrum.
    delta_sigma : float
        External scalar dimension.
    h_cache : dict
        Mapping (delta, spin) -> np.ndarray (extended H array).
    n_max : int
        Truncation parameter.
    verbose : bool
        Print progress.

    Returns
    -------
    A : np.ndarray, shape (N_operators, 66)
        Full constraint matrix.
    f_id : np.ndarray, shape (66,)
        Identity vector.
    scalar_mask : np.ndarray of bool, shape (N_operators,)
        True for scalar (l=0) operators.
    scalar_deltas : np.ndarray of float, shape (N_operators,)
        Delta value for each operator (only meaningful where scalar_mask is True).
    spinning_mask : np.ndarray of bool, shape (N_operators,)
        True for spinning (l>0) operators.
    """
    A, f_id = build_constraint_matrix_from_cache(
        spectrum, delta_sigma, h_cache, n_max
    )

    n_ops = len(spectrum)
    scalar_mask = np.array([p.spin == 0 for p in spectrum], dtype=bool)
    spinning_mask = ~scalar_mask
    scalar_deltas = np.array([p.delta for p in spectrum], dtype=np.float64)

    if verbose:
        n_scalar = np.sum(scalar_mask)
        n_spinning = np.sum(spinning_mask)
        print(f"  Constraint matrix: {A.shape[0]} operators "
              f"({n_scalar} scalars, {n_spinning} spinning)")

    return A, f_id, scalar_mask, scalar_deltas, spinning_mask


# =============================================================================
# Find epsilon bound at a single delta_sigma
# =============================================================================

def _make_sdpb_config(config: ScanConfig):
    """Build an SdpbConfig from a ScanConfig."""
    from ..lp.sdpb import SdpbConfig
    kwargs = {
        "precision": config.sdpb_precision,
        "n_cores": config.sdpb_cores,
        "timeout": config.sdpb_timeout,
        "verbose": config.verbose,
    }
    if config.sdpb_image is not None:
        kwargs["image_path"] = config.sdpb_image
    return SdpbConfig(**kwargs)


def find_eps_bound(
    delta_sigma: float,
    A: np.ndarray,
    f_id: np.ndarray,
    scalar_mask: np.ndarray,
    scalar_deltas: np.ndarray,
    spinning_mask: np.ndarray,
    config: ScanConfig,
    full_spectrum: Optional[list] = None,
) -> Tuple[float, int]:
    """
    Binary search for Δε_max at a fixed Δσ.

    Uses the precomputed full constraint matrix A and selects rows
    based on the trial gap value.

    Parameters
    ----------
    delta_sigma : float
        External scalar dimension.
    A : np.ndarray
        Full constraint matrix (all operators).
    f_id : np.ndarray
        Identity vector.
    scalar_mask, scalar_deltas, spinning_mask : np.ndarray
        Metadata from build_full_constraint_matrix.
    config : ScanConfig
        Scan configuration.
    full_spectrum : list of SpectrumPoint, optional
        Full spectrum (unused, kept for API compatibility).

    Returns
    -------
    eps_max : float
        Upper bound on Δε.
    n_iter : int
        Number of binary search iterations.

    Raises
    ------
    RuntimeError
        If the solver fails on any binary-search iteration.
    """
    sdpb_cfg = _make_sdpb_config(config) if config.backend == "sdpb" else None

    def is_excluded(gap: float) -> bool:
        # Select rows: all spinning operators + scalars with Δ >= gap
        mask = spinning_mask | (scalar_mask & (scalar_deltas >= gap - 1e-10))
        A_sub = A[mask]
        if A_sub.shape[0] == 0:
            return True

        result = check_feasibility(
            A_sub, f_id, scale=config.scale,
            backend=config.backend, sdpb_config=sdpb_cfg,
        )

        if not result.success:
            raise RuntimeError(
                f"Solver failed while testing gap={gap:.6f} at Δσ={delta_sigma:.6f}. "
                f"Failure: {result.status}"
            )
        if config.verbose:
            verdict = "EXCLUDED" if result.excluded else "ALLOWED"
            print(
                f"    gap={gap:.6f} rows={A_sub.shape[0]} -> {verdict} "
                f"({result.status})"
            )
        return result.excluded

    # Validate bisection bracket before wasting hours of compute
    if config.validate_bracket:
        lo_excluded = is_excluded(config.eps_lo)
        hi_excluded = is_excluded(config.eps_hi)

        if lo_excluded:
            raise RuntimeError(
                f"Invalid bisection bracket at Δσ={delta_sigma:.6f}: "
                f"lower bound gap={config.eps_lo:.6f} is excluded. "
                "This indicates a solver or precision issue and would force "
                "a misleading Δε_max=eps_lo result."
            )
        if not hi_excluded:
            raise RuntimeError(
                f"Invalid bisection bracket at Δσ={delta_sigma:.6f}: "
                f"upper bound gap={config.eps_hi:.6f} is still allowed. "
                "Increase eps_hi before running Stage A."
            )

    return binary_search_eps(
        is_excluded,
        lo=config.eps_lo,
        hi=config.eps_hi,
        tol=config.tolerance,
        max_iter=config.max_iter,
    )


# =============================================================================
# H cache loading
# =============================================================================

def load_h_cache_from_disk(
    spectrum: List[SpectrumPoint],
    n_max: int = N_MAX,
    verbose: bool = False,
) -> dict:
    """
    Load extended H arrays from disk cache into an in-memory dict.

    Parameters
    ----------
    spectrum : list of SpectrumPoint
        The full spectrum (determines which blocks to load).
    n_max : int
        Truncation parameter.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Mapping (delta, spin) -> np.ndarray of shape (22, 11).

    Raises
    ------
    FileNotFoundError
        If any required cache file is missing.
    """
    h_cache = {}
    unique_ops = set()
    for p in spectrum:
        unique_ops.add((round(p.delta, 8), p.spin))

    # Fast path: load from consolidated .npz archive (single NFS read instead of 520K)
    consolidated_path = CACHE_DIR / "ext_cache_consolidated.npz"
    if consolidated_path.exists():
        if verbose:
            print(f"  Loading consolidated cache: {consolidated_path}")
        data = np.load(consolidated_path)
        loaded = 0
        for key in data.files:
            delta_str, spin_str = key.rsplit("_", 1)
            delta, spin = float(delta_str), int(spin_str)
            rounded = round(delta, 8)
            if (rounded, spin) in unique_ops:
                h_cache[(rounded, spin)] = data[key]
                loaded += 1
        data.close()
        if verbose:
            print(f"  Loaded {loaded} extended block arrays from consolidated cache")
        return h_cache

    # Fallback: Use bulk directory listing instead of per-file existence checks
    cached_filenames = list_extended_cache_filenames()
    missing = []
    corrupted = 0
    for delta, spin in sorted(unique_ops):
        fname = f"ext_d{delta:.8f}_l{spin}.npy"
        if fname in cached_filenames:
            try:
                h_cache[(delta, spin)] = load_extended_h_array(delta, spin, n_max)
            except (EOFError, ValueError, OSError):
                # Corrupted file (e.g. truncated by SLURM timeout) — treat as missing
                corrupted += 1
                missing.append((delta, spin))
        else:
            missing.append((delta, spin))
    if corrupted and verbose:
        print(f"  WARNING: {corrupted} corrupted cache files, will recompute")

    if missing:
        if verbose:
            print(f"  {len(missing)} blocks not in disk cache, computing...")
        # Fall back to computing missing blocks via the LP function
        from ..lp.crossing import compute_extended_h_array
        for i, (delta, spin) in enumerate(missing):
            if verbose and (i % 100 == 0):
                print(f"    [{i+1}/{len(missing)}] (Δ={delta:.6f}, l={spin})")
            try:
                H = compute_extended_h_array(delta, spin, n_max)
                h_cache[(delta, spin)] = H
            except (ZeroDivisionError, ValueError):
                pass  # skip poles

    if verbose:
        print(f"  Loaded {len(h_cache)} extended block arrays")

    return h_cache


# =============================================================================
# CSV I/O
# =============================================================================

def write_csv_header(output_path: Path) -> None:
    """Write the CSV header for Stage A results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['delta_sigma', 'delta_eps_max'])


def append_result_to_csv(
    output_path: Path,
    delta_sigma: float,
    eps_max: float,
) -> None:
    """Append one result row to the CSV file."""
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'{delta_sigma:.6f}', f'{eps_max:.6f}'])


def load_scan_results(csv_path: Path) -> List[Tuple[float, float]]:
    """
    Load Stage A scan results from CSV.

    Returns
    -------
    list of (delta_sigma, delta_eps_max)
    """
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append((
                float(row['delta_sigma']),
                float(row['delta_eps_max']),
            ))
    return results


# =============================================================================
# Main scan loop
# =============================================================================

def run_scan(config: ScanConfig) -> List[Tuple[float, float]]:
    """
    Run the Stage A scan: compute Δε_max(Δσ) for each Δσ in the grid.

    Parameters
    ----------
    config : ScanConfig
        Scan configuration.

    Returns
    -------
    list of (delta_sigma, eps_max)
        Scan results.
    """
    tables = config.get_tables()
    sigma_grid = config.get_sigma_grid()

    if config.verbose:
        print(f"Stage A scan: {len(sigma_grid)} Δσ points "
              f"in [{config.sigma_min}, {config.sigma_max}]")
        print(f"Tables: {[t.name for t in tables]}")
        print(f"n_max = {config.n_max}, tolerance = {config.tolerance}")

    # Generate the full (ungapped) spectrum
    full_spectrum = generate_full_spectrum(tables=tables)
    if config.verbose:
        print(f"Full spectrum: {len(full_spectrum)} operators")

    # Load or compute the h_cache
    if config.verbose:
        print("Loading block cache...")

    h_cache = load_h_cache_from_disk(
        full_spectrum, n_max=config.n_max, verbose=config.verbose
    )

    # If blocks are missing and we couldn't compute them, compute in-memory
    if len(h_cache) == 0:
        if config.verbose:
            print("No disk cache found. Computing h_cache in memory...")
        h_cache = precompute_extended_blocks(full_spectrum, config.n_max,
                                             verbose=config.verbose)

    # Write CSV header
    write_csv_header(config.output)

    results = []

    for i, delta_sigma in enumerate(sigma_grid):
        if config.verbose:
            print(f"\n[{i+1}/{len(sigma_grid)}] Δσ = {delta_sigma:.4f}")

        # Build full constraint matrix for this Δσ
        A, f_id, scalar_mask, scalar_deltas, spinning_mask = \
            build_full_constraint_matrix(
                full_spectrum, delta_sigma, h_cache,
                n_max=config.n_max, verbose=config.verbose,
            )

        # Binary search for Δε_max
        eps_max, n_iter = find_eps_bound(
            delta_sigma, A, f_id,
            scalar_mask, scalar_deltas, spinning_mask,
            config,
            full_spectrum=full_spectrum,
        )

        if config.verbose:
            print(f"  Δε_max = {eps_max:.6f}  ({n_iter} iterations)")

        results.append((delta_sigma, eps_max))
        append_result_to_csv(config.output, delta_sigma, eps_max)

    if config.verbose:
        print(f"\nScan complete. Results written to {config.output}")

    return results


# =============================================================================
# Block precomputation mode
# =============================================================================

def run_precompute(config: ScanConfig) -> None:
    """
    Precompute extended block derivatives for the full spectrum.

    This is the expensive one-time computation. Results are saved to
    the disk cache for reuse across scan runs.
    """
    tables = config.get_tables()

    if config.verbose:
        print("Precomputing extended block derivatives...")
        print(f"Tables: {[t.name for t in tables]}")

    full_spectrum = generate_full_spectrum(tables=tables)
    if config.verbose:
        print(f"Full spectrum: {len(full_spectrum)} operators")

    # Extract unique (delta, spin) pairs
    unique_ops = sorted(set(
        (round(p.delta, 8), p.spin) for p in full_spectrum
    ))
    if config.verbose:
        print(f"Unique (Δ, l) pairs: {len(unique_ops)}")

    n_computed = precompute_extended_spectrum_blocks(
        unique_ops,
        n_max=config.n_max,
        skip_existing=True,
        verbose=config.verbose,
        workers=config.workers,
        shard_id=config.shard_id,
        num_shards=config.num_shards,
    )

    if config.verbose:
        print(f"\nPrecomputation complete: {n_computed} new blocks computed")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line entry point for Stage A scan."""
    parser = argparse.ArgumentParser(
        description="Stage A: compute upper bound on Δε as a function of Δσ"
    )
    parser.add_argument(
        "--sigma-min", type=float, default=DEFAULT_SIGMA_MIN,
        help=f"Minimum Δσ (default: {DEFAULT_SIGMA_MIN})"
    )
    parser.add_argument(
        "--sigma-max", type=float, default=DEFAULT_SIGMA_MAX,
        help=f"Maximum Δσ (default: {DEFAULT_SIGMA_MAX})"
    )
    parser.add_argument(
        "--sigma-step", type=float, default=DEFAULT_SIGMA_STEP,
        help=f"Δσ grid spacing (default: {DEFAULT_SIGMA_STEP})"
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_EPS_TOLERANCE,
        help=f"Binary search tolerance (default: {DEFAULT_EPS_TOLERANCE})"
    )
    parser.add_argument(
        "--output", type=str, default=str(DATA_DIR / "eps_bound.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--reduced", action="store_true",
        help="Use reduced discretization (T1-T2 only)"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for block cache (default: data/cached_blocks/)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress"
    )
    parser.add_argument(
        "--precompute-only", action="store_true",
        help="Only precompute block derivatives, don't run scan"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for precomputation (default: 1)"
    )
    parser.add_argument(
        "--shard-id", type=int, default=None,
        help="Shard index for parallel precomputation (0-based)"
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Total number of shards for parallel precomputation"
    )
    parser.add_argument(
        "--backend", type=str, default="sdpb",
        choices=["scipy", "sdpb"],
        help="LP solver backend (default: sdpb)"
    )
    parser.add_argument(
        "--sdpb-image", type=str, default=None,
        help="Path to SDPB Singularity .sif image"
    )
    parser.add_argument(
        "--sdpb-precision", type=int, default=1024,
        help="SDPB arithmetic precision in bits (default: 1024)"
    )
    parser.add_argument(
        "--sdpb-cores", type=int, default=4,
        help="Number of MPI cores for SDPB (default: 4)"
    )
    parser.add_argument(
        "--sdpb-timeout", type=int, default=600,
        help="SDPB timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--no-validate-bracket", action="store_true",
        help="Skip validating that gap=eps_lo is allowed and gap=eps_hi is "
             "excluded before binary search"
    )

    args = parser.parse_args()

    config = ScanConfig(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_step=args.sigma_step,
        tolerance=args.tolerance,
        output=Path(args.output),
        reduced=args.reduced,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        verbose=args.verbose,
        precompute_only=args.precompute_only,
        workers=args.workers,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        backend=args.backend,
        sdpb_image=Path(args.sdpb_image) if args.sdpb_image else None,
        sdpb_precision=args.sdpb_precision,
        sdpb_cores=args.sdpb_cores,
        sdpb_timeout=args.sdpb_timeout,
        validate_bracket=not args.no_validate_bracket,
    )

    if config.precompute_only:
        run_precompute(config)
    else:
        run_scan(config)


if __name__ == "__main__":
    main()
