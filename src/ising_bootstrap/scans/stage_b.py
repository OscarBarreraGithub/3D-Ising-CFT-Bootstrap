"""
Stage B scan: compute upper bound on Δε' as a function of Δσ.

For each Δσ in the scan grid, use the Stage A result Δε_max(Δσ) to impose
two gap conditions:
    1. No scalars below Δε (same as Stage A)
    2. No scalars between Δε and Δε' (the new gap)

Binary search over Δε' to find the largest second gap that is still
consistent with crossing symmetry. This yields the curve Δε'_max(Δσ),
as in Figure 6 of arXiv:1203.6064.

Binary search logic (same direction as Stage A):
    - Imposing a larger gap removes more scalars from the positivity
      constraints, making the LP easier to satisfy (fewer constraints).
    - LP feasible (excluded): the gap is too large → lower hi
    - LP infeasible (allowed): the gap is consistent → raise lo
    - At convergence: lo ≈ Δε'_max

Usage:
    python -m ising_bootstrap.scans.stage_b \\
        --eps-bound data/eps_bound.csv \\
        --sigma-min 0.50 --sigma-max 0.60 --sigma-step 0.002 \\
        --tolerance 1e-3 --output data/epsprime_bound.csv --verbose

Reference: arXiv:1203.6064, Sections 5-6 and Figure 6
"""

import argparse
import csv
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from ..config import (
    N_MAX, BINARY_SEARCH_MAX_ITER,
    DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_MAX, DEFAULT_SIGMA_STEP,
    DEFAULT_EPSPRIME_TOLERANCE,
    FULL_DISCRETIZATION, REDUCED_DISCRETIZATION,
    DATA_DIR,
    DiscretizationTable,
)
from ..spectrum.discretization import (
    SpectrumPoint, generate_full_spectrum,
)
from ..lp.solver import check_feasibility
from .stage_a import (
    binary_search_eps,
    build_full_constraint_matrix,
    load_h_cache_from_disk,
    load_scan_results,
    run_precompute as stage_a_precompute,
    ScanConfig,
)
from ..lp.constraint_matrix import precompute_extended_blocks


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StageBConfig:
    """Configuration for a Stage B scan."""
    eps_bound_path: Optional[Path] = None
    sigma_min: float = DEFAULT_SIGMA_MIN
    sigma_max: float = DEFAULT_SIGMA_MAX
    sigma_step: float = DEFAULT_SIGMA_STEP
    tolerance: float = DEFAULT_EPSPRIME_TOLERANCE
    max_iter: int = BINARY_SEARCH_MAX_ITER
    eps_prime_hi: float = 6.0     # generous upper bound for Δε'
    n_max: int = N_MAX
    tables: Optional[List[DiscretizationTable]] = None
    reduced: bool = False
    cache_dir: Optional[Path] = None
    output: Path = field(default_factory=lambda: DATA_DIR / "epsprime_bound.csv")
    verbose: bool = False
    precompute_only: bool = False
    scale: bool = True
    workers: int = 1

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
# Two-gap exclusion
# =============================================================================

def find_eps_prime_bound(
    delta_sigma: float,
    delta_eps: float,
    A: np.ndarray,
    f_id: np.ndarray,
    scalar_mask: np.ndarray,
    scalar_deltas: np.ndarray,
    spinning_mask: np.ndarray,
    config: StageBConfig,
) -> Tuple[float, int]:
    """
    Binary search for Δε'_max at a fixed Δσ with Δε fixed from Stage A.

    Uses the precomputed full constraint matrix A and selects rows
    based on two gap conditions:
        - Exclude scalars with Δ < Δε (below first gap)
        - Exclude scalars with Δε ≤ Δ < Δε' (between first and second gap)
        - Include scalars with Δ ≥ Δε' (above second gap)
        - Include all spinning operators

    Parameters
    ----------
    delta_sigma : float
        External scalar dimension.
    delta_eps : float
        Fixed first scalar gap from Stage A.
    A : np.ndarray
        Full constraint matrix (all operators).
    f_id : np.ndarray
        Identity vector.
    scalar_mask, scalar_deltas, spinning_mask : np.ndarray
        Metadata from build_full_constraint_matrix.
    config : StageBConfig
        Scan configuration.

    Returns
    -------
    eps_prime_max : float
        Upper bound on Δε'.
    n_iter : int
        Number of binary search iterations.
    """
    # Precompute the mask for scalars below the first gap (always excluded)
    scalars_below_eps = scalar_mask & (scalar_deltas < delta_eps - 1e-10)

    def is_excluded(eps_prime_trial: float) -> bool:
        # Include: spinning + scalars below Δε + scalars at or above Δε'
        # This means scalars in [Δε, Δε') are excluded (the second gap)
        scalars_above_eps_prime = scalar_mask & (
            scalar_deltas >= eps_prime_trial - 1e-10
        )
        mask = spinning_mask | scalars_below_eps | scalars_above_eps_prime
        A_sub = A[mask]
        if A_sub.shape[0] == 0:
            return True
        result = check_feasibility(A_sub, f_id, scale=config.scale)
        return result.excluded

    # Search range: just above Δε to the upper bound
    lo = delta_eps
    hi = config.eps_prime_hi

    return binary_search_eps(
        is_excluded,
        lo=lo,
        hi=hi,
        tol=config.tolerance,
        max_iter=config.max_iter,
    )


# =============================================================================
# CSV I/O
# =============================================================================

def write_csv_header(output_path: Path) -> None:
    """Write the CSV header for Stage B results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['delta_sigma', 'delta_eps', 'delta_eps_prime_max'])


def append_result_to_csv(
    output_path: Path,
    delta_sigma: float,
    delta_eps: float,
    eps_prime_max: float,
) -> None:
    """Append one result row to the CSV file."""
    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f'{delta_sigma:.6f}',
            f'{delta_eps:.6f}',
            f'{eps_prime_max:.6f}',
        ])


def load_stage_b_results(
    csv_path: Path,
) -> List[Tuple[float, float, float]]:
    """
    Load Stage B scan results from CSV.

    Returns
    -------
    list of (delta_sigma, delta_eps, delta_eps_prime_max)
    """
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append((
                float(row['delta_sigma']),
                float(row['delta_eps']),
                float(row['delta_eps_prime_max']),
            ))
    return results


# =============================================================================
# Stage A result loading
# =============================================================================

def load_eps_bound_map(
    eps_bound_path: Path,
) -> Dict[float, float]:
    """
    Load Stage A results and return a mapping from Δσ to Δε_max.

    The Δσ values are rounded to 6 decimal places for robust matching.

    Parameters
    ----------
    eps_bound_path : Path
        Path to Stage A CSV file.

    Returns
    -------
    dict
        Mapping from delta_sigma (rounded) to delta_eps_max.
    """
    stage_a_results = load_scan_results(eps_bound_path)
    return {round(ds, 6): de for ds, de in stage_a_results}


# =============================================================================
# Main scan loop
# =============================================================================

def run_scan(
    config: StageBConfig,
) -> List[Tuple[float, float, float]]:
    """
    Run the Stage B scan: compute Δε'_max(Δσ) for each Δσ in the grid.

    Parameters
    ----------
    config : StageBConfig
        Scan configuration. Must have eps_bound_path set.

    Returns
    -------
    list of (delta_sigma, delta_eps, eps_prime_max)
        Scan results.
    """
    if config.eps_bound_path is None:
        raise ValueError(
            "eps_bound_path is required for Stage B scan. "
            "Provide path to Stage A CSV with --eps-bound."
        )

    tables = config.get_tables()
    sigma_grid = config.get_sigma_grid()

    if config.verbose:
        print(f"Stage B scan: {len(sigma_grid)} Δσ points "
              f"in [{config.sigma_min}, {config.sigma_max}]")
        print(f"Tables: {[t.name for t in tables]}")
        print(f"n_max = {config.n_max}, tolerance = {config.tolerance}")
        print(f"Stage A results: {config.eps_bound_path}")

    # Load Stage A results
    sigma_to_eps = load_eps_bound_map(config.eps_bound_path)
    if config.verbose:
        print(f"Loaded {len(sigma_to_eps)} Stage A data points")

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

    if len(h_cache) == 0:
        if config.verbose:
            print("No disk cache found. Computing h_cache in memory...")
        h_cache = precompute_extended_blocks(full_spectrum, config.n_max,
                                             verbose=config.verbose)

    # Write CSV header
    write_csv_header(config.output)

    results = []
    for i, delta_sigma in enumerate(sigma_grid):
        # Look up Δε from Stage A
        ds_key = round(delta_sigma, 6)
        if ds_key not in sigma_to_eps:
            if config.verbose:
                print(f"\n[{i+1}/{len(sigma_grid)}] Δσ = {delta_sigma:.4f} "
                      f"— SKIPPED (no Stage A result)")
            continue

        delta_eps = sigma_to_eps[ds_key]

        if config.verbose:
            print(f"\n[{i+1}/{len(sigma_grid)}] Δσ = {delta_sigma:.4f}, "
                  f"Δε = {delta_eps:.6f}")

        # Build full constraint matrix for this Δσ
        A, f_id, scalar_mask, scalar_deltas, spinning_mask = \
            build_full_constraint_matrix(
                full_spectrum, delta_sigma, h_cache,
                n_max=config.n_max, verbose=config.verbose,
            )

        # Binary search for Δε'_max
        eps_prime_max, n_iter = find_eps_prime_bound(
            delta_sigma, delta_eps,
            A, f_id,
            scalar_mask, scalar_deltas, spinning_mask,
            config,
        )

        if config.verbose:
            print(f"  Δε'_max = {eps_prime_max:.6f}  ({n_iter} iterations)")

        results.append((delta_sigma, delta_eps, eps_prime_max))

        # Write result immediately (crash recovery)
        append_result_to_csv(config.output, delta_sigma, delta_eps,
                             eps_prime_max)

    if config.verbose:
        print(f"\nScan complete. Results written to {config.output}")

    return results


# =============================================================================
# Block precomputation (delegates to Stage A)
# =============================================================================

def run_precompute(config: StageBConfig) -> None:
    """
    Precompute extended block derivatives for the full spectrum.

    Delegates to Stage A precomputation since blocks are independent
    of the scan stage.
    """
    stage_a_config = ScanConfig(
        n_max=config.n_max,
        tables=config.get_tables(),
        reduced=config.reduced,
        verbose=config.verbose,
        workers=config.workers,
    )
    stage_a_precompute(stage_a_config)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line entry point for Stage B scan."""
    parser = argparse.ArgumentParser(
        description="Stage B: compute upper bound on Δε' as a function of Δσ"
    )
    parser.add_argument(
        "--eps-bound", type=str, required=True,
        help="Path to Stage A CSV file (delta_sigma, delta_eps_max)"
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
        "--tolerance", type=float, default=DEFAULT_EPSPRIME_TOLERANCE,
        help=f"Binary search tolerance (default: {DEFAULT_EPSPRIME_TOLERANCE})"
    )
    parser.add_argument(
        "--output", type=str, default=str(DATA_DIR / "epsprime_bound.csv"),
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

    args = parser.parse_args()

    config = StageBConfig(
        eps_bound_path=Path(args.eps_bound),
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
    )

    if config.precompute_only:
        run_precompute(config)
    else:
        run_scan(config)


if __name__ == "__main__":
    main()
