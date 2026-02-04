"""
Bootstrap scan module.

Implements:
- Stage A: Δε bound computation (reproducing Fig. 3 boundary)
- Stage B: Δε' bound computation (reproducing Fig. 6)
"""

from .stage_a import (
    ScanConfig,
    binary_search_eps,
    find_eps_bound,
    build_full_constraint_matrix,
    run_scan,
    run_precompute,
    load_scan_results,
)

from .stage_b import (
    StageBConfig,
    find_eps_prime_bound,
    run_scan as run_scan_stage_b,
    run_precompute as run_precompute_stage_b,
    load_stage_b_results,
    load_eps_bound_map,
)

__all__ = [
    "ScanConfig",
    "binary_search_eps",
    "find_eps_bound",
    "build_full_constraint_matrix",
    "run_scan",
    "run_precompute",
    "load_scan_results",
    "StageBConfig",
    "find_eps_prime_bound",
    "run_scan_stage_b",
    "run_precompute_stage_b",
    "load_stage_b_results",
    "load_eps_bound_map",
]
