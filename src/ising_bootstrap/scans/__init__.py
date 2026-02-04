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

__all__ = [
    "ScanConfig",
    "binary_search_eps",
    "find_eps_bound",
    "build_full_constraint_matrix",
    "run_scan",
    "run_precompute",
    "load_scan_results",
]
