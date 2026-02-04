"""
Linear programming module for the conformal bootstrap.

Implements:
- Crossing function derivatives F^{Δσ}_{Δ,l} at the crossing point
- Identity contribution derivatives
- Constraint matrix construction
- LP feasibility solver with row/column scaling

Main entry points:
- `solve_bootstrap(delta_sigma, ...)`: End-to-end bootstrap feasibility test
- `check_feasibility(A, f_id)`: Low-level LP solver
- `build_constraint_matrix(spectrum, delta_sigma)`: Matrix construction

Reference: arXiv:1203.6064, Appendix D
"""

from .crossing import (
    compute_prefactor_table,
    compute_identity_vector,
    compute_extended_h_array,
    compute_crossing_vector,
    compute_crossing_vector_fast,
    generate_extended_pairs,
    extended_pair_count,
    build_comb_cache,
)

from .constraint_matrix import (
    build_constraint_matrix,
    build_constraint_matrix_from_cache,
    precompute_extended_blocks,
)

from .solver import (
    FeasibilityResult,
    check_feasibility,
    scale_constraints,
    solve_bootstrap,
)

__all__ = [
    # Crossing derivatives
    "compute_prefactor_table",
    "compute_identity_vector",
    "compute_extended_h_array",
    "compute_crossing_vector",
    "compute_crossing_vector_fast",
    "generate_extended_pairs",
    "extended_pair_count",
    "build_comb_cache",
    # Constraint matrix
    "build_constraint_matrix",
    "build_constraint_matrix_from_cache",
    "precompute_extended_blocks",
    # Solver
    "FeasibilityResult",
    "check_feasibility",
    "scale_constraints",
    "solve_bootstrap",
]
