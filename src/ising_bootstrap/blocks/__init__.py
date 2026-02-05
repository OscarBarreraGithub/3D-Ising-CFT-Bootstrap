"""
Conformal block computation module.

Implements:
- Diagonal blocks G_{Δ,l}(z) at z=z̄ via ₃F₂ hypergeometric functions
- Spin recursion for l ≥ 2
- Derivative computation (a-derivatives and b-derivatives via Casimir)
- Block derivative caching

Main entry points:
- `block_derivatives_full(delta, spin)`: Compute all h_{m,n} for the index set
- `get_or_compute(delta, spin)`: Get block derivatives as numpy array (with caching)
- `diagonal_block_any_spin(delta, spin)`: Compute G_{Δ,l}(z) for any spin

Reference: arXiv:1203.6064
"""

# Base blocks: G_{Δ,l}(z) at z=z̄ via ₃F₂
from .diagonal_blocks import (
    spin0_block,
    spin1_block,
    diagonal_block,
)

# Spin recursion for l ≥ 2
from .spin_recursion import (
    higher_spin_block,
    diagonal_block_any_spin,
)

# z-derivatives via ₃F₂ ODE recursion
from .z_derivatives import (
    block_z_derivatives,
    spin0_block_z_derivatives,
    spin1_block_z_derivatives,
    higher_spin_block_z_derivatives,
)

# Coordinate transformation (z,z̄) <-> (a,b)
from .coordinate_transform import (
    z_zbar_to_a_b,
    a_b_to_z_zbar,
    z_derivatives_to_h_m0,
    h_m0_to_z_derivatives,
    compute_h_m0_from_block_derivs,
    crossing_point_values,
)

# Transverse (b) derivatives via Casimir recursion
from .transverse_derivs import (
    block_derivatives_full,
    block_derivatives_as_vector,
    compute_all_h_mn,
    casimir_eigenvalue,
)

# Disk caching for block derivatives
from .cache import (
    get_or_compute,
    compute_and_cache,
    precompute_spectrum_blocks,
    load_block_derivatives,
    load_block_derivatives_as_vector,
    save_block_derivatives,
    cache_exists,
    cache_stats,
    clear_cache,
    # Extended H array cache (for LP)
    extended_cache_exists,
    list_extended_cache_filenames,
    save_extended_h_array,
    load_extended_h_array,
    precompute_extended_spectrum_blocks,
)


__all__ = [
    # Base blocks
    "spin0_block",
    "spin1_block",
    "diagonal_block",
    # Spin recursion
    "higher_spin_block",
    "diagonal_block_any_spin",
    # z-derivatives
    "block_z_derivatives",
    "spin0_block_z_derivatives",
    "spin1_block_z_derivatives",
    "higher_spin_block_z_derivatives",
    # Coordinate transform
    "z_zbar_to_a_b",
    "a_b_to_z_zbar",
    "z_derivatives_to_h_m0",
    "h_m0_to_z_derivatives",
    "compute_h_m0_from_block_derivs",
    "crossing_point_values",
    # Transverse derivatives
    "block_derivatives_full",
    "block_derivatives_as_vector",
    "compute_all_h_mn",
    "casimir_eigenvalue",
    # Cache
    "get_or_compute",
    "compute_and_cache",
    "precompute_spectrum_blocks",
    "load_block_derivatives",
    "load_block_derivatives_as_vector",
    "save_block_derivatives",
    "cache_exists",
    "cache_stats",
    "clear_cache",
    # Extended H array cache
    "extended_cache_exists",
    "list_extended_cache_filenames",
    "save_extended_h_array",
    "load_extended_h_array",
    "precompute_extended_spectrum_blocks",
]
