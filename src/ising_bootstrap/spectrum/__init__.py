"""
Spectrum discretization module.

Implements:
- (m, n) index set generation with constraints m odd, m+2n ≤ 21
- Table 2 discretization (T1-T5) from the paper
- Unitarity bound checking
- Spectrum union across tables
"""

from .index_set import (
    generate_index_set,
    iter_index_set,
    index_set_size,
    is_valid_index_pair,
    get_index_position,
    get_pair_at_position,
    IndexPair,
)

from .unitarity import (
    unitarity_bound,
    satisfies_unitarity,
    check_unitarity,
    is_allowed_spin,
    validate_operator,
)

from .discretization import (
    SpectrumPoint,
    Operator,
    generate_table_points,
    generate_full_spectrum,
    spectrum_to_array,
    count_by_spin,
    count_by_table,
    get_scalars,
    get_spinning,
    build_spectrum_with_gaps,
    estimate_spectrum_size,
    get_table_info,
)

__all__ = [
    # Index set
    "generate_index_set",
    "iter_index_set",
    "index_set_size",
    "is_valid_index_pair",
    "get_index_position",
    "get_pair_at_position",
    "IndexPair",
    # Unitarity
    "unitarity_bound",
    "satisfies_unitarity",
    "check_unitarity",
    "is_allowed_spin",
    "validate_operator",
    # Discretization
    "SpectrumPoint",
    "Operator",
    "generate_table_points",
    "generate_full_spectrum",
    "spectrum_to_array",
    "count_by_spin",
    "count_by_table",
    "get_scalars",
    "get_spinning",
    "build_spectrum_with_gaps",
    "estimate_spectrum_size",
    "get_table_info",
]
