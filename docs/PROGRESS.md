# Implementation Progress

Tracking document for reproducing Figure 6 from arXiv:1203.6064.

**Last updated:** 2026-02-03
**Test suite:** 250/250 passing (20s)

---

## Milestone Summary

| # | Milestone | Status | Source Lines | Test Lines | Tests |
|---|-----------|--------|-------------|------------|-------|
| 0 | Repository scaffolding | DONE | 212 | 177 | 21 |
| 1 | Conformal block engine | DONE | 2,306 | 1,128 | 84 |
| 2 | Spectrum discretization | DONE | 933 | 733 | 88 |
| 3 | LP builder & solver | DONE | 912 | 750 | 57 |
| 4 | Stage A scan (Delta_epsilon) | NOT STARTED | 7 (stub) | 0 | 0 |
| 5 | Stage B scan (Delta_epsilon') | NOT STARTED | 0 | 0 | 0 |
| 6 | Plotting & validation | NOT STARTED | 6 (stub) | 0 | 0 |

---

## Milestone 0: Repository Scaffolding -- DONE

### Files
| File | Lines | Description |
|------|-------|-------------|
| `src/ising_bootstrap/__init__.py` | 11 | Package root |
| `src/ising_bootstrap/config.py` | 201 | All constants, Table 2 params, Ising reference values |

### Tests (21 passing)
| File | Tests | Description |
|------|-------|-------------|
| `tests/test_config.py` | 21 | Constants, crossing point, unitarity bounds, Table 2 params |

### Test Details -- `tests/test_config.py`
```
PASSED  TestConstants::test_dimension                         D = 3
PASSED  TestConstants::test_n_max                             N_MAX = 10
PASSED  TestConstants::test_alpha                             ALPHA = 0.5
PASSED  TestConstants::test_max_deriv_order                   MAX_DERIV_ORDER = 21
PASSED  TestCrossingPoint::test_z_point                       z = 1/2
PASSED  TestCrossingPoint::test_a_point                       a = 1
PASSED  TestCrossingPoint::test_b_point                       b = 0
PASSED  TestCrossingPoint::test_u_v_values                    u = v = 1/4
PASSED  TestUnitarityBounds::test_scalar_bound                Delta >= 1/2
PASSED  TestUnitarityBounds::test_spin_1_bound                Delta >= 2
PASSED  TestUnitarityBounds::test_spin_2_bound                Delta >= 3
PASSED  TestUnitarityBounds::test_general_spin_bound          Delta >= l + D - 2
PASSED  TestUnitarityBounds::test_negative_spin_raises
PASSED  TestIndexSetCount::test_count_n_max_10                66 pairs
PASSED  TestIndexSetCount::test_count_n_max_5
PASSED  TestIndexSetCount::test_count_n_max_1
PASSED  TestDiscretizationTables::test_table_1                T1: delta=2e-5, Dmax=3, Lmax=0
PASSED  TestDiscretizationTables::test_table_2                T2: delta=5e-4, Dmax=8, Lmax=6
PASSED  TestDiscretizationTables::test_table_3                T3: delta=2e-3, Dmax=22, Lmax=20
PASSED  TestDiscretizationTables::test_table_4                T4: delta=0.02, Dmax=100, Lmax=50
PASSED  TestDiscretizationTables::test_table_5                T5: delta=1.0, Dmax=500, Lmax=100
```

---

## Milestone 1: Conformal Block Engine -- DONE

Implements evaluation of h_{m,n}(Delta, l) = d_a^m d_b^n G|_{a=1,b=0} for
the full index set (66 derivatives per operator).

### Files
| File | Lines | Description |
|------|-------|-------------|
| `src/ising_bootstrap/blocks/__init__.py` | 106 | Public API exports |
| `src/ising_bootstrap/blocks/diagonal_blocks.py` | 241 | Spin-0/1 via 3F2 (Eq. 4.10-4.11) |
| `src/ising_bootstrap/blocks/spin_recursion.py` | 286 | Higher spin l>=2 (Eq. 4.9) |
| `src/ising_bootstrap/blocks/z_derivatives.py` | 635 | d^m/dz^m G via Faa di Bruno + 3F2 ODE (Eq. 4.12) |
| `src/ising_bootstrap/blocks/coordinate_transform.py` | 266 | (z,zbar) <-> (a,b), h_{m,0} from z-derivs |
| `src/ising_bootstrap/blocks/transverse_derivs.py` | 366 | Casimir recursion for h_{m,n}, n>0 (Eq. C.1) |
| `src/ising_bootstrap/blocks/cache.py` | 406 | Disk caching (NPZ) to data/cached_blocks/ |

### Key Public API
```python
from ising_bootstrap.blocks import (
    block_derivatives_full,      # dict of h_{m,n} for all 66 index pairs
    block_derivatives_as_vector,  # same as numpy array
    get_or_compute,              # cached numpy array (primary entry point)
    diagonal_block_any_spin,     # G_{Delta,l}(z) value only
)
```

### Paper Equations Implemented
| Equation | Module | Description |
|----------|--------|-------------|
| Eq. 4.10 | diagonal_blocks.py | Spin-0 block via 3F2 |
| Eq. 4.11 | diagonal_blocks.py | Spin-1 block via 3F2 |
| Eq. 4.9 | spin_recursion.py | Spin recursion (l>=2 from l=0,1) |
| Eq. 4.12 | z_derivatives.py | 3F2 ODE for z-derivative recursion |
| Eq. 4.15 | coordinate_transform.py | (z,zbar) <-> (a,b) transformation |
| Eq. C.1 | transverse_derivs.py | Casimir recursion for b-derivatives |

### Tests (84 passing)
| File | Tests | Description |
|------|-------|-------------|
| `tests/test_blocks/test_diagonal_blocks.py` | 12 | Spin-0/1 blocks, 3F2 argument, prefactor |
| `tests/test_blocks/test_spin_recursion.py` | 11 | l>=2 recursion, coefficients, stress tensor |
| `tests/test_blocks/test_z_derivatives.py` | 11 | z-derivatives, numerical cross-check |
| `tests/test_blocks/test_coordinate_transform.py` | 15 | Round-trip transforms, (u,v), chain rule |
| `tests/test_blocks/test_transverse_derivs.py` | 16 | Casimir eigenvalue, h_{m,n} structure, full n_max=10 |
| `tests/test_blocks/test_cache.py` | 15 | Save/load round-trip, filename format, cache stats |
| (4 have `@pytest.mark.slow`) | | Full 66-derivative computation at n_max=10 |

### Test Details -- `tests/test_blocks/test_diagonal_blocks.py`
```
PASSED  TestSpin0Block::test_spin0_block_positive_delta       G_{1.5,0} is finite
PASSED  TestSpin0Block::test_spin0_block_unitarity_bound      G_{0.5181,0} > 0
PASSED  TestSpin0Block::test_spin0_block_large_delta          G decreases with Delta
PASSED  TestSpin0Block::test_hyp3f2_argument_at_half          z^2/(4(z-1)) = -1/8
PASSED  TestSpin0Block::test_prefactor_base_at_half           z^2/(1-z) = 1/2
PASSED  TestSpin1Block::test_spin1_block_positive_delta       G_{2,1} is finite
PASSED  TestSpin1Block::test_spin1_block_conserved_current    G_{2,1} > 0
PASSED  TestDiagonalBlock::test_diagonal_block_spin0          Matches spin0_block
PASSED  TestDiagonalBlock::test_diagonal_block_spin1          Matches spin1_block
PASSED  TestDiagonalBlock::test_diagonal_block_negative_spin_raises
PASSED  TestBlockConsistency::test_blocks_at_crossing_point   Real-valued at z=1/2
PASSED  TestBlockConsistency::test_identity_block_is_one      (placeholder)
```

### Test Details -- `tests/test_blocks/test_spin_recursion.py`
```
PASSED  TestHigherSpinBlock::test_spin0_uses_base_formula     l=0 -> spin0_block
PASSED  TestHigherSpinBlock::test_spin1_uses_base_formula     l=1 -> spin1_block
PASSED  TestHigherSpinBlock::test_spin2_finite                G_{3,2} is finite
PASSED  TestHigherSpinBlock::test_spin4_finite                G_{5,4} is finite
PASSED  TestHigherSpinBlock::test_higher_spin_positive        G > 0 above unitarity
PASSED  TestHigherSpinBlock::test_negative_spin_raises
PASSED  TestSpinRecursionCoefficients::test_lhs_coeff_d3      (l)(2D-1) for D=3
PASSED  TestSpinRecursionCoefficients::test_first_term_coeff_d3
PASSED  TestDiagonalBlockAnySpin::test_matches_higher_spin_block
PASSED  TestCacheConsistency::test_cache_reuse                In-memory cache works
PASSED  TestStressTensorBlock::test_stress_tensor_block       Delta=D=3, l=2
```

### Test Details -- `tests/test_blocks/test_z_derivatives.py`
```
PASSED  TestSpin0ZDerivatives::test_zeroth_derivative_matches_block
PASSED  TestSpin0ZDerivatives::test_derivatives_finite        Orders 0-10 all finite
PASSED  TestSpin0ZDerivatives::test_first_derivative_numerical  vs mpmath.diff, <1e-10
PASSED  TestSpin1ZDerivatives::test_zeroth_derivative_matches_block
PASSED  TestSpin1ZDerivatives::test_derivatives_finite
PASSED  TestHigherSpinZDerivatives::test_spin2_zeroth_derivative_matches
PASSED  TestHigherSpinZDerivatives::test_spin4_derivatives_finite
PASSED  TestBlockZDerivatives::test_spin0_routing
PASSED  TestBlockZDerivatives::test_spin1_routing
PASSED  TestBlockZDerivatives::test_max_derivative_order      Correct length
PASSED  TestDerivativeConsistency::test_higher_order_numerical_check  d^2G/dz^2 vs numerical
PASSED  TestDerivativeConsistency::test_derivatives_needed_for_bootstrap  26 derivs all finite
```

### Test Details -- `tests/test_blocks/test_coordinate_transform.py`
```
PASSED  TestZZbarToAB::test_crossing_point                    z=zbar=1/2 -> a=1, b=0
PASSED  TestZZbarToAB::test_diagonal                          z=zbar -> b=0
PASSED  TestZZbarToAB::test_off_diagonal                      z!=zbar -> b>0
PASSED  TestABToZZbar::test_crossing_point                    a=1,b=0 -> z=zbar=1/2
PASSED  TestABToZZbar::test_b_zero_diagonal                   b=0 -> z=zbar=a/2
PASSED  TestABToZZbar::test_positive_b                        b>0 -> z>zbar
PASSED  TestRoundTrip::test_z_to_a_to_z                       Identity up to 1e-30
PASSED  TestRoundTrip::test_a_to_z_to_a                       Identity up to 1e-30
PASSED  TestZZbarToUV::test_crossing_point                    u=v=1/4
PASSED  TestZZbarToUV::test_formulas                          u=z*zbar, v=(1-z)(1-zbar)
PASSED  TestDerivativeConversion::test_diagonal_factor        (1/2)^m
PASSED  TestDerivativeConversion::test_z_derivs_to_h_m0       h_{m,0} = (1/2)^m d_z^m G
PASSED  TestDerivativeConversion::test_h_m0_to_z_derivs       d_z^m G = 2^m h_{m,0}
PASSED  TestDerivativeConversion::test_round_trip_conversions
PASSED  TestCrossingPointValues::test_all_keys_present
PASSED  TestCrossingPointValues::test_values_correct
PASSED  TestComputeHm0FromBlockDerivs::test_correct_length
PASSED  TestComputeHm0FromBlockDerivs::test_all_finite
PASSED  TestComputeHm0FromBlockDerivs::test_different_spins   l=0,1,2,4
```

### Test Details -- `tests/test_blocks/test_transverse_derivs.py`
```
PASSED  TestCasimirEigenvalue::test_scalar_eigenvalue          C = Delta(Delta-3)
PASSED  TestCasimirEigenvalue::test_spin2_eigenvalue           C = Delta(Delta-3)+l(l+1)
PASSED  TestCasimirEigenvalue::test_stress_tensor_eigenvalue   C = 6
PASSED  TestBlockDerivativesFull::test_correct_number_of_derivatives
PASSED  TestBlockDerivativesFull::test_all_derivatives_finite  All h_{m,n} finite
PASSED  TestBlockDerivativesFull::test_diagonal_derivatives_present
PASSED  TestBlockDerivativesFull::test_transverse_derivatives_present
PASSED  TestBlockDerivativesAsVector::test_correct_length
PASSED  TestBlockDerivativesAsVector::test_matches_dict_ordering
PASSED  TestComputeAllHmn::test_with_known_h_m0               h_{m,0} preserved
PASSED  TestDifferentSpins::test_spin0_block                   Delta=1.41, all finite
PASSED  TestDifferentSpins::test_spin2_block                   Delta=3.0, all finite
PASSED  TestDifferentSpins::test_spin4_block                   Delta=5.0, all finite
PASSED  TestFullN_max10::test_full_index_set_spin0       [slow] 66 derivs, l=0
PASSED  TestFullN_max10::test_full_index_set_spin2       [slow] 66 derivs, l=2
```

### Test Details -- `tests/test_blocks/test_cache.py`
```
PASSED  TestGetCacheFilename::test_filename_format            d1.500000_l2.npz
PASSED  TestGetCacheFilename::test_filename_precision         6 decimal places
PASSED  TestGetCacheFilename::test_returns_path_object
PASSED  TestGetCacheFilename::test_in_cache_dir
PASSED  TestCacheExists::test_nonexistent_cache
PASSED  TestSaveLoadRoundTrip::test_save_and_load_dict        Dict round-trip
PASSED  TestSaveLoadRoundTrip::test_save_and_load_vector      Vector round-trip
PASSED  TestSaveLoadRoundTrip::test_overwrite_protection      FileExistsError
PASSED  TestSaveLoadRoundTrip::test_n_max_mismatch_raises     ValueError
PASSED  TestComputeAndCache::test_computes_and_saves          Creates cache file
PASSED  TestComputeAndCache::test_loads_from_cache            Rel error < 1e-14
PASSED  TestGetOrCompute::test_returns_numpy_array            float64 ndarray
PASSED  TestGetOrCompute::test_correct_length
PASSED  TestCacheStats::test_stats_keys
PASSED  TestCacheStats::test_stats_types
```

---

## Milestone 2: Spectrum Discretization & Index Set -- DONE

Implements Table 2 from the paper: discretized spectrum of operator dimensions
and spins, plus the (m,n) index set for the linear functional.

### Files
| File | Lines | Description |
|------|-------|-------------|
| `src/ising_bootstrap/spectrum/__init__.py` | 72 | Public API exports |
| `src/ising_bootstrap/spectrum/index_set.py` | 276 | (m,n) pairs: m odd, m+2n<=21, 66 total |
| `src/ising_bootstrap/spectrum/discretization.py` | 388 | Tables T1-T5, spectrum generation, gap building |
| `src/ising_bootstrap/spectrum/unitarity.py` | 197 | Unitarity bound checks |

### Tests (88 passing)
| File | Tests | Description |
|------|-------|-------------|
| `tests/test_index_set.py` | 21 | Count=66, membership, positions, round-trip |
| `tests/test_discretization.py` | 46 | Table params, points, unitarity, gaps, helpers |
| `tests/test_config.py` (partial) | 21 | Table params also tested in config |

### Test Details -- `tests/test_index_set.py`
```
PASSED  TestGenerateIndexSet::test_count_is_66                Exactly 66 pairs
PASSED  TestGenerateIndexSet::test_count_matches_formula
PASSED  TestGenerateIndexSet::test_specific_elements_present  (1,0), (1,10), (21,0)
PASSED  TestGenerateIndexSet::test_specific_elements_absent   Even m excluded
PASSED  TestGenerateIndexSet::test_all_m_are_odd
PASSED  TestGenerateIndexSet::test_all_satisfy_constraint     m + 2n <= 21
PASSED  TestGenerateIndexSet::test_all_m_positive
PASSED  TestGenerateIndexSet::test_all_n_nonnegative
PASSED  TestGenerateIndexSet::test_detailed_count_by_m        Per-m counts
PASSED  TestIterIndexSet::test_yields_same_as_generate
PASSED  TestIterIndexSet::test_lazy_evaluation
PASSED  TestIndexSetSize::test_size_10                        66
PASSED  TestIndexSetSize::test_size_5
PASSED  TestIndexSetSize::test_agrees_with_formula
PASSED  TestIsValidIndexPair::test_valid_pairs
PASSED  TestIsValidIndexPair::test_even_m_invalid
PASSED  TestIsValidIndexPair::test_exceeds_constraint_invalid
PASSED  TestIsValidIndexPair::test_negative_values_invalid
PASSED  TestGetIndexPosition::test_first_element
PASSED  TestGetIndexPosition::test_sequential_positions
PASSED  TestGetIndexPosition::test_m3_starts_at_11
PASSED  TestGetIndexPosition::test_last_element
PASSED  TestGetIndexPosition::test_invalid_raises
PASSED  TestGetIndexPosition::test_roundtrip
PASSED  TestGetPairAtPosition::test_first_position
PASSED  TestGetPairAtPosition::test_last_position
PASSED  TestGetPairAtPosition::test_m3_first
PASSED  TestGetPairAtPosition::test_out_of_range_raises
PASSED  TestGetPairAtPosition::test_all_positions
PASSED  TestDefaultNMax::test_default_is_10
PASSED  TestDefaultNMax::test_functions_use_default
```

### Test Details -- `tests/test_discretization.py`
```
PASSED  TestUnitarityBounds::test_scalar_bound
PASSED  TestUnitarityBounds::test_spin2_bound
PASSED  TestUnitarityBounds::test_spin4_bound
PASSED  TestUnitarityBounds::test_general_spinning_bound
PASSED  TestUnitarityBounds::test_negative_spin_raises
PASSED  TestSatisfiesUnitarity::test_at_bound
PASSED  TestSatisfiesUnitarity::test_above_bound
PASSED  TestSatisfiesUnitarity::test_below_bound
PASSED  TestSatisfiesUnitarity::test_strict_at_bound
PASSED  TestCheckUnitarity::test_slightly_below_with_tolerance
PASSED  TestCheckUnitarity::test_far_below_fails
PASSED  TestIsAllowedSpin::test_even_spins_allowed
PASSED  TestIsAllowedSpin::test_odd_spins_not_allowed
PASSED  TestIsAllowedSpin::test_negative_spin_not_allowed
PASSED  TestValidateOperator::test_valid_operator
PASSED  TestValidateOperator::test_odd_spin_fails
PASSED  TestValidateOperator::test_below_unitarity_fails
PASSED  TestTable2Parameters::test_t1_parameters              delta=2e-5, Dmax=3, Lmax=0
PASSED  TestTable2Parameters::test_t2_parameters              delta=5e-4, Dmax=8, Lmax=6
PASSED  TestTable2Parameters::test_t3_parameters              delta=2e-3, Dmax=22, Lmax=20
PASSED  TestTable2Parameters::test_t4_parameters              delta=0.02, Dmax=100, Lmax=50
PASSED  TestTable2Parameters::test_t5_parameters              delta=1.0, Dmax=500, Lmax=100
PASSED  TestTable2Parameters::test_full_discretization_has_all_tables
PASSED  TestTable2Parameters::test_reduced_discretization
PASSED  TestGenerateTablePoints::test_t1_only_scalars
PASSED  TestGenerateTablePoints::test_t1_scalar_count
PASSED  TestGenerateTablePoints::test_t2_spin_range
PASSED  TestGenerateTablePoints::test_t3_spin_range
PASSED  TestGenerateTablePoints::test_all_points_satisfy_unitarity
PASSED  TestGenerateTablePoints::test_only_even_spins
PASSED  TestGenerateTablePoints::test_dimension_upper_limit
PASSED  TestGenerateTablePoints::test_gap_constraint_scalars
PASSED  TestGenerateFullSpectrum::test_uses_all_tables_by_default
PASSED  TestGenerateFullSpectrum::test_reduced_discretization
PASSED  TestGenerateFullSpectrum::test_sorted_by_spin_then_delta
PASSED  TestGenerateFullSpectrum::test_all_satisfy_unitarity
PASSED  TestGenerateFullSpectrum::test_no_exact_duplicates
PASSED  TestSpectrumHelpers::test_spectrum_to_array
PASSED  TestSpectrumHelpers::test_count_by_spin
PASSED  TestSpectrumHelpers::test_count_by_table
PASSED  TestSpectrumHelpers::test_get_scalars
PASSED  TestSpectrumHelpers::test_get_spinning
PASSED  TestBuildSpectrumWithGaps::test_no_gaps
PASSED  TestBuildSpectrumWithGaps::test_stage_a_gap           Delta_eps gap applied
PASSED  TestBuildSpectrumWithGaps::test_stage_b_two_gaps      Delta_eps + Delta_eps' gaps
PASSED  TestBuildSpectrumWithGaps::test_spinning_unaffected_by_gaps
PASSED  TestSpectrumPoint::test_as_tuple
PASSED  TestSpectrumPoint::test_attributes
PASSED  TestGetTableInfo::test_t1_info
PASSED  TestGetTableInfo::test_t2_info
PASSED  TestEstimateSpectrumSize::test_estimate_matches_actual_for_reduced
PASSED  TestEstimateSpectrumSize::test_estimate_reasonable
```

---

## Milestone 3: LP Builder & Solver -- DONE

Implements the crossing function derivatives, constraint matrix assembly,
and LP feasibility solver. Given an external dimension Delta_sigma and a
discretized spectrum, determines whether the spectrum is consistent with
crossing symmetry via linear programming.

### Files
| File | Lines | Description |
|------|-------|-------------|
| `src/ising_bootstrap/lp/__init__.py` | 61 | Public API exports |
| `src/ising_bootstrap/lp/crossing.py` | 331 | Prefactor table, identity vector, Leibniz rule (Eq. 2.6) |
| `src/ising_bootstrap/lp/constraint_matrix.py` | 218 | Matrix assembly with error handling |
| `src/ising_bootstrap/lp/solver.py` | 302 | LP feasibility via scipy.linprog (HiGHS) |

### Key Public API
```python
from ising_bootstrap.lp import (
    compute_prefactor_table,    # U^{j,k}(Delta_sigma) Taylor coefficients
    compute_identity_vector,    # F_id^{m,n} = -2 U^{m,n} for m odd (66 values)
    compute_extended_h_array,   # h_{p,q} for all p (odd+even, 132 values)
    compute_crossing_vector,    # F^{m,n}_{Delta,l} via Leibniz rule
    build_constraint_matrix,    # A (N_ops x 66) + f_id (66,)
    check_feasibility,          # LP solve: feasible -> excluded, infeasible -> allowed
    solve_bootstrap,            # End-to-end: spectrum -> matrix -> LP
    FeasibilityResult,          # Dataclass with excluded, status, alpha, etc.
)
```

### Paper Equations Implemented
| Equation | Module | Description |
|----------|--------|-------------|
| Eq. 2.6 | crossing.py | Crossing function F^{Delta_sigma}_{Delta,l}(u,v) |
| Eq. 5.3 | crossing.py | Identity contribution F_id = v^{Delta_sigma} - u^{Delta_sigma} |
| App. D | solver.py | LP feasibility formulation |
| — | crossing.py | Leibniz rule for (a,b) derivatives of u^{alpha} * h_{m,n} |

### Design Decisions

1. **Extended index set (132 pairs)**: The Leibniz rule for F^{m,n} requires
   block derivatives h_{p,q} at even m values (not just the 66 odd-m pairs in
   the standard index set). The `compute_extended_h_array` function computes all
   132 pairs with p+2q <= 21 and p >= 0, reusing `compute_h_m0_from_block_derivs`
   and `compute_all_h_mn` from the blocks module.

2. **Stable recursion for prefactor table**: The Taylor coefficients U^{j,k} of
   u^{Delta_sigma}(a,b) are computed via a recursion derived from the ODE
   w * d/dw(w^alpha) = alpha * w^alpha, avoiding the binomial series which has
   convergence issues at high order. The recursion is:
   - Column k=0: T[j+1,0] = [(2alpha-2j)T[j,0] + (2alpha-j+1)T[j-1,0]] / (j+1)
   - Column k+1: T[j,k+1] = (k-alpha)/(k+1) T[j,k] - 2 T[j-1,k+1] - T[j-2,k+1]

3. **Row/column scaling**: Constraint matrix entries span many orders of magnitude.
   Geometric mean row/column scaling (3 iterations) brings row maxima close to 1,
   dramatically improving HiGHS solver conditioning.

4. **Graceful error handling for 3F2 pole**: The spin-0 block at exactly
   Delta = alpha = 0.5 (unitarity bound for D=3) hits a pole in the 3F2
   hypergeometric function (denominator parameter b2 = Delta - alpha = 0).
   The constraint matrix builder catches `ZeroDivisionError` and skips these
   operators (leaving a zero row, which is a trivially satisfied constraint).
   This affects exactly 1 operator per discretization table that includes
   spin-0 at the exact unitarity bound. A `UserWarning` is emitted.

### Tests (57 total: 54 fast + 3 slow)
| File | Tests | Description |
|------|-------|-------------|
| `tests/test_lp/test_crossing.py` | 36 | Extended pairs, prefactor, identity, blocks, Leibniz |
| `tests/test_lp/test_solver.py` | 21 | Scaling, synthetic LP, constraint matrix, feasibility |
| (3 have `@pytest.mark.slow`) | | Full n_max=10 derivs, coarse reduced spectrum pipeline |

### Test Details -- `tests/test_lp/test_crossing.py`
```
PASSED  TestExtendedPairs::test_count_n_max_10                 132 pairs (n_max=10)
PASSED  TestExtendedPairs::test_count_function                 Small n_max values
PASSED  TestExtendedPairs::test_includes_odd_m                 Standard pairs present
PASSED  TestExtendedPairs::test_includes_even_m                Even m pairs present
PASSED  TestExtendedPairs::test_all_satisfy_constraint         p + 2q <= 21
PASSED  TestExtendedPairs::test_contains_standard_index_set    Superset of 66-pair set
PASSED  TestExtendedPairs::test_count_formula                  2(N+1)^2 - (N+1)
PASSED  TestPrefactorTable::test_shape_n_max_10                (22, 11) array
PASSED  TestPrefactorTable::test_shape_n_max_2                 (6, 3) array
PASSED  TestPrefactorTable::test_zeroth_derivative             U[0,0] = (1/4)^{Ds}
PASSED  TestPrefactorTable::test_first_a_derivative            U[1,0] analytical
PASSED  TestPrefactorTable::test_second_a_derivative           U[2,0] analytical
PASSED  TestPrefactorTable::test_first_b_derivative            U[0,1] analytical
PASSED  TestPrefactorTable::test_numerical_cross_check         vs mpmath.diff, <1e-8
PASSED  TestPrefactorTable::test_special_case_ds_half          u^{0.5} = a/2 at b=0
PASSED  TestPrefactorTable::test_all_finite
PASSED  TestIdentityDerivatives::test_shape                    66 elements
PASSED  TestIdentityDerivatives::test_shape_n_max_2            6 elements
PASSED  TestIdentityDerivatives::test_f_id_10_ds_half          F_id^{1,0} = -1
PASSED  TestIdentityDerivatives::test_f_id_30_ds_half          F_id^{3,0} = 0
PASSED  TestIdentityDerivatives::test_f_id_is_minus_2_U        F_id = -2 U^{m,n}
PASSED  TestIdentityDerivatives::test_antisymmetry_check       u<->v antisymmetry
PASSED  TestIdentityDerivatives::test_numerical_cross_check    vs mpmath.diff
PASSED  TestIdentityDerivatives::test_all_finite
PASSED  TestExtendedBlockDerivatives::test_shape               132 values
PASSED  TestExtendedBlockDerivatives::test_all_finite
PASSED  TestExtendedBlockDerivatives::test_odd_m_matches_blocks_module  vs get_or_compute
PASSED  TestExtendedBlockDerivatives::test_spin2               l=2 extended derivs
PASSED  TestExtendedBlockDerivatives::test_full_n_max_10  [slow] Full 132 derivs
PASSED  TestCrossingDerivatives::test_shape                    66 elements
PASSED  TestCrossingDerivatives::test_all_finite
PASSED  TestCrossingDerivatives::test_identity_block           G=1 -> F = F_id
PASSED  TestCrossingDerivatives::test_antisymmetry             Odd m nonzero
PASSED  TestCrossingDerivatives::test_spin2_operator           l=2, D=3.0
PASSED  TestCrossingDerivatives::test_different_delta_sigma    Varies with Ds
PASSED  TestCombCache::test_comb_cache_contains_needed_values
```

### Test Details -- `tests/test_lp/test_solver.py`
```
PASSED  TestScaling::test_scaling_preserves_shape              Shape unchanged
PASSED  TestScaling::test_scaling_improves_condition            Row maxes near 1
PASSED  TestScaling::test_identity_scaling                      f_id stays nonzero
PASSED  TestLPSolverSynthetic::test_obviously_feasible          Known feasible LP
PASSED  TestLPSolverSynthetic::test_obviously_infeasible        Known infeasible LP
PASSED  TestLPSolverSynthetic::test_mixed_constraints           Mixed signs
PASSED  TestLPSolverSynthetic::test_result_has_alpha_when_excluded  alpha vector returned
PASSED  TestLPSolverSynthetic::test_result_has_no_alpha_when_allowed  alpha is None
PASSED  TestLPSolverSynthetic::test_scaling_does_not_change_outcome  Scaling preserves result
PASSED  TestConstraintMatrix::test_matrix_shape_tiny            (8, 6) at n_max=2
PASSED  TestConstraintMatrix::test_matrix_all_finite            No NaN/Inf
PASSED  TestConstraintMatrix::test_identity_nonzero             f_id has nonzero entries
PASSED  TestConstraintMatrix::test_matrix_nonzero_rows          All operators contribute
PASSED  TestConstraintMatrix::test_cache_gives_same_result      h_cache matches direct
PASSED  TestBootstrapFeasibilityTiny::test_unconstrained_tiny_allowed  Solver runs clean
PASSED  TestBootstrapFeasibilityTiny::test_large_gap_excluded   Large gap tested
PASSED  TestBootstrapFeasibilityTiny::test_solve_bootstrap_runs End-to-end pipeline
PASSED  TestBootstrapFeasibilityReduced::test_no_gap_pipeline_runs  [slow] 219 ops, n_max=2
PASSED  TestBootstrapFeasibilityReduced::test_moderate_gap_at_ising  [slow] Gap near Ising
PASSED  TestFeasibilityResult::test_excluded_result             Dataclass fields
PASSED  TestFeasibilityResult::test_allowed_result              alpha=None when allowed
```

### Test Nuances

**Coarse discretization in slow tests**: The slow tests use custom coarse tables
(step 0.1 for scalars, 0.2 for spinning, ~219 operators) rather than the production
TABLE_1/TABLE_2 tables (~201,000 operators). This keeps test runtime under 15s while
exercising the full pipeline. As a consequence:

- `test_no_gap_pipeline_runs` does **not** assert `excluded is False` for the
  unconstrained spectrum. At n_max=2 (only 6 index pairs) with coarse discretization,
  the LP can spuriously find a functional because the grid misses operators that
  would violate it. The physical assertion "unconstrained spectrum is allowed"
  requires n_max >= ~10 and fine discretization — this will be validated in the
  production scans (Milestone 4).

- `test_moderate_gap_at_ising` verifies the solver produces a definite result
  (excluded or allowed) at Delta_sigma = 0.5182 with Delta_epsilon = 1.41, without
  asserting which outcome, since the coarse grid is not fine enough to reliably
  resolve the boundary.

---

## Milestone 4: Stage A Scan -- NOT STARTED

### What needs to be built
1. Grid over Delta_sigma in [0.5, 0.6] with step ~0.002
2. For each Delta_sigma, binary search on Delta_epsilon gap
3. Build spectrum with gap, compute constraint matrix, check LP feasibility
4. Output: CSV of (Delta_sigma, Delta_epsilon_max)

### Stub file
`src/ising_bootstrap/scans/__init__.py` (7 lines, empty)

---

## Milestone 5: Stage B Scan -- NOT STARTED

### What needs to be built
1. Load Stage A results (Delta_sigma, Delta_epsilon_max)
2. For each point, fix Delta_epsilon gap and binary search Delta_epsilon'
3. Output: CSV of (Delta_sigma, Delta_epsilon_max, Delta_epsilon_prime_max)
4. This is Figure 6

---

## Milestone 6: Plotting & Validation -- NOT STARTED

### What needs to be built
1. Load Stage B CSV
2. Plot Delta_epsilon' vs Delta_sigma (Figure 6)
3. Validate: sharp spike at Delta_sigma ~ 0.5182 with Delta_epsilon' ~ 3.84
4. Export PDF/PNG

### Stub file
`src/ising_bootstrap/plot/__init__.py` (6 lines, empty)

---

## File Inventory

### Source Code (4,384 lines)

```
src/ising_bootstrap/
  __init__.py                         11 lines
  config.py                          201 lines

  blocks/
    __init__.py                      106 lines
    diagonal_blocks.py               241 lines   Eq. 4.10-4.11
    spin_recursion.py                286 lines   Eq. 4.9
    z_derivatives.py                 635 lines   Eq. 4.12 + Faa di Bruno
    coordinate_transform.py          266 lines   Eq. 4.15
    transverse_derivs.py             366 lines   Eq. C.1
    cache.py                         406 lines   NPZ disk caching

  spectrum/
    __init__.py                       72 lines
    index_set.py                     276 lines   66 (m,n) pairs
    discretization.py                388 lines   Table 2, T1-T5
    unitarity.py                     197 lines   Bound checks

  lp/
    __init__.py                       61 lines   Public API exports
    crossing.py                      331 lines   Eq. 2.6, Leibniz rule
    constraint_matrix.py             218 lines   Matrix assembly
    solver.py                        302 lines   LP feasibility (HiGHS)

  scans/__init__.py                    7 lines   STUB
  plot/__init__.py                     6 lines   STUB
```

### Test Code (2,792 lines)

```
tests/
  __init__.py                          3 lines
  test_config.py                     177 lines   21 tests
  test_index_set.py                  283 lines   21 tests
  test_discretization.py             450 lines   46 tests

  test_blocks/
    __init__.py                        1 line
    test_diagonal_blocks.py          131 lines   12 tests
    test_spin_recursion.py           134 lines   11 tests
    test_z_derivatives.py            161 lines   11 tests
    test_coordinate_transform.py     232 lines   15 tests (incl. 4 subtests)
    test_transverse_derivs.py        193 lines   16 tests
    test_cache.py                    277 lines   15 tests

  test_lp/
    __init__.py                        0 lines
    test_crossing.py                 396 lines   36 tests (35 fast + 1 slow)
    test_solver.py                   354 lines   21 tests (19 fast + 2 slow)
```

---

## Known Limitations & Notes

1. **Cache precision**: Block derivatives cached as float64 (~15 digits).
   Computation uses mpmath with 50-digit precision. The LP solver only
   needs float64, so this is acceptable.

2. **Slow tests**: 7 tests marked `@pytest.mark.slow` compute full derivative
   sets or run the end-to-end pipeline on coarse spectra. Currently run by
   default (~2-12s each).

3. **Spin-0 at exact unitarity bound (Delta=0.5)**: The 3F2 hypergeometric
   formula has a pole at Delta = alpha = 0.5 for spin-0 in D=3 (denominator
   parameter b2 = Delta - alpha = 0). The blocks module cannot compute
   derivatives at this exact value. The LP constraint matrix builder skips
   affected operators with a `UserWarning`. This affects exactly 1 operator
   per discretization table that starts at the unitarity bound. The identity
   contribution (G=1 at Delta=0) is handled analytically via F_id = -2 U^{m,n}.

4. **Numerical cross-validation**: z-derivatives are validated against mpmath's
   numerical differentiation (mpmath.diff) to relative error < 1e-8. Prefactor
   and identity derivatives are similarly cross-checked.

5. **LP at low n_max with coarse discretization**: At n_max=2 (6 index pairs)
   with coarse operator grids, the LP may spuriously exclude unconstrained
   spectra. This is because the coarse grid misses operators whose crossing
   vectors would violate a putative functional. The physical correctness of
   the bootstrap bounds requires n_max >= ~10 and the fine discretization
   from Table 2 (production scans).

6. **Expected validation targets** (from paper):
   - At Delta_sigma ~ 0.5182: Delta_epsilon_max ~ 1.41
   - At Delta_sigma ~ 0.5182: Delta_epsilon'_max ~ 3.84
   - Sharp spike in Delta_epsilon' bound below Ising Delta_sigma
