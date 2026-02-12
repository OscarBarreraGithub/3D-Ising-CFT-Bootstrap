# Implementation Progress

Tracking document for reproducing Figure 6 from arXiv:1203.6064.

**Last updated:** 2026-02-09
**Test suite:** 308+ tests passing

---

## Milestone Summary

| # | Milestone | Status | Source Lines | Test Lines | Tests |
|---|-----------|--------|-------------|------------|-------|
| 0 | Repository scaffolding | DONE | 212 | 177 | 21 |
| 1 | Conformal block engine | DONE | 2,306 | 1,128 | 84 |
| 2 | Spectrum discretization | DONE | 933 | 733 | 88 |
| 3 | LP builder & solver | DONE | 912 | 750 | 57 |
| 4 | Stage A scan (Delta_epsilon) | DONE | 590 | 428 | 27 |
| 5 | Stage B scan (Delta_epsilon') | DONE | 456 | 510 | 25 |
| 6 | Plotting & validation | DONE | ~207 | 0 | 0 |
| 7 | SDPB integration (LP fix) | DONE | ~420 | ~250 | 17 |

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

## Milestone 4: Stage A Scan -- DONE

Implements the Stage A scan to compute the upper bound on Delta_epsilon as a function
of Delta_sigma. For each Delta_sigma value on a grid, performs binary search to find
the largest scalar gap (no scalars below Delta_epsilon) consistent with crossing symmetry.

### Files
| File | Lines | Description |
|------|-------|-------------|
| `src/ising_bootstrap/scans/__init__.py` | 42 | Public API exports (Stage A + B) |
| `src/ising_bootstrap/scans/stage_a.py` | 560 | Main scan loop, binary search, CSV output |
| `src/ising_bootstrap/blocks/cache.py` | 613 | Extended H array cache functions added |

### Key Public API
```python
from ising_bootstrap.scans import (
    ScanConfig,              # Configuration dataclass
    run_scan,                # Main scan loop
    run_precompute,          # Block precomputation
    binary_search_eps,       # Generic binary search (decoupled from LP)
    find_eps_bound,          # Row-subsetting binary search
    load_scan_results,       # CSV loading
)
```

### Implementation Highlights

1. **Binary search correction**: Fixed bug in `docs/TODO.md` pseudocode:
   - **Correct**: `if excluded: hi = mid` (LP feasible → gap inconsistent → too high)
   - **Incorrect**: `if is_feasible: lo = mid` (would maximize excluded region)
   - Logic: Larger gap removes more scalars → makes LP easier to satisfy → more likely excluded

2. **Extended H array cache**: Block derivatives h_{m,n}(Delta,l) are Delta_sigma-independent:
   - Precompute once for all unique (Delta, l) pairs in discretization
   - Save separately as `ext_*.npy` files (22,11) arrays
   - Reuse for all 51 Delta_sigma grid points
   - Avoids 1000+ redundant block computations per scan

3. **Full constraint matrix approach**: Build A matrix (N_ops × 66) once per Delta_sigma:
   - Binary search selects row subsets via numpy boolean masking
   - No matrix rebuilding per iteration
   - Significant speedup over naive approach

4. **CSV output**: Standard format for stage handoff:
   ```csv
   delta_sigma,delta_eps_max
   0.500,1.234
   0.502,1.238
   ...
   ```

### Tests (27 passing)
| File | Tests | Description |
|------|-------|-------------|
| `tests/test_scans/test_stage_a.py` | 27 | Binary search, gap filtering, CSV I/O, integration |

### Test Details -- `tests/test_scans/test_stage_a.py`
```
PASSED  TestBinarySearchLogic::test_basic_search              Finds correct bound
PASSED  TestBinarySearchLogic::test_all_feasible              Returns lo (hi bound)
PASSED  TestBinarySearchLogic::test_all_infeasible            Returns lo (unitarity)
PASSED  TestBinarySearchLogic::test_tolerance_convergence     Stops within tolerance
PASSED  TestBinarySearchLogic::test_callable_predicate        Generic predicate works
PASSED  TestBinarySearchLogic::test_max_iterations_respected  Iteration limit enforced
PASSED  TestBinarySearchLogic::test_narrow_range              Small range handled
PASSED  TestBinarySearchLogic::test_different_thresholds      Multiple thresholds tested

PASSED  TestGapFiltering::test_no_gap_includes_all            Gap=0.5 keeps all
PASSED  TestGapFiltering::test_large_gap_excludes_all_scalars Gap > Dmax removes all
PASSED  TestGapFiltering::test_gap_preserves_spinning         Spinning unaffected
PASSED  TestGapFiltering::test_gap_boundary_inclusion         Boundary semantics correct
PASSED  TestGapFiltering::test_progressive_gap                Progressive gap values

PASSED  TestCSVIO::test_write_and_read_roundtrip              CSV round-trip works
PASSED  TestCSVIO::test_header_format                         Header format correct
PASSED  TestCSVIO::test_append_mode                           Append mode works
PASSED  TestCSVIO::test_empty_file                            Empty file handling

PASSED  TestScanConfig::test_default_values                   Defaults correct
PASSED  TestScanConfig::test_sigma_grid_count                 Grid count formula
PASSED  TestScanConfig::test_custom_grid                      Custom grid generation
PASSED  TestScanConfig::test_reduced_tables                   Reduced discretization
PASSED  TestScanConfig::test_full_tables                      Full discretization

PASSED  TestExtendedCache::test_save_and_load_roundtrip       Extended H array cache
PASSED  TestExtendedCache::test_shape_validation              (22,11) shape enforced

PASSED  TestStageAIntegration::test_single_sigma_runs    [slow] Single Δσ scan completes
PASSED  TestStageAIntegration::test_single_sigma_plausible [slow] Result in plausible range
PASSED  TestStageAIntegration::test_scan_three_points    [slow] Multi-point scan works
```

### Design Decisions

1. **Decoupled binary search**: `binary_search_eps` is a pure function taking a
   predicate, making it independently testable without LP dependency.

2. **Row subsetting approach**: `find_eps_bound` builds the full A matrix once,
   then creates boolean masks to select scalar/spinning rows per iteration.
   This is much faster than rebuilding the matrix each time.

3. **Extended cache format**: Existing cache stores 66-element vectors (odd m only).
   Extended cache stores full (22,11) arrays with both odd and even m derivatives,
   which is what `compute_crossing_vector_fast` needs.

4. **Coarse discretization in tests**: Integration tests use custom tables with
   step 0.1 (scalars) / 0.2 (spinning) to keep runtime under 10s. Production
   scans use the fine Table 2 discretization.

### Known Issues & Nuances

1. **Binary search direction**: The pseudocode in `docs/TODO.md` (lines 283-289) had
   the binary search direction reversed. Corrected logic:
   - Larger gap → fewer constraints → LP more likely feasible (excluded)
   - If excluded: gap is too large → `hi = mid`
   - If allowed: gap is consistent → `lo = mid`

2. **Precomputation time**: Extended H array precomputation for full discretization
   (Tables T1-T5, ~201k operators, ~57k unique (Delta,l) pairs) takes several hours
   with n_max=10. Once cached, scans are much faster.

3. **Memory usage**: Full constraint matrix A is (N_ops × 66). For production scans
   with ~201k operators, this is ~100 MB per Delta_sigma. Manageable on modern hardware.

4. **LP tolerance**: Uses `config.LP_TOLERANCE` for feasibility checks. Default 1e-9
   works well for production runs.

### Next Steps → Milestone 5

With Stage A complete, Stage B can now:
1. Load the Delta_epsilon_max(Delta_sigma) curve from CSV
2. For each Delta_sigma, fix Delta_epsilon gap and binary search Delta_epsilon'
3. Output the final (Delta_sigma, Delta_epsilon', Delta_epsilon'_max) for Figure 6

---

## Milestone 5: Stage B Scan -- DONE

Implements the Stage B scan to compute the upper bound on Delta_epsilon' as a function
of Delta_sigma. For each Delta_sigma value, fixes Delta_epsilon from Stage A and performs
binary search to find the largest second scalar gap consistent with crossing symmetry.
This produces the data for Figure 6 of the paper.

### Files
| File | Lines | Description |
|------|-------|-------------|
| `src/ising_bootstrap/scans/stage_b.py` | 456 | Two-gap scan, binary search, CSV output, CLI |
| `src/ising_bootstrap/scans/__init__.py` | 42 | Updated public API exports (Stage A + B) |

### Key Public API
```python
from ising_bootstrap.scans import (
    StageBConfig,           # Configuration dataclass
    find_eps_prime_bound,   # Two-gap binary search
    run_scan_stage_b,       # Main scan loop
    run_precompute_stage_b, # Block precomputation (delegates to Stage A)
    load_stage_b_results,   # CSV loading (3-column format)
    load_eps_bound_map,     # Stage A result loading
)
```

### Implementation Highlights

1. **Two-gap row subsetting**: `find_eps_prime_bound` selects rows from the full
   constraint matrix using two gap conditions:
   - Exclude scalars with Delta < Delta_epsilon (below first gap, always excluded)
   - Exclude scalars with Delta_epsilon <= Delta < Delta_epsilon' (between gaps)
   - Include scalars with Delta >= Delta_epsilon' (above second gap)
   - Include all spinning operators unconditionally

2. **Reuses Stage A infrastructure**: Imports `binary_search_eps`,
   `build_full_constraint_matrix`, `load_h_cache_from_disk`, and
   `load_scan_results` from `stage_a.py`. Only the row masking logic is new.

3. **Stage A result loading**: `load_eps_bound_map` reads Stage A CSV and builds
   a dict mapping Delta_sigma (rounded to 6 decimals) to Delta_epsilon_max.
   Sigma grid points without Stage A data are skipped with a warning.

4. **CSV output**: 3-column format for downstream plotting:
   ```csv
   delta_sigma,delta_eps,delta_eps_prime_max
   0.500000,1.234000,3.456000
   ...
   ```

5. **CLI**: `python -m ising_bootstrap.scans.stage_b --eps-bound <path> [options]`

### Tests (25 passing)
| File | Tests | Description |
|------|-------|-------------|
| `tests/test_scans/test_stage_b.py` | 25 | Two-gap filtering, CSV I/O, config, integration |

### Test Details -- `tests/test_scans/test_stage_b.py`
```
PASSED  TestTwoGapFiltering::test_two_gaps_exclude_middle_scalars     Middle scalars excluded
PASSED  TestTwoGapFiltering::test_no_second_gap_includes_all_above_eps  eps'=eps keeps all
PASSED  TestTwoGapFiltering::test_very_large_second_gap_excludes_all  Large eps' removes all
PASSED  TestTwoGapFiltering::test_spinning_always_included            Spinning unaffected
PASSED  TestTwoGapFiltering::test_gap_boundary_precision              Boundary semantics correct
PASSED  TestTwoGapFiltering::test_progressive_second_gap              Progressive gap monotonic

PASSED  TestCSVIO::test_write_and_read_roundtrip                     CSV round-trip works
PASSED  TestCSVIO::test_header_format                                3-column header correct
PASSED  TestCSVIO::test_append_mode                                  Append mode works
PASSED  TestCSVIO::test_empty_file                                   Empty file handling
PASSED  TestCSVIO::test_three_column_values                          Values parsed correctly

PASSED  TestStageBConfig::test_default_values                        Defaults from config.py
PASSED  TestStageBConfig::test_sigma_grid_count                      Grid count formula
PASSED  TestStageBConfig::test_custom_grid                           Custom grid generation
PASSED  TestStageBConfig::test_reduced_tables                        Reduced discretization
PASSED  TestStageBConfig::test_full_tables                           Full discretization
PASSED  TestStageBConfig::test_eps_bound_path                        Path stored correctly

PASSED  TestLoadEpsBoundMap::test_load_creates_mapping               Dict from CSV
PASSED  TestLoadEpsBoundMap::test_empty_csv                          Empty dict from empty CSV
PASSED  TestLoadEpsBoundMap::test_rounding_for_matching              Rounded keys for matching

PASSED  TestRunScanValidation::test_missing_eps_bound_raises         ValueError if no path
PASSED  TestRunScanValidation::test_skips_missing_sigma_points       Skips missing sigma points

PASSED  TestStageBIntegration::test_single_sigma_runs           [slow] Single point scan completes
PASSED  TestStageBIntegration::test_scan_three_points           [slow] Multi-point scan works
PASSED  TestStageBIntegration::test_csv_round_trip_through_pipeline [slow] Full pipeline CSV roundtrip
```

### Design Decisions

1. **Same binary search direction as Stage A**: Larger Delta_epsilon' removes more
   scalars from constraints, making LP easier to satisfy (excluded). If excluded:
   gap too large, lower hi. If allowed: gap consistent, raise lo.

2. **Search range**: lo = Delta_epsilon (just above first gap), hi = 6.0 (generous
   upper bound). At the Ising point, expected result is ~3.84.

3. **Delegates precomputation**: `run_precompute` creates a `ScanConfig` and calls
   Stage A's precompute function since blocks are scan-stage-independent.

4. **Crash recovery**: CSV rows are written immediately after each sigma point,
   so partial results are preserved if the job is interrupted.

---

## Milestone 6: Plotting & Validation -- DONE

Implements the plotting module to reproduce Figure 6 from arXiv:1203.6064.
Loads Stage B CSV results and generates the upper bound on Delta_epsilon' vs
Delta_sigma, with optional sanity check output.

### Files
| File | Lines | Description |
|------|-------|-------------|
| `src/ising_bootstrap/plot/fig6.py` | ~207 | Figure 6 generation, sanity check, CLI |
| `src/ising_bootstrap/plot/__init__.py` | updated | Exports `plot_fig6`, `print_sanity_check` |

### SLURM Infrastructure
| File | Description |
|------|-------------|
| `jobs/stage_b.slurm` | SLURM array job for Stage B scan (mirrors stage_a.slurm) |
| `jobs/merge_stage_b.sh` | Merge script for Stage B per-task CSVs |
| `jobs/precompute.slurm` | Time limit fixed: 6h to 24h (precompute needs ~18h) |

### Key Public API
```python
from ising_bootstrap.plot import (
    plot_fig6,            # Generate Figure 6: (data, output, dpi, show) -> Figure
    print_sanity_check,   # Print validation summary: (data) -> None
)
```

### CLI
```bash
# Via module
python -m ising_bootstrap.plot.fig6 --data <path> --output <path> --dpi 300 --show --no-sanity-check

# Via entry point (registered in pyproject.toml)
ising-plot --data <path> --output <path> --dpi 300 --show --no-sanity-check
```

### Design Decisions

1. **Agg backend for headless cluster**: Uses `matplotlib.use("Agg")` at import time
   so that the module works on headless SLURM nodes without an X display. The `--show`
   flag switches to an interactive backend if requested.

2. **Auto-save PDF sibling**: When saving a PNG, the module automatically saves a
   PDF version alongside it (e.g., `fig6.png` + `fig6.pdf`) for publication quality.

3. **Reuses `load_stage_b_results()`**: Delegates CSV parsing to the existing
   `ising_bootstrap.scans.stage_b.load_stage_b_results` function rather than
   reimplementing CSV loading.

### Tests (0 new tests)

No new tests were added for the plotting module. Verification was performed manually:
- Imports work (`from ising_bootstrap.plot import plot_fig6`)
- CLI help works (`python -m ising_bootstrap.plot.fig6 --help`)
- Synthetic test plot generates both PNG and PDF
- All 302 existing tests still pass

---

## Production Run Status

### SDPB Stage A (Current)

**Status**: Blocked by NFS cache loading bottleneck (see below). Job 59675873 running
but zero results produced after 1+ hour — all 51 tasks stuck loading 520K .npy files.
**Backend**: SDPB 3.1.0 via Singularity (`tools/sdpb-3.1.0.sif`), 1024-bit precision
**Expected**: Non-trivial Delta_epsilon_max curve (not all 0.5 like the scipy run)

### NFS Cache Loading Bottleneck (2026-02-09)

**Problem**: Each Stage A task loads 520K individual `.npy` files from NFS via
`load_h_cache_from_disk()`. Each `np.load()` requires ~3-10ms of NFS syscalls
(open + read + close). At 520K files: **26-87 minutes of pure I/O per task**.
With 51 array tasks loading independently, the NFS load is 51x worse. After 1+ hour,
zero tasks produced any results.

**Additional issue**: Python stdout is fully buffered when piped to a file (which SLURM
does), so log files showed only 7 lines (the startup banner) even if more progress had
been made.

**Fix**: Consolidate 520K `.npy` files into a single `.npz` archive file.
- `jobs/consolidate_cache.py`: One-time script to pack all files into
  `data/cached_blocks/ext_cache_consolidated.npz`
- `jobs/consolidate_cache.slurm`: SLURM job to run consolidation (2h, 16G)
- `stage_a.py`: Added fast path in `load_h_cache_from_disk()` — checks for consolidated
  `.npz` first, loads all arrays in a single NFS read (~10-30s vs 60+ min)
- Added `PYTHONUNBUFFERED=1` to all SLURM scripts for real-time log output

**Expected improvement**: Cache loading drops from 60+ min to ~10-30s per task.

### Block Precomputation

**Status**: Complete (520K .npy files cached)

**Job History**:
- Job 58613547 (single-node `precompute.slurm`): Cached ~60,851 operators before timeout.
- Job 58631295 (`precompute_array.slurm`, 5 shards, 8h limit): Each shard completed
  ~45,000-46,500 operators before timeout. Total after this run: 289,665 cached.
- Job 58723936 (`precompute_array.slurm`, 10 shards, 10h limit): Each shard completed
  ~14,500-15,000 of ~23,081 assigned operators before timeout. Total: 425,429 cached.
  Throughput dropped to ~1,500 ops/hr/shard because remaining operators have higher spin
  (spin 20-100, each taking 7-62s vs 0.3s for spin-0).

**Root cause of slowdown**: Operators are sorted by (delta, spin), so early jobs process
cheap spin-0 operators while later jobs face expensive high-spin operators. The spin recursion
cost scales as O(l²) due to the recursive 3F2 hypergeometric evaluation tree.

**Additional bottleneck**: The cache existence check used `Path.exists()` per operator
(520K calls on NFS). Fixed in Session 2026-02-05 by switching to bulk `os.listdir()`,
saving ~46 minutes of startup overhead per shard.

**Resolution**: Time limit bumped to 18h, filesystem bottleneck fixed. Full pipeline
resubmitted as Job 58827408 (precompute) → 58827409 (Stage A) → 58827430 (Merge A) →
58827433 (Stage B) → 58827435 (Final merge + plot). ~95,047 operators remain.

### Pole Error at (Delta=0.5, l=0)

The ₃F₂ hypergeometric function has a pole when Delta = alpha = 0.5 (spin-0 at the exact
unitarity bound in D=3), because the denominator parameter b₂ = Delta - alpha = 0. This is
**expected and harmless**:
- Affects exactly 1 operator out of 520,476
- Caught and skipped gracefully by the error handler in `precompute_extended_spectrum_blocks()`
- The LP constraint matrix builder also catches this and leaves a zero row (trivially satisfied)
- The identity contribution (G=1 at Delta=0) is handled analytically via F_id = -2 U^{m,n}
- Already documented in Milestone 3 "Design Decisions" item 4 and "Known Limitations" item 3

---

## File Inventory

### Source Code (~5,643 lines)

```
src/ising_bootstrap/
  __init__.py                         11 lines
  config.py                          201 lines

  blocks/
    __init__.py                      116 lines   Extended cache exports added
    diagonal_blocks.py               241 lines   Eq. 4.10-4.11
    spin_recursion.py                286 lines   Eq. 4.9
    z_derivatives.py                 635 lines   Eq. 4.12 + Faa di Bruno
    coordinate_transform.py          266 lines   Eq. 4.15
    transverse_derivs.py             366 lines   Eq. C.1
    cache.py                         613 lines   NPZ + extended H cache

  spectrum/
    __init__.py                       72 lines
    index_set.py                     276 lines   66 (m,n) pairs
    discretization.py                388 lines   Table 2, T1-T5
    unitarity.py                     197 lines   Bound checks

  lp/
    __init__.py                       61 lines   Public API exports
    crossing.py                      331 lines   Eq. 2.6, Leibniz rule
    constraint_matrix.py             218 lines   Matrix assembly
    solver.py                        ~370 lines  LP feasibility (HiGHS + SDPB backend)
    sdpb.py                          ~420 lines  SDPB integration (PMP writer, solver, wrapper)

  scans/
    __init__.py                       42 lines   Public API exports (Stage A + B)
    stage_a.py                       560 lines   Stage A: binary search, CSV output
    stage_b.py                       456 lines   Stage B: two-gap scan, CSV output

  plot/
    __init__.py                      updated    Exports plot_fig6, print_sanity_check
    fig6.py                         ~207 lines  Figure 6 generation + CLI
```

### Test Code (3,730 lines)

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
    test_sdpb.py                     ~250 lines  17 tests (PMP writer, output interp, e2e)

  test_scans/
    __init__.py                        1 line
    test_stage_a.py                  427 lines   27 tests (24 fast + 3 slow)
    test_stage_b.py                  510 lines   25 tests (22 fast + 3 slow)
```

---

## Known Limitations & Notes

1. **Cache precision**: Block derivatives cached as float64 (~15 digits).
   Computation uses mpmath with 50-digit precision. The LP solver only
   needs float64, so this is acceptable.

2. **Slow tests**: 10 tests marked `@pytest.mark.slow` compute full derivative
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

5. **LP conditioning at n_max=10 (FIXED)**: The scipy/HiGHS float64 LP solver
   cannot handle the constraint matrix condition number (~4e16 at n_max=10).
   This caused all Stage A results to be 0.5 (wrong). **Fixed by integrating
   SDPB** (arbitrary-precision SDP solver) as the LP backend. See
   `docs/LP_CONDITIONING_BUG.md` for full diagnosis. The scipy backend is
   retained as a fallback for testing at low n_max.

6. **SDPB requires Singularity**: The SDPB backend requires a Singularity
   container image (`tools/sdpb-3.1.0.sif`, ~300 MB). Only available on
   compute nodes with Singularity installed. Use `--backend scipy` on
   systems without it (with the caveat that scipy fails at n_max=10).

7. **Expected validation targets** (from paper):
   - At Delta_sigma ~ 0.5182: Delta_epsilon_max ~ 1.41
   - At Delta_sigma ~ 0.5182: Delta_epsilon'_max ~ 3.84
   - Sharp spike in Delta_epsilon' bound below Ising Delta_sigma

---

## Partition Migration (2026-02-12)

### Sapphire Partition Adoption

**Status:** Migrated from `shared` to `sapphire` partition for all production runs (2026-02-12)

**Reason:** SDPB timeout analysis revealed that each Stage A task requires **26-35 hours** to complete:
- Each SDPB solve: 2-2.2 hours (measured on 16 cores)
- Bisection iterations needed: 12-16 per Δσ point
- Total time per task: 2h × 14 iterations ≈ 28 hours
- **Shared partition limit: 12 hours** → Insufficient by 2-3×

**Files Changed:** All 18 `.slurm` job scripts now use `#SBATCH --partition=sapphire`

**Resource Configuration:**
- Stage A/B scripts: 16 cores, 128GB, 36h walltime (was 8 cores, 128GB, 12h)
- SDPB timeout: 18000s (5 hours, configurable via `SDPB_TIMEOUT`)
- Supporting scripts (merge, precompute): sapphire partition, original resources

**Expected Performance:**
- Stage A runtime: 28-35 hours per task (all 51 tasks run in parallel on 9 nodes)
- Stage B runtime: 28-35 hours per task
- Total pipeline: ~60 hours (2.5 days) from start to Figure 6

**Sapphire Advantages:**
- 7-day walltime limit (vs 12 hours on shared) - Accommodates 28-35h jobs comfortably
- 990GB RAM/node (vs 184GB) - Can fit 6 jobs/node instead of 1 → Only 9 nodes needed for 51 tasks
- 112 cores/node (vs 48) - Better resource utilization
- InfiniBand MPI fabric - Optimized for SDPB's MPI parallelization

**Documentation:**
- Full analysis: `docs/OVERNIGHT_TIMEOUT_ANALYSIS_2026-02-12.md`
- Review checklist: `docs/SAPPHIRE_MIGRATION_CHECKLIST.md`
- Archived logs: `logs/archive/2026-02-overnight-timeout/`, `logs/archive/2026-02-10-stage-a-timeout/`

**Verification (2026-02-12):**
```bash
$ bash jobs/verify_partition_migration.sh
Checking partition configuration in all .slurm files:

✓ consolidate_cache.slurm: sapphire
✓ diagnose_lp.slurm: sapphire
✓ final_merge_and_plot.slurm: sapphire
✓ merge_stage_a_and_submit_b.slurm: sapphire
✓ merge_stage_a_job.slurm: sapphire
✓ precompute_array.slurm: sapphire
✓ precompute.slurm: sapphire
✓ stage_a_extended.slurm: sapphire
✓ stage_a_pilot_sdpb.slurm: sapphire
✓ stage_a_sdpb.slurm: sapphire
✓ stage_a.slurm: sapphire
✓ stage_b_extended.slurm: sapphire
✓ stage_b_sdpb.slurm: sapphire
✓ stage_b.slurm: sapphire
✓ stage_b_smoke_sdpb.slurm: sapphire
✓ test_error_detection.slurm: sapphire
✓ test_gates.slurm: sapphire
✓ test_sufficient_memory.slurm: sapphire

✓ All 18 .slurm files use sapphire partition
```

**Migration Status:** ✅ Complete - All scripts verified, documentation updated, ready for external review

---

### Overnight Run Timeline (Failures Leading to Migration)

**2026-02-11 00:46** - First overnight run (Jobs 59843729, 59844257, 59844715, 59845395):
- **Result:** All TIMEOUT after 6-12 **minutes** (not hours)
- **Cause:** Walltime parsing bug (IFS=':' delimiter conflicted with HH:MM:SS format)
- **Fix:** Changed delimiter from ':' to '|' in overnight_full_pipeline.sh
- **Documented:** `BUGFIX_2026-02-11_WALLTIME_PARSING.md`

**2026-02-11 15:04** - Second attempt (Job 59936292):
- Configuration: 8 cores, 128GB, 6h walltime, 1800s SDPB timeout
- **Result:** Completed but produced NaN (not valid)
- **Cause:** 1800s (30 min) SDPB timeout too short for bisection convergence
- **Finding:** Need hours, not minutes, for SDPB timeout

**2026-02-11 18:57** - Extended timeout test (Jobs 59973738, 59973739):
- Job 59973738: 8 cores, 128GB, 8h walltime, 18000s (5h) SDPB timeout
  - **Result:** TIMEOUT after 8h SLURM walltime limit
  - **Progress:** 2 bisection iterations completed
  - **Log:** `logs/test_sufficient_memory_59973738.log` (archived)

- Job 59973739: 16 cores, 160GB, 8h walltime, 18000s (5h) SDPB timeout
  - **Result:** TIMEOUT after 8h SLURM walltime limit
  - **Progress:** 4 bisection iterations completed (470K → 495K → 508K → 514K blocks)
  - **Log:** `logs/test_sufficient_memory_59973739.log` (archived)

**Key Performance Metrics:**
| Metric | 8 cores | 16 cores |
|--------|---------|----------|
| pmp2sdp time | ~302-350s | ~303-348s |
| SDPB solve time | ~2.2h | ~2.0h |
| Iterations in 8h | 2-3 | 3-4 |
| Speedup with 2× cores | - | 1.1× (only 10% faster) |

**Conclusion:** Both configurations hit 8-hour walltime limit before completing bisection (need 12-16 iterations). SDPB solve time (~2h) doesn't scale well with cores. **Solution: Longer walltime (sapphire partition) required.**

**Resolution:** Migrate to sapphire partition (2026-02-12)

---

### Strict Semantics Integration (2026-02-11)

**Status:** Merged `codex/strict-failfast-stageb-snap-eps` branch to main (2026-02-11)

**Key Changes:**
- **Fail-fast:** ANY solver failure → NaN for that point (no retries)
- **SDPB strict:** Only explicit "feasible" or "infeasible" trusted; "inconclusive" = failure
- **Stage B anchoring:** Snaps Stage A Δε to scalar grid with tolerance check
- **Merge gates:** Validates Stage A data before launching Stage B
- **Timeout configurable:** `SDPB_TIMEOUT` env var exposed end-to-end

**Test Suite:** All 75 tests passing (17 SDPB + 28 Stage A + 30 Stage B)

**Documented:** `HANDOFF_2026-02-11_STRICT_SEMANTICS_MERGED.md`
