# Implementation Progress

Tracking document for reproducing Figure 6 from arXiv:1203.6064.

**Last updated:** 2026-02-03
**Test suite:** 193/193 passing (7.4s)

---

## Milestone Summary

| # | Milestone | Status | Source Lines | Test Lines | Tests |
|---|-----------|--------|-------------|------------|-------|
| 0 | Repository scaffolding | DONE | 212 | 177 | 21 |
| 1 | Conformal block engine | DONE | 2,306 | 1,128 | 84 |
| 2 | Spectrum discretization | DONE | 933 | 733 | 88 |
| 3 | LP builder & solver | NOT STARTED | 8 (stub) | 0 | 0 |
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

## Milestone 3: LP Builder & Solver -- NOT STARTED

### What needs to be built
1. **Crossing function derivatives** F_{Delta,l}^{m,n} (Eq. 2.6 in README)
   - F = v^{Delta_sigma} G_sigma(u,v) - u^{Delta_sigma} G_sigma(v,u)
   - Derivatives at u=v=1/4 using h_{m,n} from blocks module
2. **Identity term** analytical derivatives of (v^d - u^d)|_{u=v=1/4}
3. **Constraint matrix** A[i, k] = F_{Delta_k, l_k}^{m_i, n_i}
   - 66 rows (index set) x N_spectrum columns
4. **LP feasibility wrapper** using scipy.optimize.linprog (HiGHS backend)
   - Exists alpha_k >= 0 such that sum_k alpha_k F_k + F_identity = 0?

### Stub file
`src/ising_bootstrap/lp/__init__.py` (8 lines, empty)

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

### Source Code (3,472 lines)

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

  lp/__init__.py                       8 lines   STUB
  scans/__init__.py                    7 lines   STUB
  plot/__init__.py                     6 lines   STUB
```

### Test Code (2,042 lines)

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
```

---

## Known Limitations & Notes

1. **Cache precision**: Block derivatives cached as float64 (~15 digits).
   Computation uses mpmath with 50-digit precision. The LP solver only
   needs float64, so this is acceptable.

2. **Slow tests**: 4 tests marked `@pytest.mark.slow` compute full 66-derivative
   sets at n_max=10. Currently run by default (~2s each).

3. **Spin-0 identity block**: The G_{0,0}(z)=1 identity block test is a
   placeholder because Delta=0 hits poles in the 3F2 formula. The identity
   contribution is handled analytically in the LP module (not yet built).

4. **Numerical cross-validation**: z-derivatives are validated against mpmath's
   numerical differentiation (mpmath.diff) to relative error < 1e-8.

5. **Expected validation targets** (from paper):
   - At Delta_sigma ~ 0.5182: Delta_epsilon_max ~ 1.41
   - At Delta_sigma ~ 0.5182: Delta_epsilon'_max ~ 3.84
   - Sharp spike in Delta_epsilon' bound below Ising Delta_sigma
