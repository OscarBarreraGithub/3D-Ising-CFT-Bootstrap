"""
Test gates for the extended-precision simplex solver.

Must ALL pass before submitting full Stage A production run.
Run via SLURM: sbatch jobs/test_gates.slurm

Gates:
    1. Synthetic LP tests (feasible/infeasible)
    2. n_max=2 cross-validation (extended vs scipy)
    3. n_max=10 no-gap -> ALLOWED
    4. n_max=10 large gap -> EXCLUDED
    5. Single-point Stage A binary search (eps_max ~ 1.4)
    6. Go/no-go timing (< 120s per LP)
"""

import sys
import time
import numpy as np

from ising_bootstrap.config import (
    N_MAX, DiscretizationTable, FULL_DISCRETIZATION,
)
from ising_bootstrap.spectrum.discretization import (
    SpectrumPoint, generate_full_spectrum,
)
from ising_bootstrap.spectrum.index_set import generate_index_set
from ising_bootstrap.lp.constraint_matrix import build_constraint_matrix
from ising_bootstrap.lp.solver import check_feasibility
from ising_bootstrap.lp.simplex import check_feasibility_extended
from ising_bootstrap.scans.stage_a import (
    ScanConfig, build_full_constraint_matrix, find_eps_bound,
    load_h_cache_from_disk,
)


COARSE_T1 = DiscretizationTable("T1_test", delta=0.1, delta_max=3, l_max=0)
COARSE_T2 = DiscretizationTable("T2_test", delta=0.2, delta_max=8, l_max=6)

PASS = 0
FAIL = 0


def gate(name):
    def decorator(fn):
        def wrapper():
            global PASS, FAIL
            print(f"\n{'='*60}")
            print(f"  GATE: {name}")
            print(f"{'='*60}")
            try:
                fn()
                PASS += 1
                print(f"  RESULT: PASS")
            except Exception as e:
                FAIL += 1
                print(f"  RESULT: FAIL - {e}")
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


@gate("1. Synthetic LP - obviously feasible")
def gate1a():
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    f_id = np.array([1.0, 0.0, 0.0])
    spectrum = [SpectrumPoint(1.0, 0, "t"), SpectrumPoint(2.0, 0, "t")]
    result = check_feasibility_extended(
        A, f_id, spectrum, 0.5, n_max=2, dps=30, dps_verify=None,
    )
    assert result.excluded is True, f"Expected EXCLUDED, got {result.status}"


@gate("1. Synthetic LP - obviously infeasible")
def gate1b():
    A = np.array([[-1.0, 0.0]])
    f_id = np.array([1.0, 0.0])
    spectrum = [SpectrumPoint(1.0, 0, "t")]
    result = check_feasibility_extended(
        A, f_id, spectrum, 0.5, n_max=2, dps=30, dps_verify=None,
    )
    assert result.excluded is False, f"Expected ALLOWED, got {result.status}"


@gate("2. n_max=2 cross-validation")
def gate2():
    tables = [COARSE_T1, COARSE_T2]
    spectrum = generate_full_spectrum(tables=tables)
    A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=2)

    result_scipy = check_feasibility(A, f_id, scale=False)
    result_ext = check_feasibility_extended(
        A, f_id, spectrum, 0.518, n_max=2, dps=30, dps_verify=None,
    )
    assert result_scipy.excluded == result_ext.excluded, (
        f"scipy says {'EXCLUDED' if result_scipy.excluded else 'ALLOWED'}, "
        f"extended says {'EXCLUDED' if result_ext.excluded else 'ALLOWED'}"
    )
    print(f"  Both agree: {'EXCLUDED' if result_ext.excluded else 'ALLOWED'}")


@gate("3. n_max=10 no-gap -> ALLOWED")
def gate3():
    tables = [COARSE_T1, COARSE_T2]
    spectrum = generate_full_spectrum(tables=tables)
    A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=10)
    print(f"  Matrix shape: {A.shape}")

    result = check_feasibility_extended(
        A, f_id, spectrum, 0.518,
        n_max=10, dps=50, dps_verify=None, verbose=True,
    )
    assert result.excluded is False, (
        f"No-gap should be ALLOWED, got {result.status}"
    )


@gate("4. n_max=10 large gap -> EXCLUDED")
def gate4():
    tables = [COARSE_T1, COARSE_T2]
    spectrum = generate_full_spectrum(tables=tables)
    A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=10)

    # Keep only spinning + scalars above gap=100
    scalar_mask = np.array([p.spin == 0 for p in spectrum])
    scalar_deltas = np.array([p.delta for p in spectrum])
    mask = ~scalar_mask | (scalar_mask & (scalar_deltas >= 100.0))
    A_sub = A[mask]
    spectrum_sub = [p for p, m in zip(spectrum, mask) if m]
    print(f"  Filtered: {A_sub.shape[0]} operators (gap=100)")

    result = check_feasibility_extended(
        A_sub, f_id, spectrum_sub, 0.518,
        n_max=10, dps=50, dps_verify=None, verbose=True,
    )
    assert result.excluded is True, (
        f"Gap=100 should be EXCLUDED, got {result.status}"
    )


@gate("5. Single-point Stage A (eps_max ~ 1.4)")
def gate5():
    tables = [COARSE_T1, COARSE_T2]
    spectrum = generate_full_spectrum(tables=tables)
    A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=10)

    scalar_mask = np.array([p.spin == 0 for p in spectrum])
    scalar_deltas = np.array([p.delta for p in spectrum])
    spinning_mask = ~scalar_mask

    config = ScanConfig(
        tolerance=0.01, max_iter=30, n_max=10,
        tables=tables, use_extended=True, dps=50, dps_verify=None,
    )

    print(f"  Running binary search at delta_sigma=0.518...")
    t0 = time.time()
    eps_max, n_iter = find_eps_bound(
        0.518, A, f_id,
        scalar_mask, scalar_deltas, spinning_mask,
        config, full_spectrum=spectrum,
    )
    elapsed = time.time() - t0
    print(f"  eps_max = {eps_max:.4f} ({n_iter} iters, {elapsed:.1f}s)")

    assert eps_max > 0.6, f"eps_max={eps_max} is too low (probably still broken)"
    assert eps_max < 2.5, f"eps_max={eps_max} is too high"
    # With coarse tables, exact value will differ from 1.41 but should be nontrivial
    print(f"  PASS: eps_max = {eps_max:.4f} is a nontrivial bound")


@gate("6. Go/no-go timing")
def gate6():
    tables = [COARSE_T1, COARSE_T2]
    spectrum = generate_full_spectrum(tables=tables)
    A, f_id = build_constraint_matrix(spectrum, 0.518, n_max=10)
    print(f"  Testing single LP solve at n_max=10 ({A.shape[0]} operators)...")

    t0 = time.time()
    result = check_feasibility_extended(
        A, f_id, spectrum, 0.518,
        n_max=10, dps=50, dps_verify=None, verbose=True,
    )
    elapsed = time.time() - t0

    print(f"  Single LP: {elapsed:.1f}s (limit: 120s)")
    assert elapsed < 120, (
        f"Single LP took {elapsed:.1f}s > 120s limit. "
        f"RECOMMENDATION: Switch to SDPB."
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  TEST GATES: Extended-Precision Simplex Solver")
    print("=" * 60)

    gate1a()
    gate1b()
    gate2()
    gate3()
    gate4()
    gate5()
    gate6()

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {PASS} passed, {FAIL} failed")
    print(f"{'='*60}")

    if FAIL > 0:
        print("\nFAILED: Aborting pipeline. Fix failures before proceeding.")
        sys.exit(1)
    else:
        print("\nALL GATES PASSED. Pipeline may proceed.")
        sys.exit(0)
