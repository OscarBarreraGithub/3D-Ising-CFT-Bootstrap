"""
Diagnostic script for the LP conditioning bug.

Tests whether the LP gives correct results at n_max=2 (well-conditioned)
with a paper-grade fine discretization, then tests component normalization
approaches to fix n_max=10.

Run via SLURM:
    sbatch jobs/diagnose_lp.slurm

Or interactively on a compute node:
    salloc -p test -t 01:00:00 --mem=8G -c 4
    conda activate ising_bootstrap
    python scripts/diagnose_lp.py
"""

import numpy as np
import sys
from math import factorial
from scipy.optimize import linprog

from ising_bootstrap.lp.crossing import (
    compute_prefactor_table, compute_identity_vector,
    compute_extended_h_array,
    compute_crossing_vector_fast, build_comb_cache,
)
from ising_bootstrap.spectrum.index_set import generate_index_set
from ising_bootstrap.blocks.cache import load_extended_h_array
from ising_bootstrap.scans.stage_a import binary_search_eps


def build_spectrum_ops(fine=False):
    """Build a list of (delta, spin) operators for testing.

    Parameters
    ----------
    fine : bool
        If True, use paper-grade fine spacing (many operators).
        If False, use coarser spacing (faster).
    """
    ops = []
    if fine:
        # T1-like: scalars [0.5, 3.0], step 2e-4
        for d in np.arange(0.5002, 3.0, 2e-4):
            ops.append((d, 0))
        # T2-like: spin 2-6, step 5e-3
        for d in np.arange(3.001, 8.0, 5e-3):
            ops.append((d, 2))
        for d in np.arange(5.001, 8.0, 5e-3):
            ops.append((d, 4))
        for d in np.arange(7.001, 8.0, 5e-3):
            ops.append((d, 6))
        # T3-like: spin 2-20, [8, 22]
        for spin in range(2, 22, 2):
            d_min = max(spin + 1.001, 8.0)
            for d in np.arange(d_min, 22.0, 0.02):
                ops.append((d, spin))
        # T4-like: spin 2-50, [22, 100]
        for spin in range(2, 52, 2):
            d_min = max(spin + 1.001, 22.0)
            for d in np.arange(d_min, 100.0, 0.2):
                ops.append((d, spin))
        # T5-like: spin 2-100, [100, 500]
        for spin in range(2, 102, 2):
            d_min = max(spin + 1.001, 100.0)
            for d in np.arange(d_min, 500.0, 1.0):
                ops.append((d, spin))
    else:
        for d in np.arange(0.52, 5.0, 0.01):
            ops.append((round(d, 8), 0))
        for d in np.arange(3.0, 15.0, 0.02):
            ops.append((round(d, 8), 2))
        for d in np.arange(5.0, 15.0, 0.05):
            ops.append((round(d, 8), 4))
        for d in np.arange(7.0, 15.0, 0.1):
            ops.append((round(d, 8), 6))
    return ops


def build_constraint_matrix_onthefly(ops, delta_sigma, n_max):
    """Build constraint matrix by computing blocks on the fly."""
    index_set = generate_index_set(n_max)
    U = compute_prefactor_table(delta_sigma, n_max)
    f_id = compute_identity_vector(delta_sigma, n_max)
    comb_cache = build_comb_cache(n_max)
    max_order = 2 * n_max + 1

    A_rows, op_d, op_s = [], [], []
    for delta, spin in ops:
        try:
            H = compute_extended_h_array(delta, spin, n_max)
            F = compute_crossing_vector_fast(H, U, index_set, comb_cache, max_order)
            if np.all(np.isfinite(F)):
                A_rows.append(F)
                op_d.append(delta)
                op_s.append(spin)
        except Exception:
            pass

    return np.array(A_rows), f_id, np.array(op_d), np.array(op_s)


def build_constraint_matrix_from_cache_fast(ops, delta_sigma, n_max):
    """Build constraint matrix from cached extended H arrays."""
    index_set = generate_index_set(n_max)
    U = compute_prefactor_table(delta_sigma, n_max)
    f_id = compute_identity_vector(delta_sigma, n_max)
    comb_cache = build_comb_cache(n_max)
    max_order = 2 * n_max + 1

    A_rows, op_d, op_s = [], [], []
    for delta, spin in ops:
        try:
            H = load_extended_h_array(delta, spin, n_max)
            F = compute_crossing_vector_fast(H, U, index_set, comb_cache, max_order)
            if np.all(np.isfinite(F)):
                A_rows.append(F)
                op_d.append(delta)
                op_s.append(spin)
        except Exception:
            pass

    return np.array(A_rows), f_id, np.array(op_d), np.array(op_s)


def check_feasibility_raw(A, f_id, tol=1e-7):
    """Solve the LP without any custom scaling."""
    n_ops, n_vars = A.shape
    res = linprog(
        np.zeros(n_vars),
        A_ub=-A, b_ub=np.zeros(n_ops),
        A_eq=f_id.reshape(1, -1), b_eq=np.array([1.0]),
        bounds=[(None, None)] * n_vars,
        method="highs",
        options={
            "presolve": True,
            "primal_feasibility_tolerance": tol,
            "dual_feasibility_tolerance": tol,
        },
    )
    return res.status == 0


def check_feasibility_normalized(A, f_id, weights, tol=1e-7):
    """Solve the LP with column normalization by given weights."""
    A_n = A / weights[np.newaxis, :]
    f_n = f_id / weights
    return check_feasibility_raw(A_n, f_n, tol)


def run_binary_search(A, f_id, op_d, op_s, check_fn, lo=0.5, hi=2.5, tol=0.005):
    """Run eps binary search using a given feasibility check function."""
    sc = op_s == 0
    sp = ~sc

    def is_excl(gap):
        mask = sp | (sc & (op_d >= gap - 1e-10))
        return check_fn(A[mask], f_id)

    return binary_search_eps(is_excl, lo, hi, tol, 30)


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# Test 1: n_max=2 with fine discretization (well-conditioned)
# ============================================================
def test_nmax2_fine():
    print_header("Test 1: n_max=2, fine discretization (compute on the fly)")

    delta_sigma = 0.518
    n_max = 2
    ops = build_spectrum_ops(fine=True)
    print(f"Building {len(ops)} operators at n_max={n_max}...")

    A, f_id, op_d, op_s = build_constraint_matrix_onthefly(ops, delta_sigma, n_max)
    sc = op_s == 0
    sp = ~sc
    print(f"Built: {A.shape[0]} operators ({np.sum(sc)} scalar, {np.sum(sp)} spinning)")

    _, S, _ = np.linalg.svd(A, full_matrices=False)
    print(f"Cond: {S[0]/S[-1]:.2e}, rank: {np.sum(S > 1e-10*S[0])}/{len(S)}")

    # Test no gap
    nogap = check_feasibility_raw(A, f_id)
    print(f"\nNo gap: {'EXCL' if nogap else 'ALLOW'} (expect ALLOW)")

    # Test individual gaps
    print("\nGap scan:")
    for gap in [0.5, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 2.0, 2.5]:
        mask = sp | (sc & (op_d >= gap - 1e-10))
        ex = check_feasibility_raw(A[mask], f_id)
        print(f"  gap={gap:.1f}: {'EXCL' if ex else 'ALLOW'} ({np.sum(mask)} ops)")

    # Binary search
    eps_max, n_iter = run_binary_search(
        A, f_id, op_d, op_s, check_feasibility_raw
    )
    print(f"\nBinary search: eps_max = {eps_max:.4f} ({n_iter} iters)")
    print(f"Expected: loose bound (n_max=2 gives weaker bounds than n_max=10)")

    return A, f_id, op_d, op_s


# ============================================================
# Test 2: n_max=10, try normalization approaches
# ============================================================
def test_nmax10_normalization():
    print_header("Test 2: n_max=10, normalization approaches")

    delta_sigma = 0.518
    n_max = 10
    index_set = generate_index_set(n_max)
    ops = build_spectrum_ops(fine=False)
    print(f"Loading {len(ops)} operators from cache at n_max={n_max}...")

    A, f_id, op_d, op_s = build_constraint_matrix_from_cache_fast(
        ops, delta_sigma, n_max
    )
    sc = op_s == 0
    sp = ~sc
    print(f"Loaded: {A.shape[0]} operators ({np.sum(sc)} scalar, {np.sum(sp)} spinning)")

    _, S, _ = np.linalg.svd(A, full_matrices=False)
    print(f"Raw cond: {S[0]/S[-1]:.2e}, rank: {np.sum(S > 1e-10*S[0])}/{len(S)}")

    # --- Approach 1: f_id component normalization ---
    # Divide each column by |f_id| for that component
    print("\n--- Approach 1: Normalize by |f_id| ---")
    w1 = np.abs(f_id)
    w1 = np.where(w1 > 1e-30, w1, 1.0)
    A_n1 = A / w1
    _, S1, _ = np.linalg.svd(A_n1, full_matrices=False)
    print(f"  Cond: {S1[0]/S1[-1]:.2e}, rank: {np.sum(S1 > 1e-10*S1[0])}/{len(S1)}")
    print(f"  Column max range: [{np.max(np.abs(A_n1),axis=0).min():.2e}, {np.max(np.abs(A_n1),axis=0).max():.2e}]")

    nogap1 = check_feasibility_normalized(A, f_id, w1)
    print(f"  No gap: {'EXCL' if nogap1 else 'ALLOW'}")

    if not nogap1:
        eps1, _ = run_binary_search(
            A, f_id, op_d, op_s,
            lambda Asub, fid: check_feasibility_normalized(Asub, fid, w1),
        )
        print(f"  Binary search: eps_max = {eps1:.4f}")

    # --- Approach 2: Empirical column normalization ---
    # Divide each column by its max absolute value in A
    print("\n--- Approach 2: Normalize by max|A column| ---")
    w2 = np.max(np.abs(A), axis=0)
    w2 = np.where(w2 > 1e-30, w2, 1.0)
    A_n2 = A / w2
    _, S2, _ = np.linalg.svd(A_n2, full_matrices=False)
    print(f"  Cond: {S2[0]/S2[-1]:.2e}, rank: {np.sum(S2 > 1e-10*S2[0])}/{len(S2)}")

    nogap2 = check_feasibility_normalized(A, f_id, w2)
    print(f"  No gap: {'EXCL' if nogap2 else 'ALLOW'}")

    # --- Approach 3: m! * n! factorial weights ---
    print("\n--- Approach 3: Normalize by m! * n! ---")
    w3 = np.array([factorial(m) * factorial(n) for m, n in index_set], dtype=np.float64)
    A_n3 = A / w3
    _, S3, _ = np.linalg.svd(A_n3, full_matrices=False)
    print(f"  Cond: {S3[0]/S3[-1]:.2e}, rank: {np.sum(S3 > 1e-10*S3[0])}/{len(S3)}")

    nogap3 = check_feasibility_normalized(A, f_id, w3)
    print(f"  No gap: {'EXCL' if nogap3 else 'ALLOW'}")

    # --- Approach 4: (2n_max+1)! / (m! * n!) "inverse" weights ---
    # Try dividing by the actual column norms of the identity vector
    print("\n--- Approach 4: Normalize by max(|f_id|, max|A col|) ---")
    w4 = np.maximum(np.abs(f_id), np.max(np.abs(A), axis=0))
    w4 = np.where(w4 > 1e-30, w4, 1.0)
    A_n4 = A / w4
    _, S4, _ = np.linalg.svd(A_n4, full_matrices=False)
    print(f"  Cond: {S4[0]/S4[-1]:.2e}, rank: {np.sum(S4 > 1e-10*S4[0])}/{len(S4)}")

    nogap4 = check_feasibility_normalized(A, f_id, w4)
    print(f"  No gap: {'EXCL' if nogap4 else 'ALLOW'}")

    # --- Approach 5: Use mpmath for the LP (extended precision simplex) ---
    print("\n--- Approach 5: Extended-precision LP via mpmath ---")
    try:
        from mpmath import mp, mpf, matrix as mpmatrix
        mp.dps = 30

        # Take a small subset for speed (first 200 operators)
        n_sub = min(200, A.shape[0])
        A_sub = A[:n_sub]

        # Convert to mpmath
        n_ops, n_vars = A_sub.shape
        A_mp = mpmatrix(n_ops, n_vars)
        for i in range(n_ops):
            for j in range(n_vars):
                A_mp[i, j] = mpf(A_sub[i, j])
        f_mp = mpmatrix(n_vars, 1)
        for j in range(n_vars):
            f_mp[j] = mpf(f_id[j])

        # Check if f_id is in the column space of A^T via least squares
        # This tells us if the normalization is reachable
        print(f"  (Testing with {n_sub} operators at 30-digit precision)")
        print(f"  f_id norm (mp): {mp.norm(f_mp)}")

        # We can't easily solve the LP at extended precision with mpmath alone.
        # But we CAN check the condition number at extended precision.
        # For that, compute A^T A and check its eigenvalues.
        print("  (Extended-precision LP solver not yet implemented)")
        print("  Condition number at extended precision would need custom simplex.")

    except ImportError:
        print("  mpmath not available")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("LP Conditioning Diagnostic")
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")

    test_nmax2_fine()
    test_nmax10_normalization()

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
