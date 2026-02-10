#!/usr/bin/env python3
"""
Resource estimation for SDPB jobs.

Usage:
    python scripts/estimate_resources.py
    python scripts/estimate_resources.py --n-max 10 --precision 1024 --cores 8
    python scripts/estimate_resources.py --check-allocation 64

Features:
- Calculate expected memory usage from config parameters
- Estimate constraint matrix size
- Suggest memory allocation
- Warn if current allocation likely insufficient
"""

import argparse
import sys
from pathlib import Path

# Add project to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ising_bootstrap.config import N_MAX, FULL_DISCRETIZATION, get_index_set_count

class ResourceEstimator:
    """Estimate computational resources needed."""

    def __init__(self, n_max: int = 10, precision: int = 1024, n_cores: int = 8):
        self.n_max = n_max
        self.precision = precision
        self.n_cores = n_cores

        # Load discretization to count operators
        from ising_bootstrap.spectrum.discretization import generate_full_spectrum
        self.spectrum = generate_full_spectrum(FULL_DISCRETIZATION)
        self.n_operators = len(self.spectrum)

        # Calculate index set size (constraints)
        self.n_constraints = get_index_set_count(n_max)

    def estimate_constraint_matrix_memory(self) -> float:
        """Estimate memory for constraint matrix in GB."""
        # Matrix is (n_operators × n_constraints) in float64
        bytes_per_element = 8
        total_bytes = self.n_operators * self.n_constraints * bytes_per_element
        return total_bytes / (1024 ** 3)  # Convert to GB

    def estimate_pmp_json_size(self) -> float:
        """Estimate PMP JSON file size in MB."""
        # Each block has ~66 polynomial coefficients at 1024-bit precision
        # 1024-bit number → ~300 decimal digits → ~350 bytes as JSON string
        bytes_per_coeff = 350
        coeffs_per_block = self.n_constraints
        total_bytes = self.n_operators * coeffs_per_block * bytes_per_coeff
        return total_bytes / (1024 ** 2)  # Convert to MB

    def estimate_sdp_binary_size(self) -> float:
        """Estimate binary SDP file size in MB (after pmp2sdp)."""
        # Empirical: binary format is ~2-3x larger than constraint matrix
        matrix_mb = self.estimate_constraint_matrix_memory() * 1024
        return matrix_mb * 2.5

    def estimate_sdpb_working_memory(self) -> float:
        """Estimate SDPB solver working memory per MPI rank in GB."""
        # SDPB uses interior-point method:
        # - Primal/dual matrices: ~2x constraint matrix size
        # - Factorizations: ~1.5x constraint matrix size
        # - Intermediate vectors: ~0.5x
        # Total per rank: ~4x constraint matrix / n_cores

        matrix_gb = self.estimate_constraint_matrix_memory()
        per_rank_gb = (matrix_gb * 4.0) / self.n_cores

        # Add precision overhead (1024-bit uses ~2x more RAM than 512-bit)
        precision_factor = self.precision / 512.0

        return per_rank_gb * precision_factor

    def estimate_total_memory(self) -> float:
        """Estimate total peak memory in GB."""
        # Components:
        # 1. Python process: constraint matrix + block cache (~2 GB)
        # 2. PMP JSON file (temporary, in page cache)
        # 3. Binary SDP files (written to /tmp, in page cache)
        # 4. SDPB working memory (all ranks)
        # 5. Filesystem overhead (~10%)

        python_base = 2.0  # GB
        matrix_gb = self.estimate_constraint_matrix_memory()
        pmp_json_gb = self.estimate_pmp_json_size() / 1024
        sdp_binary_gb = self.estimate_sdp_binary_size() / 1024
        sdpb_working_gb = self.estimate_sdpb_working_memory() * self.n_cores

        subtotal = python_base + matrix_gb + pmp_json_gb + sdp_binary_gb + sdpb_working_gb
        total_with_overhead = subtotal * 1.1

        return total_with_overhead

    def suggest_allocation(self) -> int:
        """Suggest SLURM memory allocation in GB (rounded up)."""
        estimated = self.estimate_total_memory()
        # Add 30% safety margin and round up to nearest multiple of 4
        with_margin = estimated * 1.3
        rounded = int(with_margin / 4 + 1) * 4
        return max(rounded, 16)  # Minimum 16 GB

    def print_report(self):
        """Print detailed resource report."""
        print(f"\n{'='*70}")
        print(f"Resource Estimation Report")
        print(f"{'='*70}\n")

        print(f"CONFIGURATION:")
        print(f"  n_max:              {self.n_max}")
        print(f"  Operators:          {self.n_operators:,}")
        print(f"  Constraints:        {self.n_constraints}")
        print(f"  SDPB precision:     {self.precision}-bit")
        print(f"  MPI cores:          {self.n_cores}")
        print()

        print(f"MEMORY ESTIMATES:")
        print(f"  Constraint matrix:  {self.estimate_constraint_matrix_memory():.2f} GB")
        print(f"  PMP JSON:           {self.estimate_pmp_json_size():.0f} MB")
        print(f"  Binary SDP:         {self.estimate_sdp_binary_size():.0f} MB")
        print(f"  SDPB per rank:      {self.estimate_sdpb_working_memory():.2f} GB")
        print(f"  SDPB total (all ranks): {self.estimate_sdpb_working_memory() * self.n_cores:.2f} GB")
        print()

        total_est = self.estimate_total_memory()
        suggested = self.suggest_allocation()

        print(f"TOTAL ESTIMATE:")
        print(f"  Peak usage:         {total_est:.1f} GB")
        print(f"  Suggested alloc:    {suggested} GB")
        print()

        print(f"SLURM DIRECTIVE:")
        print(f"  #SBATCH --mem={suggested}G")
        print(f"  #SBATCH --cpus-per-task={self.n_cores}")
        print()

    def check_allocation(self, current_gb: int):
        """Check if current allocation is sufficient."""
        estimated = self.estimate_total_memory()
        suggested = self.suggest_allocation()

        print(f"\nALLOCATION CHECK:")
        print(f"  Current:            {current_gb} GB")
        print(f"  Estimated peak:     {estimated:.1f} GB")
        print(f"  Suggested:          {suggested} GB")
        print()

        if current_gb >= suggested:
            print(f"  ✓ Current allocation should be sufficient")
            margin = 100 * (current_gb - estimated) / estimated
            print(f"  ✓ Safety margin:    {margin:.1f}%")
        elif current_gb >= estimated:
            print(f"  ⚠ Current allocation is close to estimated peak")
            print(f"  ⚠ Consider increasing to {suggested} GB for safety")
        else:
            shortfall = estimated - current_gb
            print(f"  ✗ INSUFFICIENT: {shortfall:.1f} GB short of estimated peak")
            print(f"  ✗ Expect OOM kills! Increase to {suggested} GB")
        print()

def main():
    parser = argparse.ArgumentParser(description="Estimate resource requirements")
    parser.add_argument("--n-max", type=int, default=10, help="Derivative truncation")
    parser.add_argument("--precision", type=int, default=1024, help="SDPB precision")
    parser.add_argument("--cores", type=int, default=8, help="MPI cores")
    parser.add_argument("--check-allocation", type=int, metavar="GB",
                       help="Check if GB allocation is sufficient")

    args = parser.parse_args()

    estimator = ResourceEstimator(args.n_max, args.precision, args.cores)
    estimator.print_report()

    if args.check_allocation:
        estimator.check_allocation(args.check_allocation)

if __name__ == "__main__":
    main()
