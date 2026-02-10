#!/usr/bin/env python3
"""
Result validation for bootstrap scans.

Usage:
    python scripts/validate_results.py --stage a
    python scripts/validate_results.py --stage b
    python scripts/validate_results.py --stage a --plot
    python scripts/validate_results.py --stage a --json report.json

Features:
- Check all CSV files for non-empty content
- Verify Δε values in physically reasonable range
- Flag anomalies (all same value, NaN, out of bounds)
- Compare consistency across tasks
- Optional: generate progress plot
"""

import argparse
import csv
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import sys

# Add project to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from ising_bootstrap.config import (
    ISING_DELTA_SIGMA, ISING_DELTA_EPSILON, ISING_DELTA_EPSILON_PRIME,
    DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_STEP
)

@dataclass
class ResultValidation:
    """Validation result for entire scan."""
    stage: str
    total_expected: int
    files_found: int
    valid_results: int
    empty_results: int
    missing_results: int

    # Value range checks
    min_value: float
    max_value: float
    mean_value: float
    std_value: float

    # Anomaly detection
    all_upper_bound: bool  # All values ≈ 2.5 (SDPB failure)
    all_lower_bound: bool  # All values ≈ 0.5 (scipy bug)
    has_nan: bool
    has_outliers: bool

    # Ising point validation (Stage A: Δσ≈0.518, expect Δε≈1.41)
    ising_task_id: Optional[int]
    ising_expected: Optional[float]
    ising_observed: Optional[float]
    ising_error_percent: Optional[float]

    # Per-task details
    task_values: List[Tuple[int, Optional[float], str]]  # (task_id, value, status)

class ResultValidator:
    """Main validator class."""

    def __init__(self, stage: str):
        self.stage = stage
        self.data_dir = Path("/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap/data")

        # Physical bounds for sanity checking
        self.LOWER_BOUND = 0.5  # Unitarity bound
        self.UPPER_BOUND = 2.5  # Hardcoded in binary search

        # Ising point expectations
        if stage == "a":
            self.ISING_SIGMA = ISING_DELTA_SIGMA
            self.ISING_EXPECTED = ISING_DELTA_EPSILON  # Δε at Ising point
            self.SIGMA_MIN = DEFAULT_SIGMA_MIN
            self.SIGMA_STEP = DEFAULT_SIGMA_STEP
        else:
            self.ISING_SIGMA = ISING_DELTA_SIGMA
            self.ISING_EXPECTED = ISING_DELTA_EPSILON_PRIME  # Δε' at Ising point
            self.SIGMA_MIN = DEFAULT_SIGMA_MIN
            self.SIGMA_STEP = DEFAULT_SIGMA_STEP

    def get_result_file(self, task_id: int) -> Path:
        """Get path to result CSV."""
        if self.stage == "a":
            return self.data_dir / f"eps_bound_{task_id}.csv"
        else:
            return self.data_dir / f"epsprime_bound_{task_id}.csv"

    def get_ising_task_id(self) -> int:
        """Calculate which task corresponds to Ising point."""
        # Task ID = round((ISING_SIGMA - SIGMA_MIN) / SIGMA_STEP)
        return round((self.ISING_SIGMA - self.SIGMA_MIN) / self.SIGMA_STEP)

    def parse_result_file(self, task_id: int) -> Tuple[Optional[float], str]:
        """Parse result value from CSV file.

        Returns:
            (value, status) where status is:
            - "valid": Non-empty, parseable value
            - "empty": File exists but only has header
            - "missing": File doesn't exist
            - "parse_error": Can't parse value
            - "nan": Value is NaN
        """
        result_file = self.get_result_file(task_id)

        if not result_file.exists():
            return (None, "missing")

        if result_file.stat().st_size < 30:  # Header only is ~27 bytes
            return (None, "empty")

        try:
            with open(result_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if len(rows) == 0:
                    return (None, "empty")

                # Stage A: column is "delta_eps_max"
                # Stage B: column is "delta_eps_prime_max"
                if self.stage == "a":
                    value = float(rows[0]["delta_eps_max"])
                else:
                    value = float(rows[0]["delta_eps_prime_max"])

                if np.isnan(value):
                    return (value, "nan")

                return (value, "valid")

        except (ValueError, KeyError, IndexError) as e:
            return (None, "parse_error")

    def validate(self, total_tasks: int = 51) -> ResultValidation:
        """Run full validation."""
        task_values = []

        for task_id in range(total_tasks):
            value, status = self.parse_result_file(task_id)
            task_values.append((task_id, value, status))

        # Count statuses
        valid_results = sum(1 for _, _, s in task_values if s == "valid")
        empty_results = sum(1 for _, _, s in task_values if s == "empty")
        missing_results = sum(1 for _, _, s in task_values if s == "missing")
        files_found = total_tasks - missing_results

        # Extract valid values for statistics
        valid_values = [v for _, v, s in task_values if s == "valid" and v is not None]

        if valid_values:
            min_val = min(valid_values)
            max_val = max(valid_values)
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
        else:
            min_val = max_val = mean_val = std_val = np.nan

        # Anomaly detection
        all_upper = valid_results > 0 and all(
            abs(v - self.UPPER_BOUND) < 0.01 for v in valid_values
        )
        all_lower = valid_results > 0 and all(
            abs(v - self.LOWER_BOUND) < 0.01 for v in valid_values
        )
        has_nan = any(s == "nan" for _, _, s in task_values)

        # Outlier detection: values > 3σ from mean
        has_outliers = False
        if valid_values and std_val > 0:
            outliers = [v for v in valid_values if abs(v - mean_val) > 3 * std_val]
            has_outliers = len(outliers) > 0

        # Ising point validation
        ising_task = self.get_ising_task_id()
        ising_value, ising_status = self.parse_result_file(ising_task)

        if ising_status == "valid" and ising_value is not None:
            ising_error = 100 * abs(ising_value - self.ISING_EXPECTED) / self.ISING_EXPECTED
        else:
            ising_error = None

        return ResultValidation(
            stage=self.stage,
            total_expected=total_tasks,
            files_found=files_found,
            valid_results=valid_results,
            empty_results=empty_results,
            missing_results=missing_results,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            std_value=std_val,
            all_upper_bound=all_upper,
            all_lower_bound=all_lower,
            has_nan=has_nan,
            has_outliers=has_outliers,
            ising_task_id=ising_task,
            ising_expected=self.ISING_EXPECTED,
            ising_observed=ising_value,
            ising_error_percent=ising_error,
            task_values=task_values,
        )

    def print_summary(self, validation: ResultValidation):
        """Print human-readable summary."""
        print(f"\n{'='*70}")
        print(f"Result Validation Report: Stage {validation.stage.upper()}")
        print(f"{'='*70}\n")

        print(f"FILE STATUS:")
        print(f"  Expected:           {validation.total_expected}")
        print(f"  Found:              {validation.files_found}")
        print(f"  Valid:              {validation.valid_results}")
        print(f"  Empty:              {validation.empty_results}")
        print(f"  Missing:            {validation.missing_results}")
        print()

        if validation.valid_results > 0:
            print(f"VALUE STATISTICS:")
            print(f"  Min:                {validation.min_value:.6f}")
            print(f"  Max:                {validation.max_value:.6f}")
            print(f"  Mean:               {validation.mean_value:.6f}")
            print(f"  Std:                {validation.std_value:.6f}")
            print()

        print(f"ANOMALY DETECTION:")
        if validation.all_upper_bound:
            print(f"  ⚠ WARNING: All values ≈ {self.UPPER_BOUND} (SDPB failure mode!)")
        elif validation.all_lower_bound:
            print(f"  ⚠ WARNING: All values ≈ {self.LOWER_BOUND} (scipy bug!)")
        else:
            print(f"  ✓ Value range looks reasonable")

        if validation.has_nan:
            print(f"  ⚠ WARNING: NaN values detected")
        else:
            print(f"  ✓ No NaN values")

        if validation.has_outliers:
            print(f"  ⚠ WARNING: Outliers detected (>3σ from mean)")
        else:
            print(f"  ✓ No statistical outliers")
        print()

        if validation.ising_observed is not None:
            print(f"ISING POINT VALIDATION:")
            print(f"  Task ID:            {validation.ising_task_id}")
            print(f"  Expected:           {validation.ising_expected:.3f}")
            print(f"  Observed:           {validation.ising_observed:.3f}")
            print(f"  Error:              {validation.ising_error_percent:.2f}%")

            if validation.ising_error_percent < 5:
                print(f"  ✓ Within 5% of expected value")
            else:
                print(f"  ⚠ WARNING: >5% deviation from expected")
            print()

        # Show problematic tasks
        problem_tasks = [
            (tid, val, status) for tid, val, status in validation.task_values
            if status != "valid"
        ]
        if problem_tasks:
            print(f"PROBLEMATIC TASKS:")
            for tid, val, status in problem_tasks[:20]:
                print(f"  Task {tid}: {status}")
            if len(problem_tasks) > 20:
                print(f"  ... and {len(problem_tasks) - 20} more")
            print()

    def plot_results(self, validation: ResultValidation, output_file: str = "validation_plot.png"):
        """Generate diagnostic plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Headless backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plot")
            return

        # Extract data
        task_ids = []
        values = []
        for tid, val, status in validation.task_values:
            if status == "valid" and val is not None:
                task_ids.append(tid)
                values.append(val)

        if not task_ids:
            print("No valid data to plot")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot results
        ax.plot(task_ids, values, 'o-', label=f"Stage {validation.stage.upper()} results")

        # Mark Ising point
        if validation.ising_observed is not None:
            ax.axvline(validation.ising_task_id, color='red', linestyle='--',
                      label=f"Ising point (task {validation.ising_task_id})")
            ax.axhline(validation.ising_expected, color='green', linestyle='--',
                      label=f"Expected: {validation.ising_expected:.3f}")

        # Mark bounds
        ax.axhline(self.LOWER_BOUND, color='gray', linestyle=':', alpha=0.5,
                  label=f"Unitarity bound: {self.LOWER_BOUND}")
        ax.axhline(self.UPPER_BOUND, color='gray', linestyle=':', alpha=0.5,
                  label=f"Upper bound: {self.UPPER_BOUND}")

        ax.set_xlabel("Task ID")
        if self.stage == "a":
            ax.set_ylabel("Δε_max")
            ax.set_title("Stage A: Upper Bound on Δε vs Δσ")
        else:
            ax.set_ylabel("Δε'_max")
            ax.set_title("Stage B: Upper Bound on Δε' vs Δσ")

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"\nPlot saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Validate result files")
    parser.add_argument("--stage", choices=["a", "b"], required=True, help="Pipeline stage")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plot")
    parser.add_argument("--json", help="Save JSON report")

    args = parser.parse_args()

    validator = ResultValidator(args.stage)
    validation = validator.validate()

    validator.print_summary(validation)

    if args.plot:
        validator.plot_results(validation)

    if args.json:
        # Convert numpy floats to Python floats for JSON serialization
        validation_dict = asdict(validation)
        with open(args.json, 'w') as f:
            json.dump(validation_dict, f, indent=2, default=str)
        print(f"\nJSON report saved to: {args.json}")

if __name__ == "__main__":
    main()
