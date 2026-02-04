#!/bin/bash
# Quick sanity check for the Ising Bootstrap on FASRC.
# Run this interactively after salloc, or submit via sbatch.
#
# Usage (interactive):
#   salloc -p test --account=iaifi_lab -c 1 -t 00:20:00 --mem=4G
#   bash jobs/quick_test.sh
#
# Usage (batch):
#   sbatch --partition=test --account=iaifi_lab --time=00:20:00 --mem=4G \
#          --output=logs/quick_test_%j.log jobs/quick_test.sh

set -e

echo "=== Ising Bootstrap: Quick Test ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# --- Prevent thread oversubscription ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- Conda activation ---
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda" >&2
    exit 1
fi
conda activate ising_bootstrap

echo "Python: $(which python)"
echo ""

# --- Test 1: Import check ---
echo "--- Test 1: Import check ---"
python -c "
from ising_bootstrap.config import N_MAX, FULL_DISCRETIZATION
from ising_bootstrap.scans.stage_a import ScanConfig, run_scan
print(f'n_max = {N_MAX}')
print(f'Tables: {[t.name for t in FULL_DISCRETIZATION]}')
print('Imports OK')
"
echo ""

# --- Test 2: Pipeline sanity check with coarse discretization ---
# Uses coarse tables (~219 operators) with n_max=2 to verify the full
# pipeline (blocks -> crossing -> LP -> binary search -> CSV) runs.
# This does NOT produce physics-quality results — that requires the
# full discretization with precomputed block cache.
echo "--- Test 2: Pipeline sanity check (coarse tables, n_max=2) ---"
python -c "
from pathlib import Path
from ising_bootstrap.config import DiscretizationTable
from ising_bootstrap.scans.stage_a import ScanConfig, run_scan

# Coarse tables matching integration test setup (~219 operators)
t1 = DiscretizationTable('T1_test', delta=0.1, delta_max=3, l_max=0)
t2 = DiscretizationTable('T2_test', delta=0.2, delta_max=8, l_max=6)

config = ScanConfig(
    sigma_min=0.51,
    sigma_max=0.53,
    sigma_step=0.01,
    tolerance=0.01,
    max_iter=30,
    n_max=2,
    tables=[t1, t2],
    output=Path('data/eps_bound_quick_test.csv'),
    verbose=True,
)
results = run_scan(config)

# Validate results
assert len(results) == 3, f'Expected 3 results, got {len(results)}'
for ds, de in results:
    assert 0.5 <= de <= 2.5, f'Unexpected eps_max={de} at sigma={ds}'
print()
print('Pipeline validation passed: 3 points computed successfully')
"

echo ""
echo "--- Results ---"
cat data/eps_bound_quick_test.csv

echo ""
echo "Quick test complete: $(date)"
echo "All checks passed."
