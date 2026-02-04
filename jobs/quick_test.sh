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

# --- Test 2: Reduced 3-point scan ---
echo "--- Test 2: Reduced 3-point scan ---"
python -m ising_bootstrap.scans.stage_a \
    --sigma-min 0.51 --sigma-max 0.53 --sigma-step 0.01 \
    --reduced --verbose --output data/eps_bound_quick_test.csv

echo ""
echo "--- Results ---"
cat data/eps_bound_quick_test.csv

echo ""
echo "Quick test complete: $(date)"
echo "All checks passed."
