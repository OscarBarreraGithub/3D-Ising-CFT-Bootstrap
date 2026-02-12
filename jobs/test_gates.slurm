#!/bin/bash
#SBATCH --job-name=ising_gates
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/test_gates_%j.log
#SBATCH --error=logs/test_gates_%j.log

# Test gates: must ALL pass before Stage A production run.
# Gates 1-6 validate the extended-precision simplex solver.
# Exit code 1 aborts the pipeline (dependency=afterok).

set -e

echo "=== Test Gates: Extended-Precision Simplex Solver ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# --- Prevent thread oversubscription ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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

cd "$SLURM_SUBMIT_DIR"

# Run test gates (exits with code 1 on any failure)
python scripts/test_gates.py

echo ""
echo "Test gates complete: $(date)"
