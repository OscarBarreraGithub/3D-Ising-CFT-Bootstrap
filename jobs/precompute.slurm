#!/bin/bash
#SBATCH --job-name=ising_precompute
#SBATCH --account=randall_lab
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/precompute_%j.log
#SBATCH --error=logs/precompute_%j.log

# Precompute extended block derivatives for the full spectrum.
# One-time job (~14 hours with 8 workers for full T1-T5 spectrum).
# Results cached to data/cached_blocks/ext_*.npy, reused by all scan tasks.
# Resumable: skips already-cached operators if resubmitted.

set -e

echo "=== Ising Bootstrap: Block Precomputation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
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

echo "Python: $(which python)"
echo ""

# --- Run precomputation ---
python -m ising_bootstrap.scans.stage_a \
    --precompute-only \
    --workers ${SLURM_CPUS_PER_TASK} \
    --verbose

echo ""
echo "Precomputation complete: $(date)"
echo "Cache files: $(ls data/cached_blocks/ext_*.npy 2>/dev/null | wc -l)"
