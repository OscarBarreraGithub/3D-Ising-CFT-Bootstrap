#!/bin/bash
#SBATCH --job-name=ising_stage_a
#SBATCH --account=iaifi_lab
#SBATCH --partition=shared
#SBATCH --array=0-50
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/stage_a_%A_%a.log
#SBATCH --error=logs/stage_a_%A_%a.log

# Stage A scan: compute Δε_max at a single Δσ point.
# Array job with 51 tasks (Δσ from 0.50 to 0.60, step 0.002).
# Requires precomputed blocks in data/cached_blocks/.

set -e

# --- Compute Δσ for this task ---
SIGMA_MIN=0.50
SIGMA_STEP=0.002
SIGMA=$(python3 -c "print(${SIGMA_MIN} + ${SLURM_ARRAY_TASK_ID} * ${SIGMA_STEP})")

echo "=== Stage A: Task ${SLURM_ARRAY_TASK_ID} / Δσ = ${SIGMA} ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

# --- Run scan for this single Δσ point ---
python -m ising_bootstrap.scans.stage_a \
    --sigma-min ${SIGMA} \
    --sigma-max ${SIGMA} \
    --sigma-step 1.0 \
    --output data/eps_bound_${SLURM_ARRAY_TASK_ID}.csv \
    --verbose

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} complete: $(date)"
cat data/eps_bound_${SLURM_ARRAY_TASK_ID}.csv
