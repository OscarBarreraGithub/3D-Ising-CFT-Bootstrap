#!/bin/bash
#SBATCH --job-name=ising_stage_b_sdpb
#SBATCH --account=randall_lab
#SBATCH --partition=shared
#SBATCH --array=0-50
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/stage_b_sdpb_%A_%a.log
#SBATCH --error=logs/stage_b_sdpb_%A_%a.log

# Stage B scan with SDPB backend.
# Each array task handles one Δσ value.
# Requires Stage A results in data/eps_bound.csv.

set -euo pipefail
export PYTHONUNBUFFERED=1

# --- Environment ---
source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

mkdir -p logs data

# --- Compute Δσ for this task ---
SIGMA_MIN=0.500
SIGMA_STEP=0.002
SIGMA=$(python3 -c "print(f'{${SIGMA_MIN} + ${SLURM_ARRAY_TASK_ID} * ${SIGMA_STEP}:.6f}')")

echo "=== Stage B (SDPB) ==="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Delta_sigma: ${SIGMA}"
echo "Start: $(date)"

# --- Run scan for this single Δσ ---
python -m ising_bootstrap.scans.stage_b \
    --eps-bound data/eps_bound.csv \
    --sigma-min "${SIGMA}" \
    --sigma-max "${SIGMA}" \
    --sigma-step 1.0 \
    --tolerance 1e-3 \
    --backend sdpb \
    --sdpb-image tools/sdpb-3.1.0.sif \
    --sdpb-precision 1024 \
    --sdpb-cores "${SLURM_CPUS_PER_TASK}" \
    --output "data/epsprime_bound_${SLURM_ARRAY_TASK_ID}.csv" \
    --verbose

echo "=== Done: $(date) ==="
