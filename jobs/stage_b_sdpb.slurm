#!/bin/bash
#SBATCH --job-name=ising_stage_b_sdpb
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
#SBATCH --array=0-50
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
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

SDPB_TIMEOUT="${SDPB_TIMEOUT:-18000}"  # 5 hours (measured SDPB solve time)
STAGE_B_TOLERANCE="${STAGE_B_TOLERANCE:-1e-3}"
EPS_SNAP_TOLERANCE="${EPS_SNAP_TOLERANCE:-1e-3}"

# --- Compute Δσ for this task ---
SIGMA_MIN=0.500
SIGMA_STEP=0.002
SIGMA=$(python3 -c "print(f'{${SIGMA_MIN} + ${SLURM_ARRAY_TASK_ID} * ${SIGMA_STEP}:.6f}')")

echo "=== Stage B (SDPB) ==="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Delta_sigma: ${SIGMA}"
echo "Timeout: ${SDPB_TIMEOUT}s"
echo "Tolerance: ${STAGE_B_TOLERANCE}"
echo "Epsilon snap tolerance: ${EPS_SNAP_TOLERANCE}"
echo "Start: $(date)"

# --- Run scan for this single Δσ ---
python -m ising_bootstrap.scans.stage_b \
    --eps-bound data/eps_bound.csv \
    --sigma-min "${SIGMA}" \
    --sigma-max "${SIGMA}" \
    --sigma-step 1.0 \
    --tolerance "${STAGE_B_TOLERANCE}" \
    --backend sdpb \
    --sdpb-image tools/sdpb-3.1.0.sif \
    --sdpb-precision 1024 \
    --sdpb-cores "${SLURM_CPUS_PER_TASK}" \
    --sdpb-timeout "${SDPB_TIMEOUT}" \
    --eps-snap-tolerance "${EPS_SNAP_TOLERANCE}" \
    --output "data/epsprime_bound_${SLURM_ARRAY_TASK_ID}.csv" \
    --verbose

echo "=== Done: $(date) ==="
