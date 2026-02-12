#!/bin/bash
#SBATCH --job-name=ising_stage_a_sdpb
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
#SBATCH --array=0-50
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/stage_a_sdpb_%A_%a.log
#SBATCH --error=logs/stage_a_sdpb_%A_%a.log

# Stage A scan with SDPB backend.
# Each array task handles one Δσ value.

set -euo pipefail
export PYTHONUNBUFFERED=1

# --- Environment ---
source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

mkdir -p logs data

SDPB_TIMEOUT="${SDPB_TIMEOUT:-18000}"  # 5 hours (measured SDPB solve time)
STAGE_A_TOLERANCE="${STAGE_A_TOLERANCE:-1e-4}"

# --- Compute Δσ for this task ---
SIGMA_MIN=0.500
SIGMA_STEP=0.002
SIGMA=$(python3 -c "print(f'{${SIGMA_MIN} + ${SLURM_ARRAY_TASK_ID} * ${SIGMA_STEP}:.6f}')")

echo "=== Stage A (SDPB) ==="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Delta_sigma: ${SIGMA}"
echo "SDPB image: tools/sdpb-3.1.0.sif"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Timeout: ${SDPB_TIMEOUT}s"
echo "Tolerance: ${STAGE_A_TOLERANCE}"
echo "Start: $(date)"

# --- Run scan for this single Δσ ---
python -m ising_bootstrap.scans.stage_a \
    --sigma-min "${SIGMA}" \
    --sigma-max "${SIGMA}" \
    --sigma-step 1.0 \
    --tolerance "${STAGE_A_TOLERANCE}" \
    --backend sdpb \
    --sdpb-image tools/sdpb-3.1.0.sif \
    --sdpb-precision 1024 \
    --sdpb-cores "${SLURM_CPUS_PER_TASK}" \
    --sdpb-timeout "${SDPB_TIMEOUT}" \
    --output "data/eps_bound_${SLURM_ARRAY_TASK_ID}.csv" \
    --verbose

echo "=== Done: $(date) ==="
