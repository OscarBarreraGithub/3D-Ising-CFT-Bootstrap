#!/bin/bash
#SBATCH --job-name=ising_stage_a_pilot
#SBATCH --account=randall_lab
#SBATCH --partition=sapphire
#SBATCH --array=0,9,18,27,36
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/stage_a_pilot_%A_%a.log
#SBATCH --error=logs/stage_a_pilot_%A_%a.log

# Stage A pilot scan with SDPB backend.
# Default pilot points are spread across the scan range.
# Override the array at submit time if needed:
#   sbatch --array=0,16,25,32,50 jobs/stage_a_pilot_sdpb.slurm

set -euo pipefail
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

mkdir -p logs data

SDPB_TIMEOUT="${SDPB_TIMEOUT:-1800}"
STAGE_A_TOLERANCE="${STAGE_A_TOLERANCE:-1e-4}"

SIGMA_MIN=0.500
SIGMA_STEP=0.002
SIGMA=$(python3 -c "print(f'{${SIGMA_MIN} + ${SLURM_ARRAY_TASK_ID} * ${SIGMA_STEP}:.6f}')")

echo "=== Stage A Pilot (SDPB) ==="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Delta_sigma: ${SIGMA}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Timeout: ${SDPB_TIMEOUT}s"
echo "Tolerance: ${STAGE_A_TOLERANCE}"
echo "Start: $(date)"

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
    --output "data/eps_bound_pilot_${SLURM_ARRAY_TASK_ID}.csv" \
    --verbose

echo "=== Done: $(date) ==="
