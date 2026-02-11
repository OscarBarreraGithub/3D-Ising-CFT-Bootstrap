#!/bin/bash
#SBATCH --job-name=ising_stage_b_smoke
#SBATCH --account=randall_lab
#SBATCH --partition=shared
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/stage_b_smoke_%j.log
#SBATCH --error=logs/stage_b_smoke_%j.log

# One-point Stage B smoke test using SDPB.
# Purpose: validate strict failure behavior and fixed-epsilon anchoring.

set -euo pipefail
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate ising_bootstrap

PROJECT_DIR="/n/holylabs/schwartz_lab/Lab/obarrera/3D-Ising-CFT-Bootstrap"
cd "$PROJECT_DIR"

mkdir -p logs data

SIGMA="${SIGMA:-0.518}"
EPS_BOUND_CSV="${EPS_BOUND_CSV:-data/eps_bound.csv}"
SDPB_TIMEOUT="${SDPB_TIMEOUT:-1800}"
STAGE_B_TOLERANCE="${STAGE_B_TOLERANCE:-1e-3}"
EPS_SNAP_TOLERANCE="${EPS_SNAP_TOLERANCE:-1e-3}"
OUTPUT_CSV="${OUTPUT_CSV:-data/stage_b_smoke.csv}"

echo "=========================================="
echo "Stage B SDPB Smoke Test"
echo "=========================================="
echo "Delta_sigma: ${SIGMA}"
echo "Stage A map: ${EPS_BOUND_CSV}"
echo "Timeout: ${SDPB_TIMEOUT}s"
echo "Tolerance: ${STAGE_B_TOLERANCE}"
echo "Epsilon snap tolerance: ${EPS_SNAP_TOLERANCE}"
echo "Output: ${OUTPUT_CSV}"
echo "Start: $(date)"
echo ""

python -m ising_bootstrap.scans.stage_b \
    --eps-bound "${EPS_BOUND_CSV}" \
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
    --output "${OUTPUT_CSV}" \
    --verbose

echo ""
echo "=== Smoke complete: $(date) ==="
cat "${OUTPUT_CSV}"
